"""Order interpretation: deterministic rules first, optional local Ollama.

STOP is always detected by rules, before and independently of any model.
"""

from __future__ import annotations

import logging
import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from clawde.brain.ollama_client import OllamaBrain
from clawde.brain.prompts import SYSTEM_PROMPT, build_user_prompt
from clawde.contracts import (
    ActionKind,
    ClarifyAction,
    ExitAction,
    GripperAction,
    GripperState,
    HomeAction,
    ObserveAction,
    PickPlaceAction,
    RearmAction,
    Scene,
    StatusAction,
    StopAction,
    is_motion,
    llm_action_schema,
    parse_llm_action,
)
from clawde.errors import ActionRejected, InterpreterError, Stopped

log = logging.getLogger(__name__)

WAKE_WORDS = {"clawde", "claude", "claudia", "clode", "cloud", "oye", "hey", "robot"}
STOP_WORDS = {"para", "parate", "alto", "detente", "stop", "quieto", "quieta", "basta"}
STOP_FILLER = {
    "ya",
    "ahora",
    "mismo",
    "por",
    "favor",
    "el",
    "brazo",
    "robot",
    "todo",
    "inmediatamente",
}
NEGATIONS = {"no", "nunca", "jamas"}

COLOR_WORDS = {
    "rojo": "red",
    "roja": "red",
    "rojos": "red",
    "rojas": "red",
    "red": "red",
    "azul": "blue",
    "azules": "blue",
    "blue": "blue",
    "verde": "green",
    "verdes": "green",
    "green": "green",
}
COLOR_LABELS = {"red": "rojo", "blue": "azul", "green": "verde"}
ZONE_WORDS = {
    "izquierda": "left",
    "izquierdo": "left",
    "left": "left",
    "derecha": "right",
    "derecho": "right",
    "right": "right",
    "centro": "center",
    "medio": "center",
    "central": "center",
    "center": "center",
}
ZONE_LABELS = {"left": "izquierda", "center": "centro", "right": "derecha"}
UNKNOWN_COLORS = {
    "amarillo",
    "amarilla",
    "negro",
    "negra",
    "blanco",
    "blanca",
    "naranja",
    "morado",
    "morada",
    "rosa",
    "gris",
}

PICK_VERBS = re.compile(
    r"\b(coge|coger|cogelo|cogela|agarra|agarrar|toma|tomar|recoge|recoger|pon|ponlo|ponla|"
    r"poner|lleva|llevar|llevalo|llevala|mueve|mover|muevelo|muevela|deja|dejalo|dejala)\b"
)
OPEN_RE = re.compile(
    r"\b(abre|abrir|abra|suelta|soltar)\b.*\b(pinza|mano|garra)\b|^(suelta|suelta lo)$"
)
# "sierra" covers Whisper seseo transcriptions of "cierra".
CLOSE_RE = re.compile(r"\b(cierra|cerrar|cierre|sierra|sierre)\b.*\b(pinza|mano|garra)\b")
HOME_RE = re.compile(r"\b(casa|inicio|home|reposo|posicion inicial)\b")
OBSERVE_RE = re.compile(r"\b(que ves|que hay|que objetos|observa|describe|ves algo|mira)\b")
STATUS_RE = re.compile(r"\b(estado|status)\b")
EXIT_RE = re.compile(r"^(salir|sal del programa|adios|terminar|termina|cerrar|apagate)$")
REARM_RE = re.compile(r"\b(rearma|rearmar|rearmate|rearme)\b")


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, drop leading wake words."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", stripped)
    tokens = cleaned.split()
    while tokens and tokens[0] in WAKE_WORDS:
        tokens.pop(0)
    return " ".join(tokens)


def is_stop_command(text: str) -> bool:
    """True only for explicit, non-negated stop utterances ('para', 'alto ya'...)."""
    tokens = normalize(text).split()
    if not tokens or NEGATIONS & set(tokens):
        return False
    vocabulary = STOP_WORDS | STOP_FILLER
    return all(t in vocabulary for t in tokens) and bool(STOP_WORDS & set(tokens))


@dataclass(frozen=True)
class ObjectQuery:
    colors: tuple[str, ...]
    unknown_colors: tuple[str, ...]
    zones: tuple[str, ...]
    number: int | None


def parse_object_query(normalized: str) -> ObjectQuery:
    tokens = normalized.split()
    colors: list[str] = []
    zones: list[str] = []
    number: int | None = None
    for i, tok in enumerate(tokens):
        if tok in COLOR_WORDS and COLOR_WORDS[tok] not in colors:
            colors.append(COLOR_WORDS[tok])
            nxt = tokens[i + 1 : i + 3]
            if nxt and nxt[0] == "numero":
                nxt = nxt[1:]
            if nxt and nxt[0].isdigit():
                number = int(nxt[0])
        elif tok in ZONE_WORDS:
            zones.append(ZONE_WORDS[tok])
    unknown = tuple(t for t in tokens if t in UNKNOWN_COLORS)
    return ObjectQuery(tuple(colors), unknown, tuple(zones), number)


def resolve_object(query: ObjectQuery, scene: Scene, object_zone: str | None) -> list[str]:
    """Return IDs of scene objects matching a single-color description."""
    if len(query.colors) != 1:
        return []
    candidates = scene.by_color(query.colors[0])
    if query.number is not None:
        candidates = [o for o in candidates if o.id == f"{query.colors[0]}_{query.number}"]
    if object_zone is not None:
        candidates = [o for o in candidates if o.zone == object_zone]
    return [o.id for o in candidates]


@dataclass(frozen=True)
class RuleOutcome:
    action: Any | None
    final: bool = False  # True: never forward to Ollama (control, negation, stop)


def _clarify(question: str) -> ClarifyAction:
    return ClarifyAction(action=ActionKind.CLARIFY, question=question)


def _pick_place(normalized: str, scene: Scene) -> RuleOutcome:
    query = parse_object_query(normalized)
    if query.unknown_colors and not query.colors:
        return RuleOutcome(_clarify("No reconozco ese color en la escena. ¿Qué objeto quieres?"))
    if len(query.colors) != 1 or len(query.zones) > 2:
        return RuleOutcome(None)  # too complex for rules
    color = query.colors[0]
    destination = query.zones[-1] if query.zones else None
    object_zone = query.zones[0] if len(query.zones) == 2 else None
    ids = resolve_object(query, scene, object_zone)
    label = COLOR_LABELS.get(color, color)
    if not ids:
        return RuleOutcome(_clarify(f"No veo ningún objeto {label} que encaje. ¿Cuál quieres?"))
    if len(ids) > 1:
        return RuleOutcome(
            _clarify(f"Hay varios objetos {label}: {', '.join(ids)}. ¿Cuál de ellos?")
        )
    if destination is None:
        return RuleOutcome(_clarify("¿Dónde lo dejo: izquierda, centro o derecha?"))
    if scene.destination(destination) is None:
        return RuleOutcome(_clarify("Ese destino no existe. Usa izquierda, centro o derecha."))
    return RuleOutcome(
        PickPlaceAction(action=ActionKind.PICK_PLACE, object_id=ids[0], destination_id=destination)
    )


def interpret_rules(text: str, scene: Scene) -> RuleOutcome:
    normalized = normalize(text)
    tokens = set(normalized.split())
    if not normalized:
        return RuleOutcome(_clarify("No he oído ninguna orden."), final=True)
    if is_stop_command(text):
        return RuleOutcome(StopAction(action=ActionKind.STOP), final=True)
    if tokens & NEGATIONS:
        return RuleOutcome(
            _clarify("Has pedido que no lo haga, así que no haré nada. ¿Qué quieres que haga?"),
            final=True,
        )
    if EXIT_RE.search(normalized):
        return RuleOutcome(ExitAction(action=ActionKind.EXIT), final=True)
    if REARM_RE.search(normalized):
        return RuleOutcome(RearmAction(action=ActionKind.REARM), final=True)
    if STATUS_RE.search(normalized):
        return RuleOutcome(StatusAction(action=ActionKind.STATUS), final=True)
    if OPEN_RE.search(normalized):
        return RuleOutcome(GripperAction(action=ActionKind.GRIPPER, state=GripperState.OPEN))
    if CLOSE_RE.search(normalized):
        return RuleOutcome(GripperAction(action=ActionKind.GRIPPER, state=GripperState.CLOSE))
    if PICK_VERBS.search(normalized):
        return _pick_place(normalized, scene)
    if HOME_RE.search(normalized):
        return RuleOutcome(HomeAction(action=ActionKind.HOME))
    if OBSERVE_RE.search(normalized):
        return RuleOutcome(ObserveAction(action=ActionKind.OBSERVE))
    return RuleOutcome(None)


def validate_against_scene(action: Any, scene: Scene, text: str) -> Any:
    """Semantic checks on a schema-valid action. Returns the (maybe clarified) action."""
    if action.action != ActionKind.PICK_PLACE:
        return action
    obj = scene.object(action.object_id)
    if obj is None:
        raise ActionRejected(f"el objeto '{action.object_id}' no existe en la escena")
    if scene.destination(action.destination_id) is None:
        raise ActionRejected(f"el destino '{action.destination_id}' no existe")
    query = parse_object_query(normalize(text))
    if query.colors and obj.color not in query.colors:
        raise ActionRejected(f"'{obj.id}' no es del color pedido")
    if len(query.colors) == 1:
        object_zone = query.zones[0] if len(query.zones) == 2 else None
        ids = resolve_object(query, scene, object_zone)
        if len(ids) > 1:
            return _clarify(f"Hay varios objetos que encajan: {', '.join(ids)}. ¿Cuál de ellos?")
        if ids and ids[0] != obj.id:
            raise ActionRejected("el objeto elegido no coincide con la descripción")
    return action


@dataclass(frozen=True)
class Interpretation:
    action: Any
    source: Literal["rules", "ollama"]
    note: str | None = None


class Interpreter:
    """Combine rules and Ollama.

    mode: "rules"  -> deterministic only.
          "hybrid" -> rules when they are sure, otherwise Ollama.
          "ollama" -> Ollama for everything except local control (stop, exit, status...).
    """

    def __init__(
        self,
        mode: Literal["rules", "hybrid", "ollama"],
        brain: OllamaBrain | None = None,
        max_retries: int = 1,
    ) -> None:
        if mode != "rules" and brain is None:
            raise ValueError("Ollama mode needs a brain")
        self.mode = mode
        self.brain = brain
        self.max_retries = max(0, min(1, max_retries))
        self._schema = llm_action_schema()

    @property
    def label(self) -> str:
        if self.mode == "rules":
            return "reglas deterministas"
        model = self.brain.config.model if self.brain else "?"
        return f"Ollama ({model}) + reglas" if self.mode == "hybrid" else f"Ollama ({model})"

    def interpret(
        self, text: str, scene: Scene, cancel: threading.Event | None = None
    ) -> Interpretation:
        outcome = interpret_rules(text, scene)
        if outcome.final:
            return Interpretation(outcome.action, "rules")
        if self.mode == "rules" or (self.mode == "hybrid" and outcome.action is not None):
            action = outcome.action or _clarify(
                "No he entendido la orden. Prueba: «¿qué ves?», «coge el objeto rojo y ponlo "
                "a la izquierda», «vuelve a casa», «abre la pinza»."
            )
            return Interpretation(action, "rules")
        return self._ask_ollama(text, scene, outcome, cancel)

    def _ask_ollama(
        self, text: str, scene: Scene, outcome: RuleOutcome, cancel: threading.Event | None
    ) -> Interpretation:
        assert self.brain is not None
        last_error = "sin respuesta"
        for attempt in range(1 + self.max_retries):
            if cancel is not None and cancel.is_set():
                raise Stopped("interpretación cancelada")
            try:
                raw = self.brain.chat_json(
                    SYSTEM_PROMPT, build_user_prompt(scene, text), self._schema
                )
                if cancel is not None and cancel.is_set():
                    raise Stopped("respuesta descartada tras STOP")
                action = validate_against_scene(parse_llm_action(raw), scene, text)
                return Interpretation(action, "ollama")
            except (InterpreterError, ActionRejected) as exc:
                last_error = str(exc)[:200]
                log.warning("Ollama attempt %d failed: %s", attempt + 1, last_error)
        if outcome.action is not None and not is_motion(outcome.action):
            return Interpretation(outcome.action, "rules", note=f"Ollama falló ({last_error})")
        raise InterpreterError(f"no ejecuto nada: {last_error}")
