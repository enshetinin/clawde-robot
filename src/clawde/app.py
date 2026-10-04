"""Application core: bounded queue, session generations, STOP latch and execution.

Flow: text -> (STOP fast path) -> queue -> interpreter -> validation -> planner
-> arm driver -> reply built from the real result.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from clawde.brain.interpreter import COLOR_LABELS, ZONE_LABELS, Interpreter, is_stop_command
from clawde.config import Settings
from clawde.contracts import ActionKind, Scene, is_motion
from clawde.errors import (
    ActionRejected,
    ClawdeError,
    HardwareNotConfigured,
    InterpreterError,
    SafetyError,
    Stopped,
)
from clawde.motion.planner import Planner
from clawde.robot.base import ArmDriver, ArmState, CancelToken
from clawde.vision.scene import SceneProvider

log = logging.getLogger(__name__)

STOP_NOTICE = (
    "PARADA por software. No es una parada de emergencia física: ante peligro, "
    "corta la alimentación del brazo."
)


@dataclass(frozen=True)
class Reply:
    text: str
    ok: bool = True
    action: str | None = None
    source: str | None = None
    exit: bool = False
    speak: bool = True


@dataclass(frozen=True)
class Job:
    text: str
    generation: int
    origin: str


class ClawdeApp:
    def __init__(
        self,
        settings: Settings,
        scene_provider: SceneProvider,
        arm: ArmDriver,
        interpreter: Interpreter,
        planner: Planner,
        on_reply: Callable[[Reply], None] | None = None,
        on_stop: Callable[[], None] | None = None,
        speaker: Any = None,
    ) -> None:
        self.settings = settings
        self.scene_provider = scene_provider
        self.arm = arm
        self.interpreter = interpreter
        self.planner = planner
        self.speaker = speaker
        self.on_reply = on_reply or (lambda reply: None)
        self.on_stop = on_stop  # e.g. cancel an ongoing recording
        self.stopped = False
        self.generation = 0
        self._token = CancelToken(0)
        self._lock = threading.RLock()
        self._queue: queue.Queue[Job | None] = queue.Queue(maxsize=settings.app.queue_maxsize)
        self._worker: threading.Thread | None = None
        self._closed = False
        self.busy = threading.Event()
        self.exit_requested = threading.Event()

    # -------------------------------------------------------------- lifecycle

    @property
    def mode_label(self) -> str:
        return "simulación" if self.arm.simulated else "REAL"

    def start(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(
                target=self._run_worker, name="clawde-worker", daemon=True
            )
            self._worker.start()

    def shutdown(self) -> None:
        """Release resources. Never moves the arm."""
        if self._closed:
            return
        self._closed = True
        self._drain_queue()
        if self._worker is not None:
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(None)
            self._worker.join(timeout=2.0)
        for closer in (
            getattr(self.speaker, "stop", None),
            self.scene_provider.close,
            self.arm.close,
        ):
            if closer is None:
                continue
            try:
                closer()
            except Exception as exc:  # cleanup must continue
                log.warning("error during shutdown: %s", exc)

    # ------------------------------------------------------------------ input

    def submit(self, text: str, origin: str = "texto") -> Reply | None:
        """Queue an order. STOP is handled immediately, without queueing or inference."""
        if is_stop_command(text):
            return self.emergency_stop(origin)
        with self._lock:
            job = Job(text, self.generation, origin)
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            return Reply("Cola llena: orden descartada. Espera a que termine la actual.", ok=False)
        return None

    def handle(self, text: str, origin: str = "texto") -> Reply:
        """Synchronous processing (single-shot commands and tests)."""
        if is_stop_command(text):
            return self.emergency_stop(origin)
        with self._lock:
            job = Job(text, self.generation, origin)
        return self._process(job) or Reply("Orden descartada tras STOP.", ok=False)

    def emergency_stop(self, origin: str = "teclado") -> Reply:
        with self._lock:
            self.generation += 1
            self._token.cancel()
            self._token = CancelToken(self.generation)
            self.stopped = True
        self._drain_queue()
        errors = []
        for action in (self.arm.stop, getattr(self.speaker, "stop", None), self.on_stop):
            if action is None:
                continue
            try:
                action()
            except Exception as exc:
                errors.append(str(exc))
        log.warning("STOP from %s (generation %d)", origin, self.generation)
        text = f"{STOP_NOTICE} Di «rearmar» para continuar."
        if errors:
            text += f" Aviso: {'; '.join(errors)}"
        return Reply(text, ok=not errors, action=ActionKind.STOP.value, source="local")

    # ----------------------------------------------------------------- worker

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            self._queue.task_done()

    def wait_idle(self, timeout_s: float) -> bool:
        """Wait until queued orders are processed (e.g. stdin closed). False on timeout."""
        deadline = time.monotonic() + timeout_s
        while self._queue.unfinished_tasks and not self.exit_requested.is_set():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def _run_worker(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            self.busy.set()
            try:
                reply = self._process(job)
            except Exception as exc:  # never kill the worker
                log.exception("unexpected error")
                reply = Reply(f"Error inesperado: {exc}. No se ha ejecutado nada más.", ok=False)
            finally:
                self.busy.clear()
            try:
                if reply is not None:
                    self.on_reply(reply)
                    if reply.exit:
                        self.exit_requested.set()
            finally:
                self._queue.task_done()

    def _current(self, generation: int) -> CancelToken | None:
        with self._lock:
            return self._token if generation == self.generation else None

    def _process(self, job: Job) -> Reply | None:
        token = self._current(job.generation)
        if token is None:
            log.info("discarding stale job: %s", job.text)
            return None
        try:
            scene = self.scene_provider.get_scene()
            result = self.interpreter.interpret(job.text, scene, token.event)
        except Stopped:
            return None
        except InterpreterError as exc:
            return Reply(f"No he podido interpretar la orden de forma segura; {exc}", ok=False)
        except ClawdeError as exc:
            return Reply(f"Error: {exc}", ok=False)
        if self._current(job.generation) is None:
            log.info("discarding late interpretation after STOP: %s", job.text)
            return None
        reply = self.execute(result.action, scene, token, result.source)
        if result.note:
            reply = Reply(
                f"{reply.text} ({result.note})",
                reply.ok,
                reply.action,
                reply.source,
                reply.exit,
                reply.speak,
            )
        return reply

    # -------------------------------------------------------------- execution

    def execute(self, action: Any, scene: Scene, token: CancelToken, source: str) -> Reply:
        kind = action.action
        if kind == ActionKind.STOP:
            return self.emergency_stop(source)
        if kind == ActionKind.EXIT:
            return Reply(
                "Hasta luego. Cerrando CLAWDE.", action=kind.value, source=source, exit=True
            )
        if kind == ActionKind.CLARIFY:
            return Reply(action.question, ok=False, action=kind.value, source=source)
        if kind == ActionKind.OBSERVE:
            return Reply(self.describe_scene(scene), action=kind.value, source=source)
        if kind == ActionKind.STATUS:
            return Reply(self.status_text(), action=kind.value, source=source)
        if kind == ActionKind.REARM:
            return self.rearm()
        if is_motion(action):
            return self._move(action, token, source)
        return Reply("Acción no soportada.", ok=False)

    def rearm(self) -> Reply:
        with self._lock:
            if not self.stopped:
                return Reply("No estaba detenido; no hace falta rearmar.", action="rearm")
            try:
                self.arm.rearm()
                state = self.arm.status().state
                if state != ArmState.IDLE:
                    return Reply(
                        f"Rearme rechazado: estado {state.value}.", ok=False, action="rearm"
                    )
                self.scene_provider.get_scene()  # revalidate perception before continuing
            except ClawdeError as exc:
                return Reply(f"Rearme fallido: {exc}", ok=False, action="rearm")
            self.stopped = False
        return Reply(f"Rearmado ({self.mode_label}). Puedes dar nuevas órdenes.", action="rearm")

    def _move(self, action: Any, token: CancelToken, source: str) -> Reply:
        kind = action.action.value
        if self.stopped:
            return Reply("Estoy detenido. Di «rearmar» antes de mover.", ok=False, action=kind)
        try:
            # Fresh scene right before planning: interpretation may have taken a while.
            scene = self.scene_provider.get_scene()
            plan = self.planner.plan(action, scene, self.arm.status())
            token.check()
            self.arm.execute(plan, token)
        except Stopped:
            return Reply("Movimiento interrumpido por STOP.", ok=False, action=kind)
        except HardwareNotConfigured as exc:
            return Reply(f"Movimiento bloqueado: {exc}", ok=False, action=kind)
        except (ActionRejected, SafetyError) as exc:
            return Reply(f"No ejecuto: {exc}", ok=False, action=kind, source=source)
        except ClawdeError as exc:
            self.emergency_stop("error de driver")
            return Reply(f"Error del brazo: {exc}. Me he detenido.", ok=False, action=kind)
        prefix = "Simulación: " if self.arm.simulated else ""
        if action.action == ActionKind.PICK_PLACE:
            dest = ZONE_LABELS.get(action.destination_id, action.destination_id)
            updated = self.scene_provider.apply_pick_place(action.object_id, action.destination_id)
            text = f"{prefix}he dejado {action.object_id} en {dest}."
            if self.arm.simulated and not updated:
                text += " El objeto real no se ha movido."
            return Reply(text, action=kind, source=source)
        if action.action == ActionKind.HOME:
            return Reply(f"{prefix}en posición de casa.", action=kind, source=source)
        verb = "abierta" if action.state.value == "open" else "cerrada"
        return Reply(f"{prefix}pinza {verb}.", action=kind, source=source)

    # ---------------------------------------------------------------- reports

    def describe_scene(self, scene: Scene) -> str:
        prefix = f"[{self.scene_provider.warning}] " if self.scene_provider.warning else ""
        if not scene.objects:
            return prefix + "No veo objetos de los colores configurados."
        parts = []
        for obj in scene.objects:
            color = COLOR_LABELS.get(obj.color, obj.color)
            zone = ZONE_LABELS.get(obj.zone or "", "?")
            where = f"{zone}, {obj.center_px[0]:.0f},{obj.center_px[1]:.0f} px"
            if obj.position_mm and scene.source != "demo":
                where += f", {obj.position_mm[0]:.0f},{obj.position_mm[1]:.0f} mm"
            parts.append(f"{obj.id} (mancha {color}, {where})")
        text = f"{prefix}Veo {len(scene.objects)} objeto(s): " + "; ".join(parts) + "."
        if scene.source == "camera" and not scene.calibrated:
            text += " Sin calibración de mesa: solo posiciones en píxeles."
        if scene.tracking_ambiguous:
            text += " Seguimiento ambiguo: no planificaré movimientos hasta tener una escena clara."
        return text

    def status_text(self) -> str:
        arm = self.arm.status()
        lines = [
            f"Modo: {self.mode_label}"
            + (f" — {self.scene_provider.warning}" if self.scene_provider.warning else ""),
            f"Intérprete: {self.interpreter.label}",
            f"Brazo: driver {arm.driver}, estado {arm.state.value}"
            + (f", pinza {arm.gripper.value}" if arm.gripper else "")
            + (f", posición {arm.location}" if arm.location else "")
            + (f", sostiene {arm.holding}" if arm.holding else ""),
            "Detenido: sí (requiere «rearmar»)" if self.stopped else "Detenido: no",
            f"Órdenes en cola: {self._queue.qsize()}/{self.settings.app.queue_maxsize}",
        ]
        lines.extend(arm.notes)
        return "\n".join(lines)
