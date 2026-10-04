"""Prueba de integración MANUAL con Ollama local (sin mover nada).

Envía órdenes de ejemplo sobre la escena demo y muestra la acción validada.
Uso: python scripts/test_ollama.py [--model qwen3.5:4b]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clawde.brain.interpreter import Interpreter  # noqa: E402
from clawde.brain.ollama_client import OllamaBrain  # noqa: E402
from clawde.config import OllamaConfig, load_settings  # noqa: E402
from clawde.errors import InterpreterError  # noqa: E402
from clawde.vision.scene import DemoSceneProvider  # noqa: E402

ORDERS = [
    "Clawde, ¿qué ves?",
    "coge el objeto rojo y ponlo a la izquierda",
    "lleva la pieza roja hacia el lado derecho",
    "coge el objeto azul y ponlo en el centro",  # dos azules: debe pedir aclaración
    "coge el objeto verde y ponlo a la derecha",  # no hay verde
    "no cierres la pinza",
    "cuéntame un chiste",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    args = parser.parse_args()
    settings = load_settings()
    cfg = settings.app.ollama
    if args.model:
        cfg = OllamaConfig.model_validate({**cfg.model_dump(), "model": args.model})
    brain = OllamaBrain(cfg)
    probe = brain.probe()
    if not probe.reachable:
        print(f"Ollama no responde en {cfg.host}: {probe.error}")
        return 1
    if not probe.has_model(cfg.model):
        print(f"Modelo {cfg.model} no instalado. Instalados: {', '.join(probe.models)}")
        return 1
    scene = DemoSceneProvider(settings.robot.demo_profile).get_scene()
    interpreter = Interpreter("ollama", brain, cfg.max_retries)
    print(f"Modelo: {cfg.model} · escena DEMO ficticia · sin movimiento\n")
    for order in ORDERS:
        start = time.monotonic()
        try:
            result = interpreter.interpret(order, scene)
            outcome = f"{result.action.model_dump(mode='json')} [{result.source}]"
            if result.note:
                outcome += f" nota: {result.note}"
        except InterpreterError as exc:
            outcome = f"RECHAZADO: {exc}"
        print(f"«{order}»\n  → {outcome} ({time.monotonic() - start:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
