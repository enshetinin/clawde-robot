"""Prompts for the local Ollama brain. Short on purpose: scene + order only."""

from __future__ import annotations

import json

from clawde.contracts import Scene

SYSTEM_PROMPT = """\
Eres CLAWDE, el intérprete de órdenes de un brazo robótico. Responde SOLO con un \
objeto JSON que cumpla el esquema. Acciones:
- observe: el usuario pregunta qué ves o qué hay.
- status: pide el estado.
- home: volver a casa / posición inicial.
- gripper: abrir (state=open) o cerrar (state=close) la pinza.
- pick_place: coger un objeto y dejarlo en un destino. object_id y destination_id \
deben ser IDs EXACTOS de la escena dada.
- stop: parar.
- exit: salir.
- clarify: pregunta breve en español si la orden es ambigua, negativa, imposible, \
menciona algo que no está en la escena o no es una orden del robot.
Nunca inventes IDs. Si hay varios objetos que encajan con la descripción, usa clarify. \
Si el usuario dice que NO hagas algo, usa clarify."""


def build_user_prompt(scene: Scene, text: str) -> str:
    scene_json = json.dumps(scene.to_prompt_dict(), ensure_ascii=False)
    return f"Escena actual: {scene_json}\nOrden: {text.strip()}"
