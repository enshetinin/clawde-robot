# Arquitectura

```text
micrófono ─► Recorder ─► check_audio ─► Whisper local ─► assess_transcription ─┐
teclado ───────────────────────────────────────────────────────────────────────┤
                                                                               ▼
                         ┌──────── is_stop_command ──► emergency_stop (inmediato)
                         │
ClawdeApp.submit(texto) ─┴─► cola limitada (Job con generación)
                                   │ worker
                                   ▼
          SceneProvider.get_scene()  (demo ficticia | cámara + HSV + tracking [+ homografía])
                                   ▼
          Interpreter: reglas ──(dudosas)──► Ollama format=JSON Schema ─► Pydantic ─► validación semántica
                                   ▼                        (¿generación vigente? si no, se descarta)
          execute(): consulta / clarify / rearm / movimiento
                                   ▼
          Planner (escena fresca, IDs, zona, pinza libre; modo real ⇒ check_real_motion_ready)
                                   ▼
          ArmDriver: SimulatedArm (pasos con CancelToken) | SerialArm (bloqueado sin cinemática)
                                   ▼
          Reply (basado en el resultado real) ─► consola + say
```

## Seguridad por capas

1. **Contratos**: unión discriminada, `extra='forbid'`, IDs con patrón, enums. No hay campos
   para pines, ángulos ni comandos.
2. **Semántica**: el objeto y el destino existen, el color coincide con la orden y no hay
   ambigüedad (dos rojos ⇒ `clarify`).
3. **Estado**: con STOP activo solo se aceptan consultas y `rearmar`. La escena caducada o el
   tracking ambiguo se rechazan.
4. **Hardware**: `--real` exige driver serie, placa, puerto, servos con límites, geometría,
   espacio de trabajo, política de par tras STOP, `calibrated: true` y un solucionador IK
   verificado. Hoy no hay ninguno, así que falla antes de abrir el puerto.
5. **Transporte**: ACK solo significa «aceptado» y el éxito requiere DONE. Sin DONE el estado
   pasa a desconocido y no se reenvía. Una desconexión implica parada sin reconexión.

## Concurrencia

- El hilo principal lee el teclado (hilo lector de stdin y cola). Ctrl+C llega siempre al
  hilo principal.
- Un *worker* procesa las órdenes. La voz graba y transcribe en otro hilo.
- STOP incrementa la generación, cancela el `CancelToken` actual, vacía la cola, detiene el
  brazo (latch), corta `say` y cancela la grabación. Los resultados de Ollama, Whisper o del
  simulador de una generación anterior se descartan.
- Ni Ollama ni Whisper se pueden interrumpir a mitad de cálculo, pero su resultado se ignora.
