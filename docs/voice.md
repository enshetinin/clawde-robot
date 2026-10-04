# Voz

## Flujo

Enter → espera a que `say` termine (más una pausa de 0.3 s) → graba N s en memoria →
comprueba el audio (vacío, corto, desbordamiento, silencio por RMS) → Whisper local →
comprueba la transcripción (texto vacío, `no_speech_prob`, `avg_logprob`, alucinaciones
conocidas) → muestra «Has dicho: …» → interpreta → ejecuta en simulación → responde
con texto y voz.

Si el audio o la transcripción se rechazan, se muestra el motivo y **no se mueve nada**.
Los umbrales de `config/voice.yaml` son heurísticas: no garantizan que la transcripción
sea correcta.

## Modelo

- faster-whisper `base` multilingüe, CPU, `int8`, idioma `es`. Los modelos `*.en` se rechazan.
  `small` es más preciso y más lento.
- Descarga explícita y única: `clawde models prepare --whisper base` (≈145 MB). Se guarda en
  `models/whisper/` (caché HF) y queda registrada en `models/whisper/manifest.json`.
- Durante el uso normal se carga con `local_files_only=True`. Si faltan los pesos, CLAWDE
  explica el comando en lugar de descargarlos sin avisar.
- El modelo se queda cargado entre órdenes. No se promete aceleración Metal con este backend.

Medido en M3 8 GB con audio generado por `say` (voz Paulina): carga 0.6–0.8 s,
transcripción de ~2 s de audio en ~0.6 s y RSS máximo ~630 MB. Transcripciones obtenidas:
«Claudia, ¿qué ves?», «Coge el objeto rojo y ponlo a la izquierda.», «Vuelve a casa.»,
«No sierres la pinza.», «Para.». Todas se interpretaron bien; «Claudia» y «sierra» están
contempladas en las reglas. **La captura con micrófono real queda pendiente de probar en tu
Mac.**

## Parada

«para», «alto», «detente», «stop» (también repetidas o con «ya» o «por favor») se resuelven
**antes** de Ollama. «No pares» no es STOP. La parada por voz arrastra la latencia de
grabación y transcripción: **no es una parada de emergencia**. Durante el procesamiento
siguen activos el teclado (`para` + Enter) y Ctrl+C. Ante peligro, corta la alimentación
del brazo.

## Síntesis

`/usr/bin/say -v <voz> -r <ritmo> -f -` con el texto por stdin (lista de argumentos, sin
shell). La voz se elige así: `tts.voice` si está instalada; si no, Mónica, Paulina u otra
`es_*`. `clawde devices voices` lista las disponibles. Si no hay ninguna voz española,
CLAWDE responde solo con texto. Para descargar una: *Ajustes → Accesibilidad → Contenido
leído → Voz del sistema → Gestionar voces*.

## Privacidad

No se guarda audio. No hay escucha permanente ni palabra de activación (ver hoja de ruta).
