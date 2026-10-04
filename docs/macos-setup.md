# Instalación en macOS

Probado en un MacBook Air M3 (8 GB) con macOS 27.0.1 y Python 3.14.8 arm64 en el `.venv`
existente. Todas las dependencias tenían wheels binarias arm64 para cp314 (ctranslate2,
onnxruntime, av, tokenizers, cffi), así que no hubo que compilar nada.

```bash
python3 -c "import platform,sys; print(sys.version, platform.machine())"   # debe ser arm64
source .venv/bin/activate
python -m pip install -e '.[voice,vision,dev]'
clawde doctor
```

## Ollama

- Abre la app Ollama o ejecuta `ollama serve`. CLAWDE solo acepta `http://127.0.0.1:11434`
  (loopback) y rechaza modelos `*cloud*`.
- `ollama list` debe mostrar el modelo configurado (`qwen3.5:4b`, 3.4 GB). Con 8 GB de RAM,
  ese modelo y Whisper base (~0.6 GB de RSS en el proceso) caben, pero cierra apps pesadas.
- Latencia observada con qwen3.5:4b: ~7–16 s por orden. No hay un tiempo garantizado. Las
  órdenes básicas se resuelven por reglas al instante.
- Para elegir otro modelo: `--model <tag>` o `ollama.model` en `config/app.yaml`. No asumas
  que un modelo admite visión: CLAWDE no envía imágenes al LLM.

## Permisos

*Ajustes del Sistema → Privacidad y seguridad → Cámara / Micrófono*: activa la app desde la
que lanzas CLAWDE (Terminal, iTerm, Zed). Tras cambiarlo, reinicia esa app.

## Zed

`.zed/tasks.json` incluye las tareas ayuda, doctor, simulación de texto y voz, cámara,
micrófono, tests y ruff (*task: spawn*). El resto de `.zed/` sigue ignorado para no versionar
preferencias personales.

## PlatformIO

`pio` está instalado, pero no hay `firmware/platformio.ini` activo hasta conocer la placa
(ver `firmware/README.md`).
