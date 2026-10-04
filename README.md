# clawde-robot

CLAWDE entiende órdenes en español por **voz y texto**, detecta **manchas de color**
con una webcam y prepara acciones para un brazo **Adeept**. Todo funciona en local:
Ollama como cerebro, faster-whisper para transcribir y `say` de macOS para hablar.
No necesita APIs de pago, claves ni cuentas.

> **Estado actual: simulación.** El brazo real **no se mueve**. La placa, los pines,
> la geometría y la calibración del Adeept están pendientes, así que el modo `--real`
> se bloquea antes de enviar ningún movimiento. Ver [Funciones](#funciones-listas-y-pendientes).

## Instalación (macOS, incremental)

Requisitos: Python 3.12+ (probado con 3.14.8 arm64), Ollama y, para el firmware, PlatformIO.
La primera vez hace falta Internet para descargar paquetes y modelos. Después todo
funciona en local con los pesos ya descargados. Las voces del sistema que no tengas
descargadas no estarán disponibles sin conexión.

```bash
cd clawde-robot
source .venv/bin/activate              # usa el entorno existente

# 1. Núcleo: CLI, simulación con texto y tests
python -m pip install -e '.[dev]'

# 2. Visión (OpenCV contrib + NumPy)
python -m pip install -e '.[vision,dev]'

# 3. Voz (sounddevice + faster-whisper)
python -m pip install -e '.[voice,vision,dev]'
clawde models prepare --whisper base   # descarga única ≈145 MB (small ≈480 MB)

# 4. Cerebro local
ollama serve                           # o abre la app Ollama
ollama list                            # el modelo por defecto es qwen3.5:4b
clawde doctor
```

`requirements.txt` y `requirements-dev.txt` solo apuntan a `pyproject.toml`, que es
la única lista de dependencias.

### Permisos de macOS

La primera vez que uses la cámara o el micrófono, macOS pedirá permiso para la
aplicación que ejecuta Python (Terminal, iTerm o **Zed**). Si los denegaste:
*Ajustes del Sistema → Privacidad y seguridad → Cámara / Micrófono* y activa esa app.
Después reiníciala.

## Comandos

```bash
clawde --help
clawde doctor                                   # diagnóstico; no abre cámara ni micrófono
clawde devices audio | serial | voices          # listar sin abrir
clawde camera --index 0                         # vista previa con etiquetas (q/Esc)
clawde voice-test --seconds 5                   # grabar + transcribir una vez
clawde run --sim --demo --text                  # escena ficticia, teclado
clawde run --sim --demo --voice                 # escena ficticia, Enter para hablar
clawde run --sim --camera 0 --voice             # cámara real, brazo simulado
clawde command --sim --demo "coge el objeto rojo y ponlo a la izquierda"
python -m pytest                                # tests (sin red, pesos ni hardware)
python -m pytest --integration                  # + prueba real con Ollama (opt-in)
python -m ruff check .
```

Opciones útiles: `--interpreter auto|rules|ollama`, `--model <tag>`, `--speak`
(leer respuestas), `--no-tts`, `--seconds N`, `--save-captures`, `--config-dir DIR`.
Las opciones de la CLI tienen prioridad sobre los YAML.

### Ejemplo de sesión

```text
$ clawde run --sim --demo --text
───── MODO DEMO: escena FICTICIA (los objetos y posiciones no son reales) ─────
Modo: simulación · Intérprete: Ollama (qwen3.5:4b) + reglas
> ¿qué ves?
CLAWDE: [MODO DEMO…] Veo 3 objeto(s): red_1 (mancha rojo, centro…); blue_1 (…izquierda…); blue_2 (…derecha…)
> coge el objeto azul y ponlo en el centro
CLAWDE: Hay varios objetos azul: blue_1, blue_2. ¿Cuál de ellos?
> coge el objeto rojo y ponlo a la izquierda
CLAWDE: Simulación: he dejado red_1 en izquierda.
> no cierres la pinza
CLAWDE: Has pedido que no lo haga, así que no haré nada. ¿Qué quieres que haga?
> para
CLAWDE: PARADA por software. No es una parada de emergencia física… Di «rearmar» para continuar.
> rearmar
> salir
```

Frases reconocidas sin Ollama: «¿qué ves?», «coge el objeto rojo y ponlo a la
izquierda» (también «el rojo de la derecha», «el rojo 2»), «vuelve a casa»,
«abre/cierra la pinza», «estado», «para / alto / detente / stop», «rearmar», «salir».

### Voz (pulsar para hablar)

En `run --voice`, **Enter** graba durante N segundos (5 por defecto), transcribe en
local, muestra el texto, lo interpreta y lo ejecuta en simulación. Si escribes texto
en lugar de pulsar Enter, se trata como una orden escrita. `c` cancela la grabación y
`para` o Ctrl+C detienen la ejecución en cualquier momento, también mientras se
transcribe o mientras Ollama responde. CLAWDE no graba mientras está hablando.
No hay escucha permanente ni palabra de activación. Detalles en [docs/voice.md](docs/voice.md).

> **La parada por voz o teclado es por software y tiene latencia.** No es una parada
> de emergencia. Con el brazo real, ten siempre a mano el **corte de alimentación**.

## Demo, simulación y real

| | `--demo` | `--sim --camera N` | `--real` |
|---|---|---|---|
| Escena | ficticia (aviso permanente) | detecciones reales de la cámara | cámara + calibración |
| Brazo | simulado | simulado (los objetos reales no se mueven) | **bloqueado** hasta completar el hardware |
| Puerto serie | nunca se abre | nunca se abre | solo tras validar la config completa |

## Funciones listas y pendientes

| Función | Estado |
|---|---|
| CLI, configuración YAML, `doctor` | ✅ |
| Intérprete determinista (reglas, negaciones, STOP sin inferencia) | ✅ |
| Ollama local con JSON Schema + validación Pydantic y semántica | ✅ (probado con qwen3.5:4b) |
| Simulador con cola limitada, STOP, generaciones y rearme explícito | ✅ |
| Detección HSV (rojo con 2 bandas, azul, verde) y tracking de IDs | ✅ (tests sintéticos) |
| Homografía de mesa (≥4 puntos, guardar/cargar, control de resolución) | ✅ (tests) |
| Voz: grabación, faster-whisper base CPU int8, `say` en español | ✅ (Whisper probado con audio de `say`) |
| Protocolo serie v1 + transporte Python con puerto simulado | ✅ (tests) |
| Firmware de diagnóstico (PING/STATUS/ARM/DISARM/STOP, sin servos) | ⚠️ sin compilar: falta la placa |
| Captura real de micrófono y cámara | ⏳ pendiente de probar en tu Mac (comandos arriba) |
| Placa Adeept, pines, límites, home, geometría, cinemática, calibración | ❌ pendiente |
| Pick & place físico | ❌ bloqueado |

## Configuración

- `config/app.yaml`: Ollama (host loopback, modelo, timeout, temperatura), intérprete,
  tamaño de cola, antigüedad máxima de la escena y captura opcional.
- `config/voice.yaml`: duración, micrófono, umbrales, modelo Whisper (`base`/`small`),
  idioma y voz TTS.
- `config/vision.yaml`: cámara, rangos HSV, áreas, tracking y archivo de calibración.
- `config/robot.yaml`: driver, hardware **pendiente** y un `demo_profile` separado
  para el simulador.

Las rutas relativas se resuelven desde la raíz del proyecto, no desde el directorio
actual. No se necesitan `.env` ni claves.

## Solución de problemas

| Síntoma | Solución |
|---|---|
| «Ollama no responde» | Abre Ollama o ejecuta `ollama serve`. Mientras tanto se usan las reglas |
| «modelo no instalado» | `ollama pull qwen3.5:4b` o `--model <tag instalado>` |
| «No hay pesos locales de Whisper» | `clawde models prepare --whisper base` |
| La cámara no abre o da imagen negra | Revisa el permiso de Cámara de Terminal/Zed, prueba `--index 1` y cierra otras apps que usen la cámara |
| «solo se ha detectado silencio» | Revisa el permiso de Micrófono, elige dispositivo con `clawde devices audio` y ajusta `recorder.min_rms` |
| «transcripción poco fiable» | Habla más cerca, usa `model: small` o ajusta los umbrales (son heurísticas) |
| No habla | `clawde devices voices`; descarga una voz en *Accesibilidad → Contenido leído* |
| «Cola llena» | Espera a que termine la orden actual (la cola está limitada a propósito) |
| `--real` bloqueado | Es lo esperado. Ver [docs/calibration.md](docs/calibration.md) |

Documentación: [arquitectura](docs/architecture.md) · [macOS](docs/macos-setup.md) ·
[voz](docs/voice.md) · [calibración](docs/calibration.md) ·
[protocolo serie](docs/serial-protocol.md) · [hoja de ruta](docs/roadmap.md) ·
[firmware](firmware/README.md)
