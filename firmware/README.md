# Firmware CLAWDE (diagnóstico, sin servos)

Este firmware implementa el protocolo serie v1 (`docs/serial-protocol.md`) **sin
controlar servos**. Sirve para verificar la placa, el puerto, el cable y el
protocolo antes de tocar motores.

| Comando | Respuesta |
|---|---|
| `PING` | `ACK`, `DONE PONG CLAWDE/1` |
| `STATUS` | `ACK`, `DONE <DISARMED/ARMED/STOPPED> SERVOS=NONE` |
| `ARM` / `DISARM` | cambia un estado lógico (no hay servos conectados) |
| `STOP` | siempre aceptado; estado `STOPPED` |
| `HOME`, `GRIP`, `MOVE` | `ERR NOT_CONFIGURED` |

Características: lectura serie no bloqueante, buffer limitado a 64 bytes, rechazo
de líneas largas o no ASCII, sin `delay()`, watchdog de comunicación (2 s) que
desarma sin ejecutar trayectorias, y evento `0 EVT BOOT` al arrancar.

## Estado

- **Placa: PENDIENTE.** No hay `platformio.ini` activo a propósito; no se asume Arduino Uno.
- **Compilación real: NO validada** (falta la placa). En el desarrollo solo se hizo
  una comprobación de sintaxis con `clang++ -fsyntax-only` contra un `Arduino.h` simulado.
- `include/robot_config.h.example` lista los datos obligatorios (pines, límites,
  home, velocidad, comportamiento de par tras STOP) y contiene `#error` hasta rellenarlo.
- Definir `CLAWDE_ENABLE_SERVOS` provoca un `#error`: el control de servos no existe aún.

## Cuando conozcas la placa

```bash
cd firmware
cp platformio.ini.example platformio.ini   # rellena platform y board
pio run                                    # compilar
pio run -t upload                          # subir (con el brazo apoyado y en reposo)
cd .. && python scripts/test_serial.py --port /dev/cu.XXXX --yes   # PING + STATUS
```

Muchas placas se **reinician al abrir el puerto serie**. La primera línea que
recibirás será `0 EVT BOOT CLAWDE/1`.
