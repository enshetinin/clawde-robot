# Protocolo serie CLAWDE v1

Líneas ASCII terminadas en `\n`, de **64 bytes como máximo** (salto incluido). 115200 baudios.

| Dirección | Formato |
|---|---|
| host → placa | `<id> <CMD>[ <arg>...]` (`id` 1–65535, args `[A-Z0-9_.-]{1,16}`) |
| placa → host | `<id> ACK`: aceptado (**no** significa éxito) |
| | `<id> DONE[ datos]`: terminado con éxito |
| | `<id> ERR <CÓDIGO>[ detalle]`: rechazado o fallido |
| | `0 EVT <NOMBRE>[ datos]`: evento espontáneo (`BOOT`, `WATCHDOG`, `MALFORMED`, `TOO_LONG`) |

## Comandos

| Comando | Efectos | Firmware de diagnóstico |
|---|---|---|
| `PING` | no | `DONE PONG CLAWDE/1` |
| `STATUS` | no | `DONE <DISARMED/ARMED/STOPPED> SERVOS=NONE` |
| `STOP` | detiene | siempre aceptado; latch `STOPPED` |
| `ARM` / `DISARM` | sí | estado lógico, sin servos |
| `HOME`, `GRIP OPEN/CLOSE`, `MOVE <cdeg...> <ms>` | movimiento | `ERR NOT_CONFIGURED` |

## Reglas del host (`serial_transport.py`)

- Espera ACK (timeout corto) y luego DONE (timeout largo). Comprueba los IDs. Las respuestas
  con otro ID se descartan como obsoletas y las malformadas o demasiado largas se cuentan y
  se ignoran.
- Un comando con efectos que se queda **sin ACK o sin DONE** deja el estado como
  **desconocido** y **no se retransmite**. Los siguientes comandos con efectos se rechazan
  hasta un `ARM` explícito.
- `STOP` se escribe de inmediato aunque otro comando esté esperando DONE. Ese comando termina
  con `Stopped`. Después hace falta `ARM`.
- Un error de lectura o escritura marca el enlace como desconectado y el estado como
  desconocido. No hay reconexión automática.
- `0 EVT WATCHDOG` activa el latch de parada en el host.

## Firmware

Lectura no bloqueante con un buffer de 64 bytes. Rechaza líneas largas y bytes no ASCII. Sin
`delay()`. El watchdog de comunicación (2 s sin tráfico) desarma, emite `EVT WATCHDOG` y nunca
inicia una trayectoria. Con el brazo armado, el host debe enviar `PING`/`STATUS` periódicamente.
Abrir el puerto puede **reiniciar** muchas placas; la primera línea será `0 EVT BOOT CLAWDE/1`.
