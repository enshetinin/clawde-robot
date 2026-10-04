"""Diagnóstico serie EXPLÍCITO: solo PING y STATUS. Nunca envía movimientos.

AVISO: muchas placas (Arduino y compatibles) se REINICIAN al abrir el puerto.
Asegúrate de que el brazo está en reposo y apoyado antes de ejecutar esto.

Uso: python scripts/test_serial.py --port /dev/cu.usbserial-XXXX --yes
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clawde.errors import ClawdeError  # noqa: E402
from clawde.robot.protocol import Command  # noqa: E402
from clawde.robot.serial_transport import SerialTransport, open_serial_port  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--port", required=True)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--boot-wait", type=float, default=2.0, help="espera tras abrir (reinicio)")
    parser.add_argument("--yes", action="store_true", help="confirmo que entiendo el aviso")
    args = parser.parse_args()
    if not args.yes:
        print("Añade --yes tras leer el aviso: la placa puede reiniciarse al abrir el puerto.")
        return 2
    transport = SerialTransport(open_serial_port(args.port, args.baudrate), ack_timeout_s=1.0)
    try:
        time.sleep(args.boot_wait)
        for command in (Command.PING, Command.STATUS):
            try:
                response = transport.send(command)
                print(f"{command}: {response.kind} {' '.join(response.fields)}")
            except ClawdeError as exc:
                print(f"{command}: fallo: {exc}")
        print(
            f"Respuestas obsoletas: {transport.stale_count}, "
            f"malformadas: {transport.malformed_count}"
        )
    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
