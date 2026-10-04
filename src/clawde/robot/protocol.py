"""CLAWDE serial protocol v1: short ASCII lines with command IDs.

Host -> device:  "<id> <CMD>[ <arg>...]\\n"      id = 1..65535
Device -> host:  "<id> ACK"                      accepted (NOT success)
                 "<id> DONE[ <data>...]"         finished successfully
                 "<id> ERR <CODE>[ <detail>...]" rejected or failed
                 "0 EVT <NAME>[ <data>...]"      unsolicited event (e.g. WATCHDOG)

Lines are at most MAX_LINE bytes including the newline. See docs/serial-protocol.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

PROTOCOL_VERSION = 1
PROTOCOL_NAME = f"CLAWDE/{PROTOCOL_VERSION}"
MAX_LINE = 64
MAX_ID = 65535


class Command(StrEnum):
    PING = "PING"
    STATUS = "STATUS"
    STOP = "STOP"
    ARM = "ARM"
    DISARM = "DISARM"
    HOME = "HOME"
    GRIP = "GRIP"
    MOVE = "MOVE"


# Commands with physical side effects: never retransmitted automatically.
EFFECTFUL = frozenset({Command.ARM, Command.HOME, Command.GRIP, Command.MOVE})
# Commands that move servos: blocked in Python unless hardware is fully validated.
MOTION = frozenset({Command.HOME, Command.GRIP, Command.MOVE})


class Kind(StrEnum):
    ACK = "ACK"
    DONE = "DONE"
    ERR = "ERR"
    EVT = "EVT"


class ProtocolError(ValueError):
    pass


_ARG_RE = re.compile(r"^[A-Z0-9_.\-]{1,16}$")
_LINE_RE = re.compile(r"^(\d{1,5}) (ACK|DONE|ERR|EVT)((?: [\x21-\x7e]+)*)$")


@dataclass(frozen=True)
class Response:
    cmd_id: int
    kind: Kind
    fields: tuple[str, ...] = ()

    @property
    def code(self) -> str | None:
        return self.fields[0] if self.kind in (Kind.ERR, Kind.EVT) and self.fields else None


def encode_command(cmd_id: int, command: Command, *args: str) -> bytes:
    if not 1 <= cmd_id <= MAX_ID:
        raise ProtocolError("ID de comando fuera de rango")
    for arg in args:
        if not _ARG_RE.match(arg):
            raise ProtocolError(f"argumento no permitido: {arg!r}")
    line = " ".join([str(cmd_id), command.value, *args]) + "\n"
    data = line.encode("ascii")
    if len(data) > MAX_LINE:
        raise ProtocolError("línea demasiado larga")
    return data


def parse_response(line: bytes) -> Response:
    if len(line) > MAX_LINE:
        raise ProtocolError("respuesta demasiado larga")
    try:
        text = line.decode("ascii").rstrip("\r\n")
    except UnicodeDecodeError as exc:
        raise ProtocolError("respuesta no ASCII") from exc
    match = _LINE_RE.match(text)
    if not match:
        raise ProtocolError(f"respuesta malformada: {text[:40]!r}")
    cmd_id = int(match.group(1))
    if cmd_id > MAX_ID:
        raise ProtocolError("ID fuera de rango")
    kind = Kind(match.group(2))
    fields = tuple(match.group(3).split())
    if kind == Kind.ERR and not fields:
        raise ProtocolError("ERR sin código")
    if (kind == Kind.EVT) != (cmd_id == 0):
        raise ProtocolError("EVT debe usar ID 0 y solo EVT puede usarlo")
    return Response(cmd_id, kind, fields)
