"""Protocol encoding and serial transport against an injected fake port."""

import threading
import time

import pytest

from clawde.errors import (
    AmbiguousOutcome,
    Disconnected,
    Stopped,
    TransportError,
    TransportTimeout,
)
from clawde.robot.protocol import (
    MAX_LINE,
    Command,
    Kind,
    ProtocolError,
    encode_command,
    parse_response,
)
from clawde.robot.serial_transport import SerialTransport


def test_encode_and_parse() -> None:
    assert encode_command(7, Command.PING) == b"7 PING\n"
    assert encode_command(8, Command.GRIP, "OPEN") == b"8 GRIP OPEN\n"
    r = parse_response(b"7 DONE PONG CLAWDE/1\r\n")
    assert (r.cmd_id, r.kind, r.fields) == (7, Kind.DONE, ("PONG", "CLAWDE/1"))
    assert parse_response(b"9 ERR NOT_CONFIGURED\n").code == "NOT_CONFIGURED"
    assert parse_response(b"0 EVT WATCHDOG DISARMED\n").code == "WATCHDOG"


@pytest.mark.parametrize("args", [("open",), ("A B",), ("1;rm",), ("X" * 17,), ("\n",)])
def test_encode_rejects_unsafe_args(args) -> None:
    with pytest.raises(ProtocolError):
        encode_command(1, Command.MOVE, *args)


def test_encode_rejects_bad_id_and_long_lines() -> None:
    with pytest.raises(ProtocolError):
        encode_command(0, Command.PING)
    with pytest.raises(ProtocolError):
        encode_command(1, Command.MOVE, *(["1234567890"] * 6))


@pytest.mark.parametrize(
    "line",
    [
        b"x ACK\n",
        b"1 OK\n",
        b"1 ERR\n",
        b"5 EVT X\n",
        b"0 DONE\n",
        b"1 ACK \xff\n",
        b"1 DONE " + b"A" * MAX_LINE + b"\n",
        b"99999 ACK\n",
    ],
)
def test_parse_rejects_malformed(line) -> None:
    with pytest.raises(ProtocolError):
        parse_response(line)


class FakePort:
    """Scripted device. `handler(cmd_id, command, args)` returns lines to emit."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.written: list[bytes] = []
        self.rx = bytearray()
        self.lock = threading.Lock()
        self.fail_read = False
        self.closed = False

    def write(self, data: bytes) -> int:
        self.written.append(data)
        parts = data.decode().split()
        out = self.handler(int(parts[0]), parts[1], parts[2:])
        with self.lock:
            for line in out:
                self.rx.extend(line if isinstance(line, bytes) else line.encode() + b"\n")
        return len(data)

    def push(self, raw: bytes) -> None:
        with self.lock:
            self.rx.extend(raw)

    def read(self, size: int = 1) -> bytes:
        if self.fail_read:
            raise OSError("device unplugged")
        with self.lock:
            if self.rx:
                chunk = bytes(self.rx[:size])
                del self.rx[:size]
                return chunk
        time.sleep(0.005)
        return b""

    def close(self) -> None:
        self.closed = True

    def commands(self) -> list[str]:
        return [w.decode().split()[1] for w in self.written]


def ok_handler(cmd_id, command, args):
    return [f"{cmd_id} ACK", f"{cmd_id} DONE"]


def transport(handler, **kw) -> tuple[SerialTransport, FakePort]:
    port = FakePort(handler)
    kw.setdefault("ack_timeout_s", 0.1)
    kw.setdefault("done_timeout_s", 0.3)
    return SerialTransport(port, **kw), port


def test_ack_is_not_done() -> None:
    t, port = transport(lambda i, c, a: [f"{i} ACK"])
    with pytest.raises(TransportTimeout):
        t.send(Command.STATUS)
    assert not t.state_unknown  # STATUS has no side effects


def test_done_after_ack_returns_done() -> None:
    t, _ = transport(lambda i, c, a: [f"{i} ACK", f"{i} DONE PONG CLAWDE/1"])
    r = t.send(Command.PING)
    assert r.kind == Kind.DONE and r.fields[0] == "PONG"


def test_lost_done_on_effectful_command_is_not_retransmitted() -> None:
    t, port = transport(lambda i, c, a: [f"{i} ACK"] if c == "HOME" else [f"{i} ACK", f"{i} DONE"])
    with pytest.raises(AmbiguousOutcome):
        t.send(Command.HOME)
    assert t.state_unknown
    assert port.commands().count("HOME") == 1
    with pytest.raises(TransportError):
        t.send(Command.HOME)  # refused locally, nothing written
    assert port.commands().count("HOME") == 1


def test_no_ack_on_effectful_command_marks_unknown() -> None:
    t, port = transport(lambda i, c, a: [])
    with pytest.raises(AmbiguousOutcome):
        t.send(Command.GRIP, "OPEN")
    assert t.state_unknown and port.commands() == ["GRIP"]


def test_stale_and_malformed_responses_ignored() -> None:
    def handler(i, c, a):
        return [b"999 DONE\n", b"garbage\n", b"Z" * (MAX_LINE * 2) + b"\n", f"{i} ACK", f"{i} DONE"]

    t, _ = transport(handler)
    assert t.send(Command.STATUS).kind == Kind.DONE
    assert t.stale_count == 1
    assert t.malformed_count >= 2


def test_err_is_reported() -> None:
    t, _ = transport(lambda i, c, a: [f"{i} ERR NOT_CONFIGURED"])
    with pytest.raises(TransportError, match="NOT_CONFIGURED"):
        t.send(Command.HOME)
    assert not t.state_unknown  # rejected before acceptance: nothing moved


def test_disconnection_means_unknown_and_no_reconnect() -> None:
    t, port = transport(ok_handler)
    port.fail_read = True
    with pytest.raises(Disconnected):
        t.send(Command.PING)
    assert t.disconnected and t.state_unknown
    port.fail_read = False
    with pytest.raises(Disconnected):
        t.send(Command.PING)
    assert port.commands() == ["PING"]


def test_stop_has_priority_over_pending_command() -> None:
    def handler(i, c, a):
        if c == "GRIP":
            return [f"{i} ACK"]  # long-running, DONE never comes
        return [f"{i} ACK", f"{i} DONE"]

    t, port = transport(handler, done_timeout_s=5.0)
    errors = []

    def run():
        try:
            t.send(Command.GRIP, "CLOSE")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    time.sleep(0.1)
    started = time.monotonic()
    response = t.stop()
    worker.join(2)
    assert time.monotonic() - started < 1.0
    assert response.kind == Kind.DONE
    assert isinstance(errors[0], Stopped)
    assert port.commands() == ["GRIP", "STOP"]
    assert t.stopped
    with pytest.raises(TransportError):
        t.send(Command.GRIP, "OPEN")  # latched until explicit ARM
    t.arm()
    assert not t.stopped


def test_watchdog_event_latches_stop() -> None:
    t, port = transport(ok_handler)
    port.push(b"0 EVT WATCHDOG DISARMED\n")
    t.send(Command.STATUS)
    assert t.stopped and t.events[-1].code == "WATCHDOG"


def test_import_opens_nothing() -> None:
    import clawde.robot.serial_transport as module

    assert module.open_serial_port.__name__ == "open_serial_port"
