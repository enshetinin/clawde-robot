"""Serial transport for protocol v1 and the (blocked) real arm driver.

- ACK only means "accepted"; success requires DONE.
- Effectful commands are never retransmitted after an ambiguous timeout.
- A lost link marks the state unknown; there is no auto-reconnect.
- STOP is written immediately, even while another command waits for DONE.
Nothing here opens a port at import time.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Protocol

from clawde.errors import (
    AmbiguousOutcome,
    Disconnected,
    KinematicsUnavailable,
    Stopped,
    TransportError,
    TransportTimeout,
)
from clawde.robot.base import ArmDriver, ArmState, ArmStatus, CancelToken, Plan
from clawde.robot.protocol import (
    EFFECTFUL,
    MAX_ID,
    MAX_LINE,
    Command,
    Kind,
    ProtocolError,
    Response,
    encode_command,
    parse_response,
)

log = logging.getLogger(__name__)


class Port(Protocol):
    def read(self, size: int = 1) -> bytes: ...
    def write(self, data: bytes) -> int | None: ...
    def close(self) -> None: ...


def open_serial_port(port: str, baudrate: int, read_timeout: float = 0.05) -> Port:
    """Open a real serial port. Note: many boards reset when the port opens."""
    import serial  # deferred

    return serial.Serial(port, baudrate, timeout=read_timeout, write_timeout=0.5)


def list_serial_ports() -> list[tuple[str, str]]:
    """List ports without opening them."""
    from serial.tools import list_ports

    return [(p.device, p.description or "") for p in list_ports.comports()]


class SerialTransport:
    def __init__(
        self,
        port: Port,
        ack_timeout_s: float = 0.5,
        done_timeout_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._port = port
        self.ack_timeout_s = ack_timeout_s
        self.done_timeout_s = done_timeout_s
        self._clock = clock
        self._cmd_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._next_id = 1
        self._buffer = bytearray()
        self._discarding = False
        self._outstanding: set[int] = set()
        self._inbox: dict[int, deque[Response]] = {}
        self.events: deque[Response] = deque(maxlen=32)
        self.stale_count = 0
        self.malformed_count = 0
        self.state_unknown = False
        self.disconnected = False
        self.stopped = False

    # ------------------------------------------------------------------ public

    def send(self, command: Command, *args: str, done_timeout_s: float | None = None) -> Response:
        if command == Command.STOP:
            return self.stop()
        self._check_link()
        if command in EFFECTFUL and (self.state_unknown or self.stopped) and command != Command.ARM:
            raise TransportError("estado desconocido o detenido: consulta STATUS y rearma (ARM)")
        with self._cmd_lock:
            if self._stop_event.is_set():
                raise Stopped("STOP en curso")
            cmd_id = self._write(command, args)
            return self._complete(cmd_id, command, done_timeout_s)

    def stop(self) -> Response:
        """Send STOP with priority. Latches `stopped` until an explicit ARM."""
        self._check_link()
        self._stop_event.set()
        self.stopped = True
        try:
            cmd_id = self._write(Command.STOP, ())
            # The waiting sender aborts within one read timeout and frees the lock.
            if not self._cmd_lock.acquire(timeout=self.done_timeout_s):
                raise TransportTimeout("no se pudo confirmar STOP")
            try:
                return self._complete(cmd_id, Command.STOP, self.ack_timeout_s * 4, priority=True)
            finally:
                self._cmd_lock.release()
        finally:
            self._stop_event.clear()

    def arm(self) -> Response:
        """Explicit re-arm. Clears the local stop latch only after DONE."""
        self._check_link()
        with self._cmd_lock:
            cmd_id = self._write(Command.ARM, ())
            response = self._complete(cmd_id, Command.ARM, None)
        self.stopped = False
        self.state_unknown = False
        return response

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self._port.close()

    # ----------------------------------------------------------------- private

    def _check_link(self) -> None:
        if self.disconnected:
            raise Disconnected("enlace serie perdido: estado del robot desconocido")

    def _allocate_id(self) -> int:
        cmd_id = self._next_id
        self._next_id = 1 if self._next_id >= MAX_ID else self._next_id + 1
        return cmd_id

    def _write(self, command: Command, args: tuple[str, ...]) -> int:
        with self._write_lock:
            cmd_id = self._allocate_id()
            data = encode_command(cmd_id, command, *args)
            self._outstanding.add(cmd_id)
            try:
                self._port.write(data)
            except OSError as exc:
                self._lose_link(exc)
        return cmd_id

    def _lose_link(self, exc: BaseException) -> None:
        self.disconnected = True
        self.state_unknown = True
        raise Disconnected(f"enlace serie perdido: {exc}") from exc

    def _complete(
        self, cmd_id: int, command: Command, done_timeout_s: float | None, priority: bool = False
    ) -> Response:
        try:
            first = self._await(cmd_id, self.ack_timeout_s, priority)
            if first is None:
                if command in EFFECTFUL:
                    self.state_unknown = True
                    raise AmbiguousOutcome(f"{command} sin ACK: estado desconocido, no se reenvía")
                raise TransportTimeout(f"{command} sin respuesta")
            if first.kind == Kind.ERR:
                raise TransportError(f"{command} rechazado: {' '.join(first.fields)}")
            if first.kind == Kind.DONE:
                return first  # some devices may answer DONE directly
            deadline = done_timeout_s if done_timeout_s is not None else self.done_timeout_s
            final = self._await(cmd_id, deadline, priority)
            if final is None:
                if command in EFFECTFUL:
                    self.state_unknown = True
                    raise AmbiguousOutcome(
                        f"{command} aceptado (ACK) sin DONE: estado desconocido, no se reenvía"
                    )
                raise TransportTimeout(f"{command} sin DONE")
            if final.kind == Kind.ERR:
                raise TransportError(f"{command} falló: {' '.join(final.fields)}")
            if final.kind != Kind.DONE:
                raise TransportError(f"{command}: respuesta inesperada {final.kind}")
            return final
        finally:
            self._outstanding.discard(cmd_id)
            self._inbox.pop(cmd_id, None)

    def _await(self, cmd_id: int, timeout_s: float, priority: bool) -> Response | None:
        deadline = self._clock() + timeout_s
        while True:
            queued = self._inbox.get(cmd_id)
            if queued:
                return queued.popleft()
            if not priority and self._stop_event.is_set():
                raise Stopped("comando interrumpido por STOP")
            if self._clock() >= deadline:
                return None
            line = self._read_line()
            if line is None:
                continue
            try:
                response = parse_response(line)
            except ProtocolError as exc:
                self.malformed_count += 1
                log.warning("discarding malformed line: %s", exc)
                continue
            if response.kind == Kind.EVT:
                self.events.append(response)
                if response.code == "WATCHDOG":
                    self.stopped = True
                continue
            if response.cmd_id == cmd_id:
                return response
            if response.cmd_id in self._outstanding:
                self._inbox.setdefault(response.cmd_id, deque(maxlen=4)).append(response)
            else:
                self.stale_count += 1
                log.warning("discarding stale response for id %d", response.cmd_id)

    def _read_line(self) -> bytes | None:
        """Read at most one bounded line. Returns None if nothing complete yet."""
        try:
            chunk = self._port.read(MAX_LINE)
        except OSError as exc:
            self._lose_link(exc)
        if chunk:
            self._buffer.extend(chunk)
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                if len(self._buffer) > MAX_LINE:
                    self._buffer.clear()
                    self._discarding = True
                    self.malformed_count += 1
                return None
            line = bytes(self._buffer[: newline + 1])
            del self._buffer[: newline + 1]
            if self._discarding:
                self._discarding = False
                continue
            if len(line) > MAX_LINE:
                self.malformed_count += 1
                continue
            return line


class SerialArm(ArmDriver):
    """Real arm over serial. Only built after `check_real_motion_ready` passed.

    Motion still requires verified kinematics; with the diagnostic firmware every
    motion command answers ERR NOT_CONFIGURED.
    """

    name = "serial"
    simulated = False

    def __init__(self, transport: SerialTransport) -> None:
        self.transport = transport

    def status(self) -> ArmStatus:
        if self.transport.disconnected:
            state = ArmState.DISCONNECTED
        elif self.transport.stopped:
            state = ArmState.STOPPED
        elif self.transport.state_unknown:
            state = ArmState.UNKNOWN
        else:
            state = ArmState.IDLE
        return ArmStatus(self.name, state, None, None, None, simulated=False)

    def execute(self, plan: Plan, token: CancelToken) -> None:
        token.check()
        raise KinematicsUnavailable("sin cinemática verificada: no se envía ningún movimiento")

    def stop(self) -> None:
        self.transport.stop()

    def rearm(self) -> None:
        self.transport.send(Command.STATUS)
        self.transport.arm()

    def close(self) -> None:
        self.transport.close()
