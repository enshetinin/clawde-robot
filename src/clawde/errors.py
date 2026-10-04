"""Exception hierarchy. Messages are user-facing and written in Spanish."""


class ClawdeError(Exception):
    """Base error for CLAWDE."""


class ConfigError(ClawdeError):
    """Invalid or missing configuration."""


class DependencyMissing(ClawdeError):
    """An optional dependency (extra) is not installed."""


class ModelNotPrepared(ClawdeError):
    """Local model weights are not available."""


class InterpreterError(ClawdeError):
    """The brain could not produce a valid action."""


class ActionRejected(ClawdeError):
    """A structurally valid action failed semantic or safety validation."""


class SafetyError(ClawdeError):
    """A motion request violates limits, workspace or freshness checks."""


class HardwareNotConfigured(ClawdeError):
    """Real hardware cannot be used: board, pins, geometry or calibration pending."""


class KinematicsUnavailable(HardwareNotConfigured):
    """No verified kinematic model for the configured arm."""


class Stopped(ClawdeError):
    """Execution was cancelled by STOP."""


class TransportError(ClawdeError):
    """Serial transport failure."""


class TransportTimeout(TransportError):
    """No response within the deadline."""


class AmbiguousOutcome(TransportError):
    """Command accepted (ACK) but completion (DONE) never arrived: state unknown."""


class Disconnected(TransportError):
    """The serial link was lost: robot state is unknown."""


class DeviceError(ClawdeError):
    """Camera, microphone or speaker failure."""


def require(module: str, extra: str) -> object:
    """Import an optional dependency or raise a helpful DependencyMissing."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise DependencyMissing(
            f"Falta la dependencia opcional '{module}'. Instálala con:\n"
            f"  python -m pip install -e '.[{extra}]'"
        ) from exc
