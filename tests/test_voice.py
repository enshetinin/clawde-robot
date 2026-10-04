"""Voice pipeline with test doubles: no microphone, no Whisper weights, no `say`."""

import subprocess
import threading

import pytest

np = pytest.importorskip("numpy")

from clawde.config import RecorderConfig, TTSConfig, VoiceConfig, WhisperConfig  # noqa: E402
from clawde.errors import ModelNotPrepared  # noqa: E402
from clawde.voice.interaction import VoiceInput  # noqa: E402
from clawde.voice.recorder import Recorder, Recording, check_audio  # noqa: E402
from clawde.voice.speaker import SAY, Speaker, Voice, choose_voice, parse_voices  # noqa: E402
from clawde.voice.transcriber import (  # noqa: E402
    Transcription,
    WhisperTranscriber,
    assess_transcription,
)

SR = 16000


def tone(seconds: float = 1.0, amplitude: float = 0.1):
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class FakeBackend:
    def __init__(self, audio, overflow=False, log=None):
        self.audio, self.overflow, self.log = audio, overflow, log
        self.calls = 0

    def record(self, seconds, sample_rate, device, cancel):
        self.calls += 1
        if self.log is not None:
            self.log.append("record")
        return self.audio, self.overflow


class FakeTranscriber:
    def __init__(self, result: Transcription | None = None):
        self.result = result or Transcription("abre la pinza", -0.2, 0.05, "es", 0.99)
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return self.result


class FakeSpeaker:
    def __init__(self, log):
        self.log = log

    def wait_idle(self, cancel=None):
        self.log.append("wait_idle")


def voice_input(audio, transcriber=None, overflow=False, log=None, speaker=None):
    cfg = VoiceConfig()
    rec = Recorder(cfg.recorder, FakeBackend(audio, overflow, log))
    return VoiceInput(cfg, rec, transcriber or FakeTranscriber(), speaker)


@pytest.mark.parametrize(
    ("audio", "overflow", "reason"),
    [
        (np.zeros(0, np.float32), False, "no se capturó"),
        (np.zeros(SR, np.float32), False, "silencio"),
        (tone(0.1), False, "corto"),
        (tone(1.0), True, "desbordamiento"),
    ],
)
def test_bad_audio_never_transcribed(audio, overflow, reason) -> None:
    transcriber = FakeTranscriber()
    result = voice_input(audio, transcriber, overflow).capture(threading.Event())
    assert not result.accepted and reason in result.rejection
    assert transcriber.calls == 0


@pytest.mark.parametrize(
    ("t", "reason"),
    [
        (Transcription("", -0.1, 0.0), "ninguna palabra"),
        (Transcription("...", -0.1, 0.0), "ninguna palabra"),
        (Transcription("abre la pinza", -0.1, 0.9), "no había voz"),
        (Transcription("abre la pinza", -2.5, 0.1), "poco fiable"),
        (Transcription("Subtítulos realizados por la comunidad de Amara.org", -0.1, 0.1), "dudosa"),
    ],
)
def test_dubious_transcriptions_rejected(t, reason) -> None:
    assert reason in assess_transcription(t, WhisperConfig())
    result = voice_input(tone(), FakeTranscriber(t)).capture(threading.Event())
    assert not result.accepted


def test_good_audio_accepted() -> None:
    result = voice_input(tone()).capture(threading.Event())
    assert result.accepted and result.text == "abre la pinza"


def test_waits_for_tts_before_recording() -> None:
    log: list[str] = []
    voice_input(tone(), log=log, speaker=FakeSpeaker(log)).capture(threading.Event())
    assert log == ["wait_idle", "record"]


def test_cancel_before_recording() -> None:
    cancel = threading.Event()
    cancel.set()
    backend_log: list[str] = []
    result = voice_input(tone(), log=backend_log).capture(cancel)
    assert not result.accepted and backend_log == []


def test_check_audio_cancelled() -> None:
    rec = Recording(tone(), SR, cancelled=True)
    assert "cancelada" in check_audio(rec, RecorderConfig())


def test_voice_text_flows_to_simulated_arm(app_factory) -> None:
    from clawde.robot.simulator import SimulatedArm

    arm = SimulatedArm(step_delay_s=0.0)
    app = app_factory(arm=arm)
    result = voice_input(
        tone(), FakeTranscriber(Transcription("Cierra la pinza.", -0.2, 0.0))
    ).capture(threading.Event())
    reply = app.handle(result.text, "voz")
    assert reply.ok and arm.gripper.value == "close"


# ----------------------------------------------------------------- whisper


def test_missing_weights_explain_prepare_command(tmp_path, monkeypatch) -> None:
    pytest.importorskip("faster_whisper")
    import faster_whisper

    def no_download(*a, **k):
        raise AssertionError("no debe cargar ni descargar")

    monkeypatch.setattr(faster_whisper, "WhisperModel", no_download)
    transcriber = WhisperTranscriber(WhisperConfig(), tmp_path)
    with pytest.raises(ModelNotPrepared, match="clawde models prepare --whisper base"):
        transcriber.load()


def test_english_only_models_rejected() -> None:
    with pytest.raises(ValueError):
        WhisperConfig(model="base.en")


# --------------------------------------------------------------------- TTS

SAY_OUTPUT = """\
Albert              en_US    # Hello! My name is Albert.
Mónica (Español (España)) es_ES    # ¡Hola! Me llamo Mónica.
Mónica (Español (España)) es_ES    # ¡Hola! Me llamo Mónica.
Paulina             es_MX    # ¡Hola! Me llamo Paulina.
"""


def test_parse_and_choose_voices() -> None:
    voices = parse_voices(SAY_OUTPUT)
    assert [v.name for v in voices] == ["Albert", "Mónica (Español (España))", "Paulina"]
    chosen, note = choose_voice(voices, None)
    assert chosen.name.startswith("Mónica") and note is None  # preferred base name
    chosen, note = choose_voice(voices, "Mónica (Español (España))")
    assert chosen.locale == "es_ES"
    chosen, note = choose_voice(voices, "Inexistente")
    assert chosen.spanish and "no está instalada" in note
    chosen, note = choose_voice([Voice("Albert", "en_US")], None)
    assert chosen is None and "solo con texto" in note


class FakeProc:
    def __init__(self):
        self.stdin_data = ""
        self.terminated = False
        self.returncode = None

        class Stdin:
            def write(inner, text):
                self.stdin_data += text

            def close(inner):
                pass

        self.stdin = Stdin()

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def test_say_uses_safe_argument_list() -> None:
    calls = []
    procs = []

    def popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        procs.append(FakeProc())
        return procs[-1]

    speaker = Speaker(TTSConfig(rate=180), Voice("Paulina", "es_MX"), popen=popen)
    text = "-v Evil; rm -rf / $(whoami)"
    speaker.speak(text)
    cmd, kwargs = calls[0]
    assert cmd == [SAY, "-v", "Paulina", "-r", "180", "-f", "-"]
    assert text not in cmd
    assert kwargs.get("shell") in (None, False)
    assert kwargs["stdin"] == subprocess.PIPE
    assert procs[0].stdin_data == text
    assert speaker.is_speaking
    speaker.stop()
    assert procs[0].terminated and not speaker.is_speaking


def test_disabled_speaker_does_nothing() -> None:
    def popen(*a, **k):
        raise AssertionError("no debe llamar a say")

    Speaker(TTSConfig(enabled=False), Voice("Paulina", "es_MX"), popen=popen).speak("hola")
    Speaker(TTSConfig(), None, popen=popen).speak("hola")


def test_stop_silences_tts(app_factory) -> None:
    class SpySpeaker:
        stopped = 0

        def stop(self):
            SpySpeaker.stopped += 1

    app = app_factory()
    app.speaker = SpySpeaker()
    app.submit("detente")
    assert SpySpeaker.stopped == 1
