"""Push-to-talk capture: wait for TTS to finish, record, check, transcribe, assess."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from clawde.config import VoiceConfig
from clawde.voice.recorder import Recorder, check_audio
from clawde.voice.speaker import Speaker
from clawde.voice.transcriber import Transcriber, Transcription, assess_transcription


@dataclass(frozen=True)
class VoiceResult:
    text: str | None
    rejection: str | None = None
    transcription: Transcription | None = None

    @property
    def accepted(self) -> bool:
        return self.text is not None


class VoiceInput:
    def __init__(
        self,
        config: VoiceConfig,
        recorder: Recorder,
        transcriber: Transcriber,
        speaker: Speaker | None = None,
    ) -> None:
        self.config = config
        self.recorder = recorder
        self.transcriber = transcriber
        self.speaker = speaker

    def capture(self, cancel: threading.Event, seconds: float | None = None) -> VoiceResult:
        if self.speaker is not None:
            self.speaker.wait_idle(cancel)  # never record while `say` is talking
        if cancel.is_set():
            return VoiceResult(None, "grabación cancelada")
        recording = self.recorder.record(seconds, cancel)
        reason = check_audio(recording, self.config.recorder)
        if reason:
            return VoiceResult(None, reason)
        if cancel.is_set():
            return VoiceResult(None, "grabación cancelada")
        transcription = self.transcriber.transcribe(recording.audio)
        if cancel.is_set():
            return VoiceResult(None, "transcripción descartada tras STOP", transcription)
        reason = assess_transcription(transcription, self.config.whisper)
        if reason:
            return VoiceResult(None, reason, transcription)
        return VoiceResult(transcription.text, None, transcription)
