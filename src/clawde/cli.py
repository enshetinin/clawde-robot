"""Command-line interface. Heavy modules are imported inside each command."""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from clawde import __version__
from clawde.config import OllamaConfig, Settings, load_settings
from clawde.errors import ClawdeError

log = logging.getLogger("clawde")


def _console() -> Any:
    from rich.console import Console

    return Console(highlight=False)


def _setup_logging(settings: Settings, verbose: bool) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    log_dir = settings.path(settings.app.log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(log_dir / "clawde.log", encoding="utf-8")
    except OSError:
        handler = logging.NullHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)


# ------------------------------------------------------------------ builders


def build_interpreter(settings: Settings, mode: str | None, model: str | None, console: Any) -> Any:
    from clawde.brain.interpreter import Interpreter
    from clawde.brain.ollama_client import OllamaBrain

    mode = mode or settings.app.interpreter
    if mode == "rules":
        return Interpreter("rules")
    cfg = settings.app.ollama
    if model:
        try:
            cfg = OllamaConfig.model_validate({**cfg.model_dump(), "model": model})
        except ValidationError as exc:
            raise ClawdeError(f"modelo no permitido '{model}': {exc.errors()[0]['msg']}") from exc
    brain = OllamaBrain(cfg)
    probe = brain.probe()
    if probe.reachable and probe.has_model(cfg.model):
        return Interpreter("hybrid" if mode == "auto" else "ollama", brain, cfg.max_retries)
    if not probe.reachable:
        reason = f"Ollama no responde en {cfg.host} (¿está abierto? `ollama serve`)"
    else:
        reason = (
            f"el modelo '{cfg.model}' no está instalado (instalados: "
            f"{', '.join(probe.models) or 'ninguno'}). Descárgalo con `ollama pull {cfg.model}`"
        )
    if mode == "ollama":
        raise ClawdeError(reason)
    console.print(f"[yellow]Aviso:[/] {reason}. Uso el intérprete de reglas deterministas.")
    return Interpreter("rules")


def build_arm(settings: Settings, real: bool) -> Any:
    if not real:
        from clawde.robot.simulator import SimulatedArm

        return SimulatedArm(step_delay_s=settings.robot.demo_profile.step_delay_s)
    from clawde.motion.planner import check_real_motion_ready
    from clawde.robot.serial_transport import SerialArm, SerialTransport, open_serial_port

    check_real_motion_ready(settings.robot)  # raises before any port is opened
    assert settings.robot.port is not None
    transport = SerialTransport(open_serial_port(settings.robot.port, settings.robot.baudrate))
    return SerialArm(transport)


def build_scene(settings: Settings, demo: bool, camera_index: int | None, save: bool) -> Any:
    if demo:
        from clawde.vision.scene import DemoSceneProvider

        return DemoSceneProvider(settings.robot.demo_profile)
    from clawde.vision.calibration import TableCalibration
    from clawde.vision.camera import Camera
    from clawde.vision.detector import ColorDetector, ObjectTracker
    from clawde.vision.scene import CameraSceneProvider

    vision = settings.vision
    calib_path = settings.path(vision.calibration_file)
    calibration = TableCalibration.load(calib_path) if calib_path.exists() else None
    detector = ColorDetector(
        vision.colors, vision.min_area_px, vision.max_area_px, vision.morph_kernel
    )
    index = vision.camera.index if camera_index is None else camera_index
    camera = Camera(index, vision.camera.width, vision.camera.height).open()
    save_dir = settings.path(settings.app.captures_dir) if save else None
    return CameraSceneProvider(
        camera, detector, ObjectTracker(vision.tracking_max_distance_px), calibration, save_dir
    )


def build_speaker(settings: Settings, enabled: bool, console: Any) -> Any:
    from clawde.voice.speaker import Speaker, choose_voice, list_voices, say_available

    if not enabled or not settings.voice.tts.enabled:
        return None
    if not say_available():
        console.print("[yellow]Síntesis de voz no disponible (requiere macOS /usr/bin/say).[/]")
        return None
    voice, note = choose_voice(list_voices(), settings.voice.tts.voice)
    if note:
        console.print(f"[yellow]{note}[/]")
    return Speaker(settings.voice.tts, voice) if voice else None


# ------------------------------------------------------------------ commands


def cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    from clawde.doctor import run_doctor

    return run_doctor(settings, _console())


def cmd_models(args: argparse.Namespace, settings: Settings) -> int:
    from clawde.voice.transcriber import prepare_model

    console = _console()
    model_dir = settings.path(settings.voice.whisper.model_dir)
    console.print(
        f"Descargando faster-whisper '{args.whisper}' en {model_dir} "
        "(base ≈ 145 MB, small ≈ 480 MB; requiere Internet una vez)…"
    )
    path = prepare_model(args.whisper, model_dir)
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    console.print(
        f"[green]Listo:[/] {path} ({size:.0f} MB). Registro en {model_dir / 'manifest.json'}"
    )
    if args.whisper != settings.voice.whisper.model:
        console.print(f'Para usarlo, pon `model: "{args.whisper}"` en config/voice.yaml.')
    return 0


def cmd_devices(args: argparse.Namespace, settings: Settings) -> int:
    console = _console()
    if args.kind == "audio":
        from clawde.voice.recorder import list_input_devices

        devices = list_input_devices()
        if not devices:
            console.print("No hay dispositivos de entrada de audio.")
        for d in devices:
            mark = " [green](predeterminado)[/]" if d["default"] else ""
            console.print(f"{d['index']:>3}  {d['name']}  canales={d['channels']}{mark}")
        console.print("Configura el micrófono en config/voice.yaml → recorder.device")
    elif args.kind == "serial":
        from clawde.robot.serial_transport import list_serial_ports

        ports = list_serial_ports()
        if not ports:
            console.print("No se detectan puertos serie.")
        for device, desc in ports:
            console.print(f"{device}  {desc}")
        console.print("(Solo listado: no se ha abierto ningún puerto.)")
    else:
        from clawde.voice.speaker import choose_voice, list_voices

        voices = list_voices()
        chosen, note = choose_voice(voices, settings.voice.tts.voice)
        for v in voices:
            if v.spanish:
                mark = " [green]← en uso[/]" if chosen and v.name == chosen.name else ""
                console.print(f"{v.name}  {v.locale}{mark}")
        if note:
            console.print(f"[yellow]{note}[/]")
        console.print("Elige una con tts.voice en config/voice.yaml (nombre exacto).")
    return 0


def cmd_camera(args: argparse.Namespace, settings: Settings) -> int:
    from clawde.vision.calibration import TableCalibration
    from clawde.vision.camera import Camera, destroy_windows, draw_labels, save_capture
    from clawde.vision.detector import ColorDetector, ObjectTracker

    console = _console()
    vision = settings.vision
    detector = ColorDetector(
        vision.colors, vision.min_area_px, vision.max_area_px, vision.morph_kernel
    )
    tracker = ObjectTracker(vision.tracking_max_distance_px)
    calib_path = settings.path(vision.calibration_file)
    calibration = TableCalibration.load(calib_path) if calib_path.exists() else None
    preview = not args.no_preview
    cv2: Any = None
    if preview:
        import cv2
    console.print(
        f"Cámara {args.index}: "
        + (
            "ventana de vista previa (q/Esc para salir)."
            if preview
            else "sin ventana, Ctrl+C sale."
        )
    )
    camera = Camera(args.index, vision.camera.width, vision.camera.height)
    try:
        camera.open()
        frame = camera.read()
        height, width = frame.shape[:2]
        console.print(f"Resolución: {width}x{height}")
        if calibration:
            calibration.ensure_resolution(width, height)
        start = last_print = time.monotonic()
        while args.seconds is None or time.monotonic() - start < args.seconds:
            frame = camera.read()
            labelled, ambiguous = tracker.update(detector.detect(frame))
            if time.monotonic() - last_print >= 1.0:
                last_print = time.monotonic()
                desc = ", ".join(
                    f"{oid}@{d.center_px[0]:.0f},{d.center_px[1]:.0f}"
                    + (
                        " ({:.0f},{:.0f} mm)".format(*calibration.pixel_to_mm(*d.center_px))
                        if calibration
                        else ""
                    )
                    for oid, d in labelled
                )
                console.print(
                    f"{len(labelled)} objeto(s): {desc or '-'}"
                    + (" [ambiguo]" if ambiguous else "")
                )
            if args.save:
                path = save_capture(
                    draw_labels(frame.copy(), labelled), settings.path(settings.app.captures_dir)
                )
                console.print(f"Captura guardada: {path}")
                args.save = False
            if preview:
                cv2.imshow("CLAWDE camera", draw_labels(frame, labelled))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        if preview:
            destroy_windows()
    return 0


def cmd_voice_test(args: argparse.Namespace, settings: Settings) -> int:
    from clawde.voice.recorder import Recorder, check_audio
    from clawde.voice.transcriber import WhisperTranscriber, assess_transcription

    console = _console()
    rec_cfg = settings.voice.recorder
    if args.device is not None:
        rec_cfg = rec_cfg.model_copy(update={"device": _device_arg(args.device)})
    transcriber = WhisperTranscriber(
        settings.voice.whisper, settings.path(settings.voice.whisper.model_dir)
    )
    console.print("Cargando Whisper local…")
    t0 = time.monotonic()
    transcriber.load()
    console.print(f"Modelo cargado en {time.monotonic() - t0:.1f} s. {_rss_text()}")
    console.print(f"[bold]Habla ahora durante {args.seconds:.0f} s…[/] (Ctrl+C cancela)")
    cancel = threading.Event()
    try:
        recording = Recorder(rec_cfg).record(args.seconds, cancel)
    except KeyboardInterrupt:
        cancel.set()
        console.print("Cancelado.")
        return 1
    reason = check_audio(recording, rec_cfg)
    console.print(f"Audio: {recording.duration_s:.1f} s, desbordamiento={recording.overflowed}")
    if reason:
        console.print(f"[yellow]Rechazado:[/] {reason}")
        return 1
    t0 = time.monotonic()
    result = transcriber.transcribe(recording.audio)
    console.print(
        f"Transcripción ({time.monotonic() - t0:.1f} s): «{result.text}»\n"
        f"avg_logprob={result.avg_logprob:.2f} no_speech={result.no_speech_prob:.2f} "
        f"idioma={result.language} ({(result.language_probability or 0):.2f}). {_rss_text()}"
    )
    reason = assess_transcription(result, settings.voice.whisper)
    if reason:
        console.print(f"[yellow]Se rechazaría como orden:[/] {reason}")
        return 1
    console.print("[green]Transcripción aceptable como orden.[/]")
    return 0


def cmd_calibrate(args: argparse.Namespace, settings: Settings) -> int:
    import json

    from clawde.vision.calibration import TableCalibration

    console = _console()
    data = json.loads(Path(args.points).read_text(encoding="utf-8"))
    pairs = data["points"]
    calibration = TableCalibration.from_points(
        [tuple(p["px"]) for p in pairs],
        [tuple(p["mm"]) for p in pairs],
        tuple(data["image_size"]),
        int(data.get("camera_index", settings.vision.camera.index)),
        settings.vision.max_calibration_rms_mm,
        data.get("camera_name"),
    )
    out = Path(args.output) if args.output else settings.path(settings.vision.calibration_file)
    calibration.save(out)
    console.print(
        f"[green]Calibración guardada[/] en {out}: {calibration.num_points} puntos, "
        f"error RMS {calibration.rms_error_mm:.2f} mm, "
        f"{calibration.image_width}x{calibration.image_height}."
    )
    return 0


def _rss_text() -> str:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3
        return f"Memoria máx. del proceso: {mb:.0f} MB."
    except Exception:
        return ""


def _device_arg(value: str) -> int | str:
    return int(value) if value.isdigit() else value


# ---------------------------------------------------------------- run / command


class Presenter:
    def __init__(self, console: Any, speaker: Any) -> None:
        self.console = console
        self.speaker = speaker

    def __call__(self, reply: Any) -> None:
        style = "green" if reply.ok else "yellow"
        source = f" [dim]({reply.source})[/]" if reply.source else ""
        self.console.print(f"[bold {style}]CLAWDE:[/] {reply.text}{source}")
        if self.speaker is not None and reply.speak:
            self.speaker.speak(reply.text.split("\n")[0])


def _build_app(args: argparse.Namespace, settings: Settings, console: Any, tts: bool) -> Any:
    from clawde.app import ClawdeApp
    from clawde.motion.planner import Planner

    if args.real and args.demo:
        raise ClawdeError("--demo solo está permitido en simulación (--sim)")
    demo = args.demo or args.camera is None
    interpreter = build_interpreter(settings, args.interpreter, args.model, console)
    arm = build_arm(settings, args.real)
    try:
        scene = build_scene(settings, demo, args.camera, getattr(args, "save_captures", False))
    except BaseException:
        arm.close()
        raise
    speaker = build_speaker(settings, tts, console)
    planner = Planner(settings.robot, settings.app.scene_max_age_s, real=args.real)
    app = ClawdeApp(settings, scene, arm, interpreter, planner, speaker=speaker)
    app.on_reply = Presenter(console, speaker)
    if scene.warning:
        console.rule(f"[bold red]{scene.warning}")
    console.print(f"Modo: {app.mode_label} · Intérprete: [bold]{interpreter.label}[/]")
    return app


def cmd_command(args: argparse.Namespace, settings: Settings) -> int:
    console = _console()
    app = _build_app(args, settings, console, tts=args.speak)
    try:
        reply = app.handle(args.text, "comando")
        app.on_reply(reply)
        if app.speaker is not None:
            app.speaker.wait_idle()
        return 0 if reply.ok else 1
    except KeyboardInterrupt:
        app.on_reply(app.emergency_stop("Ctrl+C"))
        return 130
    finally:
        app.shutdown()


def _stdin_reader(lines: queue.Queue[str | None]) -> None:
    for line in sys.stdin:
        lines.put(line.rstrip("\n"))
    lines.put(None)


HELP_TEXT = (
    "Ejemplos: «¿qué ves?», «coge el objeto rojo y ponlo a la izquierda», «vuelve a casa», "
    "«abre la pinza», «estado», «para», «rearmar», «salir»."
)


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    console = _console()
    voice_input = None
    if args.voice:
        voice_input = _build_voice(settings, args, console)
    app = _build_app(args, settings, console, tts=args.voice and not args.no_tts or args.speak)
    if voice_input is not None:
        voice_input.speaker = app.speaker
    capture_cancel = threading.Event()
    capturing = threading.Event()

    def cancel_capture() -> None:
        capture_cancel.set()

    app.on_stop = cancel_capture
    console.print(HELP_TEXT)
    if args.voice:
        console.print(
            "[bold]Voz:[/] pulsa Enter para grabar "
            f"{settings.voice.recorder.seconds if args.seconds is None else args.seconds:.0f} s. "
            "Escribe una orden para usar texto, «c» cancela la grabación, «para» detiene, "
            "Ctrl+C detiene y sale."
        )
    else:
        console.print(
            "[bold]Texto:[/] escribe una orden y pulsa Enter. "
            "«para» detiene, Ctrl+C detiene y sale."
        )

    def voice_job() -> None:
        assert voice_input is not None
        try:
            console.print("[bold cyan]● Grabando…[/]")
            result = voice_input.capture(capture_cancel, args.seconds)
            if result.accepted:
                console.print(f"[cyan]Has dicho:[/] «{result.text}»")
                reply = app.submit(result.text, "voz")
                if reply is not None:
                    app.on_reply(reply)
            else:
                heard = f" (oído: «{result.transcription.text}»)" if result.transcription else ""
                console.print(
                    f"[yellow]Voz rechazada:[/] {result.rejection}{heard}. No se mueve nada."
                )
        except ClawdeError as exc:
            console.print(f"[red]Error de voz:[/] {exc}")
        finally:
            capturing.clear()

    lines: queue.Queue[str | None] = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(lines,), daemon=True).start()
    app.start()
    try:
        while not app.exit_requested.is_set():
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:  # stdin closed: finish queued orders, STOP still works (Ctrl+C)
                app.wait_idle(timeout_s=settings.app.ollama.timeout_s * 2 + 30)
                break
            text = line.strip()
            lowered = text.lower()
            if args.voice and text == "":
                if capturing.is_set():
                    console.print("Ya estoy grabando.")
                    continue
                capture_cancel.clear()
                capturing.set()
                threading.Thread(target=voice_job, name="clawde-voice", daemon=True).start()
                continue
            if not text:
                continue
            if lowered in {"c", "cancelar"}:
                capture_cancel.set()
                console.print(
                    "Grabación cancelada." if capturing.is_set() else "Nada que cancelar."
                )
                continue
            if lowered in {"q", "quit", "exit"}:
                break
            reply = app.submit(text, "texto")
            if reply is not None:
                app.on_reply(reply)
    except KeyboardInterrupt:
        app.on_reply(app.emergency_stop("Ctrl+C"))
    finally:
        capture_cancel.set()
        app.shutdown()
        console.print("CLAWDE cerrado.")
    return 0


def _build_voice(settings: Settings, args: argparse.Namespace, console: Any) -> Any:
    from clawde.voice.interaction import VoiceInput
    from clawde.voice.recorder import Recorder
    from clawde.voice.transcriber import WhisperTranscriber

    transcriber = WhisperTranscriber(
        settings.voice.whisper, settings.path(settings.voice.whisper.model_dir)
    )
    console.print(f"Cargando Whisper '{settings.voice.whisper.model}' local (CPU int8)…")
    transcriber.load()
    console.print(_rss_text())
    return VoiceInput(settings.voice, Recorder(settings.voice.recorder), transcriber, None)


# ------------------------------------------------------------------- parser


def _add_runtime_args(p: argparse.ArgumentParser) -> None:
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--sim", action="store_true", default=True, help="brazo simulado (por defecto)"
    )
    mode.add_argument(
        "--real", action="store_true", help="driver real (requiere configuración validada)"
    )
    scene = p.add_mutually_exclusive_group()
    scene.add_argument(
        "--demo", action="store_true", help="escena FICTICIA con objetos rojos y azules"
    )
    scene.add_argument("--camera", type=int, metavar="N", help="usar la cámara N para la escena")
    p.add_argument(
        "--interpreter", choices=["auto", "rules", "ollama"], help="por defecto: app.yaml"
    )
    p.add_argument("--model", help="modelo Ollama local (sobrescribe app.yaml)")
    p.add_argument("--speak", action="store_true", help="leer respuestas con `say`")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clawde",
        description="CLAWDE: órdenes en español por voz y texto, visión local y brazo simulado.",
    )
    parser.add_argument("--version", action="version", version=f"clawde {__version__}")
    parser.add_argument(
        "--config-dir", help="directorio de configuración (por defecto ./config del proyecto)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="registro detallado en logs/")
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMANDO")

    sub.add_parser("doctor", help="diagnóstico del entorno (no abre cámara ni micrófono)")

    models = sub.add_parser("models", help="gestión de modelos locales")
    models_sub = models.add_subparsers(dest="models_cmd", required=True)
    prep = models_sub.add_parser("prepare", help="descarga explícita de pesos de Whisper")
    prep.add_argument("--whisper", default="base", choices=["tiny", "base", "small", "medium"])

    devices = sub.add_parser("devices", help="listar dispositivos sin abrirlos")
    devices.add_argument("kind", choices=["audio", "serial", "voices"])

    camera = sub.add_parser("camera", help="probar la cámara y la detección de color")
    camera.add_argument("--index", type=int, default=0)
    camera.add_argument("--seconds", type=float, help="duración (por defecto hasta q/Esc)")
    camera.add_argument("--no-preview", action="store_true", help="sin ventana")
    camera.add_argument("--save", action="store_true", help="guardar UNA captura en captures/")

    voice = sub.add_parser("voice-test", help="grabar y transcribir una vez (sin robot)")
    voice.add_argument("--seconds", type=float, default=5.0)
    voice.add_argument("--device", help="índice o nombre del micrófono")

    run = sub.add_parser("run", help="sesión interactiva")
    _add_runtime_args(run)
    io = run.add_mutually_exclusive_group()
    io.add_argument(
        "--text", action="store_true", default=True, help="entrada por teclado (por defecto)"
    )
    io.add_argument("--voice", action="store_true", help="pulsar Enter para hablar")
    run.add_argument("--seconds", type=float, help="duración de cada grabación")
    run.add_argument("--no-tts", action="store_true", help="no hablar en modo voz")
    run.add_argument("--save-captures", action="store_true", help="guardar imágenes de cada escena")

    command = sub.add_parser("command", help="ejecutar una sola orden de texto")
    _add_runtime_args(command)
    command.add_argument("text", help="orden en español entre comillas")

    calib = sub.add_parser("calibrate", help="calcular homografía de mesa desde un JSON de puntos")
    calib.add_argument("--points", required=True, help="JSON con image_size y points[{px, mm}]")
    calib.add_argument("--output", help="ruta de salida (por defecto vision.yaml)")
    return parser


COMMANDS = {
    "doctor": cmd_doctor,
    "models": cmd_models,
    "devices": cmd_devices,
    "camera": cmd_camera,
    "voice-test": cmd_voice_test,
    "run": cmd_run,
    "command": cmd_command,
    "calibrate": cmd_calibrate,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config_dir)
        _setup_logging(settings, args.verbose)
        return COMMANDS[args.cmd](args, settings)
    except ClawdeError as exc:
        _console().print(f"[bold red]Error:[/] {exc}")
        return 2
    except KeyboardInterrupt:
        return 130
