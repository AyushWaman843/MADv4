import logging
import tempfile
from pathlib import Path

import requests

from ..config import Config


LOGGER = logging.getLogger(__name__)
_LOCAL_MODEL = None


def _load_local_model():
    global _LOCAL_MODEL
    if _LOCAL_MODEL is not None:
        return _LOCAL_MODEL

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Local Whisper is not installed yet. Install backend requirements to enable offline transcription."
        ) from exc

    _LOCAL_MODEL = WhisperModel(
        Config.LOCAL_WHISPER_MODEL,
        device=Config.LOCAL_WHISPER_DEVICE,
        compute_type=Config.LOCAL_WHISPER_COMPUTE_TYPE,
    )
    LOGGER.info(
        "Loaded local faster-whisper model %s on %s with compute type %s.",
        Config.LOCAL_WHISPER_MODEL,
        Config.LOCAL_WHISPER_DEVICE,
        Config.LOCAL_WHISPER_COMPUTE_TYPE,
    )
    return _LOCAL_MODEL


def _transcribe_with_local_whisper(file_storage) -> str:
    model = _load_local_model()

    suffix = Path(file_storage.filename or "prompt.m4a").suffix or ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_path = Path(temp_audio.name)
        file_storage.stream.seek(0)
        temp_audio.write(file_storage.stream.read())

    try:
        segments, _ = model.transcribe(
            str(temp_path),
            task="transcribe",
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        temp_path.unlink(missing_ok=True)

    if not text:
        raise RuntimeError("Local Whisper returned an empty transcription.")
    return text


def _transcribe_with_openai(file_storage) -> str:
    if not Config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required when TRANSCRIPTION_PROVIDER=openai.")

    file_storage.stream.seek(0)
    response = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {Config.OPENAI_API_KEY}"},
        data={"model": Config.WHISPER_MODEL},
        files={
            "file": (
                file_storage.filename or "prompt.m4a",
                file_storage.stream,
                file_storage.mimetype or "application/octet-stream",
            )
        },
        timeout=120,
    )

    if not response.ok:
        LOGGER.error(
            "OpenAI Whisper transcription failed with status %s for file %s and model %s. Response: %s",
            response.status_code,
            file_storage.filename,
            Config.WHISPER_MODEL,
            response.text[:500],
        )
        response.raise_for_status()

    text = str((response.json() or {}).get("text") or "").strip()
    if not text:
        raise RuntimeError("OpenAI Whisper returned an empty transcription.")
    return text


def transcribe_audio(file_storage) -> str:
    if file_storage is None or not getattr(file_storage, "filename", ""):
        raise ValueError("An audio file is required.")

    provider = str(Config.TRANSCRIPTION_PROVIDER or "local").strip().lower()
    if provider == "openai":
        return _transcribe_with_openai(file_storage)

    return _transcribe_with_local_whisper(file_storage)
