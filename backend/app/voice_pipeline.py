"""
Voice pipeline utilities for local ASR and TTS integration.

This module keeps ASR/TTS concerns separate from API routes so the
WebSocket endpoint stays focused on orchestration.
"""

from __future__ import annotations

import asyncio
import base64
import io
import importlib
import inspect
import logging
import os
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


class AudioConverter:
    """Converts browser audio chunks to model-friendly WAV (16k mono)."""

    async def to_wav_16k(self, audio_bytes: bytes, source_extension: str = "webm") -> bytes:
        """
        Convert arbitrary audio bytes to PCM WAV 16kHz mono using ffmpeg.

        Browser microphone captures often arrive as WebM/Opus chunks.
        Most CPU ASR runtimes perform best with normalized WAV input.
        """
        if not audio_bytes:
            raise ValueError("Audio payload is empty")

        suffix = f".{source_extension or 'webm'}"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_file = tmp_path / f"input{suffix}"
            output_file = tmp_path / "output.wav"
            input_file.write_bytes(audio_bytes)

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(input_file),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                str(output_file),
            ]

            try:
                process = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "ffmpeg executable not found. Install ffmpeg and ensure it is in PATH."
                ) from exc

            if process.returncode != 0:
                stderr = (process.stderr or "").strip()
                logger.error("ffmpeg conversion failed: %s", stderr)
                short_err = stderr.splitlines()[-1] if stderr else "unknown ffmpeg error"
                raise RuntimeError(f"Failed to convert audio to WAV using ffmpeg: {short_err}")

            return output_file.read_bytes()


class MoonshineASRService:
    """
    Mooshine/Moonshine ASR adapter.

    Runs ASR directly in-process (no HTTP APIs). The model is loaded once and
    reused across requests to reduce CPU warm-up overhead.
    """

    def __init__(self, max_concurrency: int = 4):
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._transcriber = None
        self._transcriber_lock = threading.Lock()

    def _get_transcriber(self):
        # Lazy import keeps app startup working even before voice deps are installed.
        if self._transcriber is not None:
            return self._transcriber

        with self._transcriber_lock:
            if self._transcriber is not None:
                return self._transcriber

            try:
                moonshine_voice = importlib.import_module("moonshine_voice")
            except ImportError as exc:
                raise RuntimeError(
                    "moonshine-voice is not installed. Install with: pip install moonshine-voice"
                ) from exc

            model_path, model_arch = moonshine_voice.get_model_for_language("en")
            self._transcriber = moonshine_voice.Transcriber(model_path=model_path, model_arch=model_arch)
            return self._transcriber

    async def transcribe(self, wav_audio_bytes: bytes) -> str:
        """Transcribe WAV audio to text using local Moonshine runtime."""
        if not wav_audio_bytes:
            return ""

        def run_local_asr() -> str:
            try:
                np = importlib.import_module("numpy")
                sf = importlib.import_module("soundfile")
            except ImportError as exc:
                raise RuntimeError(
                    "Missing ASR audio dependencies. Install with: pip install numpy soundfile"
                ) from exc

            transcriber = self._get_transcriber()

            audio_buffer = io.BytesIO(wav_audio_bytes)
            audio_data, sample_rate = sf.read(audio_buffer, dtype="float32")

            # Down-mix multi-channel input to mono for stable ASR behavior.
            if isinstance(audio_data, np.ndarray) and audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            transcript = transcriber.transcribe_without_streaming(
                audio_data.tolist(), sample_rate=int(sample_rate), flags=0
            )
            lines = getattr(transcript, "lines", [])
            text_parts = [getattr(line, "text", "") for line in lines if getattr(line, "text", "")]
            return " ".join(text_parts).strip()

        async with self._semaphore:
            return await asyncio.to_thread(run_local_asr)


class PiperTTSService:
    """
    Piper TTS adapter.

    Runs Piper synthesis via its local CLI and returns WAV bytes for
    incremental playback on the frontend.
    """

    def __init__(self, max_concurrency: int = 4):
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._voice = None
        self._voice_lock = threading.Lock()

        self._model_path = os.getenv("PIPER_MODEL_PATH", "").strip()
        self._config_path = os.getenv("PIPER_CONFIG_PATH", "").strip()
        self._speaker = os.getenv("PIPER_SPEAKER", "").strip()
        self._length_scale = os.getenv("PIPER_LENGTH_SCALE", "1.0").strip()

    def _get_voice(self):
        if self._voice is not None:
            return self._voice

        with self._voice_lock:
            if self._voice is not None:
                return self._voice

            if not self._model_path:
                raise RuntimeError(
                    "PIPER_MODEL_PATH is not set. Point it to your Piper .onnx model file."
                )

            model = Path(self._model_path)
            if not model.exists():
                raise RuntimeError(f"Piper model not found at: {model}")

            if self._config_path and not Path(self._config_path).exists():
                raise RuntimeError(f"Piper config not found at: {self._config_path}")

            try:
                piper_voice = importlib.import_module("piper.voice")
            except ImportError as exc:
                raise RuntimeError(
                    "piper-tts is not installed in the backend environment. Install with: pip install piper-tts"
                ) from exc

            PiperVoice = getattr(piper_voice, "PiperVoice", None)
            if PiperVoice is None:
                raise RuntimeError("Installed piper-tts package does not expose PiperVoice")

            load_sig = inspect.signature(PiperVoice.load)
            load_kwargs = {}
            if self._config_path and "config_path" in load_sig.parameters:
                load_kwargs["config_path"] = self._config_path

            self._voice = PiperVoice.load(self._model_path, **load_kwargs)
            return self._voice

    async def synthesize_wav(self, text: str) -> bytes:
        """Synthesize a text fragment into WAV audio bytes using local Piper."""
        if not text.strip():
            return b""

        def run_local_tts() -> bytes:
            voice = self._get_voice()
            synthesis_config = None
            try:
                piper_config = importlib.import_module("piper.config")
                SynthesisConfig = getattr(piper_config, "SynthesisConfig", None)
                if SynthesisConfig is not None:
                    synthesis_config = SynthesisConfig()
                    synthesis_config.length_scale = float(self._length_scale)
                    if self._speaker:
                        synthesis_config.speaker_id = int(self._speaker)
            except Exception:
                # Fall back to Piper defaults if config helpers are unavailable.
                synthesis_config = None

            chunks = list(voice.synthesize(text, syn_config=synthesis_config))
            if not chunks:
                return b""

            sample_rate = int(getattr(chunks[0], "sample_rate", 22050))
            sample_width = int(getattr(chunks[0], "sample_width", 2))
            sample_channels = int(getattr(chunks[0], "sample_channels", 1))
            pcm_bytes = b"".join(getattr(chunk, "audio_int16_bytes", b"") for chunk in chunks)

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(sample_channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_bytes)

            return wav_buffer.getvalue()

        async with self._semaphore:
            return await asyncio.to_thread(run_local_tts)

    async def synthesize_chunked_b64(self, text: str, chunk_size: int = 24576) -> AsyncGenerator[str, None]:
        """
        Yield base64-encoded WAV chunks for WebSocket JSON transport.

        Each yielded item is a complete WAV payload that the frontend can play
        directly. Do not split a single WAV by bytes because only the first
        byte-range contains a valid WAV header.
        """
        wav_bytes = await self.synthesize_wav(text)
        if not wav_bytes:
            return

        # Keep chunk_size arg for backward compatibility with callers.
        _ = chunk_size
        yield base64.b64encode(wav_bytes).decode("ascii")


def should_flush_sentence(text_buffer: str, first_fragment: bool = False) -> bool:
    """
    Decide when to flush text to TTS.

    Strategy:
    - First fragment: flush earlier to reduce time-to-first-audio.
    - Later fragments: flush less often to avoid robotic, over-segmented speech.
    """
    if not text_buffer:
        return False
    stripped = text_buffer.strip()
    if len(stripped) < 5:
        return False

    if stripped.endswith((".", "?", "!", "\n")):
        # Avoid ultra-short chunks; they sound abrupt with process-based TTS.
        return len(stripped) >= 16

    # Reduce initial latency by allowing an early first fragment.
    if first_fragment:
        if len(stripped) >= 24 and stripped.endswith((" ", ",", ";", ":")):
            return True
        return len(stripped) >= 36

    # After first audio starts, prefer larger chunks for smoother speech.
    if len(stripped) >= 120 and stripped.endswith((" ", ",", ";", ":")):
        return True

    return len(stripped) >= 180
