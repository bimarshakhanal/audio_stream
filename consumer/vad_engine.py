"""Silero VAD integration and speech segment builder for streaming chunks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

LOGGER = logging.getLogger(__name__)


class SileroVADEngine:
    """Chunk-level Silero VAD wrapper.

    This engine evaluates each incoming chunk independently and returns
    whether speech is present in that chunk.
    """

    def __init__(
        self,
        sample_rate_hz: int = 16_000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 30,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms

        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "silero-vad is required. Install with: pip install silero-vad"
            ) from exc

        self._get_speech_timestamps = get_speech_timestamps
        self._model = load_silero_vad()

    def is_speech(self, chunk: np.ndarray) -> bool:
        """Return True if the chunk contains speech according to Silero VAD."""
        if len(chunk) == 0:
            return False

        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        timestamps = self._get_speech_timestamps(
            chunk,
            self._model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate_hz,
            min_speech_duration_ms=self.min_speech_duration_ms,
        )
        return len(timestamps) > 0


@dataclass
class SpeechSegmentBuilder:
    """Maintains speech state and writes finalized speech segments to WAV."""

    sample_rate_hz: int
    silence_threshold_ms: int = 700
    min_speech_seconds: float = 2.0
    output_dir: Path = Path("debug_chunks")
    _in_speech: bool = field(default=False, init=False)
    _segment_chunks: list[np.ndarray] = field(default_factory=list, init=False)
    _speech_samples: int = field(default=0, init=False)
    _silence_samples: int = field(default=0, init=False)
    _segment_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def silence_threshold_samples(self) -> int:
        return int(self.sample_rate_hz * (self.silence_threshold_ms / 1000.0))

    def _save_wav(self, audio: np.ndarray, file_path: Path) -> None:
        audio_clipped = np.clip(audio, -1.0, 1.0)
        sf.write(
            file=str(file_path),
            data=audio_clipped,
            samplerate=self.sample_rate_hz,
            subtype="PCM_16",
        )

    def _finalize_segment(self) -> Optional[Path]:
        if not self._segment_chunks:
            self._reset_state()
            return None

        duration_seconds = self._speech_samples / float(self.sample_rate_hz)
        full_segment = np.concatenate(self._segment_chunks)

        if duration_seconds < self.min_speech_seconds:
            LOGGER.info(
                "Speech ended (discarded): duration=%.2fs (< %.2fs)",
                duration_seconds,
                self.min_speech_seconds,
            )
            self._reset_state()
            return None

        self._segment_index += 1
        filename = f"chunk_{self._segment_index:03d}.wav"
        file_path = self.output_dir / filename
        self._save_wav(full_segment, file_path)

        LOGGER.info(
            "Speech ended: duration=%.2fs file=%s",
            duration_seconds,
            file_path,
        )
        self._reset_state()
        return file_path

    def _reset_state(self) -> None:
        self._in_speech = False
        self._segment_chunks = []
        self._speech_samples = 0
        self._silence_samples = 0

    def process_chunk(
        self,
        chunk: np.ndarray,
        is_speech: bool,
    ) -> Optional[Path]:
        """Update speech state with a chunk; save file when a segment ends."""
        chunk_samples = len(chunk)

        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                LOGGER.info("Speech started")

            self._segment_chunks.append(chunk)
            self._speech_samples += chunk_samples
            self._silence_samples = 0
            return None

        if not self._in_speech:
            return None

        self._segment_chunks.append(chunk)
        self._silence_samples += chunk_samples

        if self._silence_samples >= self.silence_threshold_samples:
            return self._finalize_segment()

        return None

    def finalize_on_stream_end(self) -> Optional[Path]:
        """Finalize current segment when stream ends."""
        if not self._in_speech:
            return None
        return self._finalize_segment()
