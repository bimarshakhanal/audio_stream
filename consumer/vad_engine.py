"""Silero VAD integration and speech segment builder for streaming chunks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
import soundfile as sf
import torch

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = load_silero_vad()
        self._model.to(self.device)
        LOGGER.info(f"Silero VAD loaded on device: {self.device}")

    def is_speech(self, chunk: np.ndarray) -> bool:
        """Return True if the chunk contains speech according to Silero VAD."""
        if len(chunk) == 0:
            return False

        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        chunk_tensor = torch.from_numpy(chunk).to(self.device)
        timestamps = self._get_speech_timestamps(
            chunk_tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate_hz,
            min_speech_duration_ms=self.min_speech_duration_ms,
        )
        return len(timestamps) > 0

    def get_speech_segments(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """Get speech segments from full audio array.

        Returns list of (start_sample, end_sample) tuples.
        """
        if len(audio) == 0:
            return []

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        audio_tensor = torch.from_numpy(audio).to(self.device)
        timestamps = self._get_speech_timestamps(   
            audio_tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate_hz,
            min_speech_duration_ms=self.min_speech_duration_ms,
        )
        return [(ts['start'], ts['end']) for ts in timestamps]


@dataclass
class SpeechSegmentBuilder:
    """Maintains speech state and writes finalized speech segments to WAV."""

    sample_rate_hz: int
    silence_threshold_ms: int = 700
    min_speech_seconds: float = 1.0
    output_dir: Path = Path("debug_chunks")
    _in_speech: bool = field(default=False, init=False)
    _segment_chunks: list[np.ndarray] = field(default_factory=list, init=False)
    _speech_samples: int = field(default=0, init=False)
    _silence_samples: int = field(default=0, init=False)
    _segment_start_time: float = field(default=0.0, init=False)

    _global_segment_index: ClassVar[int] = 0

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

    def _save_metadata(
        self,
        file_path: Path,
        start_time: float,
        end_time: float,
        speaker_id: str,
    ) -> None:
        """Save optional JSON metadata for the chunk."""
        metadata = {
            "filename": file_path.name,
            "start_time": start_time,
            "end_time": end_time,
            "speaker_id": speaker_id,
        }
        metadata_path = file_path.with_suffix(".json")
        import json
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _finalize_segment(self, speaker_id: str) -> Optional[Path]:
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

        self.__class__._global_segment_index += 1
        filename = f"chunk_{self.__class__._global_segment_index:03d}.wav"
        file_path = self.output_dir / filename
        print("fiLE PATH: ", file_path, flush=True)
        self._save_wav(full_segment, file_path)

        end_time = self._segment_start_time + duration_seconds
        self._save_metadata(
            file_path, self._segment_start_time, end_time, speaker_id
        )

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
        chunk_start_time: float,
        speaker_id: str,
    ) -> Optional[Path]:
        """Update speech state with a chunk; save file when a segment ends."""
        chunk_samples = len(chunk)

        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._segment_start_time = chunk_start_time
                LOGGER.info("Speech started at %.2fs", chunk_start_time)

            self._segment_chunks.append(chunk)
            self._speech_samples += chunk_samples
            self._silence_samples = 0
            return None

        if not self._in_speech:
            return None

        self._segment_chunks.append(chunk)
        self._silence_samples += chunk_samples

        if self._silence_samples >= self.silence_threshold_samples:
            return self._finalize_segment(speaker_id)

        return None

    def finalize_on_stream_end(self, speaker_id: str) -> Optional[Path]:
        """Finalize current segment when stream ends."""
        if not self._in_speech:
            return None
        return self._finalize_segment(speaker_id)
