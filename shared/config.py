"""Shared configuration for the audio streaming prototype."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AudioSettings:
    sample_rate_hz: int = 16_000
    chunk_ms: int = 100
    buffer_seconds: int = 6
    step_seconds: int = 3

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate_hz * (self.chunk_ms / 1000.0))

    @property
    def buffer_samples(self) -> int:
        return self.sample_rate_hz * self.buffer_seconds

    @property
    def step_samples(self) -> int:
        return self.sample_rate_hz * self.step_seconds


@dataclass(frozen=True)
class RuntimeSettings:
    reconnect_delay_seconds: float = float(
        os.getenv("RECONNECT_DELAY_SECONDS", "2.0")
    )
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    buffer_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("BUFFER_TIMEOUT_SECONDS", "30"))
    )
    vad_silence_threshold_ms: int = field(
        default_factory=lambda: int(os.getenv("VAD_SILENCE_THRESHOLD_MS", "700"))
    )
    vad_min_speech_seconds: float = field(
        default_factory=lambda: float(os.getenv("VAD_MIN_SPEECH_SECONDS", "2.0"))
    )


@dataclass(frozen=True)
class WebSocketSettings:
    host: str = os.getenv("WS_HOST", "127.0.0.1")
    port: int = int(os.getenv("WS_PORT", "8765"))
    reconnect_delay_seconds: float = float(
        os.getenv("WS_RECONNECT_DELAY", "2.0")
    )


def configure_logging(level: str = "INFO") -> None:
    """Set process-wide logging format and level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
