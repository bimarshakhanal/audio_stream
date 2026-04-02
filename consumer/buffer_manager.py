"""Rolling buffer implementation for overlapped inference windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class RollingBufferManager:
    """Maintains a rolling buffer and emits fixed-size windows.

    The manager emits a 6-second window every 3 seconds (configurable),
    preserving overlap by retaining the latest samples.
    """

    sample_rate_hz: int
    buffer_seconds: float
    step_seconds: float
    _buffer: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32),
        init=False,
    )
    _total_received_samples: int = field(default=0, init=False)
    _next_trigger_sample: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.buffer_samples = int(self.sample_rate_hz * self.buffer_seconds)
        self.step_samples = int(self.sample_rate_hz * self.step_seconds)
        self._next_trigger_sample = self.buffer_samples

    def add_chunk(self, chunk: np.ndarray) -> List[np.ndarray]:
        """Append a chunk and return ready inference windows."""
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        self._buffer = np.concatenate([self._buffer, chunk])
        self._total_received_samples += len(chunk)

        if len(self._buffer) > self.buffer_samples:
            self._buffer = self._buffer[-self.buffer_samples:]

        windows: List[np.ndarray] = []
        while (
            self._total_received_samples >= self._next_trigger_sample
            and len(self._buffer) == self.buffer_samples
        ):
            windows.append(self._buffer.copy())
            self._next_trigger_sample += self.step_samples

        return windows

    @property
    def current_duration_seconds(self) -> float:
        return len(self._buffer) / float(self.sample_rate_hz)
