"""Background inference worker and placeholder model integration."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


def run_qwen_inference(audio_array: np.ndarray) -> dict:
    """Placeholder for future Qwen2Audio integration.

    Keep this function isolated so a real model call can replace this logic
    without changing receiver or buffering flows.
    """
    # Simulate non-trivial model runtime.
    time.sleep(0.2)
    duration_s = len(audio_array) / 16_000.0
    return {
        "status": "ok",
        "duration_seconds": round(duration_s, 3),
        "num_samples": int(len(audio_array)),
    }


class InferenceWorker(threading.Thread):
    """Consumes audio windows from a queue and runs inference async."""

    def __init__(self, job_queue: "queue.Queue[Optional[np.ndarray]]") -> None:
        super().__init__(name="inference-worker", daemon=True)
        self._queue = job_queue
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)

    def run(self) -> None:
        LOGGER.info("Inference worker started")
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                self._queue.task_done()
                break

            try:
                result = run_qwen_inference(item)
                LOGGER.info("Inference completed: %s", result)
            except Exception:  # pylint: disable=broad-except
                LOGGER.exception("Inference failed")
            finally:
                self._queue.task_done()

        LOGGER.info("Inference worker stopped")
