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
    """Placeholder for Qwen2Audio model inference.

    This function is isolated so the real Qwen2Audio integration can be added
    later without modifying the receiver or buffering logic.

    Args:
        audio_array: numpy float32 array of audio samples (16kHz mono)

    Returns:
        Dictionary with keys:
            - transcript: (str) Transcribed text from audio
            - technical_qa: (str) Technical question answering on content
            - response_reasoning: (str) Reasoning for the technical response
            - answer_rating: (str) Quality/confidence rating (e.g., "high", "medium", "low")
            - follow_up_question: (str) Suggested follow-up question based on content

    Note: This is a placeholder. Replace this with actual Qwen2Audio model call.
    The real integration should:
    1. Load audio into Qwen2Audio model
    2. Run inference to get transcript and analysis
    3. Return results in the schema above

    To avoid blocking audio reception, this is called from InferenceWorker
    (a background thread) which consumes from a queue.
    """
    # TODO: Replace with actual Qwen2Audio model inference
    # For now, simulate with delays and mock data

    duration_s = len(audio_array) / 16_000.0

    # Simulate model processing time (0.2s + 0.01s per second of audio)
    processing_time = 0.2 + (duration_s * 0.01)
    time.sleep(processing_time)

    # Mock response (replace with real model output)
    return {
        "transcript": "[PLACEHOLDER] Transcribed text from audio...",
        "technical_qa": "[PLACEHOLDER] Technical question answering response...",
        "response_reasoning": "[PLACEHOLDER] Reasoning for the technical response...",
        "answer_rating": "medium",
        "follow_up_question": "[PLACEHOLDER] Suggested follow-up question...",
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
                audio_duration = len(item) / 16_000.0
                LOGGER.info("Processing audio chunk: duration=%.2fs samples=%d", audio_duration, len(item))
                
                result = run_qwen_inference(item)
                
                LOGGER.info(
                    "Inference completed: transcript=%s, rating=%s",
                    result.get("transcript", "N/A")[:50],
                    result.get("answer_rating", "N/A"),
                )
            except Exception:
                LOGGER.exception("Inference failed")
            finally:
                self._queue.task_done()

        LOGGER.info("Inference worker stopped")
