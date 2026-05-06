"""Background inference worker and placeholder model integration."""

from __future__ import annotations

import logging
import queue
import threading
import time
import copy
from typing import Optional, List, Dict

import numpy as np

LOGGER = logging.getLogger(__name__)

from consumer.infer_utils import USER_PROMPT
# Import results server for broadcasting
try:
    from consumer.results_server import get_results_server
except ImportError:
    get_results_server = None  # type: ignore


def run_qwen_inference(
    audio_array: np.ndarray,
    speaker_id: str = "unknown",
    history: Optional[List[Dict]] = None,
) -> dict:
    """Placeholder for Qwen2Audio model inference.

    This function is isolated so the real Qwen2Audio integration can be added
    later without modifying the receiver or buffering logic.

    Args:
        audio_array: numpy float32 array of audio samples (16kHz mono)
        speaker_id: identifier of the speaker (e.g., "speaker_1", "speaker_2")
        history: optional list of past inference result dicts (most recent last)

    Returns:
        Dictionary with keys:
            - transcript: (str) Transcribed text from audio
            - technical_qa: (bool) Whether this contains a technical QA intent/answer
            - response_reasoning: (str) 1-2 sentence reasoning for the answer
            - answer_rating: (str) One of 'poor', 'satisfactory', 'excellent'
            - follow_up_question: (str) Suggested follow-up question based on content

    Note: This is a placeholder. Replace this with actual Qwen2Audio model call.
    The real integration should:
    1. Load audio into Qwen2Audio model
    2. Provide conversation/history context as an input
    3. Run inference to get transcript and analysis
    4. Return results in the schema above

    To avoid blocking audio reception, this is called from InferenceWorker
    (a background thread) which consumes from a queue.
    """
    duration_s = len(audio_array) / 16_000.0

    # Simulate model processing time (0.2s + 0.01s per second of audio)
    processing_time = 0.2 + (duration_s * 0.01)
    time.sleep(processing_time)

    # Example mock response that uses history length and speaker_id to vary output.
    history_len = len(history) if history is not None else 0

    return {
        "transcript": f"Transcribed text from audio... (speaker={speaker_id}, history={history_len})",
        "technical_qa": bool(history_len % 2 == 0),
        "response_reasoning": "Reasoning for the technical response...",
        "answer_rating": "satisfactory",
        "follow_up_question": "This is a follow-up question?",
    }


class InferenceWorker(threading.Thread):
    """Consumes audio windows from a queue and runs inference async."""

    def __init__(
        self,
        job_queue: "queue.Queue[Optional[np.ndarray]]",
        speaker_id: str = "unknown",
        history_max: int = 10,
    ) -> None:
        super().__init__(name=f"inference-worker-speaker{speaker_id}", daemon=True)
        self._queue = job_queue
        self._speaker_id = speaker_id
        self._stop_event = threading.Event()
        self._chunk_counter = 0
        self._history_max = int(history_max)
        self._history: List[Dict] = []

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)

    def run(self) -> None:
        LOGGER.info("Inference worker started for speaker %s", self._speaker_id)
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
                LOGGER.info(
                    "Processing audio chunk (speaker=%s): duration=%.2fs samples=%d",
                    self._speaker_id, 
                    audio_duration,
                    len(item),
                )
                
                # Provide a snapshot of history to the model so it can use past outputs
                history_snapshot = copy.deepcopy(self._history)
                result = run_qwen_inference(item, speaker_id=self._speaker_id, history=history_snapshot)

                # Add metadata
                result["speaker_id"] = self._speaker_id
                result["chunk_number"] = self._chunk_counter
                result["audio_duration_seconds"] = round(audio_duration, 3)
                self._chunk_counter += 1

                # Append a copy of the result to the in-memory history (bounded)
                try:
                    self._history.append(copy.deepcopy(result))
                    if len(self._history) > self._history_max:
                        # drop oldest
                        self._history.pop(0)
                except Exception as exc:
                    LOGGER.exception(
                        "Failed to update history for speaker %s (%s)", self._speaker_id, exc
                    )
                
                LOGGER.info(
                    "Inference completed (speaker=%s): transcript=%s, rating=%s",
                    self._speaker_id,
                    result.get("transcript", "N/A")[:50],
                    result.get("answer_rating", "N/A"),
                )
                
                # Broadcast result via WebSocket
                if get_results_server is not None:
                    server = get_results_server()
                    server.broadcast_result_async(result)
                        
            except Exception:  # pylint: disable=broad-except
                LOGGER.exception("Inference failed (speaker=%s)", self._speaker_id)
            finally:
                self._queue.task_done()

        LOGGER.info("Inference worker stopped for speaker %s", self._speaker_id)
