"""Background inference worker and placeholder model integration."""

from __future__ import annotations

import numpy as np
import os
import json

import logging
import queue
import threading
import time
import copy
from typing import Optional, List, Dict

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
)
import partial_json_parser

# Toggle to enable/disable calling the Qwen inference path. Set to '0'/'false' to disable.
load_dotenv()
_env_flag = os.getenv("ENABLE_QWEN_INFERENCE", "1")
ENABLE_QWEN_INFERENCE = str(_env_flag).strip().lower() in ("1", "true", "yes", "on")

# Toggle whether broadcasting results to the WebSocket results server is enabled.
# Set env var ENABLE_RESULTS_SERVER=0/false to disable broadcasting.
_env_rs_flag = os.getenv("ENABLE_RESULTS_SERVER", "1")
ENABLE_RESULTS_SERVER = str(_env_rs_flag).strip().lower() in ("1", "true", "yes", "on")


LOGGER = logging.getLogger(__name__)

from consumer.infer_utils import USER_PROMPT
# Import results server for broadcasting
try:
    from consumer.results_server import get_results_server
except ImportError:
    get_results_server = None  # type: ignore

# Shared history storage: flat list of all results from all speakers
shared_history: List[Dict] = []
shared_history_lock = threading.Lock()
MAX_HISTORY_SIZE = 8  # total items across all speakers


def format_history(history: List[Dict]) -> str:
    """
    Format conversation history to use as context to Qwen Model
    """
    context = ""
    for item in history:
        speaker = "inteviewer" if item["speaker"] == 1 else "candidate"
        context += f"{speaker}: {item['transcript']}\n"
    return context.strip()


def load_model() -> tuple:
    """Load Qwen2Audio processor and LoRA-adapted model.

    Reads `MODEL_NAME` and `LORA_PATH` from a `.env` file or environment.

    Returns:
        (processor, model)
    """
    # Load env vars if present
    try:
        load_dotenv()
    except Exception:
        # noop if dotenv not available
        pass

    model_name = os.getenv("MODEL_NAME") or os.getenv("QWEN2AUDIO_MODEL_NAME")
    lora_path = os.getenv("LORA_PATH") or os.getenv("QWEN2AUDIO_LORA_PATH")
    model_cache_path = os.getenv("QWEN_CACHE_DIR")

    if not model_name:
        raise RuntimeError("MODEL_NAME not set in environment or .env")
    if not lora_path:
        raise RuntimeError("LORA_PATH not set in environment or .env")

    LOGGER.info("Loading model and processor... model=%s lora=%s", model_name, lora_path)

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=model_cache_path)

    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", cache_dir=model_cache_path
    )

    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    LOGGER.info("Model and processor loaded successfully")
    return processor, model


class InferenceWorker(threading.Thread):
    """Consumes audio chunks (audio_array, speaker_id) tuples from a shared queue."""

    def __init__(self, job_queue: "queue.Queue") -> None:
        super().__init__(name="inference-worker-shared", daemon=True)
        self._queue = job_queue
        self._stop_event = threading.Event()
        self._chunk_counters: Dict[str, int] = {}  # per-speaker counters
        # Only load model if inference is enabled
        if ENABLE_QWEN_INFERENCE:
            self.processor, self.model = load_model()
        else:
            LOGGER.info("Qwen inference disabled; skipping model loading")
            self.processor = None
            self.model = None

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)

    def run_qwen_inference(
        self,
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

        To avoid blocking audio reception, this is called from InferenceWorker
        (a background thread) which consumes from a queue.
        """

        hist = history or []
        history_text = format_history(hist)
        prompt = USER_PROMPT.replace("<PREVIOUS CONTEXT>", history_text)

        convo = [{"role": "user", "content": [
            {"type": "text", "text": prompt}, 
            {"type": "audio", "audio": audio_array}
            ]}]
        
        text = self.processor.apply_chat_template(convo, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audios=[audio_array], return_tensors="pt", sampling_rate=16000)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generate_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
            )
        parsed = partial_json_parser.loads(response[0])

        print("Parsed Output: ", parsed)
        return {
            "speaker": speaker_id,
            "transcript": parsed.get("transcript") or "No transcript",
            "technical_qa": parsed.get("is_technical_qa") or False,
            "response_reasoning": parsed.get("response_reasoning") or "Reasoning unavailable.",
            "answer_rating": parsed.get("answer_rating") or "satisfactory",
            "follow_up_question": parsed.get("follow_up_question") or "No follow-up question.",
        }

    def run(self) -> None:
        LOGGER.info("Shared inference worker started")
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                self._queue.task_done()
                break

            try:
                # Backwards-compatible: support queued items of form
                # (audio_array, speaker_id) or (audio_array, speaker_id, json_path)
                json_path = None
                if isinstance(item, tuple) and len(item) == 3:
                    audio_array, speaker_id, json_path = item
                else:
                    audio_array, speaker_id = item
                audio_duration = len(audio_array) / 16_000.0

                if speaker_id not in self._chunk_counters:
                    self._chunk_counters[speaker_id] = 0

                LOGGER.info(
                    "Processing audio chunk (speaker=%s): duration=%.2fs samples=%d",
                    speaker_id,
                    audio_duration,
                    len(audio_array),
                )

                # Get a snapshot of shared history
                with shared_history_lock:
                    hist_snapshot = copy.deepcopy(shared_history)

                if ENABLE_QWEN_INFERENCE:
                    result = self.run_qwen_inference(
                        audio_array, speaker_id=speaker_id, history=hist_snapshot
                    )
                else:
                    LOGGER.info("Qwen inference disabled; emitting placeholder (speaker=%s)", speaker_id)
                    result = {
                        "speaker": speaker_id,
                        "transcript": "",
                        "technical_qa": False,
                        "response_reasoning": "",
                        "answer_rating": "disabled",
                        "follow_up_question": "",
                    }

                # Add metadata
                result["speaker_id"] = speaker_id
                result["chunk_number"] = self._chunk_counters[speaker_id]
                result["audio_duration_seconds"] = round(audio_duration, 3)
                self._chunk_counters[speaker_id] += 1

                # Append to shared history (bounded total)
                with shared_history_lock:
                    shared_history.append(copy.deepcopy(result))
                    if len(shared_history) > MAX_HISTORY_SIZE:
                        shared_history.pop(0)

                LOGGER.info(
                    "Inference completed (speaker=%s): transcript=%s, rating=%s",
                    speaker_id,
                    result.get("transcript", "N/A")[:50],
                    result.get("answer_rating", "N/A"),
                )

                # Save model output into the chunk's JSON metadata file if provided
                if json_path:
                    try:
                        try:
                            with open(json_path, "r", encoding="utf-8") as fh:
                                existing = json.load(fh)
                        except Exception:
                            existing = {}

                        existing["model_output"] = result
                        with open(json_path, "w", encoding="utf-8") as fh:
                            json.dump(existing, fh, indent=2)
                    except Exception:
                        LOGGER.exception("Failed to write model output to %s", json_path)

                # Broadcast result via WebSocket (only when enabled)
                if ENABLE_RESULTS_SERVER and get_results_server is not None:
                    server = get_results_server()
                    server.broadcast_result_async(result)

            except Exception:  # pylint: disable=broad-except
                LOGGER.exception("Inference failed")
            finally:
                self._queue.task_done()

        LOGGER.info("Shared inference worker stopped")
