"""Background inference worker and placeholder model integration."""

from __future__ import annotations

import numpy as np
import os

import logging
import queue
import threading
import time
import copy
from typing import Optional, List, Dict

from dotenv import load_dotenv
from peft import PeftModel
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    TextIteratorStreamer,
)
import partial_json_parser

# Toggle to enable/disable calling the Qwen inference path. Set to '0'/'false' to disable.
load_dotenv()
_env_flag = os.getenv("ENABLE_QWEN_INFERENCE", "1")
ENABLE_QWEN_INFERENCE = str(_env_flag).strip().lower() in ("1", "true", "yes", "on")


LOGGER = logging.getLogger(__name__)

from consumer.infer_utils import USER_PROMPT
# Import results server for broadcasting
try:
    from consumer.results_server import get_results_server
except ImportError:
    get_results_server = None  # type: ignore


def format_history(history):
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
        history: Optional[List[Dict]] = [],
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

        history_text = format_history(history)
        prompt = USER_PROMPT.replace("<PREVIOUS CONTEXT>", history_text)

        convo = [{"role": "user", "content": [
            {"type": "text", "text": prompt}, 
            {"type": "audio", "audio": "test.wav"}
            ]}]
        
        text = self.processor.apply_chat_template(convo, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=audio_array, return_tensors="pt", sampling_rate=16000).to(self.model.device)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generate_ids = self.model.generate(**inputs, max_length=2048, temperature=1e-3)
        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
            )
        parsed = partial_json_parser.loads(response[0])
        return {
            "speaker": speaker_id,
            "transcript": parsed["transcript"] or "No transcript",
            "technical_qa": parsed["is_technical_qa"] or False,
            "response_reasoning": parsed["response_reasoning"] or "Reasoning for the technical response...",
            "answer_rating": parsed["answer_rating"] or "satisfactory",
            "follow_up_question": parsed["answer_rating"] or "This is a follow-up question?",
        }

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
                if ENABLE_QWEN_INFERENCE:
                    result = self.run_qwen_inference(item, speaker_id=self._speaker_id, history=history_snapshot)
                else:
                    LOGGER.info("Qwen inference disabled by env; emitting placeholder result (speaker=%s)", self._speaker_id)
                    result = {
                        "speaker": self._speaker_id,
                        "transcript": "",
                        "technical_qa": False,
                        "response_reasoning": "",
                        "answer_rating": "disabled",
                        "follow_up_question": "",
                    }

                print("Result: ", result)

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
