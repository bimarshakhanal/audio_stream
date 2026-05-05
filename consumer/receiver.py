"""WebSocket consumer service.

Receives speaker-tagged chunks, buffers them per speaker, applies VAD on full
buffers, and saves finalized speech segments with metadata in per-speaker dirs.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import queue
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import soundfile as sf
import websockets

from consumer.inference_worker import InferenceWorker
from consumer.vad_engine import SileroVADEngine
from shared.config import (
    AudioSettings,
    RuntimeSettings,
    WebSocketSettings,
    configure_logging,
)

LOGGER = logging.getLogger(__name__)


class WebSocketChunkReceiver(threading.Thread):
    """Reads audio chunks from WebSocket with reconnect handling."""

    def __init__(
        self,
        websocket_settings: WebSocketSettings,
        reconnect_delay: float,
        on_chunk: Callable[[Dict[str, Any]], None],
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="websocket-receiver", daemon=True)
        self._settings = websocket_settings
        self._reconnect_delay = reconnect_delay
        self._on_chunk = on_chunk
        self._stop_event = stop_event
        self._url = f"ws://{self._settings.host}:{self._settings.port}"

    async def _run_client(self) -> None:
        while not self._stop_event.is_set():
            try:
                LOGGER.info("Connecting to producer at %s", self._url)
                async with websockets.connect(self._url, max_size=None) as ws:
                    LOGGER.info("Connected to producer")
                    async for raw_message in ws:
                        if self._stop_event.is_set():
                            break
                        chunk_fields = json.loads(raw_message)
                        self._on_chunk(chunk_fields)
            except websockets.ConnectionClosed:
                LOGGER.warning("WebSocket closed; reconnecting")
            except OSError:
                LOGGER.exception("WebSocket connection error; reconnecting")
            except Exception:  # pylint: disable=broad-except
                LOGGER.exception("Unexpected receiver error")

            if self._stop_event.is_set():
                break

            await asyncio.sleep(self._reconnect_delay)

    def run(self) -> None:
        asyncio.run(self._run_client())


class StreamingConsumerApp:
    """Coordinates receiver, per-speaker buffering, VAD, and saving."""

    def __init__(
        self,
        websocket_settings: Optional[WebSocketSettings] = None,
        run_name: str = "default",
    ) -> None:
        self.run_name = run_name
        self.websocket_settings = websocket_settings or WebSocketSettings()
        self.audio_settings = AudioSettings()
        self.runtime_settings = RuntimeSettings()

        self.stop_event = threading.Event()

        self.speaker_ids = ["1", "2"]
        self.speaker_vads: Dict[str, SileroVADEngine] = {
            speaker_id: SileroVADEngine(
                sample_rate_hz=self.audio_settings.sample_rate_hz,
            )
            for speaker_id in self.speaker_ids
        }

        self.speaker_buffers: Dict[str, list[np.ndarray]] = {
            speaker_id: [] for speaker_id in self.speaker_ids
        }
        self.speaker_buffer_samples: Dict[str, int] = {
            speaker_id: 0 for speaker_id in self.speaker_ids
        }
        self.speaker_buffer_start_time: Dict[str, Optional[float]] = {
            speaker_id: None for speaker_id in self.speaker_ids
        }
        self.speaker_in_speech: Dict[str, bool] = {
            speaker_id: False for speaker_id in self.speaker_ids
        }
        self.speaker_silence_samples: Dict[str, int] = {
            speaker_id: 0 for speaker_id in self.speaker_ids
        }
        self.speaker_chunk_ids: Dict[str, int] = {
            speaker_id: 0 for speaker_id in self.speaker_ids
        }

        self.buffer_timeout_samples = int(
            self.runtime_settings.buffer_timeout_seconds * self.audio_settings.sample_rate_hz
        )

        self.inference_queue: queue.Queue = queue.Queue()
        self.inference_worker = InferenceWorker(self.inference_queue)

        self.receiver = WebSocketChunkReceiver(
            websocket_settings=self.websocket_settings,
            reconnect_delay=self.runtime_settings.reconnect_delay_seconds,
            on_chunk=self._handle_chunk,
            stop_event=self.stop_event,
        )

    def _decode_audio(self, payload_b64: str) -> np.ndarray:
        raw = base64.b64decode(payload_b64)
        return np.frombuffer(raw, dtype=np.float32)

    def _speaker_output_dir(self, speaker_id: str) -> Path:
        return Path("chunks") / self.run_name / f"speaker_{speaker_id}"

    def _reset_speaker_buffer(self, speaker_id: str) -> None:
        self.speaker_buffers[speaker_id] = []
        self.speaker_buffer_samples[speaker_id] = 0
        self.speaker_buffer_start_time[speaker_id] = None
        self.speaker_in_speech[speaker_id] = False
        self.speaker_silence_samples[speaker_id] = 0

    def _flush_speaker_buffer(self, speaker_id: str) -> None:
        buffered_chunks = self.speaker_buffers[speaker_id]
        if not buffered_chunks:
            return

        full_audio = np.concatenate(buffered_chunks)
        buffer_start_time = self.speaker_buffer_start_time[speaker_id] or 0.0

        LOGGER.info(
            "Flushing speaker=%s buffer: samples=%d duration=%.2fs",
            speaker_id,
            len(full_audio),
            len(full_audio) / self.audio_settings.sample_rate_hz,
        )

        speech_segments = self.speaker_vads[speaker_id].get_speech_segments(full_audio)
        if not speech_segments:
            LOGGER.info("No speech detected for speaker %s in flushed buffer", speaker_id)
            self._reset_speaker_buffer(speaker_id)
            return

        speech_audio_parts = [full_audio[start:end] for start, end in speech_segments]
        filtered_audio = np.concatenate(speech_audio_parts)

        self.speaker_chunk_ids[speaker_id] += 1
        chunk_id = self.speaker_chunk_ids[speaker_id]
        output_dir = self._speaker_output_dir(speaker_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        wav_path = output_dir / f"chunk_{chunk_id:04d}.wav"
        json_path = wav_path.with_suffix(".json")

        first_segment_start = speech_segments[0][0]
        last_segment_end = speech_segments[-1][1]
        metadata = {
            "filename": wav_path.name,
            "speaker_id": speaker_id,
            "chunk_id": chunk_id,
            "start_time": buffer_start_time
            + (first_segment_start / self.audio_settings.sample_rate_hz),
            "end_time": buffer_start_time
            + (last_segment_end / self.audio_settings.sample_rate_hz),
            "num_samples": int(len(filtered_audio)),
            "duration_seconds": float(
                len(filtered_audio) / self.audio_settings.sample_rate_hz
            ),
        }

        sf.write(
            file=str(wav_path),
            data=np.clip(filtered_audio, -1.0, 1.0),
            samplerate=self.audio_settings.sample_rate_hz,
            subtype="PCM_16",
        )
        with open(json_path, "w", encoding="utf-8") as file_handle:
            json.dump(metadata, file_handle, indent=2)

        self.inference_queue.put(filtered_audio)
        LOGGER.info(
            "Saved speaker=%s chunk=%d path=%s samples=%d",
            speaker_id,
            chunk_id,
            wav_path,
            len(filtered_audio),
        )

        self._reset_speaker_buffer(speaker_id)

    def _handle_chunk(self, chunk_fields: Dict[str, Any]) -> None:
        event_type = chunk_fields.get("event", "chunk")
        if event_type == "eos":
            LOGGER.info("Received end-of-stream event from producer")
            self._flush_all_pending()
            self.stop_event.set()
            return

        speaker_id = chunk_fields.get("speaker")
        if not speaker_id:
            LOGGER.warning("Skipping chunk: missing speaker ID")
            return

        if speaker_id not in self.speaker_vads:
            LOGGER.warning("Skipping chunk for unknown speaker: %s", speaker_id)
            return

        chunk_id = chunk_fields.get("chunk_id")
        timestamp = chunk_fields.get("timestamp")
        start_time = float(chunk_fields.get("start_time", "0"))
        payload_b64 = chunk_fields.get("payload")

        if not payload_b64:
            LOGGER.warning("Skipping chunk %s: missing payload", chunk_id)
            return

        audio_chunk = self._decode_audio(payload_b64)
        LOGGER.info(
            "Chunk arrived: speaker=%s id=%s timestamp=%s start_time=%.2fs samples=%d",
            speaker_id,
            chunk_id,
            timestamp,
            start_time,
            len(audio_chunk),
        )

        if not self.speaker_buffers[speaker_id]:
            self.speaker_buffer_start_time[speaker_id] = start_time

        self.speaker_buffers[speaker_id].append(audio_chunk)
        self.speaker_buffer_samples[speaker_id] += len(audio_chunk)

        is_speech = self.speaker_vads[speaker_id].is_speech(audio_chunk)
        LOGGER.debug("Speech detected for speaker %s: %s", speaker_id, is_speech)

        if is_speech:
            self.speaker_in_speech[speaker_id] = True
            self.speaker_silence_samples[speaker_id] = 0
        elif self.speaker_in_speech[speaker_id]:
            self.speaker_silence_samples[speaker_id] += len(audio_chunk)

        if self.speaker_buffer_samples[speaker_id] >= self.buffer_timeout_samples:
            LOGGER.info(
                "Buffer timeout for speaker %s: %d samples accumulated (>= %.0fs); flushing",
                speaker_id,
                self.speaker_buffer_samples[speaker_id],
                self.runtime_settings.buffer_timeout_seconds,
            )
            self._flush_speaker_buffer(speaker_id)
            return

        # During continuous buffering (up to 30s), we don't flush on silence.
        # Silence detection would break long utterances with natural pauses.
        # Only the buffer timeout (30s) triggers a flush during active recording.

    def _flush_all_pending(self) -> None:
        for speaker_id in self.speaker_ids:
            if self.speaker_buffers[speaker_id]:
                self._flush_speaker_buffer(speaker_id)

    def _install_signal_handlers(self) -> None:
        def _graceful_stop(_signum: int, _frame: Any) -> None:
            LOGGER.info("Shutdown signal received")
            self.stop_event.set()

        signal.signal(signal.SIGINT, _graceful_stop)
        signal.signal(signal.SIGTERM, _graceful_stop)

    def run(self) -> None:
        self._install_signal_handlers()
        self.inference_worker.start()
        self.receiver.start()

        LOGGER.info("Consumer is running. Waiting for chunks...")
        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
        finally:
            self.stop_event.set()
            self.receiver.join(timeout=3)
            self.inference_worker.stop()
            self.inference_worker.join(timeout=3)
            LOGGER.info("Consumer shutdown complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebSocket audio chunk consumer")
    parser.add_argument(
        "--host",
        default=WebSocketSettings().host,
        help="Producer WebSocket host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=WebSocketSettings().port,
        help="Producer WebSocket port",
    )
    parser.add_argument(
        "--run-name",
        default="default",
        help="Unique name for this run (used in output directory)",
    )
    parser.add_argument(
        "--log-level",
        default=RuntimeSettings().log_level,
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def main() -> None:
    import dotenv

    dotenv.load_dotenv()
    args = parse_args()
    configure_logging(args.log_level)
    LOGGER.info("Starting consumer service")
    app = StreamingConsumerApp(
        websocket_settings=WebSocketSettings(host=args.host, port=args.port),
        run_name=args.run_name,
    )
    app.run()


if __name__ == "__main__":
    main()
