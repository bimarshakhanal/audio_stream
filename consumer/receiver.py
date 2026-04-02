"""WebSocket consumer service.

Receives chunks over WebSocket and builds speech segments via VAD.
"""

from __future__ import annotations

import argparse
import base64
import asyncio
import json
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import websockets

from consumer.vad_engine import SileroVADEngine, SpeechSegmentBuilder
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
    """Coordinates receiver and VAD-based speech segment building."""

    def __init__(
        self,
        websocket_settings: Optional[WebSocketSettings] = None,
    ) -> None:
        self.websocket_settings = websocket_settings or WebSocketSettings()
        self.audio_settings = AudioSettings()
        self.runtime_settings = RuntimeSettings()

        self.stop_event = threading.Event()
        self.vad_engine = SileroVADEngine(
            sample_rate_hz=self.audio_settings.sample_rate_hz,
        )
        self.segment_builder = SpeechSegmentBuilder(
            sample_rate_hz=self.audio_settings.sample_rate_hz,
            silence_threshold_ms=700,
            min_speech_seconds=2.0,
            output_dir=Path("debug_chunks"),
        )
        self.receiver = WebSocketChunkReceiver(
            websocket_settings=self.websocket_settings,
            reconnect_delay=self.runtime_settings.reconnect_delay_seconds,
            on_chunk=self._handle_chunk,
            stop_event=self.stop_event,
        )

    def _decode_audio(self, payload_b64: str) -> np.ndarray:
        raw = base64.b64decode(payload_b64)
        return np.frombuffer(raw, dtype=np.float32)

    def _handle_chunk(self, chunk_fields: Dict[str, Any]) -> None:
        event_type = chunk_fields.get("event", "chunk")
        if event_type == "eos":
            LOGGER.info("Received end-of-stream event from producer")
            self.segment_builder.finalize_on_stream_end()
            self.stop_event.set()
            return

        chunk_id = chunk_fields.get("chunk_id")
        ts = chunk_fields.get("timestamp")
        payload_b64 = chunk_fields.get("payload")

        if not payload_b64:
            LOGGER.warning("Skipping chunk %s: missing payload", chunk_id)
            return

        audio_chunk = self._decode_audio(payload_b64)
        LOGGER.info(
            "Chunk arrived: id=%s timestamp=%s samples=%d",
            chunk_id,
            ts,
            len(audio_chunk),
        )

        has_speech = self.vad_engine.is_speech(audio_chunk)
        self.segment_builder.process_chunk(audio_chunk, has_speech)

    def _install_signal_handlers(self) -> None:
        def _graceful_stop(_signum: int, _frame: Any) -> None:
            LOGGER.info("Shutdown signal received")
            self.stop_event.set()

        signal.signal(signal.SIGINT, _graceful_stop)
        signal.signal(signal.SIGTERM, _graceful_stop)

    def run(self) -> None:
        self._install_signal_handlers()
        self.receiver.start()

        LOGGER.info("Consumer is running. Waiting for chunks...")
        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
        finally:
            self.stop_event.set()
            self.receiver.join(timeout=3)
            LOGGER.info("Consumer shutdown complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WebSocket audio chunk consumer"
    )
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
        "--log-level",
        default=RuntimeSettings().log_level,
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    LOGGER.info("Starting consumer service")
    app = StreamingConsumerApp(
        websocket_settings=WebSocketSettings(
            host=args.host,
            port=args.port,
        )
    )
    app.run()


if __name__ == "__main__":
    main()
