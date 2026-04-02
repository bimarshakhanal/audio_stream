"""Producer service that reads MP4 audio and streams via WebSocket."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import signal
import subprocess
import time
import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import websockets

from shared.config import (
    AudioSettings,
    RuntimeSettings,
    WebSocketSettings,
    configure_logging,
)

LOGGER = logging.getLogger(__name__)


class Mp4AudioStreamer:
    """Extracts audio from MP4 and streams fixed-size chunks in real time."""

    def __init__(
        self,
        input_path: Path,
        websocket_settings: WebSocketSettings,
        audio_settings: AudioSettings,
        runtime_settings: RuntimeSettings,
    ) -> None:
        self.input_path = input_path
        self.websocket_settings = websocket_settings
        self.audio_settings = audio_settings
        self.runtime_settings = runtime_settings
        self.stop_event = asyncio.Event()

    def _load_audio_float32(self) -> np.ndarray:
        """Decode MP4 audio into mono 16kHz PCM float32 using ffmpeg.

        Future extension: replace this batch decode with a true frame-by-frame
        live source reader (RTSP/WebRTC/microphone).
        """
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(self.input_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.audio_settings.sample_rate_hz),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ]

        LOGGER.info("Decoding audio from MP4: %s", self.input_path)
        result = subprocess.run(cmd, check=True, capture_output=True)
        audio = np.frombuffer(result.stdout, dtype=np.float32)
        LOGGER.info(
            "Decoded %.2fs audio (%d samples)",
            len(audio) / self.audio_settings.sample_rate_hz,
            len(audio),
        )
        return audio

    def _make_chunk_message(self, chunk_id: int, chunk: np.ndarray) -> str:
        payload = base64.b64encode(
            chunk.astype(np.float32).tobytes()
        ).decode("utf-8")
        message = {
            "event": "chunk",
            "chunk_id": str(chunk_id),
            "timestamp": str(time.time()),
            "payload": payload,
            "sample_rate_hz": str(self.audio_settings.sample_rate_hz),
        }
        return json.dumps(message)

    def _make_eos_message(self) -> str:
        message = {
            "event": "eos",
            "timestamp": str(time.time()),
            "payload": "",
        }
        return json.dumps(message)

    def _install_signal_handlers(
        self,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        def _graceful_stop(_signum: int, _frame: Any) -> None:
            LOGGER.info("Shutdown signal received")
            loop.call_soon_threadsafe(self.stop_event.set)

        signal.signal(signal.SIGINT, _graceful_stop)
        signal.signal(signal.SIGTERM, _graceful_stop)

    async def _stream_to_client(
        self,
        websocket: Any,
    ) -> None:
        audio = self._load_audio_float32()
        chunk_samples = self.audio_settings.chunk_samples
        chunk_duration_s = self.audio_settings.chunk_ms / 1000.0

        start_wall = time.monotonic()
        chunk_id = 0

        for offset in range(0, len(audio), chunk_samples):
            if self.stop_event.is_set():
                break

            chunk = audio[offset: offset + chunk_samples]
            if len(chunk) == 0:
                continue

            await websocket.send(self._make_chunk_message(chunk_id, chunk))
            LOGGER.info("Emitted chunk id=%d samples=%d", chunk_id, len(chunk))

            chunk_id += 1
            next_deadline = start_wall + chunk_id * chunk_duration_s
            sleep_time = max(0.0, next_deadline - time.monotonic())
            await asyncio.sleep(sleep_time)

        await websocket.send(self._make_eos_message())
        LOGGER.info("End-of-stream event sent")
        LOGGER.info("Producer finished")

    async def _handler(
        self,
        websocket: Any,
    ) -> None:
        LOGGER.info("Consumer connected over WebSocket")
        try:
            await self._stream_to_client(websocket)
        except websockets.ConnectionClosed:
            LOGGER.warning("Consumer disconnected during streaming")
        finally:
            self.stop_event.set()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._install_signal_handlers(loop)

        async with websockets.serve(
            self._handler,
            self.websocket_settings.host,
            self.websocket_settings.port,
            max_size=None,
        ):
            LOGGER.info(
                "WebSocket producer listening at ws://%s:%d",
                self.websocket_settings.host,
                self.websocket_settings.port,
            )
            LOGGER.info("Waiting for a consumer connection...")
            await self.stop_event.wait()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream MP4 audio to WebSocket in real time"
    )
    parser.add_argument("--input", required=True, help="Path to MP4 file")
    parser.add_argument(
        "--host",
        default=WebSocketSettings().host,
        help="WebSocket host to bind",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=WebSocketSettings().port,
        help="WebSocket port to bind",
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

    streamer = Mp4AudioStreamer(
        input_path=Path(args.input),
        websocket_settings=WebSocketSettings(
            host=args.host,
            port=args.port,
        ),
        audio_settings=AudioSettings(),
        runtime_settings=RuntimeSettings(),
    )
    asyncio.run(streamer.run())


if __name__ == "__main__":
    main()
