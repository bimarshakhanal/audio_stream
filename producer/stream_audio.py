"""Producer service that reads WAV audio files and streams via WebSocket."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import signal
import time
import asyncio
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import websockets

from shared.config import (
    AudioSettings,
    RuntimeSettings,
    WebSocketSettings,
    configure_logging,
)

LOGGER = logging.getLogger(__name__)


class MultiSpeakerStreamer:
    """Streams multiple WAV audio files as fixed-size chunks in real time."""

    def __init__(
        self,
        input_paths: Dict[str, Path],
        websocket_settings: WebSocketSettings,
        audio_settings: AudioSettings,
        runtime_settings: RuntimeSettings,
    ) -> None:
        self.input_paths = input_paths
        self.websocket_settings = websocket_settings
        self.audio_settings = audio_settings
        self.runtime_settings = runtime_settings
        self.stop_event = asyncio.Event()

    def _load_audio_float32(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        audio, sr = sf.read(str(path), dtype="float32")
        if sr != self.audio_settings.sample_rate_hz:
            raise ValueError(
                f"Sample rate mismatch: {sr} != {self.audio_settings.sample_rate_hz}"
            )

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        return audio

    def _make_chunk_message(
        self,
        speaker_id: str,
        chunk_id: int,
        chunk: np.ndarray,
        start_time: float,
    ) -> str:
        payload = base64.b64encode(chunk.astype(np.float32).tobytes()).decode("utf-8")
        message = {
            "event": "chunk",
            "speaker": speaker_id,
            "chunk_id": str(chunk_id),
            "timestamp": str(time.time()),
            "start_time": str(start_time),
            "payload": payload,
            "sample_rate_hz": str(self.audio_settings.sample_rate_hz),
        }
        return json.dumps(message)

    def _make_eos_message(self) -> str:
        return json.dumps({"event": "eos", "timestamp": str(time.time()), "payload": ""})

    def _install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        def _graceful_stop(_signum: int, _frame: Any) -> None:
            LOGGER.info("Shutdown signal received")
            loop.call_soon_threadsafe(self.stop_event.set)

        signal.signal(signal.SIGINT, _graceful_stop)
        signal.signal(signal.SIGTERM, _graceful_stop)

    async def _stream_to_client(self, websocket: Any) -> None:
        speaker_audios: Dict[str, np.ndarray] = {}
        for speaker_id, path in self.input_paths.items():
            speaker_audios[speaker_id] = self._load_audio_float32(path)
            duration = len(speaker_audios[speaker_id]) / self.audio_settings.sample_rate_hz
            LOGGER.info(
                "Loaded %s: %.2fs audio (%d samples)",
                path,
                duration,
                len(speaker_audios[speaker_id]),
            )

        chunk_samples = self.audio_settings.chunk_samples
        chunk_duration_s = self.audio_settings.chunk_ms / 1000.0

        speaker_chunks: Dict[str, List[np.ndarray]] = {}
        for speaker_id, audio in speaker_audios.items():
            chunks: List[np.ndarray] = []
            for offset in range(0, len(audio), chunk_samples):
                chunk = audio[offset: offset + chunk_samples]
                if len(chunk) == 0:
                    continue
                chunks.append(chunk)
            speaker_chunks[speaker_id] = chunks

        max_chunks = max(len(chunks) for chunks in speaker_chunks.values())

        start_wall = time.monotonic()
        global_chunk_id = 0

        for chunk_idx in range(max_chunks):
            if self.stop_event.is_set():
                break

            for speaker_id in sorted(self.input_paths.keys()):
                if chunk_idx >= len(speaker_chunks[speaker_id]):
                    continue

                chunk = speaker_chunks[speaker_id][chunk_idx]
                start_time = chunk_idx * chunk_duration_s

                await websocket.send(
                    self._make_chunk_message(speaker_id, global_chunk_id, chunk, start_time)
                )
                LOGGER.info(
                    "Emitted chunk speaker=%s id=%d start_time=%.2fs samples=%d",
                    speaker_id,
                    global_chunk_id,
                    start_time,
                    len(chunk),
                )

                global_chunk_id += 1

                next_deadline = start_wall + global_chunk_id * chunk_duration_s
                sleep_time = max(0.0, next_deadline - time.monotonic())
                await asyncio.sleep(sleep_time)

        await websocket.send(self._make_eos_message())
        LOGGER.info("End-of-stream event sent")

    async def _handler(self, websocket: Any) -> None:
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
        description="Stream WAV audio files to WebSocket in real time"
    )
    parser.add_argument(
        "--speaker1",
        default=os.getenv("SPEAKER1_PATH", None),
        required=False,
        help="Path to speaker 1 WAV file (or set SPEAKER1_PATH in .env)",
    )
    parser.add_argument(
        "--speaker2",
        default=os.getenv("SPEAKER2_PATH", None),
        required=False,
        help="Path to speaker 2 WAV file (or set SPEAKER2_PATH in .env)",
    )
    parser.add_argument(
        "--host",
        default=WebSocketSettings().host,
        help="WebSocket host to bind (or set WS_HOST in .env)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=WebSocketSettings().port,
        help="WebSocket port to bind (or set WS_PORT in .env)",
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

    speaker1 = args.speaker1 or os.getenv('SPEAKER1_PATH')
    speaker2 = args.speaker2 or os.getenv('SPEAKER2_PATH')

    if not speaker1 or not speaker2:
        raise ValueError('Both speaker1 and speaker2 paths are required either via CLI or .env')

    input_paths = {
        "1": Path(speaker1),
        "2": Path(speaker2),
    }

    streamer = MultiSpeakerStreamer(
        input_paths=input_paths,
        websocket_settings=WebSocketSettings(host=args.host, port=args.port),
        audio_settings=AudioSettings(),
        runtime_settings=RuntimeSettings(),
    )
    asyncio.run(streamer.run())


if __name__ == "__main__":
    main()
