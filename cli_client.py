"""
Minimal WebSocket listener (no Streamlit).

- Connects to a WebSocket server.
- Receives JSON messages from the model.
- Stores messages in ordered format.
- Periodically saves results to JSON and CSV.

Expected incoming JSON example:
{
    "speaker_id": "speaker_1",
    "chunk_number": 3,
    "transcript": "Hello world",
    "technical_qa": true,
    "response_reasoning": "Correct technical explanation.",
    "answer_rating": "excellent",
    "follow_up_question": "Can you explain more?",
    "audio_duration_seconds": 4.2
}
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import websockets

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@dataclass
class Message:
    speaker_id: str
    timestamp: str
    chunk_number: int
    transcript: str
    technical_qa: bool
    response_reasoning: str
    answer_rating: str
    follow_up_question: str
    audio_duration: float


class ResultStore:
    """Stores incoming messages and exports them."""

    def __init__(self) -> None:
        self.messages: dict[str, list[Message]] = defaultdict(list)

    def add(self, result: dict) -> None:
        speaker_id = str(result.get("speaker_id", "unknown"))

        msg = Message(
            speaker_id=speaker_id,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            chunk_number=int(result.get("chunk_number", 0)),
            transcript=result.get("transcript", ""),
            technical_qa=bool(result.get("technical_qa", False)),
            response_reasoning=result.get("response_reasoning", ""),
            answer_rating=str(
                result.get("answer_rating", "satisfactory")
            ).strip().lower(),
            follow_up_question=result.get("follow_up_question", ""),
            audio_duration=float(result.get("audio_duration_seconds", 0.0)),
        )

        self.messages[speaker_id].append(msg)

    def get_all_ordered(self) -> list[Message]:
        """Return all messages sorted by timestamp and chunk number."""
        all_messages = [
            message
            for speaker_messages in self.messages.values()
            for message in speaker_messages
        ]

        all_messages.sort(
            key=lambda m: (m.timestamp, m.chunk_number, m.speaker_id)
        )
        return all_messages

    def save_json(self, path: str | Path = "results.json") -> None:
        ordered = self.get_all_ordered()
        data = [asdict(msg) for msg in ordered]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        LOGGER.info("Saved JSON to %s", path)

    def save_csv(self, path: str | Path = "results.csv") -> None:
        ordered = self.get_all_ordered()
        df = pd.DataFrame([asdict(msg) for msg in ordered])
        df.to_csv(path, index=False)

        LOGGER.info("Saved CSV to %s", path)

    def print_latest(self, n: int = 5) -> None:
        ordered = self.get_all_ordered()

        print("\n" + "=" * 80)
        print(f"Total messages: {len(ordered)}")
        print("=" * 80)

        for msg in ordered[-n:]:
            print(
                f"[{msg.timestamp}] "
                f"{msg.speaker_id} "
                f"(chunk {msg.chunk_number}, rating={msg.answer_rating})"
            )
            print(f"Transcript: {msg.transcript}")
            print(f"Reasoning : {msg.response_reasoning}")
            print(f"Follow-up : {msg.follow_up_question}")
            print("-" * 80)


async def listen_forever(uri: str, store: ResultStore) -> None:
    """Listen to WebSocket server and store incoming results."""
    while True:
        try:
            LOGGER.info("Connecting to %s", uri)

            async with websockets.connect(uri, max_size=None) as websocket:
                LOGGER.info("Connected")

                async for raw_message in websocket:
                    try:
                        result = json.loads(raw_message)
                    except json.JSONDecodeError:
                        LOGGER.warning("Invalid JSON received")
                        continue

                    store.add(result)

                    # Show latest message(s)
                    store.print_latest(n=3)

                    # Save after every message
                    store.save_json("results/results.json")
                    store.save_csv("results/results.csv")

        except OSError as exc:
            LOGGER.error("Connection error: %s", exc)

        except Exception as exc:
            LOGGER.exception("Unexpected error: %s", exc)

        LOGGER.info("Reconnecting in 1 second...")
        await asyncio.sleep(1)


async def main() -> None:
    uri = "ws://127.0.0.1:8766"
    store = ResultStore()

    try:
        await listen_forever(uri, store)
    except KeyboardInterrupt:
        LOGGER.info("Stopping...")
        store.save_json("results/results_final.json")
        store.save_csv("resulst/results_final.csv")


if __name__ == "__main__":
    asyncio.run(main())