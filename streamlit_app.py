"""Minimal Streamlit UI for Qwen chunk outputs."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from collections import defaultdict
from datetime import datetime

import streamlit as st
import websockets

LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="Qwen Output",
    layout="wide",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = defaultdict(list)  # {speaker_id: [messages]}
if "connected" not in st.session_state:
    st.session_state.connected = False
if "ws_uri" not in st.session_state:
    st.session_state.ws_uri = "ws://127.0.0.1:8766"
if "listener_thread" not in st.session_state:
    st.session_state.listener_thread = None
if "listener_stop_event" not in st.session_state:
    st.session_state.listener_stop_event = None
if "incoming_queue" not in st.session_state:
    st.session_state.incoming_queue = queue.Queue()
if "last_error" not in st.session_state:
    st.session_state.last_error = ""


async def _listen_forever(uri: str, stop_event: threading.Event, out_queue: queue.Queue) -> None:
    """Background websocket listener that pushes received results into a queue."""
    while not stop_event.is_set():
        try:
            async with websockets.connect(uri, max_size=None) as websocket:
                LOGGER.info("Connected to results server: %s", uri)
                out_queue.put({"_control": "connected"})
                while not stop_event.is_set():
                    try:
                        raw_message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    except TimeoutError:
                        continue
                    result = json.loads(raw_message)
                    out_queue.put({"_control": "data", "payload": result})
        except OSError as exc:
            out_queue.put({"_control": "error", "payload": str(exc)})
            await asyncio.sleep(1.0)
        except json.JSONDecodeError as exc:
            LOGGER.error("Failed to decode message: %s", exc)
        except Exception as exc:  # pylint: disable=broad-except
            out_queue.put({"_control": "error", "payload": str(exc)})
            await asyncio.sleep(1.0)


def _listener_thread_target(uri: str, stop_event: threading.Event, out_queue: queue.Queue) -> None:
    """Thread target that runs websocket listener on its own event loop."""
    asyncio.run(_listen_forever(uri, stop_event, out_queue))


def start_listener(uri: str) -> None:
    """Start background websocket listener."""
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_listener_thread_target,
        args=(uri, stop_event, st.session_state.incoming_queue),
        name="streamlit-results-listener",
        daemon=True,
    )
    thread.start()
    st.session_state.listener_stop_event = stop_event
    st.session_state.listener_thread = thread


def stop_listener() -> None:
    """Stop background websocket listener."""
    stop_event = st.session_state.listener_stop_event
    thread = st.session_state.listener_thread
    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=2)
    st.session_state.listener_stop_event = None
    st.session_state.listener_thread = None
    st.session_state.connected = False


def drain_incoming_results() -> int:
    """Drain received websocket messages from queue into session state."""
    drained = 0
    while True:
        try:
            item = st.session_state.incoming_queue.get_nowait()
        except queue.Empty:
            break

        drained += 1
        kind = item.get("_control")
        if kind == "connected":
            st.session_state.connected = True
            st.session_state.last_error = ""
            continue
        if kind == "error":
            st.session_state.last_error = item.get("payload", "Unknown websocket error")
            continue
        if kind != "data":
            continue

        result = item.get("payload", {})
        speaker_id = str(result.get("speaker_id", "unknown"))
        message_data = {
            "speaker_id": speaker_id,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "chunk_number": result.get("chunk_number", 0),
            "transcript": result.get("transcript", "N/A"),
            "technical_qa": bool(result.get("technical_qa", False)),
            "response_reasoning": result.get("response_reasoning", "N/A"),
            "answer_rating": str(result.get("answer_rating", "satisfactory")).strip().lower(),
            "follow_up_question": result.get("follow_up_question", "N/A"),
            "audio_duration": result.get("audio_duration_seconds", 0),
        }
        st.session_state.messages[speaker_id].append(message_data)
    return drained


def _rating_badge(rating: str) -> str:
    rating = (rating or "satisfactory").lower()
    if rating == "excellent":
        return "🟢 Excellent"
    if rating == "poor":
        return "🔴 Poor"
    return "🟡 Satisfactory"


def display_message(message: dict) -> None:
    """Display model output in a UI-friendly compact card."""
    speaker = message.get("speaker_id", "unknown")
    technical_flag = "✅ True" if message.get("technical_qa", False) else "❌ False"
    rating_label = _rating_badge(message.get("answer_rating", "satisfactory"))

    with st.container(border=True):
        st.markdown(f"**Speaker:** {speaker}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**TechnicalQA:** {technical_flag}")
        with col2:
            st.markdown(f"**Answer Rating:** {rating_label}")

        st.markdown("**Transcript**")
        st.write(message["transcript"])

        st.markdown("**Reasoning**")
        st.write(message["response_reasoning"])

        st.markdown("**Followup**")
        st.write(message["follow_up_question"])


def main() -> None:
    """Main Streamlit app."""
    st.title("Qwen Output")

    drained_count = drain_incoming_results()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        ws_uri = st.text_input(
            "Results Server URI",
            value=st.session_state.ws_uri,
            help="WebSocket URI of the results server",
        )
        st.session_state.ws_uri = ws_uri
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Connect", use_container_width=True):
                if st.session_state.listener_thread is None:
                    start_listener(ws_uri)
                st.session_state.connected = True
        
        with col2:
            if st.button("Disconnect", use_container_width=True):
                stop_listener()

        if st.button("Clear", use_container_width=True):
            st.session_state.messages = defaultdict(list)

        if st.session_state.connected:
            st.success("Connected")
        else:
            st.warning("Disconnected")

        if st.session_state.last_error:
            st.error(f"WebSocket: {st.session_state.last_error}")

        if drained_count:
            st.caption(f"Received {drained_count}")
    
    # Main output area
    if not st.session_state.messages:
        st.write("Waiting for results...")
    else:
        all_messages = []
        for speaker_id in sorted(st.session_state.messages.keys()):
            for message in st.session_state.messages[speaker_id]:
                all_messages.append(message)

        all_messages.sort(key=lambda x: x["timestamp"], reverse=True)

        for message in all_messages:
            display_message(message)

    if st.session_state.connected:
        time.sleep(0.7)
        st.rerun()


if __name__ == "__main__":
    main()
