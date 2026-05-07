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
import streamlit.components.v1 as components
import websockets

LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="",
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
        return "🟢"
    if rating == "poor":
        return "🔴"
    return "🟡"


def _speaker_icon(speaker_id: str) -> str:
    """Return an icon based on speaker ID."""
    speaker_id_lower = str(speaker_id).lower()
    if "0" in speaker_id_lower or "speaker1" in speaker_id_lower:
        return "👤"
    return "👥"


def _render_message_html(message: dict, index: int) -> str:
    """Return HTML for a single message (used inside a scrollable container)."""
    speaker = message.get("speaker_id", "unknown")
    speaker_icon = _speaker_icon(speaker)
    technical_flag = "✅" if message.get("technical_qa", False) else "❌"
    rating_label = _rating_badge(message.get("answer_rating", "satisfactory"))
    transcript = message["transcript"]
    reasoning = message["response_reasoning"]
    followup = message["follow_up_question"]

    # Alternate background colors for dark theme
    bg_color = "#1e1e1e" if index % 2 == 0 else "#262626"
    text_color = "#e0e0e0"
    accent_color = "#4a9eff"

    html = f"""
    <div style="background-color: {bg_color}; padding: 12px; border-radius: 8px; margin: 6px 0; color: {text_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div style="font-weight: bold; font-size: 14px; color: {accent_color};">
                {speaker_icon} <span style="margin-left: 6px;">{speaker}</span>
            </div>
            <div style="display: flex; gap: 10px; font-size: 13px; color: #b0b0b0;">
                <span>QA: {technical_flag}</span>
                <span>Rating: {rating_label}</span>
            </div>
        </div>
        <div style="margin: 8px 0; padding: 6px; background: rgba(74, 158, 255, 0.06); border-left: 3px solid {accent_color}; border-radius: 4px;">
            <strong style="color: {accent_color};">Transcript:</strong> <em style="color: {text_color};">{transcript}</em>
        </div>
        <div style="margin: 6px 0; display: flex; gap: 8px;">
            <span>💭</span>
            <span style="color: #c0c0c0;">{reasoning}</span>
        </div>
        <div style="margin: 6px 0; display: flex; gap: 8px;">
            <span>❓</span>
            <span style="color: #c0c0c0;">{followup}</span>
        </div>
    </div>
    """
    return html


def display_message(message: dict, index: int) -> None:
    """Backward-compatible wrapper that renders a message to Streamlit."""
    html = _render_message_html(message, index)
    st.markdown(html, unsafe_allow_html=True)


def main() -> None:
    """Main Streamlit app."""
    st.title("Live Audio Inference")

    drained_count = drain_incoming_results()

    # Fixed header (appears above the scrollable chat area)
    header_bg = "#0f1720"
    header_text = "#e6eef8"
    conn_status = "Connected" if st.session_state.connected else "Disconnected"
    conn_color = "#2ecc71" if st.session_state.connected else "#ff6b6b"
    header_html = f"""
    <div style="position: sticky; top: 0; z-index: 999; background: {header_bg}; padding: 12px; border-radius: 6px; margin-bottom: 8px;">
        <div style="display:flex; justify-content:space-between; align-items:center; color: {header_text};">
            <div style="font-size:18px; font-weight:700;">#</div>
            <div style="display:flex; gap:12px; align-items:center;">
                <div style="font-size:13px; color:{conn_color};">●</div>
                <div style="font-size:13px; color:{header_text};">{conn_status}</div>
            </div>
        </div>
    </div>
    """
    components.html(header_html, height=80, scrolling=False)

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
    
    # Main output area: render messages inside a fixed-height scrollable container
    if not st.session_state.messages:
        st.write("Waiting for results...")
    else:
        all_messages = []
        for speaker_id in sorted(st.session_state.messages.keys()):
            for message in st.session_state.messages[speaker_id]:
                all_messages.append(message)

        all_messages.sort(key=lambda x: x["timestamp"])

        # Build HTML for all messages and render inside a scrollable div
        html_parts = []
        for idx, message in enumerate(all_messages):
            html_parts.append(_render_message_html(message, idx))

        chat_html = (
            "<div style='height:70vh; overflow-y:auto; padding-right:12px;'>"
            + "".join(html_parts)
            + "</div>"
        )

        # Use a taller iframe so the chat area reaches further down the screen.
        # If your screen is larger, increase this value (e.g. 900 or 1000).
        components.html(chat_html, height=900, scrolling=True)

    if st.session_state.connected:
        time.sleep(0.7)
        st.rerun()


if __name__ == "__main__":
    main()
