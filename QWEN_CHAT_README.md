## Qwen Audio Analysis Chat Interface

This system captures real-time audio analysis results from the Qwen model and displays them in a chat-like Streamlit interface.

### Architecture

1. **Producer** (`producer/stream_audio.py`)
   - Streams audio chunks via WebSocket

2. **Consumer** (`consumer/receiver.py`)
   - Receives and buffers audio chunks per speaker
   - Applies VAD filtering
   - Saves speech segments
   - **New**: Starts a WebSocket server on port 8766 for results broadcasting

3. **Inference Worker** (`consumer/inference_worker.py`)
   - Per-speaker inference threads
   - Runs Qwen placeholder on each audio chunk
   - Broadcasts results to all connected clients via WebSocket

4. **Results Server** (`consumer/results_server.py`)
   - WebSocket server that broadcasts inference results
   - Handles multiple concurrent clients
   - Broadcasting happens asynchronously, non-blocking

5. **Streamlit App** (`streamlit_app.py`)
   - WebSocket client that connects to results server
   - Displays results in real-time, grouped by speaker
   - Chat-like UI with speaker identification

### Running the System

#### 1. Terminal 1: Start the Producer
```bash
python -m producer.stream_audio
```

#### 2. Terminal 2: Start the Consumer (with results server)
```bash
python -m consumer.receiver --run-name demo
```

The consumer will:
- Listen for audio chunks from producer
- Start inference workers for each speaker
- Start the results server on `ws://127.0.0.1:8766`

#### 3. Terminal 3: Start the Streamlit App
```bash
streamlit run streamlit_app.py
```

Opens browser at `http://localhost:8501`

### Streamlit UI Features

- **Sidebar Configuration**:
  - Set WebSocket server URI (default: `ws://127.0.0.1:8766`)
  - Connect/Disconnect buttons
  - Clear history button
  - Statistics dashboard

- **Main Chat Area**:
  - Tabbed view for each speaker
  - "All Speakers" combined tab
  - Chat-like message display with:
    - Timestamp
    - Chunk number
    - Audio duration
    - Answer rating (high/medium/low)
    - Transcript
    - Technical Q&A
    - Response reasoning
    - Follow-up question

### Output Format (per chunk)

Each Qwen result contains:
```json
{
  "speaker_id": "1",
  "chunk_number": 0,
  "transcript": "...",
  "technical_qa": "...",
  "response_reasoning": "...",
  "answer_rating": "high|medium|low",
  "follow_up_question": "...",
  "audio_duration_seconds": 25.4
}
```

### Non-Blocking Design

- **Audio reception**: Not blocked by inference
- **Inference**: Runs in per-speaker background threads
- **Results broadcast**: Asynchronous via WebSocket
- **Streamlit UI**: Receives updates without polling

### Replacing the Placeholder

To integrate real Qwen2Audio model:

1. Edit `consumer/inference_worker.py`
2. Replace the body of `run_qwen_inference(audio_array)` with actual model inference
3. Return a dict with the same keys: `transcript`, `technical_qa`, `response_reasoning`, `answer_rating`, `follow_up_question`
4. The rest of the pipeline stays the same

Example:
```python
def run_qwen_inference(audio_array: np.ndarray) -> dict:
    # Load your Qwen2Audio model here
    # result = model.transcribe_and_analyze(audio_array)
    return {
        "transcript": result["text"],
        "technical_qa": result["qa"],
        "response_reasoning": result["reasoning"],
        "answer_rating": result["confidence"],
        "follow_up_question": result["follow_up"],
    }
```

### Configuration

Via `.env`:
- `BUFFER_TIMEOUT_SECONDS=35` — How long to buffer before flushing
- `VAD_SILENCE_THRESHOLD_MS=700` — Silence detection threshold
- `VAD_MIN_SPEECH_SECONDS=2.0` — Minimum speech duration
- `WS_HOST=127.0.0.1`, `WS_PORT=8765` — Producer WebSocket
- `LOG_LEVEL=INFO` — Logging verbosity
