Build a Python prototype that simulates a live audio streaming system for testing an audio inference pipeline.

Project requirements:

1. Input source:

* Use an MP4 file as input.
* Extract audio from the MP4 file.
* Resample audio to 16kHz mono PCM float32.

2. Streaming simulator service:

* Simulate live streaming by splitting audio into fixed chunks of 200ms.
* Emit chunks in real-time using actual wall-clock pacing (`time.sleep(chunk_duration)`).
* Each chunk should include metadata:

  * chunk_id
  * timestamp
  * audio payload

3. Transport layer:

* Use Redis as the communication layer between producer and consumer.
* Producer pushes chunks into a Redis queue or stream.
* Consumer reads chunks continuously.

4. Consumer service:

* Maintain a rolling audio buffer.
* Buffer size = 6 seconds.
* Sliding window step = 3 seconds overlap.
* When buffer becomes full:

  * send buffered audio to a placeholder inference function.

5. Inference integration:

* Create a placeholder function named `run_qwen_inference(audio_array)` that receives numpy audio samples.
* Keep this function isolated so Qwen2Audio integration can be added later.

6. Concurrency:

* Receiver should run independently from inference.
* Use queue/threading so inference does not block chunk reception.

7. Output:

* Print:

  * when chunks arrive
  * current buffer duration
  * when inference is triggered

8. Code structure:
   Organize code into modules:

project/
├── producer/
│   └── stream_mp4.py
├── consumer/
│   ├── receiver.py
│   ├── buffer_manager.py
│   ├── inference_worker.py
├── shared/
│   └── config.py

9. Implementation constraints:

* Use clean production-style Python.
* Use classes where appropriate.
* Add logging.
* Add comments for future extension to real streaming input.

10. Bonus:

* Add graceful shutdown.
* Add Redis reconnect handling.

Generate complete runnable code for all files.
