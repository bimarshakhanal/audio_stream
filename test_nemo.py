#!/usr/bin/env python
"""Test script for NeMo diarization."""

import sys
import os

sep = "=" * 70
print(sep, file=sys.stderr)
print("Starting test_nemo.py", file=sys.stderr)
print(sep, file=sys.stderr)
print("Python version:", sys.version, file=sys.stderr)
print("Current directory:", os.getcwd(), file=sys.stderr)
file_exists = os.path.exists('data/sample.wav')
print("Input file exists:", file_exists, file=sys.stderr)

# Test imports
try:
    print("Importing nemo_diarization...", file=sys.stderr)
    from nemo_diarization import NemoDiarizationEngine
    print("OK: Import successful", file=sys.stderr)
except Exception as err:
    print("ERROR: Import failed:", err, file=sys.stderr)
    sys.exit(1)

# Test engine creation
try:
    print("\nCreating engine...", file=sys.stderr)
    model_name = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    engine = NemoDiarizationEngine(num_speakers=2,
                                   model_name=model_name)
    print("OK: Engine created", file=sys.stderr)
except Exception as err:
    print("ERROR: Engine creation failed:", err, file=sys.stderr)
    sys.exit(1)

# Test audio loading
try:
    print("\nLoading audio...", file=sys.stderr)
    engine.load_audio("data/sample.wav")
    shape = engine.audio_data.shape
    rate = engine.sample_rate
    print("OK: Audio loaded:", shape, "samples at", rate, "Hz",
          file=sys.stderr)
except Exception as err:
    print("ERROR: Audio loading failed:", err, file=sys.stderr)
    sys.exit(1)

# Test diarization
try:
    print("\nRunning diarization...", file=sys.stderr)
    speaker_intervals = engine.run_diarization("data/sample.wav")
    print("OK: Diarization complete", file=sys.stderr)
    print("Speaker intervals:", speaker_intervals, file=sys.stderr)
except Exception as err:
    print("ERROR: Diarization failed:", err, file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# Test audio reconstruction
try:
    print("\nReconstructing audio...", file=sys.stderr)
    output_files = engine.reconstruct_speaker_audio(
        speaker_intervals,
        "diarization_output"
    )
    print("OK: Audio reconstruction complete", file=sys.stderr)
    print("Output files:", output_files, file=sys.stderr)
except Exception as err:
    print("ERROR: Audio reconstruction failed:", err, file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

# Verify output files
try:
    print("\nVerifying output files...", file=sys.stderr)
    for speaker_id, filepath in output_files.items():
        exists = os.path.exists(filepath)
        size = os.path.getsize(filepath) if exists else 0
        print("  Speaker", speaker_id, ":", filepath, file=sys.stderr)
        print("    Exists:", exists, ", Size:", size, "bytes",
              file=sys.stderr)
except Exception as err:
    print("ERROR: Verification failed:", err, file=sys.stderr)
    sys.exit(1)

print("\n" + sep, file=sys.stderr)
print("All tests passed!", file=sys.stderr)
print(sep, file=sys.stderr)
