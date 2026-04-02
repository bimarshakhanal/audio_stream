# Speaker Diarization System Architecture

## Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT AUDIO FILE                             │
│              audio/sample.wav (2 minutes, mono)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DIARIZATION ENGINE                            │
│        (SpeakerDiarizationEngine class)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Load Audio                                                  │
│     └─ Read WAV file → numpy array                             │
│     └─ Store sample rate (e.g., 16kHz)                         │
│                                                                 │
│  2. Load Pretrained Model                                       │
│     └─ pyannote/speaker-diarization-community-1 (~300MB)      │
│     └─ Available on first run (automatic download)             │
│                                                                 │
│  3. Run Diarization                                             │
│     └─ Detect speaker boundaries and speaker IDs              │
│     └─ Output: List of (speaker_id, start_time, end_time)     │
│                                                                 │
│  4. Extract Speaker Intervals                                   │
│     └─ Speaker 0: [(t1, t2), (t3, t4), ...]                   │
│     └─ Speaker 1: [(t5, t6), (t7, t8), ...]                   │
│                                                                 │
│  5. Reconstruct Audio Tracks                                    │
│     └─ Initialize Speaker 0 array: zeros (full duration)       │
│     └─ Initialize Speaker 1 array: zeros (full duration)       │
│     └─ Fill Speaker 0: Copy audio where Speaker 0 speaks       │
│     └─ Fill Speaker 1: Copy audio where Speaker 1 speaks       │
│                                                                 │
│  6. Save Output Files                                           │
│     └─ speaker_0_full.wav (with silence during Speaker 1)      │
│     └─ speaker_1_full.wav (with silence during Speaker 0)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
┌──────────────────────┐         ┌──────────────────────┐
│ speaker_0_full.wav   │         │ speaker_1_full.wav   │
├──────────────────────┤         ├──────────────────────┤
│ Duration: 2:00       │         │ Duration: 2:00       │
│ Sample Rate: 16kHz   │         │ Sample Rate: 16kHz   │
│ Channels: 1 (mono)   │         │ Channels: 1 (mono)   │
│                      │         │                      │
│ Contains Speaker 0   │         │ Contains Speaker 1   │
│ audio + silence      │         │ audio + silence      │
│ where Speaker 1 is   │         │ where Speaker 0 is   │
│ speaking             │         │ speaking             │
└──────────────────────┘         └──────────────────────┘
```

## Audio Array Reconstruction Example

```
Original Audio Timeline (2 minutes = 1,920,000 samples @ 16kHz)
┌─────────────────────────────────────────────────────────────┐
│ S0  S1  S0  S1  S0  S0  S1  S1  S0  S1  S0                  │
│ (Speaker IDs detected by diarization)                       │
└─────────────────────────────────────────────────────────────┘

↓ Reconstruct Speaker 0 Track
┌─────────────────────────────────────────────────────────────┐
│[A] [0] [C] [0] [E] [F] [0] [0] [I] [0] [K]                 │
│ ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲                  │
│ Original audio at Speaker 0 intervals, silence (0) elsewhere │
│ Result: speaker_0_full.wav                                   │
└─────────────────────────────────────────────────────────────┘

↓ Reconstruct Speaker 1 Track
┌─────────────────────────────────────────────────────────────┐
│[0] [B] [0] [D] [0] [0] [G] [H] [0] [J] [0]                 │
│ ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲   ▲                  │
│ Original audio at Speaker 1 intervals, silence (0) elsewhere │
│ Result: speaker_1_full.wav                                   │
└─────────────────────────────────────────────────────────────┘

Key: [A], [B], etc. = audio segments
     [0] = silence (zeros in array)
```

## Class Architecture

```
SpeakerDiarizationEngine
│
├─ Attributes:
│  ├─ pipeline: Pyannote Pipeline (diarization model)
│  ├─ diarization: Diarization results
│  ├─ audio_data: numpy array (loaded audio)
│  ├─ sample_rate: int (Hz)
│  ├─ num_speakers: int (always 2 in this case)
│  └─ model_name: str (model identifier)
│
├─ Public Methods:
│  ├─ __init__(): Initialize engine and load model
│  ├─ load_audio(): Load WAV file into memory
│  ├─ run_diarization(): Execute diarization pipeline
│  ├─ reconstruct_speaker_audio(): Create separated tracks
│  ├─ print_summary(): Display statistics and results
│  └─ main(): Complete workflow orchestration
│
└─ Private Methods:
   ├─ _load_pipeline(): Download and initialize model
   └─ _extract_speaker_intervals(): Parse diarization output
```

## Data Flow

```
1. User calls: python speaker_diarization.py
   │
   ├─→ main() initializes SpeakerDiarizationEngine
   │   └─→ _load_pipeline() loads pyannote model
   │
   ├─→ load_audio("audio/sample.wav")
   │   └─→ Read file, validate, store in self.audio_data
   │
   ├─→ run_diarization("audio/sample.wav")
   │   ├─→ pipeline(audio_path, num_speakers=2)
   │   ├─→ _extract_speaker_intervals()
   │   └─→ Return dict: {0: [(t1,t2), ...], 1: [(t3,t4), ...]}
   │
   ├─→ reconstruct_speaker_audio(intervals, output_dir)
   │   ├─→ Initialize zero arrays for each speaker
   │   ├─→ Loop over speaker_intervals
   │   │   ├─→ Convert times to sample indices
   │   │   └─→ Copy audio segments to respective arrays
   │   ├─→ Save speaker_0_full.wav
   │   ├─→ Save speaker_1_full.wav
   │   └─→ Return {0: path0, 1: path1}
   │
   └─→ print_summary(intervals, output_files)
       └─→ Log statistics and file locations
```

## Time Complexity & Memory Usage

| Operation | Complexity | Memory |
|-----------|-----------|--------|
| Load audio | O(n) | O(n) |
| Diarization | O(n) | O(n) model + O(n) intermediate |
| Reconstruct | O(n) | O(2n) (two output arrays) |
| Save files | O(n) | O(n) buffer |
| **Total** | **O(n)** | **~2-3x audio size** |

Where n = total number of audio samples (~1.9M for 2min @ 16kHz)

## Error Handling Hierarchy

```
main()
├─ FileNotFoundError
│  └─ "Audio file not found: audio/sample.wav"
│
├─ RuntimeError
│  └─ "Audio not loaded. Call load_audio() first."
│
├─ Exception (Load Pipeline)
│  └─ "Failed to load pipeline: {error_details}"
│
├─ Exception (Run Diarization)
│  └─ "Diarization failed: {error_details}"
│
└─ Exception (Save Files)
   └─ "Failed to save {output_path}: {error_details}"

All exceptions are logged with level ERROR and re-raised
```

## Configuration & Customization

```python
# Easy to modify in speaker_diarization.py:

INPUT_AUDIO_PATH = "audio/sample.wav"
    │── Change to different input file

OUTPUT_DIR = "diarization_output"
    │── Change to different output directory

NUM_SPEAKERS = 2
    │── Modify for different number of speakers
    │── (Note: model trained for variable speakers)

PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"
    │── Use different model version if available
```

## Dependencies Graph

```
speaker_diarization.py
│
├─ pyannote.audio
│  ├─ torch (PyTorch deep learning framework)
│  ├─ torchaudio
│  └─ librosa
│
├─ soundfile
│  └─ libsndfile (system library)
│
├─ numpy
│  └─ BLAS/LAPACK (optimized linear algebra)
│
└─ logging (Python stdlib)
```
