# Speaker Diarization System

A modular Python implementation for speaker diarization using `pyannote.audio`, which separates audio into speaker-specific tracks while preserving the full timeline with silence gaps.

## Features

- **2-Speaker Diarization**: Automatically detects and separates audio from exactly 2 speakers
- **Full Timeline Preservation**: Each output file maintains the full 2-minute duration with silence where the other speaker is active
- **Modular Design**: Clean, well-documented class-based architecture
- **Comprehensive Logging**: Detailed logs of speaker intervals, speaking durations, and output paths
- **Error Handling**: Robust exception handling with informative error messages
- **Automatic Directory Creation**: Output folder created automatically if it doesn't exist

## Requirements

The following packages are required (see `requirements.txt`):

- `pyannote.audio >= 3.0.0` - Speaker diarization model
- `torch >= 2.0.0` - Deep learning framework
- `numpy >= 1.24` - Numerical computing
- `soundfile >= 0.12.1` - Audio file I/O

## Installation

1. Ensure your audio file is placed at `audio/sample.wav`

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

  Note: The first run of pyannote will download the pretrained model (~300MB)
  and requires a Hugging Face token with access to the model.

## Usage

### Basic Execution

```bash
python speaker_diarization.py
```

### Configuration

Edit the constants at the top of `speaker_diarization.py` to customize:

```python
INPUT_AUDIO_PATH = "audio/sample.wav"      # Path to input audio
OUTPUT_DIR = "diarization_output"          # Output directory
NUM_SPEAKERS = 2                           # Number of speakers to detect
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"  # Model to use
```

## Output

The script generates two files in the `diarization_output/` directory:

- **`speaker_0_full.wav`**: Audio containing only Speaker 0 with silence during Speaker 1's turns
- **`speaker_1_full.wav`**: Audio containing only Speaker 1 with silence during Speaker 0's turns

Both files:
- Preserve the full 2-minute timeline
- Maintain the original sample rate (e.g., 16kHz)
- Keep the original mono channel format

## Logging Output

The script provides detailed console logging including:

1. **Pipeline Initialization**: Model loading status
2. **Audio Loading**: File path, sample rate, and duration
3. **Diarization Progress**: Number of speakers detected, segments per speaker
4. **Segment Details**: Exact time intervals for each speaking segment
5. **Reconstruction Progress**: Processing status for each speaker
6. **Output Summary**: Total speaking duration and output file paths

### Log Levels

- **INFO**: General processing information (default)
- **WARNING**: Non-critical issues (e.g., speaker with no detected intervals)
- **DEBUG**: Detailed segment information (requires setting level to DEBUG)
- **ERROR**: Critical failures with exception details

## Code Structure

### Main Components

```
SpeakerDiarizationEngine
├── __init__()                 - Initialize engine and load model
├── _load_pipeline()           - Load pyannote diarization pipeline
├── load_audio()               - Load and validate audio file
├── run_diarization()          - Execute diarization on audio
├── _extract_speaker_intervals() - Parse diarization results
├── reconstruct_speaker_audio() - Generate separated speaker tracks
└── print_summary()            - Display results and statistics
```

### Key Methods

#### `load_audio(audio_path: str) -> Tuple[np.ndarray, int]`
Loads the WAV file and validates it exists. Returns audio samples and sample rate.

#### `run_diarization(audio_path: str) -> Dict`
Executes the diarization pipeline and returns speaker intervals.

#### `reconstruct_speaker_audio(speaker_intervals: Dict) -> Dict[int, str]`
Creates two output files by:
1. Initializing zero-filled arrays for each speaker
2. Copying audio segments where each speaker is detected
3. Saving reconstructed tracks to disk

## Error Handling

The system includes comprehensive error handling for:

- Missing audio files
- Invalid audio formats
- Pipeline loading failures
- Audio saving failures
- Model download issues (first run)

## Performance Considerations

- **Model Download**: First run downloads ~300MB model (one-time)
- **Processing Time**: Varies by audio length (typically 30-60 seconds for 2-minute audio)
- **Memory Usage**: Peak usage ~1-2GB depending on audio length and model
- **GPU Support**: Automatically uses GPU if available (via torch)

## Example Workflow

```python
# Initialize engine
engine = SpeakerDiarizationEngine(num_speakers=2)

# Load audio
engine.load_audio("audio/sample.wav")

# Run diarization
intervals = engine.run_diarization("audio/sample.wav")

# Reconstruct separated audio
output_files = engine.reconstruct_speaker_audio(intervals)

# Display results
engine.print_summary(intervals, output_files)
```

## Troubleshooting

### "Audio file not found"
- Ensure `audio/sample.wav` exists in the project directory
- Check the file path in `INPUT_AUDIO_PATH`

### "Failed to load pipeline"
- Verify torch is installed: `pip install torch`
- Check internet connection (model download required on first run)
- Try manual model download:
  ```bash
  python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained('pyannote/speaker-diarization-community-1')"
  ```

### "No intervals detected for Speaker X"
- The model may not have detected this speaker
- Verify the audio contains both speakers speaking
- Check audio quality and loudness levels

## License

This implementation uses the pyannote.audio model which requires proper attribution.
See https://github.com/pyannote/pyannote-audio for model licensing details.
