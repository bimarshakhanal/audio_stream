"""
Speaker Diarization using Nvidia NeMo Streaming Sortformer v2.1

Uses the NVIDIA Streaming Sortformer diarization model from Hugging Face:
https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1

This is a state-of-the-art streaming speaker diarization model
trained on 5000+ hours of diverse audio.

Output format:
- 2 full-length WAV files (speaker_0_full.wav, speaker_1_full.wav)
- Full audio timeline preserved
- Silence inserted where other speaker is active
"""

import logging
import os
from typing import Dict, Tuple, List
from collections import defaultdict

import numpy as np
import torch
import soundfile as sf
from dotenv import load_dotenv

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# ============================================================================
# Configuration Constants
# ============================================================================

INPUT_AUDIO_PATH = "data/full_audio6.wav"
OUTPUT_DIR = "diarization_output/sample6"
NUM_SPEAKERS = 2
NEMO_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"

# Streaming configuration (in 80ms frames)
# Very high latency: good for offline processing
CHUNK_LEN = 25
CHUNK_RIGHT_CONTEXT = 10
FIFO_LEN = 20
SPKCACHE_UPDATE_PERIOD = 50
SPKCACHE_LEN = 64

# Interval filtering
MAX_GAP_TO_MERGE = 0.4
MIN_SEGMENT_DURATION = 0.25


# ============================================================================
# NeMo Sortformer Diarization Engine
# ============================================================================

class NemoDiarizationEngine:
    """
    Speaker diarization using NVIDIA NeMo Streaming Sortformer v2.1.
    
    Features:
    - State-of-the-art streaming diarization model
    - Trained on 5000+ hours of diverse audio
    - Handles up to 4 speakers
    - Excellent on overlapping speech
    - Full-timeline reconstruction with silence preservation
    """

    def __init__(
        self,
        num_speakers: int = NUM_SPEAKERS,
        model_name: str = NEMO_MODEL,
        device: str = None
    ):
        """
        Initialize NeMo Sortformer diarization engine.

        Args:
            num_speakers: Expected number of speakers (max 4)
            model_name: NeMo model identifier
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        logger.info(
            "Initializing NeMo Sortformer Diarization Engine"
        )

        if num_speakers > 4:
            logger.warning(
                "Model trained for max 4 speakers, "
                "but %d requested. Performance may degrade.",
                num_speakers
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.num_speakers = num_speakers
        self.model_name = model_name
        self.device = torch.device(device)
        self.diar_model = None
        self.audio_data = None
        self.sample_rate = None

        self._load_model()
        self._setup_streaming_config()

    def _load_model(self) -> None:
        """Load Sortformer diarization model from Hugging Face."""
        try:
            logger.info("Importing NeMo Sortformer model...")
            from nemo.collections.asr.models import (
                SortformerEncLabelModel
            )

            logger.info(
                "Loading Sortformer model from Hugging Face: %s",
                self.model_name
            )

            self.diar_model = SortformerEncLabelModel.from_pretrained(
                self.model_name,
                map_location=self.device
            )
            self.diar_model.eval()

            logger.info(
                "Sortformer model loaded successfully on %s",
                self.device
            )

        except Exception as e:
            logger.error("Failed to load Sortformer model: %s", e)
            raise

    def _setup_streaming_config(self) -> None:
        """Configure streaming parameters for diarization."""
        try:
            logger.info("Configuring streaming parameters...")

            self.diar_model.sortformer_modules.chunk_len = CHUNK_LEN
            self.diar_model.sortformer_modules.chunk_right_context = (
                CHUNK_RIGHT_CONTEXT
            )
            self.diar_model.sortformer_modules.fifo_len = FIFO_LEN
            self.diar_model.sortformer_modules.spkcache_update_period = (
                SPKCACHE_UPDATE_PERIOD
            )
            self.diar_model.sortformer_modules.spkcache_len = SPKCACHE_LEN
            self.diar_model.sortformer_modules._check_streaming_parameters()

            logger.info("Streaming configuration complete")
            logger.info("  Chunk length: %dms", CHUNK_LEN * 80)
            logger.info("  Right context: %dms", CHUNK_RIGHT_CONTEXT * 80)
            logger.info("  FIFO length: %dms", FIFO_LEN * 80)
            logger.info(
                "  Cache update period: %dms",
                SPKCACHE_UPDATE_PERIOD * 80
            )

        except Exception as e:
            logger.error("Failed to setup streaming config: %s", e)
            raise

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            audio_path: Path to input WAV file

        Returns:
            Tuple of (audio_array, sample_rate)

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(audio_path):
            error_msg = f"Audio file not found: {audio_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info("Loading audio from: %s", audio_path)
            audio_data, sample_rate = sf.read(audio_path)

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Ensure 16kHz sample rate (NeMo requirement)
            if sample_rate != 16000:
                logger.info(
                    "Resampling from %dHz to 16000Hz",
                    sample_rate
                )
                import librosa
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=16000
                )
                sample_rate = 16000

            duration = len(audio_data) / sample_rate
            logger.info(
                "Audio loaded - Sample rate: %sHz, Duration: %.2fs",
                sample_rate,
                duration
            )

            self.audio_data = audio_data
            self.sample_rate = sample_rate
            return audio_data, sample_rate

        except Exception as e:
            logger.error("Failed to load audio: %s", e)
            raise

    def run_diarization(self, audio_path: str) -> Dict:
        """
        Run Sortformer diarization on audio.

        Args:
            audio_path: Path to input WAV file

        Returns:
            Dictionary with speaker intervals

        Raises:
            Exception: If diarization fails
        """
        try:
            logger.info("Running Sortformer diarization...")

            if self.audio_data is None or self.sample_rate is None:
                self.load_audio(audio_path)

            # Run diarization
            logger.info("Performing speaker diarization inference...")
            predicted_output = self.diar_model.diarize(
                audio=[self.audio_data],
                batch_size=1,
                sample_rate=self.sample_rate
            )

            # diarize() returns nested list:
            # [[segment_string1, segment_string2, ...]]
            # Each segment_string is "start_time end_time speaker_label"
            logger.info(
                "Diarization output type: %s",
                type(predicted_output).__name__
            )
            if isinstance(predicted_output, (list, tuple)):
                logger.info("Output is list/tuple with %d items",
                            len(predicted_output))
                # predicted_output[0] is list of segment strings
                segment_strings = predicted_output[0]
            else:
                logger.error(
                    "Unexpected diarization output format: %s",
                    type(predicted_output)
                )
                raise ValueError(
                    f"Unexpected output format: {type(predicted_output)}"
                )

            logger.info("Diarization completed successfully")
            return self._extract_intervals(segment_strings)

        except Exception as e:
            logger.error("Diarization failed: %s", e)
            raise

    def _extract_intervals(self, segment_strings) -> Dict:
        """
        Extract speaker intervals from NeMo segment strings.

        Args:
            segment_strings: List of strings in format
                "start_time end_time speaker_label"
                e.g., "1.600 2.640 speaker_0"

        Returns:
            Dictionary with speaker intervals
        """
        speaker_intervals = defaultdict(list)

        try:
            # NeMo returns list of string segments
            logger.info(
                "Extracting intervals from %d segment strings",
                len(segment_strings)
            )

            for segment_str in segment_strings:
                try:
                    # Parse "start_time end_time speaker_label"
                    parts = segment_str.strip().split()
                    if len(parts) != 3:
                        logger.warning(
                            "Skipping malformed segment: %s",
                            segment_str
                        )
                        continue

                    start_time_str, end_time_str, speaker_label = parts
                    start_time = float(start_time_str)
                    end_time = float(end_time_str)

                    # Extract speaker number from label
                    # (speaker_0, speaker_1, etc)
                    try:
                        speaker_num = int(
                            str(speaker_label).split('_')[-1]
                        )
                    except (ValueError, IndexError):
                        speaker_num = int(speaker_label) \
                            if str(speaker_label).isdigit() else 0

                    if speaker_num < self.num_speakers:
                        speaker_intervals[speaker_num].append(
                            (start_time, end_time)
                        )
                        logger.debug(
                            "Speaker %d: [%.3f, %.3f]",
                            speaker_num,
                            start_time,
                            end_time
                        )
                except ValueError as err:
                    logger.error(
                        "Failed to parse segment '%s': %s",
                        segment_str,
                        err
                    )
                    continue

            # Sort intervals for each speaker
            for speaker_id in speaker_intervals:
                speaker_intervals[speaker_id].sort(key=lambda x: x[0])

            # Normalize intervals
            for speaker_id in list(speaker_intervals.keys()):
                normalized = self._normalize_intervals(
                    speaker_intervals[speaker_id]
                )
                if normalized:
                    speaker_intervals[speaker_id] = normalized
                else:
                    logger.warning(
                        "Speaker %d has no intervals after normalization",
                        speaker_id
                    )
                    del speaker_intervals[speaker_id]

            # Log results
            logger.info("Detected %d speakers", len(speaker_intervals))
            for speaker_id, intervals in sorted(
                speaker_intervals.items()
            ):
                total_duration = sum(
                    end - start for start, end in intervals
                )
                logger.info(
                    "  Speaker %d: %d segments, "
                    "Total duration: %.2fs",
                    speaker_id,
                    len(intervals),
                    total_duration
                )

            if not speaker_intervals:
                logger.warning(
                    "No speaker intervals detected. "
                    "Input segments: %s",
                    segment_strings
                )

            return dict(speaker_intervals)

        except Exception as e:
            logger.error("Failed to extract intervals: %s", e)
            logger.exception("Full traceback:")
            raise

    def _normalize_intervals(
        self,
        intervals: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Normalize intervals by merging close segments
        and filtering short ones.

        Args:
            intervals: List of (start_time, end_time) tuples

        Returns:
            Normalized list of intervals
        """
        if not intervals:
            return []

        intervals = sorted(intervals, key=lambda x: x[0])
        merged = []
        current_start, current_end = intervals[0]

        for start, end in intervals[1:]:
            gap = start - current_end
            if gap <= MAX_GAP_TO_MERGE:
                # Merge close segments
                current_end = max(current_end, end)
            else:
                # Save previous segment if long enough
                if current_end - current_start >= MIN_SEGMENT_DURATION:
                    merged.append((current_start, current_end))
                # Start new segment
                current_start, current_end = start, end

        # Save final segment
        if current_end - current_start >= MIN_SEGMENT_DURATION:
            merged.append((current_start, current_end))

        return merged

    def reconstruct_speaker_audio(
        self,
        speaker_intervals: Dict,
        output_dir: str = OUTPUT_DIR
    ) -> Dict[int, str]:
        """
        Reconstruct full-length audio for each speaker
        with silence preservation.

        Args:
            speaker_intervals: Dictionary with speaker intervals
            output_dir: Output directory for WAV files

        Returns:
            Dictionary mapping speaker_id to output file path
        """
        if self.audio_data is None:
            error_msg = "Audio not loaded. Call load_audio() first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        os.makedirs(output_dir, exist_ok=True)
        logger.info("Output directory ready: %s", output_dir)

        output_files = {}
        total_samples = len(self.audio_data)
        audio_duration = total_samples / self.sample_rate

        # Initialize silence-filled arrays
        speaker_audio = {
            speaker_id: np.zeros(
                total_samples,
                dtype=self.audio_data.dtype
            )
            for speaker_id in range(self.num_speakers)
        }

        logger.info(
            "Reconstructing %d speaker tracks "
            "(duration: %.2fs)...",
            self.num_speakers,
            audio_duration
        )

        # Fill with audio segments
        for speaker_id in range(self.num_speakers):
            if speaker_id not in speaker_intervals:
                logger.warning(
                    "Speaker %d has no detected intervals",
                    speaker_id
                )
                continue

            logger.info("Processing speaker %d...", speaker_id)

            for start_time, end_time in speaker_intervals[speaker_id]:
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)

                start_sample = max(0, min(start_sample, total_samples))
                end_sample = max(0, min(end_sample, total_samples))

                if start_sample < end_sample:
                    speaker_audio[speaker_id][
                        start_sample:end_sample
                    ] = self.audio_data[start_sample:end_sample]

            # Save to file
            output_path = os.path.join(
                output_dir,
                f"speaker_{speaker_id}_full.wav"
            )
            try:
                sf.write(
                    output_path,
                    speaker_audio[speaker_id],
                    self.sample_rate
                )
                output_files[speaker_id] = output_path
                logger.info("Saved: %s", output_path)
            except Exception as e:
                logger.error("Failed to save %s: %s", output_path, e)
                raise

        return output_files

    def print_summary(
        self,
        speaker_intervals: Dict,
        output_files: Dict[int, str]
    ) -> None:
        """Print diarization summary."""
        logger.info("=" * 70)
        logger.info("DIARIZATION SUMMARY (NeMo Streaming Sortformer v2.1)")
        logger.info("=" * 70)

        for speaker_id in range(self.num_speakers):
            logger.info("\nSpeaker %d:", speaker_id)
            if speaker_id in speaker_intervals:
                intervals = speaker_intervals[speaker_id]
                total_duration = sum(
                    end - start for start, end in intervals
                )
                logger.info(
                    "  Number of segments: %d",
                    len(intervals)
                )
                logger.info(
                    "  Total speaking duration: %.2fs",
                    total_duration
                )
                output_file = output_files.get(speaker_id, "N/A")
                logger.info(
                    "  Output file: %s",
                    output_file
                )
            else:
                logger.info("  No intervals detected")

        logger.info("=" * 70)
        logger.info("All output files:")
        for speaker_id, path in sorted(output_files.items()):
            logger.info("  %s", path)
        logger.info("=" * 70)


# ============================================================================
# Main Execution
# ============================================================================

def main() -> None:
    """Main execution."""
    try:
        engine = NemoDiarizationEngine(
            num_speakers=NUM_SPEAKERS,
            model_name=NEMO_MODEL
        )

        engine.load_audio(INPUT_AUDIO_PATH)
        speaker_intervals = engine.run_diarization(INPUT_AUDIO_PATH)
        print("Speaker intervals extracted successfully:", speaker_intervals)
        output_files = engine.reconstruct_speaker_audio(
            speaker_intervals,
            OUTPUT_DIR
        )
        engine.print_summary(speaker_intervals, output_files)

        logger.info("Processing completed successfully!")

    except FileNotFoundError as e:
        logger.error("File error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise


if __name__ == "__main__":
    main()
