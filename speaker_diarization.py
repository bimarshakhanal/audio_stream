"""
Speaker Diarization Module

This module handles speaker diarization using pyannote.audio, splitting audio
into speaker-specific tracks with silence preservation for the full timeline.
"""

import logging
import os
from collections import defaultdict
from typing import Tuple, Dict

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env (e.g. HF_TOKEN)
load_dotenv()


# ============================================================================
# Configuration Constants
# ============================================================================

INPUT_AUDIO_PATH = "audio/sample.wav"
OUTPUT_DIR = "diarization_output"
NUM_SPEAKERS = 2
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


# ============================================================================
# Diarization Engine Class
# ============================================================================

class SpeakerDiarizationEngine:
    """
    Handles speaker diarization and speaker-separated audio reconstruction.
    """

    def __init__(
        self,
        model_name: str = PYANNOTE_MODEL,
        num_speakers: int = NUM_SPEAKERS
    ):
        """
        Initialize the diarization engine.

        Args:
            model_name: Path or identifier for pyannote model
            num_speakers: Expected number of speakers
        """
        logger.info(
            "Initializing SpeakerDiarizationEngine with model: %s",
            model_name
        )
        self.model_name = model_name
        self.num_speakers = num_speakers
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.pipeline = None
        self.diarization = None
        self.audio_data = None
        self.sample_rate = None

        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """Load the pyannote diarization pipeline."""
        try:
            logger.info("Loading pyannote diarization pipeline...")
            token = os.getenv("HF_TOKEN")

            if token:
                self.pipeline = Pipeline.from_pretrained(
                    self.model_name,
                    token=token
                )
            else:
                logger.warning(
                    "No Hugging Face token found in HF_TOKEN; "
                    "attempting public access."
                )
                self.pipeline = Pipeline.from_pretrained(self.model_name)

            self.pipeline.to(self.device)
            logger.info("Pipeline loaded successfully on %s", self.device)
        except Exception as e:
            logger.error("Failed to load pipeline: %s", e)
            raise

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            audio_path: Path to input WAV file

        Returns:
            Tuple of (audio_array, sample_rate)

        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: If audio loading fails
        """
        if not os.path.exists(audio_path):
            error_msg = f"Audio file not found: {audio_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info("Loading audio from: %s", audio_path)
            audio_data, sample_rate = sf.read(audio_path)
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
        Run speaker diarization on audio file.

        Args:
            audio_path: Path to input WAV file

        Returns:
            Dictionary containing diarization results and speaker
            intervals
        """
        try:
            logger.info("Running diarization...")
            with ProgressHook() as hook:
                self.diarization = self.pipeline(
                    audio_path,
                    num_speakers=self.num_speakers,
                    hook=hook
                )
            logger.info("Diarization completed successfully")
            return self._extract_speaker_intervals()
        except NameError as e:
            if "AudioDecoder" not in str(e):
                logger.error("Diarization failed: %s", e)
                raise

            logger.warning(
                "AudioDecoder backend unavailable. Falling back to in-memory "
                "waveform input via soundfile."
            )

            waveform_np, sample_rate = sf.read(
                audio_path,
                dtype="float32",
                always_2d=False
            )

            # soundfile returns (frames, channels)
            # for multi-channel audio
            if waveform_np.ndim == 2:
                waveform_np = waveform_np.mean(axis=1)

            waveform = torch.from_numpy(waveform_np).unsqueeze(0)
            file_payload = {
                "waveform": waveform,
                "sample_rate": sample_rate,
            }

            with ProgressHook() as hook:
                self.diarization = self.pipeline(
                    file_payload,
                    num_speakers=self.num_speakers,
                    hook=hook
                )
            logger.info("Diarization completed successfully (fallback mode)")
            return self._extract_speaker_intervals()
        except Exception as e:
            logger.error("Diarization failed: %s", e)
            raise

    def _extract_speaker_intervals(self) -> Dict:
        """
        Extract speaker intervals from diarization results.

        Returns:
            Dictionary with speaker IDs as keys and list of
            (start_time, end_time) tuples
        """
        speaker_intervals = defaultdict(list)
        speaker_map = {}

        diarization_tracks = getattr(
            self.diarization,
            "speaker_diarization",
            self.diarization
        )

        for turn, _, speaker in diarization_tracks.itertracks(
            yield_label=True
        ):
            if speaker not in speaker_map:
                speaker_map[speaker] = len(speaker_map)

            speaker_id = speaker_map[speaker]
            if speaker_id < self.num_speakers:
                speaker_intervals[speaker_id].append(
                    (turn.start, turn.end)
                )

        # Log detected intervals
        logger.info("Detected %d speakers", len(speaker_map))
        for speaker_label, speaker_id in sorted(
            speaker_map.items(),
            key=lambda item: item[1]
        ):
            intervals = speaker_intervals.get(speaker_id, [])
            total_duration = sum(
                end - start for start, end in intervals
            )
            logger.info(
                "  Speaker %s -> speaker_%d: %d segments, Total duration: "
                "%.2fs",
                speaker_label,
                speaker_id,
                len(intervals),
                total_duration
            )
            for i, (start, end) in enumerate(intervals):
                logger.debug(
                    "    Segment %d: [%.2fs - %.2fs]",
                    i,
                    start,
                    end
                )

        return speaker_intervals

    def reconstruct_speaker_audio(
        self,
        speaker_intervals: Dict,
        output_dir: str = OUTPUT_DIR
    ) -> Dict[int, str]:
        """
        Reconstruct full-length audio for each speaker with silence.

        Args:
            speaker_intervals: Dictionary with speaker IDs and their
                time intervals
            output_dir: Directory to save output WAV files

        Returns:
            Dictionary mapping speaker_id to output file path
        """
        if self.audio_data is None:
            error_msg = "Audio not loaded. Call load_audio() first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Output directory ready: %s", output_dir)

        output_files = {}
        total_samples = len(self.audio_data)
        audio_duration = total_samples / self.sample_rate

        # Initialize arrays for each speaker (filled with silence)
        speaker_audio = {
            speaker_id: np.zeros(
                total_samples,
                dtype=self.audio_data.dtype
            )
            for speaker_id in range(self.num_speakers)
        }

        logger.info(
            "Reconstructing %d speaker tracks (duration: %.2fs)...",
            self.num_speakers,
            audio_duration
        )

        # Fill arrays with audio segments for each speaker
        for speaker_id in range(self.num_speakers):
            if speaker_id not in speaker_intervals:
                logger.warning(
                    "Speaker %d has no detected intervals",
                    speaker_id
                )
                continue

            logger.info("Processing speaker %d...", speaker_id)
            for start_time, end_time in speaker_intervals[speaker_id]:
                # Convert time in seconds to sample indices
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)

                # Clamp to valid range
                start_sample = max(0, min(start_sample, total_samples))
                end_sample = max(0, min(end_sample, total_samples))

                # Copy audio segment to speaker track
                if start_sample < end_sample:
                    speaker_audio[speaker_id][start_sample:end_sample] = (
                        self.audio_data[start_sample:end_sample]
                    )

            # Save speaker audio to file
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
        """
        Print detailed summary of diarization results.

        Args:
            speaker_intervals: Dictionary with speaker IDs and their
                time intervals
            output_files: Dictionary mapping speaker_id to output file
                path
        """
        logger.info("=" * 70)
        logger.info("DIARIZATION SUMMARY")
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
    """Main execution function."""
    try:
        # Initialize diarization engine
        engine = SpeakerDiarizationEngine(
            model_name=PYANNOTE_MODEL,
            num_speakers=NUM_SPEAKERS
        )

        # Load audio
        engine.load_audio(INPUT_AUDIO_PATH)

        # Run diarization
        speaker_intervals = engine.run_diarization(INPUT_AUDIO_PATH)

        # Reconstruct speaker audio
        output_files = engine.reconstruct_speaker_audio(
            speaker_intervals,
            OUTPUT_DIR
        )

        # Print summary
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
