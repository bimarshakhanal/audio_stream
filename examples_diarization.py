"""
Example usage and integration patterns for the Speaker Diarization module.

This file demonstrates various ways to use the SpeakerDiarizationEngine
in different scenarios.
"""

import logging

import speaker_diarization


def _build_engine(num_speakers=2, model_name=None):
    """Return a configured diarization engine instance."""
    # pylint: disable=not-callable
    engine_cls: type | None = getattr(
        speaker_diarization,
        "SpeakerDiarizationEngine",
        None
    )
    if engine_cls is None:
        raise RuntimeError(
            "speaker_diarization.py does not define "
            "SpeakerDiarizationEngine."
        )

    if model_name is None:
        return engine_cls(num_speakers=num_speakers)

    return engine_cls(
        model_name=model_name,
        num_speakers=num_speakers
    )


# ============================================================================
# Example 1: Basic Usage (Recommended for most users)
# ============================================================================

def example_basic_usage():
    """
    Basic usage: Load audio, run diarization, save output, print summary.
    This is the simplest way to use the diarization engine.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70 + "\n")

    # Initialize engine
    engine = _build_engine(num_speakers=2)

    # Load audio
    audio_data, sample_rate = engine.load_audio("data/full_audio.wav")
    print(f"Loaded audio: {len(audio_data)} samples @ {sample_rate}Hz")

    # Run diarization
    intervals = engine.run_diarization("data/full_audio.wav")
    # Reconstruct separated audio
    output_files = engine.reconstruct_speaker_audio(
        intervals,
        output_dir="diarization_output"
    )

    # Display results
    engine.print_summary(intervals, output_files)


# ============================================================================
# Example 2: Custom Configuration
# ============================================================================

def example_custom_configuration():
    """
    Custom usage: Specify different input/output paths and model.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Configuration")
    print("="*70 + "\n")

    # Create engine with specific model
    engine = _build_engine(
        model_name="pyannote/speaker-diarization-community-1",
        num_speakers=2
    )

    # Load audio from custom path
    custom_audio_path = "data/custom_audio.wav"
    try:
        _audio_data, _sample_rate = engine.load_audio(custom_audio_path)
        intervals = engine.run_diarization(custom_audio_path)

        # Save to custom output directory
        output_files = engine.reconstruct_speaker_audio(
            intervals,
            output_dir="custom_output"
        )

        engine.print_summary(intervals, output_files)
    except FileNotFoundError as e:
        print(f"File not found: {e}")


# ============================================================================
# Example 3: Advanced: Processing and Analysis
# ============================================================================

def example_advanced_analysis():
    """
    Advanced usage: Access diarization results for custom analysis.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Advanced Analysis")
    print("="*70 + "\n")

    engine = _build_engine(num_speakers=2)
    engine.load_audio("audio/sample.wav")
    intervals = engine.run_diarization("audio/sample.wav")

    # Custom analysis of intervals
    print("\nCustom Analysis:")
    print("-" * 70)

    total_duration = len(engine.audio_data) / engine.sample_rate
    print(f"Total audio duration: {total_duration:.2f}s")

    for speaker_id in range(engine.num_speakers):
        if speaker_id in intervals:
            speaker_intervals = intervals[speaker_id]

            # Calculate statistics
            speaking_duration = sum(
                end - start for start, end in speaker_intervals
            )
            percentage = (speaking_duration / total_duration) * 100
            avg_segment_length = (
                speaking_duration / len(speaker_intervals)
            )

            print(f"\nSpeaker {speaker_id}:")
            print(f"  Segments: {len(speaker_intervals)}")
            print(f"  Total speaking: {speaking_duration:.2f}s")
            print(f"  Speaking percentage: {percentage:.1f}%")
            print(f"  Avg segment length: {avg_segment_length:.2f}s")
            print(f"  First segment: {speaker_intervals[0]}")
            print(f"  Last segment: {speaker_intervals[-1]}")


# ============================================================================
# Example 4: Batch Processing Multiple Files
# ============================================================================

def example_batch_processing():
    """
    Process multiple audio files in sequence.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Processing")
    print("="*70 + "\n")

    audio_files = [
        "audio/sample1.wav",
        "audio/sample2.wav",
        "audio/sample3.wav",
    ]

    engine = _build_engine(num_speakers=2)

    results = {}
    for audio_file in audio_files:
        try:
            print(f"\nProcessing: {audio_file}")
            engine.load_audio(audio_file)
            intervals = engine.run_diarization(audio_file)

            output_files = engine.reconstruct_speaker_audio(
                intervals,
                output_dir=f"output_{audio_file.split('/')[-1].split('.')[0]}"
            )

            results[audio_file] = output_files
            print("  ✓ Completed successfully")

        except FileNotFoundError:
            print(f"  ✗ File not found: {audio_file}")
        except (RuntimeError, ValueError, OSError) as e:
            print(f"  ✗ Error processing {audio_file}: {e}")

    # Summary
    print("\n" + "="*70)
    print("Batch Processing Summary")
    print("="*70)
    print(f"Processed: {len(results)} files successfully")


# ============================================================================
# Example 5: Integration with Logging
# ============================================================================

def example_with_logging():
    """
    Configure custom logging for detailed debugging.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Logging Configuration")
    print("="*70 + "\n")

    # Configure detailed logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('diarization.log')
        ]
    )

    logger = logging.getLogger('example')
    logger.info("Starting diarization with detailed logging...")

    engine = _build_engine(num_speakers=2)
    engine.load_audio("audio/sample.wav")
    intervals = engine.run_diarization("audio/sample.wav")
    output_files = engine.reconstruct_speaker_audio(intervals)
    engine.print_summary(intervals, output_files)

    logger.info("Diarization completed. Check diarization.log for details")


# ============================================================================
# Example 6: Error Handling
# ============================================================================

def example_error_handling():
    """
    Demonstrate robust error handling patterns.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Error Handling")
    print("="*70 + "\n")

    engine = _build_engine(num_speakers=2)

    # Scenario 1: File not found
    print("Scenario 1: File not found")
    try:
        engine.load_audio("nonexistent/audio.wav")
    except FileNotFoundError as e:
        print(f"  Caught FileNotFoundError: {e}")

    # Scenario 2: Audio not loaded before processing
    print("\nScenario 2: Processing without loading audio")
    try:
        engine2 = _build_engine(num_speakers=2)
        engine2.reconstruct_speaker_audio({})
    except RuntimeError as e:
        print(f"  Caught RuntimeError: {e}")

    # Scenario 3: Successful error recovery
    print("\nScenario 3: Successful recovery")
    try:
        engine.load_audio("audio/sample.wav")
        intervals = engine.run_diarization("audio/sample.wav")
        output_files = engine.reconstruct_speaker_audio(intervals)
        print("Successfully processed audio")
        print(f"  Output files: {list(output_files.values())}")
    except (RuntimeError, ValueError, OSError) as e:
        print(f"  Unexpected error: {e}")


# ============================================================================
# Example 7: Using Engine Methods Independently
# ============================================================================

def example_independent_methods():
    """
    Demonstrate using individual engine methods for modular workflows.
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Independent Method Usage")
    print("="*70 + "\n")

    engine = _build_engine(num_speakers=2)

    # Step 1: Load audio
    audio_data, sample_rate = engine.load_audio("audio/sample.wav")
    print(f"Audio loaded: {len(audio_data)} samples, {sample_rate}Hz")

    # Step 2: Run diarization only
    intervals = engine.run_diarization("audio/sample.wav")
    print(f"Diarization complete: {len(intervals)} speakers detected")

    # Step 3: Could process intervals further before reconstruction
    # (e.g., apply confidence thresholds, merge nearby segments, etc.)

    # Step 4: Reconstruct with modified intervals
    output_files = engine.reconstruct_speaker_audio(
        intervals,
        output_dir="diarization_output"
    )
    print(f"Audio reconstructed to: {output_files}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# Speaker Diarization Examples")
    print("#"*70)

    # Run selected examples (uncomment to run)

    example_basic_usage()

    # Uncomment to run additional examples:
    # example_custom_configuration()
    # example_advanced_analysis()
    # example_batch_processing()
    # example_with_logging()
    # example_error_handling()
    # example_independent_methods()

    print("\n" + "#"*70)
    print("# All examples completed")
    print("#"*70 + "\n")
