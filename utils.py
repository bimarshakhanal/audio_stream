import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import soundfile as sf


def plot_all_speakers_turns(
    speaker_dir,
    original_wav_dir,
    save_path=None,
    show=True,
):
    """
    Plot:
      1. Original waveform for each speaker (before VAD chunking)
      2. VAD speech timeline for each speaker

    Expected structure:
        speaker_dir/
            SPEAKER_00_metadata.json
            SPEAKER_01_metadata.json
            ...

        original_wav_dir/
            SPEAKER_00.wav
            SPEAKER_01.wav
            ...

    Args:
        speaker_dir (str | Path): Directory containing *_metadata.json files.
        original_wav_dir (str | Path): Directory containing original speaker WAVs.
        save_path (str | Path | None): Optional output image path.
        show (bool): Whether to display the plot.
    """
    speaker_dir = Path(speaker_dir)
    original_wav_dir = Path(original_wav_dir)

    # --------------------------------------------------
    # Load metadata files
    # --------------------------------------------------
    metadata_files = sorted(speaker_dir.glob("*_metadata.json"))
    metadata_files = [
        p for p in metadata_files
        if p.name != "combined_metadata.json"
    ]

    if not metadata_files:
        raise FileNotFoundError(
            f"No speaker metadata files found in {speaker_dir}"
        )

    all_metadata = {}

    for metadata_file in metadata_files:
        speaker_name = metadata_file.stem.replace("_metadata", "")

        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if metadata:
            all_metadata[speaker_name] = metadata

    if not all_metadata:
        raise ValueError("All metadata files are empty.")

    speakers = list(all_metadata.keys())
    n_speakers = len(speakers)

    # --------------------------------------------------
    # Load original waveforms
    # --------------------------------------------------
    waveforms = {}
    max_end_time = 0.0
    total_turns = 0
    total_speech = 0.0

    for speaker in speakers:
        wav_path = original_wav_dir / f"{speaker}.wav"

        if wav_path.exists():
            audio, sr = sf.read(wav_path)

            # Convert stereo to mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            times = np.arange(len(audio)) / sr
            waveforms[speaker] = (times, audio)

            if len(times) > 0:
                max_end_time = max(max_end_time, times[-1])
        else:
            waveforms[speaker] = None

        # Stats from metadata
        metadata = all_metadata[speaker]
        total_turns += len(metadata)
        total_speech += sum(seg["duration_sec"] for seg in metadata)

        if metadata:
            max_end_time = max(
                max_end_time,
                max(seg["end_sec"] for seg in metadata)
            )

    # --------------------------------------------------
    # Create figure: 2 rows per speaker
    # --------------------------------------------------
    fig, axes = plt.subplots(
        nrows=n_speakers * 2,
        ncols=1,
        figsize=(20, max(4, n_speakers * 3.5)),
        sharex=True,
        constrained_layout=True,
    )

    if n_speakers == 1:
        axes = np.array(axes).reshape(2)

    # --------------------------------------------------
    # Plot each speaker
    # --------------------------------------------------
    for i, speaker in enumerate(speakers):
        ax_wave = axes[2 * i]
        ax_vad = axes[2 * i + 1]

        # Original waveform
        waveform_data = waveforms.get(speaker)

        if waveform_data is not None:
            times, audio = waveform_data
            ax_wave.plot(times, audio, linewidth=0.5)
        else:
            ax_wave.text(
                0.5,
                0.5,
                f"{speaker}.wav not found in {original_wav_dir}",
                ha="center",
                va="center",
                transform=ax_wave.transAxes,
            )

        ax_wave.set_ylabel("Amplitude")
        ax_wave.set_title(f"{speaker} - Original Waveform")
        ax_wave.grid(True, axis="x", which="major", alpha=0.4)
        ax_wave.grid(True, axis="x", which="minor", alpha=0.15)

        # VAD segments
        for seg in all_metadata[speaker]:
            ax_vad.barh(
                y=0,
                width=seg["duration_sec"],
                left=seg["start_sec"],
                height=0.6,
                alpha=0.85,
            )

        ax_vad.set_yticks([0])
        ax_vad.set_yticklabels(["Speech"])
        ax_vad.set_ylabel("VAD")
        ax_vad.set_title(f"{speaker} - Detected Speech Segments")
        ax_vad.grid(True, axis="x", which="major", alpha=0.4)
        ax_vad.grid(True, axis="x", which="minor", alpha=0.15)

    # --------------------------------------------------
    # Shared x-axis formatting
    # --------------------------------------------------
    for ax in np.ravel(axes):
        ax.xaxis.set_major_locator(MultipleLocator(30))
        ax.xaxis.set_minor_locator(MultipleLocator(15))
        ax.set_xlim(0, max_end_time)

    axes[-1].set_xlabel("Time (seconds)")

    # --------------------------------------------------
    # Figure title
    # --------------------------------------------------
    fig.suptitle(
        (
            f"{n_speakers} speakers | "
            f"{total_turns} turns | "
            f"Total speech: {total_speech:.2f} sec | "
            f"Conversation length: {max_end_time:.2f} sec"
        ),
        fontsize=14,
        fontweight="bold",
    )

    # --------------------------------------------------
    # Save figure
    # --------------------------------------------------
    if save_path is None:
        save_path = speaker_dir / "all_speakers_waveform_and_timeline.png"
    else:
        save_path = Path(save_path)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return all_metadata