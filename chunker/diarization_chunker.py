"""Diarization-aware chunker using Silero VAD.

This module reads diarized audio inputs (speaker folders or audio files),
runs VAD to extract speech segments, and writes sequentially numbered
WAV files named 1.wav, 2.wav, ... across all speakers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf

from consumer.vad_engine import SileroVADEngine

LOGGER = logging.getLogger(__name__)


class DiarizationChunker:
    def __init__(
        self,
        sample_rate_hz: int = 16_000,
        output_dir: Path | str = "diarized_chunks",
        min_speech_seconds: float = 0.5,
        max_chunk_seconds: float | None = None,
        max_silence_between_ms: int = 1000,
    ) -> None:
        self.sample_rate_hz = int(sample_rate_hz)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vad = SileroVADEngine(sample_rate_hz=self.sample_rate_hz)
        self._global_index = 0
        self.min_speech_seconds = float(min_speech_seconds)
        # max_chunk_seconds: if None, do not split long segments
        self.max_chunk_seconds = None if max_chunk_seconds is None else float(max_chunk_seconds)
        self.max_silence_between_ms = int(max_silence_between_ms)

    def _load_audio(self, path: Path) -> np.ndarray:
        data, sr = sf.read(str(path))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != self.sample_rate_hz:
            data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=self.sample_rate_hz)
        # ensure float32 in -1..1
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 1.0:
            data = data / np.max(np.abs(data))
        return data

    def _save_segment(self, audio: np.ndarray, speaker_id: str, start_sample: int, end_sample: int) -> Path:
        duration = (end_sample - start_sample) / float(self.sample_rate_hz)
        if duration < self.min_speech_seconds:
            LOGGER.debug("Skipping short segment %.3fs", duration)
            return None  # type: ignore[return-value]

        self._global_index += 1
        filename = f"{self._global_index}.wav"
        file_path = self.output_dir / filename
        sf.write(str(file_path), audio.astype(np.float32), samplerate=self.sample_rate_hz, subtype="PCM_16")

        metadata = {
            "filename": file_path.name,
            "speaker_id": speaker_id,
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),
            "start_time_sec": float(start_sample) / self.sample_rate_hz,
            "end_time_sec": float(end_sample) / self.sample_rate_hz,
        }
        meta_path = file_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        LOGGER.info("Wrote segment %s (speaker=%s dur=%.3fs)", file_path, speaker_id, duration)
        return file_path

    def process_file(self, file_path: Path, speaker_id: str | None = None) -> Iterable[Path]:
        file_path = Path(file_path)
        speaker_id = speaker_id or file_path.stem
        audio = self._load_audio(file_path)
        segments = self.vad.get_speech_segments(audio)
        # merge short gaps and split long merged regions into max_chunk_seconds
        segments = self._merge_and_split_segments(segments)
        out_paths = []
        for start, end in segments:
            # clip to bounds
            start = max(0, int(start))
            end = min(len(audio), int(end))
            seg_audio = audio[start:end]
            out = self._save_segment(seg_audio, speaker_id, start, end)
            if out is not None:
                out_paths.append(out)
        return out_paths

    def process_diarized_dir(self, input_path: Path) -> list[Path]:
        input_path = Path(input_path)
        written = []
        if not input_path.exists():
            raise FileNotFoundError(f"{input_path} not found")

        # If the directory contains subdirectories, treat each subdir as a speaker folder
        subdirs = [p for p in input_path.iterdir() if p.is_dir()]
        if subdirs:
            # collect speaker files from each subdir (assume aligned timelines)
            speaker_files = []
            for sp in sorted(subdirs):
                # find a single audio file per speaker folder (or multiple; we'll take the first matching)
                audio_files = [f for f in sorted(sp.iterdir()) if f.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".ogg")]
                if not audio_files:
                    continue
                speaker_files.append((sp.name, audio_files[0]))
        else:
            # no subdirs: treat top-level files as separate speaker-aligned tracks
            speaker_files = []
            for f in sorted(input_path.iterdir()):
                if not f.is_file():
                    continue
                if f.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
                    speaker_files.append((f.stem, f))

        if not speaker_files:
            return []

        # Load all speaker tracks, resample and normalize. Truncate to shortest length.
        audios = []
        names = []
        for name, path in speaker_files:
            a = self._load_audio(path)
            audios.append(a)
            names.append(name)

        min_len = min(len(a) for a in audios)
        audios = [a[:min_len] for a in audios]

        # Create a combined mix for turn-detection (average to avoid clipping)
        mix = np.stack(audios, axis=0).sum(axis=0) / float(len(audios))

        # run VAD on combined mix to get conversation turns
        segments = self.vad.get_speech_segments(mix)
        segments = self._merge_and_split_segments(segments)

        # Precompute per-speaker speech intervals to know active speakers per turn
        speaker_intervals = [self._merge_and_split_segments(self.vad.get_speech_segments(a)) for a in audios]

        def _intersects(a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
            return not (a_e <= b_s or b_e <= a_s)

        # Save each global turn in chronological order with combined audio and metadata of active speakers
        for start, end in segments:
            start = max(0, int(start))
            end = min(min_len, int(end))
            if end <= start:
                continue

            # determine active speakers in this time window
            active = []
            for idx, intervals in enumerate(speaker_intervals):
                for s, e in intervals:
                    if _intersects(start, end, s, e):
                        active.append(names[idx])
                        break

            # If a single active speaker detected, use that speaker's track for the chunk
            if len(active) == 1:
                idx = names.index(active[0])
                chunk_audio = audios[idx][start:end]
                speaker_id = active[0]
            else:
                # multiple or no active speakers: mix all tracks for the chunk
                chunk_audio = np.stack([a[start:end] for a in audios], axis=0).sum(axis=0) / float(len(audios))
                speaker_id = ",".join(active) if active else "mixed"
            out = self._save_segment(chunk_audio, speaker_id, start, end)
            if out is not None:
                written.append(out)

        return written

    def _merge_and_split_segments(self, segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Merge segments separated by small silences and split if longer than max chunk length.

        segments are (start_sample, end_sample) tuples in samples.
        """
        if not segments:
            return []

        # sort by start
        segs = sorted([(int(s), int(e)) for s, e in segments], key=lambda x: x[0])
        silence_samples = int(self.sample_rate_hz * (self.max_silence_between_ms / 1000.0))
        # if max_chunk_seconds is None or <= 0, do not split by length
        if self.max_chunk_seconds is None or self.max_chunk_seconds <= 0:
            max_chunk_samples = None
        else:
            max_chunk_samples = int(self.sample_rate_hz * float(self.max_chunk_seconds))

        merged: list[tuple[int, int]] = []
        cur_s, cur_e = segs[0]
        for s, e in segs[1:]:
            # if gap small, merge
            if s - cur_e <= silence_samples:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))


        # split merged segments that exceed max_chunk_samples (if set)
        out: list[tuple[int, int]] = []
        for s, e in merged:
            length = e - s
            if max_chunk_samples is None or length <= max_chunk_samples:
                out.append((s, e))
                continue

            # split into consecutive chunks of up to max_chunk_samples
            start = s
            while start < e:
                end = min(e, start + max_chunk_samples)
                out.append((start, end))
                start = end

        return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk diarized audio using Silero VAD")
    parser.add_argument("input", help="Input file or directory containing diarized audio or speaker folders")
    parser.add_argument("--out", default="diarized_chunks", help="Output directory for chunks")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--min", type=float, default=2, help="Minimum speech seconds to keep segment")
    args = parser.parse_args()

    chunker = DiarizationChunker(sample_rate_hz=args.sr, output_dir=Path(args.out), min_speech_seconds=args.min)
    inp = Path(args.input)
    if inp.is_file():
        chunker.process_file(inp)
    else:
        chunker.process_diarized_dir(inp)
