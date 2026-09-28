"""Leakage-safe music-envelope retrieval from EEG.

The task does not use genre labels for learning.  A ridge decoder is fit on
training sessions to predict a song's audio RMS envelope from its EEG.  For a
held-out EEG song, the predicted envelope is compared with every candidate
audio envelope from the same held-out session; top-1 retrieval is the metric.

The XDF ``Song`` stream, rather than a reconstructed shuffle order, supplies
the exact EEG-row-to-GTZAN-file mapping.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyxdf
import scipy.signal
import soundfile as sf
from sklearn.linear_model import Ridge

from code.classification.foundation_embedding_svm import load_headset_csv_session


DEFAULT_SAMPLE_RATE = 250
DEFAULT_ENVELOPE_RATE = 50


@dataclass(frozen=True)
class SongRecord:
    session: int
    row_index: int
    genre: str
    song_id: str
    audio_path: Path


def _parse_sessions(value: str) -> list[int]:
    sessions: list[int] = []
    for item in value.split(","):
        if "-" in item:
            start, end = (int(part) for part in item.split("-", maxsplit=1))
            sessions.extend(range(start, end + 1))
        elif item:
            sessions.append(int(item))
    return sorted(set(sessions))


def _song_stream(path: Path) -> list[tuple[str, str]]:
    streams, _ = pyxdf.load_xdf(path)
    for stream in streams:
        if stream["info"]["name"] == ["Song"]:
            # The acquisition code emits one empty Song event after playback.
            return [(str(row[0]), str(row[1])) for row in stream["time_series"] if row[0] and row[1]]
    raise ValueError(f"No Song stream found in {path}.")


def build_manifest(
    sessions: Iterable[int], xdf_dir: Path, eeg_csv_dir: Path, gtzan_dir: Path
) -> list[SongRecord]:
    """Recover the exact CSV-row-to-audio-file association from XDF metadata."""
    records: list[SongRecord] = []
    for session in sessions:
        xdf_path = xdf_dir / f"sub-P001_ses-S{session:03d}_task-Default_run-001_eeg.xdf"
        if not xdf_path.exists():
            raise FileNotFoundError(f"XDF recording not found: {xdf_path}")
        csv_path = eeg_csv_dir / f"{session:03d}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"EEG CSV not found: {csv_path}")
        song_events = _song_stream(xdf_path)
        frame = pd.read_csv(csv_path, index_col=0)
        if len(song_events) != len(frame):
            raise ValueError(
                f"Session {session}: XDF has {len(song_events)} non-empty Song events but CSV has {len(frame)} rows."
            )
        for row_index, ((song_id, genre), csv_genre) in enumerate(zip(song_events, frame["genre"].astype(str))):
            if genre != csv_genre:
                raise ValueError(
                    f"Session {session}, row {row_index}: XDF genre {genre!r} disagrees with CSV genre {csv_genre!r}."
                )
            audio_path = gtzan_dir / genre / f"{genre}.{song_id}.wav"
            if not audio_path.exists():
                raise FileNotFoundError(f"Session {session}, row {row_index}: matching audio file not found: {audio_path}")
            records.append(SongRecord(session, row_index, genre, song_id, audio_path))
    return records


def _audio_envelope(path: Path, frame_rate: int) -> np.ndarray:
    waveform, sample_rate = sf.read(path, always_2d=True)
    mono = waveform.mean(axis=1)
    # GTZAN clips are nominally 30 seconds.  Fixed frames make candidates comparable.
    frames = round(len(mono) / sample_rate * frame_rate)
    chunks = np.array_split(mono, frames)
    envelope = np.asarray([np.sqrt(np.mean(chunk**2)) for chunk in chunks], dtype=np.float64)
    return (envelope - envelope.mean()) / (envelope.std() + 1e-8)


def _eeg_envelope_features(trial: np.ndarray, target_frames: int) -> np.ndarray:
    """Resample seven normalized EEG channels to the audio-envelope frame rate."""
    trial = (trial - trial.mean(axis=1, keepdims=True)) / (trial.std(axis=1, keepdims=True) + 1e-8)
    return scipy.signal.resample(trial, target_frames, axis=1).T  # (time, channels)


def _aligned_pairs(eeg: np.ndarray, audio: np.ndarray, delay_frames: int) -> tuple[np.ndarray, np.ndarray]:
    # EEG follows acoustic input, so use later EEG frames to predict the earlier audio envelope.
    if delay_frames == 0:
        return eeg, audio
    return eeg[delay_frames:], audio[:-delay_frames]


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    # GTZAN files differ by a few samples/duration frames. Retrieval compares
    # the common time span so candidates remain valid without time padding.
    length = min(len(first), len(second))
    first, second = first[:length], second[:length]
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return -1.0
    return float(np.corrcoef(first, second)[0, 1])


def retrieval_score_matrix(
    model: Ridge, eeg_songs: list[np.ndarray], audio_songs: list[np.ndarray], delay_frames: int
) -> np.ndarray:
    """Return correlation scores: rows are EEG songs, columns are audio songs."""
    all_scores = []
    for index, eeg in enumerate(eeg_songs):
        features, _ = _aligned_pairs(eeg, audio_songs[index], delay_frames)
        predicted = model.predict(features)
        all_scores.append([
            _correlation(predicted, _aligned_pairs(eeg, candidate, delay_frames)[1])
            for candidate in audio_songs
        ])
    return np.asarray(all_scores)


def _metrics_from_score_matrix(scores: np.ndarray, matched_audio_indices: np.ndarray | None = None) -> dict[str, object]:
    """Compute retrieval metrics for one specified EEG/audio matching."""
    if matched_audio_indices is None:
        matched_audio_indices = np.arange(len(scores))
    ranks, correlations = [], []
    for index, audio_index in enumerate(matched_audio_indices):
        rank = int(np.sum(scores[index] > scores[index, audio_index]) + 1)
        ranks.append(rank)
        correlations.append(float(scores[index, audio_index]))
    return {
        "top1_accuracy": float(np.mean(np.asarray(ranks) == 1)),
        "mean_reciprocal_rank": float(np.mean(1.0 / np.asarray(ranks))),
        "mean_true_correlation": float(np.mean(correlations)),
        "ranks": ranks,
        "candidate_count": len(scores),
        "chance_top1_accuracy": 1.0 / len(scores),
    }


def retrieval_metrics(model: Ridge, eeg_songs: list[np.ndarray], audio_songs: list[np.ndarray], delay_frames: int) -> dict[str, object]:
    return _metrics_from_score_matrix(retrieval_score_matrix(model, eeg_songs, audio_songs, delay_frames))


def permutation_test(scores: np.ndarray, permutations: int, rng: np.random.Generator) -> dict[str, object]:
    """Test observed matching against random one-to-one audio identities."""
    observed = _metrics_from_score_matrix(scores)
    null_top1 = np.empty(permutations)
    null_mrr = np.empty(permutations)
    for permutation_index in range(permutations):
        null_metrics = _metrics_from_score_matrix(scores, rng.permutation(len(scores)))
        null_top1[permutation_index] = null_metrics["top1_accuracy"]
        null_mrr[permutation_index] = null_metrics["mean_reciprocal_rank"]
    return {
        "permutations": permutations,
        "top1_p_value_one_sided": float((1 + np.sum(null_top1 >= observed["top1_accuracy"])) / (permutations + 1)),
        "mrr_p_value_one_sided": float((1 + np.sum(null_mrr >= observed["mean_reciprocal_rank"])) / (permutations + 1)),
        "null_top1_mean": float(null_top1.mean()),
        "null_mrr_mean": float(null_mrr.mean()),
    }


def load_session_pairs(
    session: int, records: list[SongRecord], eeg_csv_dir: Path, sample_rate: int,
    frame_rate: int, denoise_csv: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    trials, _ = load_headset_csv_session(
        session, eeg_csv_dir, sample_rate, minimum_duration_seconds=5.0, denoise=denoise_csv
    )
    session_records = [record for record in records if record.session == session]
    if len(trials) != len(session_records):
        raise ValueError(f"Session {session}: manifest and EEG trial counts differ.")
    audio = [_audio_envelope(record.audio_path, frame_rate) for record in session_records]
    eeg = [_eeg_envelope_features(trial, len(envelope)) for trial, envelope in zip(trials, audio)]
    return eeg, audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xdf-dir", type=Path, default=Path("data/eeg-data"))
    parser.add_argument("--eeg-csv-dir", type=Path, default=Path("data/eeg_data/csv"))
    parser.add_argument("--gtzan-dir", type=Path, default=Path("data/genres_original"))
    parser.add_argument("--train-sessions", default="0-15")
    parser.add_argument("--validation-session", type=int, default=16)
    parser.add_argument("--test-session", type=int, default=17)
    parser.add_argument("--test-sessions", help="Final test sessions, e.g. '18-19'; requires fixed hyperparameters.")
    parser.add_argument("--fixed-delay-ms", type=int, help="Previously selected delay; disables validation selection.")
    parser.add_argument("--fixed-alpha", type=float, help="Previously selected ridge alpha; disables validation selection.")
    parser.add_argument("--permutations", type=int, default=0, help="Permutation count for each final test session.")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--envelope-rate", type=int, default=DEFAULT_ENVELOPE_RATE)
    parser.add_argument("--max-delay-ms", type=int, default=500)
    parser.add_argument("--delay-step-ms", type=int, default=50)
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100")
    parser.add_argument("--max-training-frames", type=int, default=150000)
    parser.add_argument("--denoise-csv", action="store_true")
    parser.add_argument("--manifest-output", type=Path, default=Path("data/eeg_audio_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/eeg_audio_retrieval_results.json"))
    parser.add_argument("--seed", type=int, default=1997)
    args = parser.parse_args()
    train_sessions = _parse_sessions(args.train_sessions)
    final_mode = args.test_sessions is not None
    fixed_parameters_given = args.fixed_delay_ms is not None and args.fixed_alpha is not None
    if final_mode != fixed_parameters_given:
        parser.error("Use --test-sessions, --fixed-delay-ms, and --fixed-alpha together for final evaluation.")
    test_sessions = _parse_sessions(args.test_sessions) if final_mode else [args.test_session]
    used_sessions = train_sessions + ([] if final_mode else [args.validation_session]) + test_sessions
    if len(set(used_sessions)) != len(used_sessions):
        parser.error("Training, validation, and test sessions must be distinct.")
    records = build_manifest(used_sessions, args.xdf_dir, args.eeg_csv_dir, args.gtzan_dir)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "session": item.session, "row_index": item.row_index, "genre": item.genre,
        "song_id": item.song_id, "audio_path": str(item.audio_path),
    } for item in records]).to_csv(args.manifest_output, index=False)
    pairs = {session: load_session_pairs(session, records, args.eeg_csv_dir, args.sample_rate, args.envelope_rate, args.denoise_csv) for session in used_sessions}
    train_eeg = [song for session in train_sessions for song in pairs[session][0]]
    train_audio = [song for session in train_sessions for song in pairs[session][1]]
    rng = np.random.default_rng(args.seed)
    if final_mode:
        delay_ms, alpha = args.fixed_delay_ms, args.fixed_alpha
        delay_frames = round(delay_ms / 1000 * args.envelope_rate)
        X, y = zip(*[_aligned_pairs(eeg, audio, delay_frames) for eeg, audio in zip(train_eeg, train_audio)])
        X_all, y_all = np.concatenate(X), np.concatenate(y)
        if len(X_all) > args.max_training_frames:
            subset = rng.choice(len(X_all), args.max_training_frames, replace=False)
            X_all, y_all = X_all[subset], y_all[subset]
        model = Ridge(alpha=alpha).fit(X_all, y_all)
        final_tests = {}
        for session in test_sessions:
            test_eeg, test_audio = pairs[session]
            scores = retrieval_score_matrix(model, test_eeg, test_audio, delay_frames)
            metrics = _metrics_from_score_matrix(scores)
            if args.permutations:
                metrics["permutation_test"] = permutation_test(scores, args.permutations, rng)
            final_tests[str(session)] = metrics
        result = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "eeg_to_audio_envelope_retrieval_final_locked",
            "mapping": "XDF Song stream to GTZAN genre.song_id.wav", "genre_labels_used_for_training": False,
            "split": {"train_sessions": train_sessions, "final_test_sessions": test_sessions},
            "fixed_from_development": {"delay_ms": delay_ms, "ridge_alpha": alpha},
            "final_held_out_tests": final_tests, "manifest_path": str(args.manifest_output),
        }
    else:
        validation_eeg, validation_audio = pairs[args.validation_session]
        test_eeg, test_audio = pairs[args.test_session]
        candidates = []
        for delay_ms in range(0, args.max_delay_ms + 1, args.delay_step_ms):
            delay_frames = round(delay_ms / 1000 * args.envelope_rate)
            X, y = zip(*[_aligned_pairs(eeg, audio, delay_frames) for eeg, audio in zip(train_eeg, train_audio)])
            X_all, y_all = np.concatenate(X), np.concatenate(y)
            if len(X_all) > args.max_training_frames:
                subset = rng.choice(len(X_all), args.max_training_frames, replace=False)
                X_all, y_all = X_all[subset], y_all[subset]
            for alpha in [float(value) for value in args.alphas.split(",")]:
                model = Ridge(alpha=alpha).fit(X_all, y_all)
                metrics = retrieval_metrics(model, validation_eeg, validation_audio, delay_frames)
                candidates.append((metrics["top1_accuracy"], metrics["mean_reciprocal_rank"], delay_ms, alpha, model, metrics))
        _, _, delay_ms, alpha, model, validation_metrics = max(candidates, key=lambda item: (item[0], item[1]))
        test_metrics = retrieval_metrics(model, test_eeg, test_audio, round(delay_ms / 1000 * args.envelope_rate))
        result = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "eeg_to_audio_envelope_retrieval",
            "mapping": "XDF Song stream to GTZAN genre.song_id.wav", "genre_labels_used_for_training": False,
            "split": {"train_sessions": train_sessions, "validation_session": args.validation_session, "test_session": args.test_session},
            "selected_on_validation": {"delay_ms": delay_ms, "ridge_alpha": alpha, "metrics": validation_metrics},
            "held_out_test": test_metrics,
            "manifest_path": str(args.manifest_output),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    if final_mode:
        print(f"Using fixed delay={delay_ms} ms, alpha={alpha}; trained on sessions {train_sessions}")
        for session, metrics in result["final_held_out_tests"].items():
            p_value = metrics.get("permutation_test", {}).get("mrr_p_value_one_sided")
            p_text = "" if p_value is None else f"; MRR permutation p={p_value:.4f}"
            print(f"Final test session {int(session):02d} top-1={metrics['top1_accuracy']:.4f}; chance={metrics['chance_top1_accuracy']:.4f}{p_text}")
    else:
        print(f"Selected delay={delay_ms} ms, alpha={alpha} from validation session {args.validation_session:02d}")
        print(f"Held-out test top-1 retrieval={test_metrics['top1_accuracy']:.4f}; chance={test_metrics['chance_top1_accuracy']:.4f}")
    print(f"Saved manifest to {args.manifest_output} and results to {args.output}")


if __name__ == "__main__":
    main()
