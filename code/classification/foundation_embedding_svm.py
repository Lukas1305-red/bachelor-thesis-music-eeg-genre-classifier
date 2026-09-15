"""Leakage-safe SVM evaluation on frozen BioCodec EEG embeddings.

The encoder is never fine-tuned.  Each trial is normalized independently and
embedded independently, so evaluating a trial never uses statistics from a
different trial.  SVM scaling and hyper-parameter selection happen inside the
nested CV folds only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


DEFAULT_RAW_DATA_PATH = Path("data/eeg_data_features/pickle")
DEFAULT_SAMPLE_RATE = 250
DEFAULT_WINDOW_SECONDS = 5.0
CSV_CHANNEL_PATTERN = re.compile(r"np\.float(?:32|64)\(([^)]*)\)")


class BioCodecEEGEmbeddingExtractor:
    """Extract fixed-length, frozen BioCodec embeddings from EEG trials."""

    def __init__(self, checkpoint_path: str | Path, sample_rate: int, name: str, device: str | None = None):
        try:
            import torch
        except ImportError as exc:  # keeps non-embedding project code importable
            raise ImportError("Foundation-embedding mode requires PyTorch. Install torch first.") from exc

        biocodec_path = os.environ.get("BIOCODEC_PATH")
        if biocodec_path and biocodec_path not in sys.path:
            sys.path.append(biocodec_path)
        try:
            from biocodec.model import BioCodecModel  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Could not import BioCodec. Set BIOCODEC_PATH to the BioCodec repository "
                "or install its package."
            ) from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = str(checkpoint_path)
        self.model_name = name
        # BioCodec-release calls its 250 Hz EEG architecture
        # ``_get_optimized_model`` (the EMG architecture is separate).
        eeg_factory = getattr(BioCodecModel, "_get_optimized_model", None)
        if eeg_factory is None:
            raise AttributeError(
                "This BioCodec checkout has no _get_optimized_model EEG factory. Use the "
                "BioCodec-release version that matches the supplied checkpoint."
            )
        self.model = eeg_factory(
            sample_rate=sample_rate, signal_normalize=False, name=name
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        if any(key.startswith("_orig_mod.") for key in state_dict):
            state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _embed_normalized_signal(self, signal: np.ndarray) -> np.ndarray:
        """Encode one already-normalized ``(channels, time)`` signal."""
        tensor = self.torch.from_numpy(signal.astype(np.float32, copy=False)).unsqueeze(1).to(self.device)
        encoded = self.model.encoder(tensor)  # (channels, embedding_dim, time)
        if encoded.ndim != 3:
            raise ValueError(f"Unexpected encoder output shape: {tuple(encoded.shape)}")

        # Unlike the example's accidental q=0 upper threshold, these are actual top/bottom quartiles.
        q75 = self.torch.quantile(encoded, 0.75, dim=2, keepdim=True)
        q25 = self.torch.quantile(encoded, 0.25, dim=2, keepdim=True)
        top = self.torch.nanmean(self.torch.where(encoded >= q75, encoded, self.torch.nan), dim=2)
        bottom = self.torch.nanmean(self.torch.where(encoded <= q25, encoded, self.torch.nan), dim=2)
        return self.torch.cat((top, bottom), dim=1).flatten().cpu().numpy()

    @staticmethod
    def _normalize_trial(signal: np.ndarray) -> np.ndarray:
        """Z-score channels using only the complete current song/trial."""
        if signal.ndim != 2 or signal.shape[0] != 7:
            raise ValueError(f"Expected a (7, time) trial, got {signal.shape}.")
        if not np.isfinite(signal).all():
            raise ValueError("Signals contain NaN or infinity; clean the raw EEG data before embedding.")
        return (signal - signal.mean(axis=1, keepdims=True)) / (
            signal.std(axis=1, keepdims=True) + 1e-8
        )

    def transform_windowed_trials(
        self, trials: list[np.ndarray], sample_rate: int, window_seconds: float
    ) -> tuple[np.ndarray, list[int]]:
        """Pool 5-second BioCodec embeddings into one embedding per song.

        Each song is normalized once before it is split into complete windows.
        Windows are never separate SVM examples, so a song cannot leak from a
        training fold into a test fold.
        """
        window_samples = round(sample_rate * window_seconds)
        if window_samples <= 0:
            raise ValueError("window_seconds must be positive.")

        embeddings: list[np.ndarray] = []
        window_counts: list[int] = []
        with self.torch.no_grad():
            for trial_index, trial in enumerate(trials):
                normalized_trial = self._normalize_trial(trial)
                n_windows = normalized_trial.shape[1] // window_samples
                if n_windows == 0:
                    raise ValueError(
                        f"Trial {trial_index} is shorter than one {window_seconds:g}-second window: "
                        f"{normalized_trial.shape[1]} samples."
                    )
                embeddings_for_trial = [
                    self._embed_normalized_signal(
                        normalized_trial[:, start : start + window_samples]
                    )
                    for start in range(0, n_windows * window_samples, window_samples)
                ]
                embeddings.append(np.mean(embeddings_for_trial, axis=0))
                window_counts.append(n_windows)
        return np.stack(embeddings), window_counts


def load_headset_raw_session(
    session_number: int,
    raw_data_path: str | Path = DEFAULT_RAW_DATA_PATH,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    minimum_duration_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load the same session labels as the current headset classifier, plus raw EEG."""
    path = Path(raw_data_path) / f"{session_number:03d}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Raw session file not found: {path}")
    frame = pd.read_pickle(path)
    eeg_columns = sorted(
        (column for column in frame.columns if column.startswith("EEG_") and "Ch3" not in column),
        key=lambda column: int("".join(filter(str.isdigit, column)) or 0),
    )
    if len(eeg_columns) != 7:
        raise ValueError(f"Expected the seven non-reference EEG channels in {path}, found {eeg_columns}.")
    if "genre" not in frame:
        raise ValueError(f"{path} has no 'genre' column.")

    n_samples = round(sample_rate * minimum_duration_seconds)
    trials = []
    for row_index, row in frame.iterrows():
        trial = np.asarray([row[column] for column in eeg_columns], dtype=np.float64)
        if trial.ndim != 2 or trial.shape[0] != 7 or trial.shape[1] < n_samples:
            raise ValueError(
                f"Session {session_number}, trial {row_index} must have at least {n_samples} samples "
                f"in seven channels; got {trial.shape}."
            )
        trials.append(trial)
    return trials, frame["genre"].astype(str).to_numpy()


def _parse_csv_signal(value: object) -> np.ndarray:
    """Parse this project's ``[np.float32(...), ...]`` CSV representation."""
    if not isinstance(value, str):
        raise ValueError(f"Expected a serialized EEG array in the CSV, got {type(value).__name__}.")
    # The files were written from Python lists containing np.float32 scalars,
    # which are not valid JSON or Python literals.  Convert only that wrapper;
    # np.fromstring is fast and does not evaluate data from the CSV.
    numeric_text = CSV_CHANNEL_PATTERN.sub(r"\1", value).strip().strip("[]")
    signal = np.fromstring(numeric_text, sep=",", dtype=np.float64)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("CSV contains an empty, non-numeric, NaN, or infinite EEG signal.")
    return signal


def _denoise_csv_trials(
    trials: list[np.ndarray], sample_rate: int, remove_ica_artifacts: bool
) -> list[np.ndarray]:
    """Filter a whole CSV session and optionally apply the existing session ICA.

    CSV rows are already song-segmented, so the original inter-song gaps are not
    available.  Concatenating rows is still preferable to fitting ICA separately
    on short trials; filter transients at row joins are avoided by splitting only
    after filtering the complete per-channel sequence.
    """
    from code.denoising.denoising import artifact_removal_ica, bandpass, notch_filter

    lengths = [trial.shape[1] for trial in trials]
    session = np.concatenate(trials, axis=1)  # (channels, session samples)
    filtered = np.stack(
        [
            notch_filter(bandpass(channel, [0.5, 100], sample_rate), 50, sample_rate)
            for channel in session
        ]
    )
    if remove_ica_artifacts:
        filtered = artifact_removal_ica(filtered)
    split_points = np.cumsum(lengths)[:-1]
    return list(np.split(filtered, split_points, axis=1))


def load_headset_csv_session(
    session_number: int,
    raw_data_path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    minimum_duration_seconds: float = DEFAULT_WINDOW_SECONDS,
    denoise: bool = False,
    remove_ica_artifacts: bool = False,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Load a session from ``data/eeg_data/csv`` into BioCodec's (trial,C,T) shape."""
    path = Path(raw_data_path) / f"{session_number:03d}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV session file not found: {path}")
    frame = pd.read_csv(path, index_col=0)
    eeg_columns = [f"EEG_Ch{channel}" for channel in range(1, 9) if channel != 3]
    missing = [column for column in eeg_columns if column not in frame]
    if missing or "genre" not in frame:
        raise ValueError(f"{path} must contain genre and {eeg_columns}; missing {missing}.")

    full_trials = []
    for row_index, row in frame.iterrows():
        trial = np.stack([_parse_csv_signal(row[column]) for column in eeg_columns])
        if trial.shape[0] != 7:
            raise ValueError(f"Session {session_number}, trial {row_index} has invalid channel shape {trial.shape}.")
        full_trials.append(trial)
    if denoise:
        full_trials = _denoise_csv_trials(full_trials, sample_rate, remove_ica_artifacts)

    n_samples = round(sample_rate * minimum_duration_seconds)
    trials = []
    for row_index, trial in enumerate(full_trials):
        if trial.shape[1] < n_samples:
            raise ValueError(
                f"Session {session_number}, trial {row_index} must have at least {n_samples} samples; "
                f"got {trial.shape}."
            )
        trials.append(trial)
    return trials, frame["genre"].astype(str).to_numpy()


def _svm_pipeline(params: dict[str, Any], random_state: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(class_weight="balanced", random_state=random_state, **params)),
    ])


def _select_params(X: np.ndarray, y: np.ndarray, param_list: list[dict[str, Any]], inner_cv: StratifiedKFold, random_state: int) -> dict[str, Any]:
    best_params, best_score = param_list[0], -np.inf
    for params in param_list:
        scores = []
        for train_index, validation_index in inner_cv.split(X, y):
            classifier = _svm_pipeline(params, random_state)
            classifier.fit(X[train_index], y[train_index])
            scores.append(f1_score(y[validation_index], classifier.predict(X[validation_index]), average="macro", zero_division=0))
        score = float(np.mean(scores))
        if score > best_score:
            best_params, best_score = params, score
    return best_params


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def evaluate_frozen_embeddings(
    extractor: BioCodecEEGEmbeddingExtractor,
    num_subjects: int = 20,
    raw_data_path: str | Path = DEFAULT_RAW_DATA_PATH,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    n_outer: int = 5,
    n_inner: int = 4,
    random_state: int = 1997,
    input_format: str = "pickle",
    denoise_csv: bool = False,
    remove_ica_artifacts: bool = False,
) -> dict[str, Any]:
    """Run subject-local nested CV and return a JSON-safe audit report."""
    param_list = list(ParameterGrid([
        {"C": [0.01, 0.1, 1, 10, 100], "kernel": ["linear"]},
        {"C": [0.01, 0.1, 1, 10, 100], "kernel": ["rbf"], "gamma": ["scale", 0.001, 0.01, 0.1]},
    ]))
    subject_results = []
    # Keep genre names for the global report. Numeric encodings are local to a
    # session and therefore must not be combined across sessions.
    all_y_true: list[str] = []
    all_y_pred: list[str] = []

    for subject in range(num_subjects):
        if input_format == "csv":
            trials, genre_labels = load_headset_csv_session(
                subject, raw_data_path, sample_rate, window_seconds,
                denoise=denoise_csv, remove_ica_artifacts=remove_ica_artifacts,
            )
        else:
            trials, genre_labels = load_headset_raw_session(
                subject, raw_data_path, sample_rate, window_seconds
            )
        encoder = LabelEncoder()
        y = encoder.fit_transform(genre_labels)
        minimum_class_count = int(np.bincount(y).min())
        session_outer_folds = min(n_outer, minimum_class_count)
        if session_outer_folds < 2:
            raise ValueError(
                f"Session {subject} needs at least two examples of every class for stratified CV; "
                f"the smallest class has {minimum_class_count}."
            )
        if session_outer_folds != n_outer:
            print(
                f"Session {subject:02d}: using {session_outer_folds} outer folds instead of "
                f"{n_outer}; its smallest genre has only {minimum_class_count} songs."
            )

        print(f"========= SUBJECT {subject:02d}: extracting frozen embeddings =========")
        X, window_counts = extractor.transform_windowed_trials(
            trials, sample_rate, window_seconds
        )
        outer_cv = StratifiedKFold(
            n_splits=session_outer_folds, shuffle=True, random_state=random_state
        )
        folds, subject_true, subject_pred = [], [], []
        for fold, (train_validation_index, test_index) in enumerate(outer_cv.split(X, y), start=1):
            X_train_validation, y_train_validation = X[train_validation_index], y[train_validation_index]
            inner_folds = min(n_inner, int(np.bincount(y_train_validation).min()))
            if inner_folds < 2:
                raise ValueError(
                    f"Session {subject}, outer fold {fold} has too few training examples "
                    "of a class for inner stratified CV."
                )
            inner_cv = StratifiedKFold(
                n_splits=inner_folds, shuffle=True, random_state=random_state
            )
            best_params = _select_params(X_train_validation, y_train_validation, param_list, inner_cv, random_state)
            classifier = _svm_pipeline(best_params, random_state)
            classifier.fit(X_train_validation, y_train_validation)
            prediction = classifier.predict(X[test_index])
            accuracy = float(accuracy_score(y[test_index], prediction))
            macro_f1 = float(f1_score(y[test_index], prediction, average="macro", zero_division=0))
            print(f"  fold {fold}/{session_outer_folds} | accuracy={accuracy:.4f} | macro_f1={macro_f1:.4f} | params={best_params}")
            folds.append({"fold": fold, "inner_folds": inner_folds, "accuracy": accuracy, "macro_f1": macro_f1, "best_params": best_params})
            subject_true.extend(y[test_index])
            subject_pred.extend(prediction)

        report = classification_report(subject_true, subject_pred, target_names=encoder.classes_.tolist(), output_dict=True, zero_division=0)
        mean_accuracy = float(np.mean([fold["accuracy"] for fold in folds]))
        mean_macro_f1 = float(np.mean([fold["macro_f1"] for fold in folds]))
        print(f"Subject {subject:02d} summary | accuracy={mean_accuracy:.4f} | macro_f1={mean_macro_f1:.4f}")
        subject_results.append({
            "subject": subject,
            "classes": encoder.classes_.tolist(),
            "embedding_shape": list(X.shape),
            "windows_per_song": window_counts,
            "outer_folds": session_outer_folds,
            "mean_accuracy": mean_accuracy,
            "mean_macro_f1": mean_macro_f1,
            "folds": folds,
            "classification_report": report,
        })
        all_y_true.extend(encoder.inverse_transform(subject_true))
        all_y_pred.extend(encoder.inverse_transform(subject_pred))

    overall_accuracy = float(accuracy_score(all_y_true, all_y_pred))
    overall_macro_f1 = float(f1_score(all_y_true, all_y_pred, average="macro", zero_division=0))
    print(f"========= OVERALL | accuracy={overall_accuracy:.4f} | macro_f1={overall_macro_f1:.4f} =========")
    return _jsonable({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "frozen_biocodec_eeg_embeddings_svm",
        "evaluation": {
            "scheme": "per-subject nested stratified cross-validation",
            "outer_folds_requested": n_outer,
            "inner_folds_requested": n_inner,
            "folds_are_reduced_per_session_when_a_class_has_too_few_songs": True,
            "random_state": random_state,
            "leakage_safeguards": [
                "The pretrained encoder is frozen and receives each trial independently.",
                "Z-score normalization is per song and per channel before windowing; no test-set statistics are used.",
                "All windows of a song are mean-pooled into one embedding before CV, so no song spans folds.",
                "StandardScaler is fit only inside each inner/final SVM training fold.",
                "SVM hyperparameters are selected only on the outer training partition; outer test trials are never used for selection.",
            ],
        },
        "data": {"raw_data_path": str(raw_data_path), "input_format": input_format, "csv_denoised": denoise_csv if input_format == "csv" else None, "csv_ica_artifact_removal": remove_ica_artifacts if input_format == "csv" else None, "sample_rate_hz": sample_rate, "window_seconds": window_seconds, "window_pooling": "mean", "num_subjects": num_subjects},
        "foundation_model": {"name": extractor.model_name, "checkpoint_path": extractor.checkpoint_path, "device": extractor.device, "fine_tuned": False},
        "overall": {"accuracy": overall_accuracy, "macro_f1": overall_macro_f1, "classification_report": classification_report(all_y_true, all_y_pred, output_dict=True, zero_division=0)},
        "subjects": subject_results,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to the pretrained BioCodec EEG checkpoint (.pt).")
    parser.add_argument("--model-name", required=True, help="BioCodec model name used when creating the EEG architecture.")
    parser.add_argument("--raw-data-path", type=Path, default=DEFAULT_RAW_DATA_PATH)
    parser.add_argument("--input-format", choices=["pickle", "csv"], default="pickle")
    parser.add_argument("--denoise-csv", action="store_true", help="For CSV input, apply 0.5-100 Hz band-pass and 50 Hz notch filtering to the whole session before embedding.")
    parser.add_argument("--remove-ica-artifacts", action="store_true", help="For denoised CSV input, also use the project's session-level ICA artifact removal.")
    parser.add_argument("--output", type=Path, default=Path("foundation_embedding_svm_results.json"))
    parser.add_argument("--num-subjects", type=int, default=20)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS, help="Length of each complete BioCodec window; song-level embeddings are the mean of all window embeddings.")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=1997)
    args = parser.parse_args()
    extractor = BioCodecEEGEmbeddingExtractor(args.checkpoint, args.sample_rate, args.model_name)
    if args.remove_ica_artifacts and not args.denoise_csv:
        parser.error("--remove-ica-artifacts requires --denoise-csv")
    results = evaluate_frozen_embeddings(extractor, args.num_subjects, args.raw_data_path, args.sample_rate, args.window_seconds, args.outer_folds, args.inner_folds, args.random_state, args.input_format, args.denoise_csv, args.remove_ica_artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved full JSON report to {args.output}")


if __name__ == "__main__":
    main()
