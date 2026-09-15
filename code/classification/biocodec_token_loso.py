"""Leave-one-session-out music-genre classification from frozen BioCodec RVQ tokens.

Unlike ``foundation_embedding_svm.py``, this experiment preserves BioCodec's
discrete residual-vector-quantizer (RVQ) codes and their temporal order.  The
codec and codebook embeddings remain frozen; only a small temporal/channel
Transformer classifier is trained.  Every five-second window of a song has the
same label during training, but logits are mean-pooled to one prediction per
song before validation and testing.  Sessions are never mixed between splits.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from code.classification.foundation_embedding_svm import (
    BioCodecEEGEmbeddingExtractor,
    DEFAULT_SAMPLE_RATE,
    load_headset_csv_session,
)


DEFAULT_WINDOW_SECONDS = 5.0
DEFAULT_CACHE_DIR = Path("data/biocodec_tokens")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


class RVQTokenTransformer:
    """Small classifier head modeled on BioCodec's time/channel token head."""

    def __new__(
        cls,
        codebook_weights: list[np.ndarray],
        channels: int,
        timesteps: int,
        classes: int,
        device: str,
        dropout: float,
    ):
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                dim = int(codebook_weights[0].shape[1])
                self.codebooks = nn.ModuleList(
                    [nn.Embedding.from_pretrained(torch.as_tensor(weight, dtype=torch.float32), freeze=True)
                     for weight in codebook_weights]
                )
                self.time_position = nn.Parameter(torch.zeros(1, timesteps, dim))
                self.channel_position = nn.Parameter(torch.zeros(1, channels, dim))
                time_layer = nn.TransformerEncoderLayer(
                    d_model=dim, nhead=4, dim_feedforward=dim * 4,
                    dropout=dropout, activation="gelu", batch_first=True,
                )
                channel_layer = nn.TransformerEncoderLayer(
                    d_model=dim, nhead=4, dim_feedforward=dim * 4,
                    dropout=dropout, activation="gelu", batch_first=True,
                )
                self.time_transformer = nn.TransformerEncoder(time_layer, num_layers=1)
                self.channel_transformer = nn.TransformerEncoder(channel_layer, num_layers=1)
                self.classifier = nn.Sequential(
                    nn.LayerNorm(channels * dim),
                    nn.Dropout(dropout),
                    nn.Linear(channels * dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, classes),
                )

            def forward(self, codes: torch.Tensor) -> torch.Tensor:
                # codes: (batch, channels, time, codebooks)
                if codes.ndim != 4:
                    raise ValueError(f"Expected RVQ codes shaped (B,C,T,Q), got {tuple(codes.shape)}.")
                if codes.shape[-1] != len(self.codebooks):
                    raise ValueError("Unexpected number of RVQ codebooks.")
                x = sum(embedding(codes[..., index]) for index, embedding in enumerate(self.codebooks))
                batch, channels_, time, dim = x.shape
                if channels_ != channels or time != timesteps:
                    raise ValueError(
                        f"Expected ({channels}, {timesteps}) channels/timesteps, got ({channels_}, {time})."
                    )
                x = x.reshape(batch * channels, time, dim)
                x = self.time_transformer(x + self.time_position)
                x = x.reshape(batch, channels, time, dim).permute(0, 2, 1, 3)
                x = x.reshape(batch * time, channels, dim)
                x = self.channel_transformer(x + self.channel_position)
                x = x.reshape(batch, time, channels, dim).mean(dim=1).flatten(start_dim=1)
                return self.classifier(x)

        return _Model().to(device)


def _token_cache_path(cache_dir: Path, session: int) -> Path:
    return cache_dir / f"session_{session:03d}_5s_rvq.npz"


def build_or_load_session_tokens(
    session: int,
    extractor: BioCodecEEGEmbeddingExtractor,
    raw_data_path: Path,
    cache_dir: Path,
    sample_rate: int,
    window_seconds: float,
    denoise_csv: bool,
    remove_ica_artifacts: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(songs, windows, channels, tokens, codebooks)`` and labels."""
    cache_path = _token_cache_path(cache_dir, session)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return cached["codes"], cached["labels"].astype(str)

    trials, labels = load_headset_csv_session(
        session, raw_data_path, sample_rate, window_seconds,
        denoise=denoise_csv, remove_ica_artifacts=remove_ica_artifacts,
    )
    window_samples = round(sample_rate * window_seconds)
    all_song_codes: list[np.ndarray] = []
    with extractor.torch.no_grad():
        for song_index, trial in enumerate(trials):
            normalized = extractor._normalize_trial(trial)
            n_windows = normalized.shape[1] // window_samples
            if n_windows == 0:
                raise ValueError(f"Session {session}, song {song_index} has no complete window.")
            windows = normalized[:, : n_windows * window_samples].reshape(7, n_windows, window_samples)
            windows = windows.transpose(1, 0, 2)  # (windows, channels, samples)
            # BioCodec is a single-channel codec. Treat channels as its batch dimension.
            window_codes = []
            for window in windows:
                tensor = extractor.torch.from_numpy(window.astype(np.float32, copy=False)).unsqueeze(1).to(extractor.device)
                frames = extractor.model.encode(tensor)
                codes = extractor.torch.cat([frame_codes for frame_codes, _ in frames], dim=-1)
                window_codes.append(codes.permute(0, 2, 1).cpu().numpy().astype(np.uint16))
            all_song_codes.append(np.stack(window_codes))

    window_counts = {codes.shape[0] for codes in all_song_codes}
    if len(window_counts) != 1:
        raise ValueError(
            f"Session {session} has differing complete-window counts {sorted(window_counts)}; "
            "use a fixed crop before this experiment."
        )
    codes = np.stack(all_song_codes)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, codes=codes, labels=labels.astype(str))
    return codes, labels.astype(str)


def _song_logits(model: Any, song_codes: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    """Return mean window logits, one row per song."""
    import torch

    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(song_codes), batch_size):
            codes = torch.as_tensor(song_codes[start : start + batch_size], dtype=torch.long, device=device)
            batch, windows, channels, timesteps, codebooks = codes.shape
            logits = model(codes.reshape(batch * windows, channels, timesteps, codebooks))
            results.append(logits.reshape(batch, windows, -1).mean(dim=1).cpu().numpy())
    return np.concatenate(results)


def _train_fold(
    train_codes: np.ndarray,
    train_y: np.ndarray,
    validation_codes: np.ndarray,
    validation_y: np.ndarray,
    codebook_weights: list[np.ndarray],
    device: str,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    writer: Any | None = None,
    tensorboard_prefix: str = "",
) -> tuple[Any, dict[str, float], list[dict[str, float]]]:
    import torch
    import torch.nn.functional as functional

    _, _, channels, timesteps, _ = train_codes.shape
    model = RVQTokenTransformer(codebook_weights, channels, timesteps, len(np.unique(train_y)), device, dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state, best_metrics, stale_epochs = None, {"macro_f1": -1.0, "accuracy": 0.0, "loss": float("inf"), "epoch": 0}, 0
    history: list[dict[str, float]] = []
    # Windows are training examples only; validation/test metrics pool them by song.
    windows_per_song = train_codes.shape[1]
    flat_codes = train_codes.reshape(-1, channels, timesteps, train_codes.shape[-1])
    flat_y = np.repeat(train_y, windows_per_song)

    for epoch in range(1, epochs + 1):
        order = np.random.permutation(len(flat_y))
        model.train()
        training_loss_sum = 0.0
        training_examples = 0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            codes = torch.as_tensor(flat_codes[index], dtype=torch.long, device=device)
            labels = torch.as_tensor(flat_y[index], dtype=torch.long, device=device)
            optimizer.zero_grad()
            logits = model(codes)
            loss = functional.cross_entropy(logits, labels, label_smoothing=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            training_loss_sum += float(loss.item()) * len(index)
            training_examples += len(index)

        validation_logits = _song_logits(model, validation_codes, device, batch_size)
        validation_pred = validation_logits.argmax(axis=1)
        metrics = {
            "train_loss": training_loss_sum / training_examples,
            "validation_loss": float(functional.cross_entropy(
                torch.as_tensor(validation_logits), torch.as_tensor(validation_y, dtype=torch.long)
            ).item()),
            "macro_f1": float(f1_score(validation_y, validation_pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(validation_y, validation_pred)),
            "epoch": epoch,
        }
        history.append(metrics)
        if writer is not None:
            writer.add_scalar(f"{tensorboard_prefix}/train/loss", metrics["train_loss"], epoch)
            writer.add_scalar(f"{tensorboard_prefix}/validation/loss", metrics["validation_loss"], epoch)
            writer.add_scalar(f"{tensorboard_prefix}/validation/macro_f1", metrics["macro_f1"], epoch)
            writer.add_scalar(f"{tensorboard_prefix}/validation/accuracy", metrics["accuracy"], epoch)
            writer.flush()
        if metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_state, best_metrics, stale_epochs = copy.deepcopy(model.state_dict()), {
                "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"],
                "loss": metrics["validation_loss"], "epoch": epoch,
            }, 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("No classifier checkpoint was selected.")
    model.load_state_dict(best_state)
    return model, best_metrics, history


def run_memorization_sanity_check(
    extractor: BioCodecEEGEmbeddingExtractor,
    raw_data_path: Path,
    cache_dir: Path,
    output: Path,
    session: int,
    sample_rate: int,
    window_seconds: float,
    denoise_csv: bool,
    remove_ica_artifacts: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    tensorboard_dir: Path | None,
) -> dict[str, Any]:
    """Try to memorize one session; this is a training-pipeline diagnostic only."""
    import torch
    import torch.nn.functional as functional

    _set_seed(seed)
    print(f"Encoding/loading session {session:02d} RVQ tokens for memorization check...")
    codes, labels = build_or_load_session_tokens(
        session, extractor, raw_data_path, cache_dir, sample_rate, window_seconds,
        denoise_csv, remove_ica_artifacts,
    )
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    codebook_weights = [
        layer.codebook.detach().cpu().numpy().copy()
        for layer in extractor.model.quantizer.vq.layers
    ]
    _, _, channels, timesteps, codebooks = codes.shape
    # Remove regularizers deliberately: the model should be able to overfit
    # this small fixed dataset if the tokens and training path are sound.
    model = RVQTokenTransformer(
        codebook_weights, channels, timesteps, len(encoder.classes_), extractor.device, dropout=0.0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    flat_codes = codes.reshape(-1, channels, timesteps, codebooks)
    flat_y = np.repeat(y, codes.shape[1])
    writer = None
    tensorboard_run_dir = None
    if tensorboard_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        tensorboard_run_dir = tensorboard_dir / f"biocodec_token_memorization_session_{session:02d}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        writer = SummaryWriter(log_dir=str(tensorboard_run_dir))
        writer.add_text("experiment/purpose", "Diagnostic only: deliberately memorize one session; never report as held-out performance.")

    best_state, best_epoch, history = None, 0, []
    for epoch in range(1, epochs + 1):
        order = np.random.permutation(len(flat_y))
        model.train()
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            batch_codes = torch.as_tensor(flat_codes[index], dtype=torch.long, device=extractor.device)
            batch_y = torch.as_tensor(flat_y[index], dtype=torch.long, device=extractor.device)
            optimizer.zero_grad()
            loss = functional.cross_entropy(model(batch_codes), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(index)

        logits = _song_logits(model, codes, extractor.device, batch_size)
        prediction = logits.argmax(axis=1)
        metrics = {
            "epoch": epoch,
            "window_train_loss": loss_sum / len(flat_y),
            "song_train_loss": float(functional.cross_entropy(
                torch.as_tensor(logits), torch.as_tensor(y, dtype=torch.long)
            ).item()),
            "song_train_accuracy": float(accuracy_score(y, prediction)),
            "song_train_macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        }
        history.append(metrics)
        if writer is not None:
            for name, value in metrics.items():
                if name != "epoch":
                    writer.add_scalar(f"memorization/{name}", value, epoch)
            writer.flush()
        if best_state is None or metrics["song_train_accuracy"] >= history[best_epoch - 1]["song_train_accuracy"]:
            best_state, best_epoch = copy.deepcopy(model.state_dict()), epoch

    model.load_state_dict(best_state)
    if writer is not None:
        writer.close()
    final = history[-1]
    best = history[best_epoch - 1]
    result = _jsonable({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "biocodec_rvq_token_memorization_sanity_check",
        "warning": "This is an intentional same-session overfit diagnostic, not a valid generalization result.",
        "data": {"session": session, "songs": len(y), "windows_per_song": int(codes.shape[1])},
        "configuration": {
            "codec_frozen": True, "codebook_embeddings_frozen": True,
            "dropout": 0.0, "weight_decay": 0.0, "label_smoothing": 0.0,
            "epochs": epochs, "learning_rate": learning_rate,
            "tensorboard_log_dir": str(tensorboard_run_dir) if tensorboard_run_dir else None,
        },
        "best_epoch": best_epoch,
        "best": best,
        "final": final,
        "history": history,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(
        f"Memorization check session {session:02d} | final song accuracy={final['song_train_accuracy']:.4f} "
        f"| best song accuracy={best['song_train_accuracy']:.4f} at epoch {best_epoch}"
    )
    print(f"Saved sanity-check JSON report to {output}")
    return result


def evaluate_within_session_cv(
    extractor: BioCodecEEGEmbeddingExtractor,
    raw_data_path: Path,
    cache_dir: Path,
    output: Path,
    session: int,
    sample_rate: int,
    window_seconds: float,
    denoise_csv: bool,
    remove_ica_artifacts: bool,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    tensorboard_dir: Path | None,
) -> dict[str, Any]:
    """Nested song-level CV within one recording session.

    Each outer fold holds out 20% of songs for testing.  A distinct stratified
    portion of the remaining songs is used only for early stopping.  All
    windows belonging to a song remain together in every partition.
    """
    _set_seed(seed)
    print(f"Encoding/loading session {session:02d} RVQ tokens for within-session CV...")
    codes, labels = build_or_load_session_tokens(
        session, extractor, raw_data_path, cache_dir, sample_rate, window_seconds,
        denoise_csv, remove_ica_artifacts,
    )
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    minimum_class_count = int(np.bincount(y).min())
    outer_folds = min(5, minimum_class_count)
    if outer_folds < 3:
        raise ValueError("Within-session nested CV needs at least three songs in every genre.")
    codebook_weights = [
        layer.codebook.detach().cpu().numpy().copy()
        for layer in extractor.model.quantizer.vq.layers
    ]
    writer = None
    tensorboard_run_dir = None
    if tensorboard_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        tensorboard_run_dir = tensorboard_dir / f"biocodec_token_within_session_{session:02d}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        writer = SummaryWriter(log_dir=str(tensorboard_run_dir))
        writer.add_text("experiment/purpose", "Nested song-level CV within one session; test songs are never used for fitting or early stopping.")

    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    all_true: list[int] = []
    all_pred: list[int] = []
    folds = []
    for fold, (train_validation_index, test_index) in enumerate(outer_cv.split(codes, y), start=1):
        y_train_validation = y[train_validation_index]
        inner_folds = min(outer_folds - 1, int(np.bincount(y_train_validation).min()))
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold)
        train_relative, validation_relative = next(inner_cv.split(codes[train_validation_index], y_train_validation))
        train_index = train_validation_index[train_relative]
        validation_index = train_validation_index[validation_relative]
        print(f"========= SESSION {session:02d} | outer fold {fold}/{outer_folds} =========")
        model, validation_metrics, history = _train_fold(
            codes[train_index], y[train_index], codes[validation_index], y[validation_index],
            codebook_weights, extractor.device, epochs, patience, batch_size,
            learning_rate, weight_decay, dropout, writer=writer,
            tensorboard_prefix=f"session_{session:02d}/fold_{fold}",
        )
        test_logits = _song_logits(model, codes[test_index], extractor.device, batch_size)
        test_pred = test_logits.argmax(axis=1)
        accuracy = float(accuracy_score(y[test_index], test_pred))
        macro_f1 = float(f1_score(y[test_index], test_pred, average="macro", zero_division=0))
        if writer is not None:
            writer.add_scalar(f"session_{session:02d}/fold_{fold}/test/accuracy", accuracy, 0)
            writer.add_scalar(f"session_{session:02d}/fold_{fold}/test/macro_f1", macro_f1, 0)
            writer.flush()
        print(f"Outer fold {fold}/{outer_folds} | accuracy={accuracy:.4f} | macro_f1={macro_f1:.4f} | selected_epoch={validation_metrics['epoch']}")
        folds.append({
            "fold": fold, "train_song_count": len(train_index), "validation_song_count": len(validation_index),
            "test_song_count": len(test_index), "validation": validation_metrics,
            "test_accuracy": accuracy, "test_macro_f1": macro_f1, "training_history": history,
        })
        all_true.extend(y[test_index].tolist())
        all_pred.extend(test_pred.tolist())
    if writer is not None:
        writer.close()

    result = _jsonable({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "frozen_biocodec_rvq_tokens_within_session_nested_song_cv",
        "data": {"session": session, "songs": len(y), "windows_per_song": int(codes.shape[1]), "tensorboard_log_dir": str(tensorboard_run_dir) if tensorboard_run_dir else None},
        "evaluation": {
            "scheme": "outer stratified song CV; nested stratified song validation for early stopping",
            "outer_folds": outer_folds,
            "safeguards": [
                "All windows from each song remain in one train, validation, or test partition.",
                "Outer test songs are never used for fitting or early stopping.",
                "Codec and codebook embeddings remain frozen.",
            ],
        },
        "overall": {
            "accuracy": float(accuracy_score(all_true, all_pred)),
            "macro_f1": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
            "classification_report": classification_report(all_true, all_pred, target_names=label_encoder.classes_.tolist(), output_dict=True, zero_division=0),
        },
        "folds": folds,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"Saved within-session CV JSON report to {output}")
    return result


def evaluate_loso(
    extractor: BioCodecEEGEmbeddingExtractor,
    raw_data_path: Path,
    cache_dir: Path,
    output: Path,
    num_sessions: int,
    sample_rate: int,
    window_seconds: float,
    denoise_csv: bool,
    remove_ica_artifacts: bool,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    tensorboard_dir: Path | None,
    test_session: int | None,
    validation_session: int | None,
) -> dict[str, Any]:
    _set_seed(seed)
    device = extractor.device
    sessions = []
    for session in range(num_sessions):
        print(f"Encoding/loading session {session:02d} RVQ tokens...")
        sessions.append(build_or_load_session_tokens(
            session, extractor, raw_data_path, cache_dir, sample_rate, window_seconds,
            denoise_csv, remove_ica_artifacts,
        ))

    # Labels are shared across sessions but encoded once so logits align globally.
    label_encoder = LabelEncoder().fit(np.concatenate([labels for _, labels in sessions]))
    encoded_sessions = [(codes, label_encoder.transform(labels)) for codes, labels in sessions]
    codebook_weights = [
        layer.codebook.detach().cpu().numpy().copy()
        for layer in extractor.model.quantizer.vq.layers
    ]
    all_true: list[int] = []
    all_pred: list[int] = []
    fold_results = []
    writer = None
    tensorboard_run_dir = None
    if tensorboard_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        tensorboard_run_dir = tensorboard_dir / f"biocodec_token_loso_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        writer = SummaryWriter(log_dir=str(tensorboard_run_dir))
        writer.add_text("experiment/config", json.dumps({
            "window_seconds": window_seconds, "epochs": epochs, "patience": patience,
            "batch_size": batch_size, "learning_rate": learning_rate, "weight_decay": weight_decay,
            "dropout": dropout, "seed": seed,
        }, indent=2))
    if test_session is not None and not 0 <= test_session < num_sessions:
        raise ValueError(f"test_session must be between 0 and {num_sessions - 1}.")
    if validation_session is not None and not 0 <= validation_session < num_sessions:
        raise ValueError(f"validation_session must be between 0 and {num_sessions - 1}.")
    test_sessions = [test_session] if test_session is not None else list(range(num_sessions))
    if validation_session is not None and test_session is None:
        raise ValueError("--validation-session can only be used together with --test-session.")

    for held_out_session in test_sessions:
        fold_validation_session = validation_session if validation_session is not None else (held_out_session + 1) % num_sessions
        if fold_validation_session == held_out_session:
            raise ValueError("test_session and validation_session must be different.")
        training_sessions = [index for index in range(num_sessions) if index not in {held_out_session, fold_validation_session}]
        train_codes = np.concatenate([encoded_sessions[index][0] for index in training_sessions])
        train_y = np.concatenate([encoded_sessions[index][1] for index in training_sessions])
        validation_codes, validation_y = encoded_sessions[fold_validation_session]
        test_codes, test_y = encoded_sessions[held_out_session]
        print(f"========= TEST SESSION {held_out_session:02d} | validation session {fold_validation_session:02d} =========")
        model, validation_metrics, history = _train_fold(
            train_codes, train_y, validation_codes, validation_y, codebook_weights, device,
            epochs, patience, batch_size, learning_rate, weight_decay, dropout,
            writer=writer, tensorboard_prefix=f"fold_{held_out_session:02d}",
        )
        test_logits = _song_logits(model, test_codes, device, batch_size)
        test_pred = test_logits.argmax(axis=1)
        accuracy = float(accuracy_score(test_y, test_pred))
        macro_f1 = float(f1_score(test_y, test_pred, average="macro", zero_division=0))
        if writer is not None:
            writer.add_scalar(f"fold_{held_out_session:02d}/test/accuracy", accuracy, 0)
            writer.add_scalar(f"fold_{held_out_session:02d}/test/macro_f1", macro_f1, 0)
            writer.flush()
        print(f"Test session {held_out_session:02d} | accuracy={accuracy:.4f} | macro_f1={macro_f1:.4f} | selected_epoch={validation_metrics['epoch']}")
        fold_results.append({
            "test_session": held_out_session,
            "validation_session": fold_validation_session,
            "training_sessions": training_sessions,
            "validation": validation_metrics,
            "training_history": history,
            "test_accuracy": accuracy,
            "test_macro_f1": macro_f1,
        })
        all_true.extend(test_y.tolist())
        all_pred.extend(test_pred.tolist())

    if writer is not None:
        writer.close()
    result = _jsonable({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "frozen_biocodec_rvq_tokens_temporal_channel_transformer",
        "data": {
            "raw_data_path": str(raw_data_path), "cache_dir": str(cache_dir),
            "sample_rate_hz": sample_rate, "window_seconds": window_seconds,
            "csv_denoised": denoise_csv, "csv_ica_artifact_removal": remove_ica_artifacts,
            "sessions": num_sessions, "evaluated_test_sessions": test_sessions,
            "tensorboard_log_dir": str(tensorboard_run_dir) if tensorboard_run_dir else None,
        },
        "model": {
            "codec_frozen": True, "codebook_embeddings_frozen": True,
            "trainable_head": "one temporal Transformer + one channel Transformer + MLP classifier",
            "window_logit_pooling": "mean", "device": device,
        },
        "evaluation": {
            "scheme": "leave-one-session-out; next session is validation; remaining sessions train",
            "safeguards": [
                "All windows from a song are pooled before validation and test metrics.",
                "Test and validation sessions are excluded from classifier fitting and early stopping.",
                "Codec and RVQ codebook embeddings are frozen in every fold.",
            ],
        },
        "overall": {
            "accuracy": float(accuracy_score(all_true, all_pred)),
            "macro_f1": float(f1_score(all_true, all_pred, average="macro", zero_division=0)),
            "classification_report": classification_report(all_true, all_pred, target_names=label_encoder.classes_.tolist(), output_dict=True, zero_division=0),
        },
        "folds": fold_results,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"Saved full JSON report to {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="eeg_optimized")
    parser.add_argument("--raw-data-path", type=Path, default=Path("data/eeg_data/csv"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=Path("data/biocodec_token_loso_results.json"))
    parser.add_argument("--num-sessions", type=int, default=20)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--denoise-csv", action="store_true")
    parser.add_argument("--remove-ica-artifacts", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1997)
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("data/tensorboard"))
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--test-session", type=int, help="Run one development fold with this held-out session instead of all leave-one-session-out folds.")
    parser.add_argument("--validation-session", type=int, help="Session used only for early stopping in --test-session mode; defaults to the next session.")
    parser.add_argument("--sanity-check-session", type=int, help="Deliberately overfit one session to verify the token classifier can memorize it. This is not an evaluation.")
    parser.add_argument("--within-session-cv-session", type=int, help="Run nested song-level CV within one session to test unseen songs under the same recording conditions.")
    args = parser.parse_args()
    if args.remove_ica_artifacts and not args.denoise_csv:
        parser.error("--remove-ica-artifacts requires --denoise-csv")
    special_modes = sum(value is not None for value in (args.sanity_check_session, args.within_session_cv_session))
    if special_modes and (args.test_session is not None or args.validation_session is not None or special_modes > 1):
        parser.error("--sanity-check-session and --within-session-cv-session cannot be combined with each other or session holdout options")
    extractor = BioCodecEEGEmbeddingExtractor(args.checkpoint, args.sample_rate, args.model_name)
    if args.sanity_check_session is not None:
        run_memorization_sanity_check(
            extractor=extractor,
            raw_data_path=args.raw_data_path,
            cache_dir=args.cache_dir,
            output=args.output,
            session=args.sanity_check_session,
            sample_rate=args.sample_rate,
            window_seconds=args.window_seconds,
            denoise_csv=args.denoise_csv,
            remove_ica_artifacts=args.remove_ica_artifacts,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            tensorboard_dir=None if args.disable_tensorboard else args.tensorboard_dir,
        )
        return
    if args.within_session_cv_session is not None:
        evaluate_within_session_cv(
            extractor=extractor,
            raw_data_path=args.raw_data_path,
            cache_dir=args.cache_dir,
            output=args.output,
            session=args.within_session_cv_session,
            sample_rate=args.sample_rate,
            window_seconds=args.window_seconds,
            denoise_csv=args.denoise_csv,
            remove_ica_artifacts=args.remove_ica_artifacts,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            seed=args.seed,
            tensorboard_dir=None if args.disable_tensorboard else args.tensorboard_dir,
        )
        return
    evaluate_loso(
        extractor=extractor,
        raw_data_path=args.raw_data_path,
        cache_dir=args.cache_dir,
        output=args.output,
        num_sessions=args.num_sessions,
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        denoise_csv=args.denoise_csv,
        remove_ica_artifacts=args.remove_ica_artifacts,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        seed=args.seed,
        tensorboard_dir=None if args.disable_tensorboard else args.tensorboard_dir,
        test_session=args.test_session,
        validation_session=args.validation_session,
    )


if __name__ == "__main__":
    main()
