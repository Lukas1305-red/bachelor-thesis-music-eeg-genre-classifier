"""Train an EEG-only genre model with audio-guided latent alignment.

Audio is available *only* while fitting the model.  A trainable audio encoder
maps the concurrently played waveform to a latent sequence and an InfoNCE loss
aligns it with a delayed latent sequence from the EEG encoder.  The exported
checkpoint deliberately contains only the EEG encoder and genre classifier:
audio is neither required nor accepted by ``predict_eeg_only``.

Splits are whole recording sessions.  The XDF ``Song`` stream is used to prove
the row-to-audio mapping, and no validation/test audio enters model selection
or prediction.  This is an auxiliary/self-supervision experiment, not a
bimodal classifier.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.signal
import soundfile as sf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from code.classification.eeg_audio_retrieval import _parse_sessions, build_manifest
from code.classification.foundation_embedding_svm import DEFAULT_SAMPLE_RATE, load_headset_csv_session


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Small dependency-free mel filterbank; audio features are training-only."""
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(sample_rate / 2), n_mels + 2)) / sample_rate).astype(int)
    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, centre, right = bins[index : index + 3]
        centre = max(centre, left + 1)
        right = max(right, centre + 1)
        bank[index, left:centre] = np.linspace(0, 1, centre - left, endpoint=False)
        bank[index, centre:right] = np.linspace(1, 0, right - centre, endpoint=False)
    return bank


def audio_log_mel(path: Path, frames: int, n_mels: int) -> np.ndarray:
    """Return fixed-rate log-mel features as ``(frames, mel)``."""
    waveform, sample_rate = sf.read(path, always_2d=True)
    waveform = waveform.mean(axis=1)
    n_fft = 1024
    _, _, spectrum = scipy.signal.stft(waveform, fs=sample_rate, nperseg=n_fft, noverlap=768, boundary=None)
    mel = _mel_filterbank(sample_rate, n_fft, n_mels) @ (np.abs(spectrum) ** 2)
    features = np.log(mel + 1e-8).T
    features = scipy.signal.resample(features, frames, axis=0).astype(np.float32)
    return (features - features.mean(axis=0, keepdims=True)) / (features.std(axis=0, keepdims=True) + 1e-6)


def eeg_frames(trial: np.ndarray, frames: int) -> np.ndarray:
    """Per-song normalized EEG, resampled to the common latent frame rate."""
    normalized = (trial - trial.mean(axis=1, keepdims=True)) / (trial.std(axis=1, keepdims=True) + 1e-6)
    return scipy.signal.resample(normalized, frames, axis=1).astype(np.float32)


@dataclass(frozen=True)
class SongExample:
    eeg: np.ndarray  # (channels, time)
    audio: np.ndarray | None  # (time, mel); absent outside training
    label: str | int
    session: int


def load_examples(
    sessions: Iterable[int], xdf_dir: Path, eeg_csv_dir: Path, gtzan_dir: Path,
    sample_rate: int, frame_rate: int, duration_seconds: float, n_mels: int, denoise_csv: bool, include_audio: bool = True,
) -> list[SongExample]:
    sessions = list(sessions)
    manifest = build_manifest(sessions, xdf_dir, eeg_csv_dir, gtzan_dir)
    frames = round(frame_rate * duration_seconds)
    examples: list[SongExample] = []
    unencoded: list[tuple[np.ndarray, np.ndarray | None, int, str]] = []
    for session in sessions:
        trials, session_labels = load_headset_csv_session(
            session, eeg_csv_dir, sample_rate, minimum_duration_seconds=duration_seconds, denoise=denoise_csv
        )
        records = [item for item in manifest if item.session == session]
        if len(trials) != len(records):
            raise ValueError(f"Session {session}: manifest has {len(records)} songs but EEG has {len(trials)} trials.")
        for trial, label, record in zip(trials, session_labels.astype(str), records):
            audio = audio_log_mel(record.audio_path, frames, n_mels) if include_audio else None
            unencoded.append((eeg_frames(trial, frames), audio, session, label))
    for eeg, audio, session, label in unencoded:
        examples.append(SongExample(eeg, audio, label, session))
    return examples


def make_model(channels: int, n_mels: int, latent_dim: int, classes: int, dropout: float, device: str) -> Any:
    import torch.nn as nn

    class EEGOnlyLatentClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.eeg_encoder = nn.Sequential(
                nn.Conv1d(channels, latent_dim, kernel_size=7, padding=3), nn.GELU(),
                nn.Conv1d(latent_dim, latent_dim, kernel_size=5, padding=2), nn.GELU(), nn.Dropout(dropout),
            )
            self.genre_classifier = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, classes))
            # This module is intentionally auxiliary.  It is excluded from exported checkpoints.
            self.audio_encoder = nn.Sequential(nn.Linear(n_mels, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))

        def eeg_latents(self, eeg: Any) -> Any:
            return self.eeg_encoder(eeg).transpose(1, 2)

        def predict_eeg_only(self, eeg: Any) -> Any:
            """The sole inference API: EEG ``(batch, channels, time)`` -> genre logits."""
            return self.genre_classifier(self.eeg_latents(eeg).mean(dim=1))

        def forward(self, eeg: Any, audio: Any) -> tuple[Any, Any, Any]:
            eeg_latent = self.eeg_latents(eeg)
            audio_latent = self.audio_encoder(audio)
            return self.genre_classifier(eeg_latent.mean(dim=1)), eeg_latent, audio_latent

        def eeg_only_state_dict(self) -> dict[str, Any]:
            return {key: value for key, value in self.state_dict().items() if not key.startswith("audio_encoder.")}

    return EEGOnlyLatentClassifier().to(device)


def alignment_loss(eeg_latent: Any, audio_latent: Any, lag_frames: int, samples_per_song: int, temperature: float) -> Any:
    """Cross-modal InfoNCE using only paired, delayed training frames."""
    import torch
    import torch.nn.functional as F

    if not 0 <= lag_frames < eeg_latent.shape[1]:
        raise ValueError("lag_frames must be non-negative and shorter than the sequence.")
    eeg_latent, audio_latent = eeg_latent[:, lag_frames:], audio_latent[:, : eeg_latent.shape[1] - lag_frames]
    count = min(samples_per_song, eeg_latent.shape[1])
    indices = torch.randperm(eeg_latent.shape[1], device=eeg_latent.device)[:count]
    eeg_selected = F.normalize(eeg_latent[:, indices].reshape(-1, eeg_latent.shape[-1]), dim=1)
    audio_selected = F.normalize(audio_latent[:, indices].reshape(-1, audio_latent.shape[-1]), dim=1)
    logits = eeg_selected @ audio_selected.T / temperature
    targets = torch.arange(len(eeg_selected), device=eeg_latent.device)
    return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


def eeg_metrics(model: Any, examples: list[SongExample], batch_size: int, device: str) -> dict[str, float]:
    """Evaluate strictly with EEG; audio tensors are never constructed here."""
    import torch

    predictions, targets = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            eeg = torch.as_tensor(np.stack([item.eeg for item in batch]), device=device)
            predictions.extend(model.predict_eeg_only(eeg).argmax(dim=1).cpu().tolist())
            targets.extend(item.label for item in batch)
    return {"accuracy": float(accuracy_score(targets, predictions)), "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xdf-dir", type=Path, default=Path("data/eeg-data"))
    parser.add_argument("--eeg-csv-dir", type=Path, default=Path("data/eeg_data/csv"))
    parser.add_argument("--gtzan-dir", type=Path, default=Path("data/genres_original"))
    parser.add_argument("--train-sessions", default="0-15")
    parser.add_argument("--validation-session", type=int, default=16)
    parser.add_argument("--test-session", type=int, default=17)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--frame-rate", type=int, default=10)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--n-mels", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--lag-ms", type=int, default=100)
    parser.add_argument("--alignment-weight", type=float, default=0.2)
    parser.add_argument("--alignment-frames-per-song", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--denoise-csv", action="store_true")
    parser.add_argument("--seed", type=int, default=1997)
    parser.add_argument("--output", type=Path, default=Path("data/eeg_audio_alignment_results.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/eeg_only_audio_aligned_model.pt"))
    args = parser.parse_args()
    train_sessions = _parse_sessions(args.train_sessions)
    if len(set(train_sessions + [args.validation_session, args.test_session])) != len(train_sessions) + 2:
        parser.error("Training, validation, and test sessions must be distinct.")
    if args.alignment_weight < 0 or args.frame_rate <= 0:
        parser.error("alignment-weight must be non-negative and frame-rate must be positive.")

    import torch
    import torch.nn.functional as F

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = load_examples(train_sessions, args.xdf_dir, args.eeg_csv_dir, args.gtzan_dir, args.sample_rate, args.frame_rate, args.duration_seconds, args.n_mels, args.denoise_csv, include_audio=True)
    validation = load_examples([args.validation_session], args.xdf_dir, args.eeg_csv_dir, args.gtzan_dir, args.sample_rate, args.frame_rate, args.duration_seconds, args.n_mels, args.denoise_csv, include_audio=False)
    test = load_examples([args.test_session], args.xdf_dir, args.eeg_csv_dir, args.gtzan_dir, args.sample_rate, args.frame_rate, args.duration_seconds, args.n_mels, args.denoise_csv, include_audio=False)
    if any(item.audio is not None for item in validation + test):
        raise RuntimeError("Safety check failed: validation/test examples must not contain audio features.")
    # Fit label coding on training sessions only.  This also rejects a final
    # session containing a genre unseen during model fitting.
    label_encoder = LabelEncoder().fit([str(item.label) for item in train])
    class_names = label_encoder.classes_.tolist()
    def encode(items: list[SongExample]) -> list[SongExample]:
        return [SongExample(item.eeg, item.audio, int(label_encoder.transform([str(item.label)])[0]), item.session) for item in items]
    train, validation, test = encode(train), encode(validation), encode(test)
    model = make_model(7, args.n_mels, args.latent_dim, len(class_names), args.dropout, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    lag_frames = round(args.lag_ms / 1000 * args.frame_rate)
    best_state, best_validation, history = None, -np.inf, []
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = [train[index] for index in order[start : start + args.batch_size]]
            eeg = torch.as_tensor(np.stack([item.eeg for item in batch]), device=device)
            if any(item.audio is None for item in batch):
                raise RuntimeError("Training examples must contain paired audio features.")
            audio = torch.as_tensor(np.stack([item.audio for item in batch]), device=device)
            labels = torch.as_tensor([item.label for item in batch], device=device)
            logits, eeg_latent, audio_latent = model(eeg, audio)
            genre_loss = F.cross_entropy(logits, labels)
            auxiliary_loss = alignment_loss(eeg_latent, audio_latent, lag_frames, args.alignment_frames_per_song, args.temperature)
            loss = genre_loss + args.alignment_weight * auxiliary_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_metrics = eeg_metrics(model, validation, args.batch_size, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **validation_metrics})
        if validation_metrics["macro_f1"] > best_validation:
            best_validation = validation_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.eeg_only_state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state, strict=False)
    test_metrics = eeg_metrics(model, test, args.batch_size, device)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "class_names": class_names, "frame_rate": args.frame_rate, "latent_dim": args.latent_dim, "inference": "EEG only; no audio encoder or audio input"}, args.checkpoint)
    result = {"mode": "eeg_only_genre_classifier_audio_guided_training", "created_at_utc": datetime.now(timezone.utc).isoformat(), "device": device, "validation_best_macro_f1": float(best_validation), "test_eeg_only": test_metrics, "history": history, "class_names": class_names, "checkpoint": str(args.checkpoint), "configuration": vars(args) | {"xdf_dir": str(args.xdf_dir), "eeg_csv_dir": str(args.eeg_csv_dir), "gtzan_dir": str(args.gtzan_dir)}, "leakage_safeguards": ["Whole sessions are mutually exclusive across train, validation, and test.", "XDF Song events validate every EEG/audio pairing.", "Audio features are loaded only for training sessions and the auxiliary alignment loss.", "Validation and test call predict_eeg_only and their examples contain no audio features.", "The exported state_dict explicitly excludes audio_encoder parameters."], "inference_contract": "EEG tensor (batch, 7, time) only"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"Best validation macro-F1={best_validation:.4f}; EEG-only test macro-F1={test_metrics['macro_f1']:.4f}")
    print(f"Saved EEG-only checkpoint to {args.checkpoint} and report to {args.output}")


if __name__ == "__main__":
    main()
