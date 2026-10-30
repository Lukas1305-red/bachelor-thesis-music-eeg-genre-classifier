"""Frozen BioCodec EEG representations with training-only audio alignment.

The deployment model is EEG-only: ``EEG -> frozen BioCodec -> projection ->
genre classifier``.  Paired played audio is used exclusively in training as an
auxiliary contrastive target; it is never constructed during validation/test
and is excluded from the exported checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.signal
import soundfile as sf
from sklearn.preprocessing import LabelEncoder

from code.classification.eeg_audio_latent_alignment import (
    SongExample, _parse_sessions, alignment_loss, eeg_metrics, set_seed,
)
from code.classification.eeg_audio_retrieval import build_manifest
from code.classification.foundation_embedding_svm import (
    BioCodecEEGEmbeddingExtractor, DEFAULT_SAMPLE_RATE, load_headset_csv_session,
)


def select_device(requested: str) -> str:
    """Choose a supported accelerator, preferring CUDA then Apple MPS."""
    import torch

    available_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if available_mps else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if requested == "mps" and not available_mps:
        raise ValueError("MPS was requested but is unavailable. Use --device cpu or install a native Apple-Silicon PyTorch build.")
    return requested


class MERTAudioEmbeddingExtractor:
    """Frozen, time-resolved MERT teacher used solely for training examples."""

    def __init__(self, model_id: str, layer: int, frame_rate: int, device: str) -> None:
        try:
            import torch
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
        except ImportError as exc:
            raise ImportError(
                "MERT alignment requires torch and transformers==4.38. Install the optional MERT dependencies "
                "documented in code/classification/README.md."
            ) from exc
        self.torch, self.layer, self.frame_rate = torch, layer, frame_rate
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.sample_rate = int(self.processor.sampling_rate)

    def encode(self, path: Path, duration_seconds: float) -> np.ndarray:
        waveform, sample_rate = sf.read(path, always_2d=True)
        waveform = waveform.mean(axis=1)
        source_samples = round(sample_rate * duration_seconds)
        waveform = np.pad(waveform[:source_samples], (0, max(0, source_samples - len(waveform))))
        target_samples = round(self.sample_rate * duration_seconds)
        waveform = scipy.signal.resample(waveform, target_samples) if sample_rate != self.sample_rate else waveform
        waveform = waveform.astype(np.float32)
        inputs = self.processor(waveform, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            output = self.model(**inputs, output_hidden_states=True)
        hidden_states = output.hidden_states
        if not -len(hidden_states) <= self.layer < len(hidden_states):
            raise ValueError(f"MERT layer {self.layer} is unavailable; model returned {len(hidden_states)} hidden-state layers.")
        features = hidden_states[self.layer].squeeze(0).float().cpu().numpy()
        frames = round(self.frame_rate * duration_seconds)
        return scipy.signal.resample(features, frames, axis=0).astype(np.float16)


def load_biocodec_examples(
    sessions: Iterable[int], xdf_dir: Path, eeg_csv_dir: Path, gtzan_dir: Path,
    sample_rate: int, duration_seconds: float, denoise_csv: bool, include_audio: bool = True,
    mert: MERTAudioEmbeddingExtractor | None = None, mert_cache_dir: Path | None = None,
) -> list[SongExample]:
    """Load synchronized trials; EEG remains at BioCodec's native sample rate."""
    sessions = list(sessions)
    records = build_manifest(sessions, xdf_dir, eeg_csv_dir, gtzan_dir)
    if include_audio and (mert is None or mert_cache_dir is None):
        raise ValueError("Training examples require a MERT extractor and cache directory.")
    samples = round(sample_rate * duration_seconds)
    examples: list[SongExample] = []
    progress = None
    if include_audio:
        from tqdm.auto import tqdm
        progress = tqdm(total=len(records), desc="MERT teacher features", unit="song", dynamic_ncols=True)
    for session in sessions:
        print(f"[data] Loading EEG session {session:03d} ({'with MERT targets' if include_audio else 'EEG only'})", flush=True)
        trials, labels = load_headset_csv_session(
            session, eeg_csv_dir, sample_rate, minimum_duration_seconds=duration_seconds, denoise=denoise_csv
        )
        session_records = [item for item in records if item.session == session]
        if len(trials) != len(session_records):
            raise ValueError(f"Session {session}: manifest has {len(session_records)} songs but EEG has {len(trials)} trials.")
        for trial, label, record in zip(trials, labels.astype(str), session_records):
            # Per-song normalization avoids sharing EEG statistics across songs/splits.
            trial = trial[:, :samples]
            trial = (trial - trial.mean(axis=1, keepdims=True)) / (trial.std(axis=1, keepdims=True) + 1e-6)
            cache_path = None if mert_cache_dir is None else mert_cache_dir / f"session_{session:03d}" / f"{record.genre}_{record.song_id}.npy"
            if not include_audio:
                audio = None
            elif cache_path is not None and cache_path.exists():
                audio = np.load(cache_path, allow_pickle=False)
                assert progress is not None
                progress.set_postfix_str("cache hit")
            else:
                assert mert is not None and cache_path is not None
                assert progress is not None
                progress.set_postfix_str(f"encoding {record.genre}.{record.song_id}")
                audio = mert.encode(record.audio_path, duration_seconds)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, audio)
            examples.append(SongExample(trial.astype(np.float32), audio, label, session))
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()
    return examples


def make_model(biocodec_encoder: Any, channels: int, mert_dim: int, latent_dim: int, classes: int, dropout: float, device: str) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FrozenBioCodecAudioAlignedClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.biocodec_encoder = biocodec_encoder
            for parameter in self.biocodec_encoder.parameters():
                parameter.requires_grad_(False)
            self.biocodec_encoder.eval()
            # LazyLinear infers BioCodec's latent dimension on the first EEG batch.
            self.eeg_projection = nn.Sequential(nn.LazyLinear(latent_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(latent_dim, latent_dim))
            self.genre_classifier = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, classes))
            # Auxiliary-only module: never part of eeg_only_state_dict().
            self.audio_projection = nn.Sequential(nn.Linear(mert_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            # Keep frozen BioCodec deterministic, including any BatchNorm/Dropout layers.
            self.biocodec_encoder.eval()
            return self

        def eeg_latents(self, eeg: Any) -> Any:
            if eeg.ndim != 3 or eeg.shape[1] != channels:
                raise ValueError(f"Expected EEG shaped (batch, {channels}, time), got {tuple(eeg.shape)}.")
            batch, channel_count, samples = eeg.shape
            with torch.no_grad():
                encoded = self.biocodec_encoder(eeg.reshape(batch * channel_count, 1, samples))
            if encoded.ndim != 3:
                raise ValueError(f"Unexpected BioCodec encoder shape {tuple(encoded.shape)}.")
            # BioCodec is single-channel; preserve its time axis and aggregate electrode embeddings.
            encoded = encoded.reshape(batch, channel_count, encoded.shape[1], encoded.shape[2]).mean(dim=1).transpose(1, 2)
            return self.eeg_projection(encoded)

        def predict_eeg_only(self, eeg: Any) -> Any:
            """The only inference API; it accepts no audio argument."""
            return self.genre_classifier(self.eeg_latents(eeg).mean(dim=1))

        def forward(self, eeg: Any, audio: Any | None = None) -> tuple[Any, Any, Any | None]:
            eeg_latent = self.eeg_latents(eeg)
            logits = self.genre_classifier(eeg_latent.mean(dim=1))
            if audio is None:
                return logits, eeg_latent, None
            audio_latent = self.audio_projection(audio.float())
            # Audio frames are resampled only inside the training-only auxiliary branch.
            audio_latent = F.interpolate(audio_latent.transpose(1, 2), size=eeg_latent.shape[1], mode="linear", align_corners=False).transpose(1, 2)
            return logits, eeg_latent, audio_latent

        def eeg_only_state_dict(self) -> dict[str, Any]:
            return {key: value for key, value in self.state_dict().items() if not key.startswith("audio_projection.")}

    return FrozenBioCodecAudioAlignedClassifier().to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pretrained BioCodec checkpoint.")
    parser.add_argument("--model-name", required=True, help="BioCodec EEG architecture name, normally eeg_optimized.")
    parser.add_argument("--xdf-dir", type=Path, default=Path("data/eeg-data"))
    parser.add_argument("--eeg-csv-dir", type=Path, default=Path("data/eeg_data/csv"))
    parser.add_argument("--gtzan-dir", type=Path, default=Path("data/genres_original"))
    parser.add_argument("--train-sessions", default="0-15")
    parser.add_argument("--validation-session", type=int, default=16)
    parser.add_argument("--test-session", type=int, default=17)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--mert-model", default="m-a-p/MERT-v1-95M", help="Frozen Hugging Face MERT teacher checkpoint.")
    parser.add_argument("--mert-layer", type=int, default=8, help="MERT hidden-state layer used as the audio teacher.")
    parser.add_argument("--mert-frame-rate", type=int, default=10, help="Temporal rate after downsampling MERT features.")
    parser.add_argument("--mert-cache-dir", type=Path, default=Path("data/mert_features"))
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--lag-ms", type=int, default=100)
    parser.add_argument("--alignment-weight", type=float, default=0.02, help="Weight of the MERT alignment loss; use 0 for the EEG-only BioCodec baseline.")
    parser.add_argument("--alignment-frames-per-song", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15, help="Stop after this many epochs without validation macro-F1 improvement; 0 disables early stopping.")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum validation macro-F1 improvement required to reset patience.")
    parser.add_argument("--log-every-epochs", type=int, default=1, help="Print training and EEG-only validation metrics every N epochs.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--denoise-csv", action="store_true")
    parser.add_argument("--seed", type=int, default=1997)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Training device; auto prefers CUDA, then Apple MPS, then CPU.")
    parser.add_argument("--output", type=Path, default=Path("data/biocodec_audio_alignment_results.json"))
    parser.add_argument("--output-checkpoint", type=Path, default=Path("data/biocodec_eeg_only_audio_aligned_model.pt"))
    args = parser.parse_args()
    train_sessions = _parse_sessions(args.train_sessions)
    if len(set(train_sessions + [args.validation_session, args.test_session])) != len(train_sessions) + 2:
        parser.error("Training, validation, and test sessions must be distinct.")
    if args.alignment_weight < 0 or args.mert_frame_rate <= 0:
        parser.error("alignment-weight must be non-negative and mert-frame-rate must be positive.")
    if args.log_every_epochs <= 0 or args.patience < 0 or args.min_delta < 0:
        parser.error("log-every-epochs must be positive; patience and min-delta must be non-negative.")

    import torch
    import torch.nn.functional as F

    set_seed(args.seed)
    try:
        device = select_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    use_alignment = args.alignment_weight > 0
    print(f"[setup] Device: {device}; alignment={'MERT-guided' if use_alignment else 'disabled (EEG-only baseline)'}", flush=True)
    if use_alignment:
        print(f"[setup] Preparing MERT features: {args.mert_model}, layer {args.mert_layer}, {args.mert_frame_rate} Hz...", flush=True)
        feature_start = time.monotonic()
        mert = MERTAudioEmbeddingExtractor(args.mert_model, args.mert_layer, args.mert_frame_rate, device)
        cache_root = args.mert_cache_dir / args.mert_model.replace("/", "--") / f"layer_{args.mert_layer}_rate_{args.mert_frame_rate}"
        train = load_biocodec_examples(train_sessions, args.xdf_dir, args.eeg_csv_dir, args.gtzan_dir, args.sample_rate, args.duration_seconds, args.denoise_csv, include_audio=True, mert=mert, mert_cache_dir=cache_root)
        print(f"[setup] Training MERT features ready in {time.monotonic() - feature_start:.1f}s ({len(train)} songs).", flush=True)
    else:
        print("[setup] Loading training EEG only; MERT is not loaded or cached.", flush=True)
        train = load_biocodec_examples(train_sessions, args.xdf_dir, args.eeg_csv_dir, args.gtzan_dir, args.sample_rate, args.duration_seconds, args.denoise_csv, include_audio=False)
    print("[setup] Loading validation and test EEG only (no audio/MERT features)...", flush=True)
    validation = load_biocodec_examples([args.validation_session], args.xdf_dir, args.eeg_csv_dir, args.gtzan_dir, args.sample_rate, args.duration_seconds, args.denoise_csv, include_audio=False)
    test = load_biocodec_examples([args.test_session], args.xdf_dir, args.eeg_csv_dir, args.gtzan_dir, args.sample_rate, args.duration_seconds, args.denoise_csv, include_audio=False)
    if any(item.audio is not None for item in validation + test):
        raise RuntimeError("Safety check failed: validation/test examples must not contain audio features.")
    # MERT has finished producing frozen teacher targets and is not retained by
    # the trainable/deployed model.
    if use_alignment:
        del mert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    label_encoder = LabelEncoder().fit([str(item.label) for item in train])
    class_names = label_encoder.classes_.tolist()
    def encode(items: list[SongExample]) -> list[SongExample]:
        return [SongExample(item.eeg, item.audio, int(label_encoder.transform([str(item.label)])[0]), item.session) for item in items]
    train, validation, test = encode(train), encode(validation), encode(test)
    extractor = BioCodecEEGEmbeddingExtractor(args.checkpoint, args.sample_rate, args.model_name, device=device)
    mert_dim = int(train[0].audio.shape[-1]) if use_alignment and train and train[0].audio is not None else 1
    if use_alignment and mert_dim <= 0:
        raise RuntimeError("No MERT training features were loaded.")
    model = make_model(extractor.model.encoder, 7, mert_dim, args.latent_dim, len(class_names), args.dropout, device)
    if any(parameter.requires_grad for parameter in model.biocodec_encoder.parameters()):
        raise RuntimeError("Safety check failed: BioCodec encoder must remain frozen.")
    # Only the projection, classifier, and training-only audio branch are optimized.
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=1e-4)
    best_state, best_validation, history, epochs_without_improvement = None, -np.inf, [], 0
    rng = np.random.default_rng(args.seed)
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"[train] Starting {args.epochs} epochs on {len(train)} training songs; validation has {len(validation)} EEG-only songs.", flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        order, losses, genre_losses, alignment_losses, correct, count = rng.permutation(len(train)), [], [], [], 0, 0
        for start in range(0, len(order), args.batch_size):
            batch = [train[index] for index in order[start : start + args.batch_size]]
            eeg = torch.as_tensor(np.stack([item.eeg for item in batch]), device=device)
            labels = torch.as_tensor([item.label for item in batch], device=device)
            if use_alignment:
                if any(item.audio is None for item in batch):
                    raise RuntimeError("MERT-guided training examples must contain paired MERT features.")
                audio = torch.as_tensor(np.stack([item.audio for item in batch]), device=device)
                logits, eeg_latent, audio_latent = model(eeg, audio)
                assert audio_latent is not None
                lag_frames = round(args.lag_ms / 1000 * eeg_latent.shape[1] / args.duration_seconds)
                auxiliary_loss = alignment_loss(eeg_latent, audio_latent, lag_frames, args.alignment_frames_per_song, args.temperature)
            else:
                logits, _, _ = model(eeg)
                auxiliary_loss = torch.zeros((), device=device)
            genre_loss = F.cross_entropy(logits, labels)
            loss = genre_loss + args.alignment_weight * auxiliary_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
            genre_losses.append(float(genre_loss.detach().cpu()))
            alignment_losses.append(float(auxiliary_loss.detach().cpu()))
            correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
            count += len(labels)
        validation_metrics = eeg_metrics(model, validation, args.batch_size, device)
        train_metrics = {"train_loss": float(np.mean(losses)), "train_genre_loss": float(np.mean(genre_losses)), "train_alignment_loss": float(np.mean(alignment_losses)), "train_accuracy": correct / count}
        history.append({"epoch": epoch, **train_metrics, **validation_metrics})
        if validation_metrics["macro_f1"] > best_validation + args.min_delta:
            best_validation = validation_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.eeg_only_state_dict().items()}
            epochs_without_improvement = 0
            # Preserve the best EEG-only model even if a later run is interrupted.
            torch.save({"state_dict": best_state, "class_names": class_names, "inference": "EEG only: frozen BioCodec encoder + EEG projection + genre classifier", "mert_included": False, "audio_projection_included": False}, args.output_checkpoint)
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % args.log_every_epochs == 0 or epoch == args.epochs:
            print(
                f"[train] epoch {epoch:03d}/{args.epochs}: total={np.mean(losses):.4f}, genre={np.mean(genre_losses):.4f}, "
                f"alignment={np.mean(alignment_losses):.4f}, train_accuracy={correct / count:.4f}, "
                f"val_accuracy={validation_metrics['accuracy']:.4f}, val_macro_f1={validation_metrics['macro_f1']:.4f}, "
                f"best_f1={best_validation:.4f}, {time.monotonic() - epoch_start:.1f}s",
                flush=True,
            )
        if args.patience and epochs_without_improvement >= args.patience:
            print(f"[train] Early stopping after {args.patience} epochs without validation macro-F1 improvement.", flush=True)
            break
    assert best_state is not None
    model.load_state_dict(best_state, strict=False)
    test_metrics = eeg_metrics(model, test, args.batch_size, device)
    print(f"[test] EEG-only test: accuracy={test_metrics['accuracy']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}", flush=True)
    if any(key.startswith("audio_projection.") for key in best_state):
        raise RuntimeError("Safety check failed: an audio weight entered the EEG-only checkpoint.")
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "class_names": class_names, "inference": "EEG only: frozen BioCodec encoder + EEG projection + genre classifier", "mert_included": False, "audio_projection_included": False}, args.output_checkpoint)
    result = {"mode": "frozen_biocodec_eeg_only_genre_classifier_mert_guided_training", "created_at_utc": datetime.now(timezone.utc).isoformat(), "validation_best_macro_f1": float(best_validation), "test_eeg_only": test_metrics, "history": history, "class_names": class_names, "checkpoint": str(args.output_checkpoint), "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "leakage_safeguards": ["BioCodec encoder parameters are frozen and its encoder remains in eval mode.", "MERT is frozen and is used only to precompute training-session targets.", "Whole recording sessions are mutually exclusive across train, validation, and test.", "Validation and test use predict_eeg_only; their examples contain no MERT/audio features.", "Export verification rejects every audio_projection parameter; MERT is not saved."], "inference_contract": "EEG only: (batch, 7, samples)"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Best validation macro-F1={best_validation:.4f}; EEG-only test macro-F1={test_metrics['macro_f1']:.4f}")
    print(f"Verified EEG-only export: {args.output_checkpoint}")


if __name__ == "__main__":
    main()
