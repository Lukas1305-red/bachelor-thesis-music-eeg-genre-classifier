# Classification

This subsystem holds all logic and functions that are important for performing classification on the MuSiCGen dataset and the DEAP dataset. The different steps of this classification task are split between different files.

## Functionality

### Classification Pipeline
![](../../thesis/figures/chapter05/classification_pipeline_dark.png#gh-dark-mode-only)
![](../../thesis/figures/chapter05/classification_pipeline.png#gh-light-mode-only)

### Data loading
[data.py](data.py) and [deap_data.py](deap_data.py) are responsible for loading the MuSiCGen dataset and DEAP dataset respectively. Beforehand, all `inf` or `nan` values are filtered out.

### Feature selector
To choose the optimal subset of features, feature selectors are also implemented and can be found in [feature_selector.py](feature_selector.py). In this file you find a `FeatureSelectorFromModel`, which trains a model and uses the ranking of feature importances as a selection criteria and an `MRMRSelector`, choosing the features that share maximum relevancy and minimal redundancy. Mainly, the `MRMRSelector` is used in the case of this thesis.

### Learning environment
In [learning.py](learning.py), a `GridSearch` or `RandomizedSearch` is setup to find the optimal parameters. Furthermore, to gain more insights about the model training, [MLflow](https://mlflow.org/) is setup to log important training metrics like confusion matrices or training accuracy.

### Classification
[classification.py](classification.py) accomodates all functions concerned with performing the Hyper Parameter Tuning by combining all previously mentioned components together. By either loading the MuSiCGen dataset or the DEAP dataset, the data is split with a `StratifiedKFold` into five parts. The best estimator's accuracy and paramaters are then logged and printed to the command line.

Special cases are [deap_classification.py](deap_classification.py) and [headset_classification.py](headset_classification.py), where the hyperparameters are tuned per subject/session and the models are chosen directly including the paramater grid.

### Frozen foundation-embedding SVM mode

[foundation_embedding_svm.py](foundation_embedding_svm.py) is the headset mode for a pretrained BioCodec EEG encoder followed by an SVM. It reads raw session tables, filters CSV sessions when requested, applies per-song/per-channel z-scoring, divides each song into complete five-second windows, and mean-pools their frozen BioCodec embeddings into one fixed-length song embedding. It does not fine-tune the foundation model.

The resulting JSON contains outer-fold accuracy, macro-F1, selected SVM parameters, per-session and global classification reports, and split details. Hyperparameters are selected inside each outer training partition, and `StandardScaler` is fitted on training folds only.

Clone the official [BioCodec release](https://github.com/klean2050/BioCodec-release), set `BIOCODEC_PATH` to that checkout, and run it with its included `biocodec_ckpt.pt` checkpoint. The matching EEG architecture name is `eeg_optimized`:

```bash
python -m code.classification.foundation_embedding_svm \
  --checkpoint "$BIOCODEC_PATH/biocodec_ckpt.pt" \
  --model-name eeg_optimized \
  --output foundation_embedding_svm_results.json
```

It is also available from `headset_classification.py` as `--mode foundation_embeddings` with the same `--checkpoint`, `--model-name`, and `--output` arguments.

### Using the exported raw CSV files

`data/eeg_data/csv` contains one row per song, with each `EEG_Ch*` cell holding
the complete serialized waveform.  It is raw data, so use the CSV loader with
the filter step enabled.  This matches the minimal preprocessing used for
BioCodec EEG pre-training (0.5--100 Hz band-pass, 50 Hz notch, then the
pipeline's per-song/channel z-score):

```bash
python -m code.classification.foundation_embedding_svm \
  --checkpoint "$BIOCODEC_PATH/biocodec_ckpt.pt" \
  --model-name eeg_optimized \
  --input-format csv \
  --raw-data-path data/eeg_data/csv \
  --denoise-csv \
  --window-seconds 5 \
  --output foundation_embedding_svm_csv_results.json
```

Add `--remove-ica-artifacts` only after comparing the filtered-only and ICA
variants under the same nested-CV splits. The CSVs have already lost the
between-song gaps, so ICA is applied to the concatenated session. Each song is
normalized before it is split into five-second windows, and its window
embeddings are pooled before the SVM split; windows are never independent CV
examples. Session 10 has one fewer jazz song, so the evaluator automatically
uses valid 4-outer/3-inner folds for that session and records this in the JSON.

### BioCodec RVQ-token Transformer experiment

[biocodec_token_loso.py](biocodec_token_loso.py) retains BioCodec's discrete
RVQ code sequences rather than reducing continuous encoder activations to
summary statistics. The codec and its codebook embeddings stay frozen; a small
temporal Transformer, channel Transformer, and MLP head are trained with
leave-one-session-out evaluation. One session is held out for testing, the next
for early stopping, and the other 18 sessions supply training songs. Window
logits are mean-pooled before song-level validation and test metrics.

It writes train/validation loss and validation metrics for every epoch, plus
held-out test metrics, to a timestamped TensorBoard run under
`data/tensorboard` by default. Start TensorBoard with
`./.venv/bin/tensorboard --logdir data/tensorboard` and open
`http://localhost:6006`. The JSON output also stores each fold's full epoch
history for retrospective plotting.

For faster development, add `--test-session 17 --validation-session 16` to run
one fold only. The other 18 sessions remain training data. Treat this as a
development result, not a final test result.

Before trusting a poor validation result, run the memorization diagnostic with
`--sanity-check-session 0`. It deliberately removes regularization and trains
and scores on the same session. A high same-session score confirms that the
token extraction and training path can learn; it must never be reported as a
held-out result.

To test unseen songs recorded under the same conditions, use
`--within-session-cv-session 0`. It performs nested song-level CV: every outer
test fold holds out 20% of songs, another distinct song split is reserved for
early stopping, and all windows from a song stay in one partition.

### EEG-to-audio envelope retrieval

[eeg_audio_retrieval.py](eeg_audio_retrieval.py) tests auditory tracking
without training on genre labels. It extracts the exact song ID from every XDF
`Song` stream, builds a validated EEG-row-to-GTZAN-file manifest, fits a Ridge
decoder from EEG to the audio RMS envelope using training sessions only, selects
the neural delay and regularization on one validation session, and retrieves the
matching audio among all songs in a held-out test session.

After development selection, use its locked final-evaluation mode rather than
selecting parameters again. For the current experiment, delay 50 ms and ridge
alpha 0.01 were selected on session 16; session 17 was used for development
evaluation, so sessions 18--19 remain final test data:

```bash
./.venv/bin/python -m code.classification.eeg_audio_retrieval \
  --train-sessions 0-16 --test-sessions 18-19 \
  --fixed-delay-ms 50 --fixed-alpha 0.01 --denoise-csv \
  --permutations 10000 \
  --output data/eeg_audio_retrieval_final_results.json
```

This refits only on sessions 0--16. For each final session, the permutation
test retains its fitted predictions and retrieval score matrix but randomly
reassigns the matched audio identities, yielding a one-sided empirical p-value
for top-1 accuracy and mean reciprocal rank.

### Audio-guided, EEG-only latent classifier

[eeg_audio_latent_alignment.py](eeg_audio_latent_alignment.py) implements a
training-only audio auxiliary task inspired by cross-modal contrastive
alignment.  It learns time-resolved EEG and audio latents for each known
EEG--song pairing, with an InfoNCE alignment loss after a configurable neural
delay.  The classifier is **not bimodal**: audio is used only in training to
shape the EEG representation.  Validation and test call an EEG-only inference
method, and exported checkpoints exclude the audio encoder entirely.

The initial audio representation is log-mel audio features encoded by a small
auxiliary encoder.  This avoids downloading a foundation model while providing
a reproducible baseline.  A later experiment can replace that branch with a
frozen music foundation embedding without changing the EEG-only inference
contract.

```bash
./.venv/bin/python -m code.classification.eeg_audio_latent_alignment \
  --train-sessions 0-15 --validation-session 16 --test-session 17 \
  --lag-ms 100 --alignment-weight 0.2 --denoise-csv \
  --output data/eeg_audio_alignment_results.json \
  --checkpoint data/eeg_only_audio_aligned_model.pt
```

Select the lag, alignment weight, and architecture using validation sessions,
then reserve unseen sessions for the final single evaluation.  Do not supply
audio at deployment: the saved model accepts EEG shaped `(batch, 7, time)`
only.

### Frozen BioCodec with MERT-guided alignment

[biocodec_audio_latent_alignment.py](biocodec_audio_latent_alignment.py) tests
training-only MERT guidance. BioCodec's pretrained EEG encoder is frozen
and remains in evaluation mode. A frozen MERT-v1-95M music encoder provides
time-resolved teacher embeddings only for training sessions; MERT features are
cached locally to avoid repeated inference. A trainable EEG projection head
aligns to them, and the genre head uses EEG embeddings. The exported model
contains BioCodec, the EEG projection, and the genre classifier—but neither
MERT nor an audio branch—and validation/test use EEG only.

Setup from the repository root:

```bash
# BioCodec is needed locally by the existing EEG embedding code.
mkdir -p external
git clone https://github.com/klean2050/BioCodec-release.git external/BioCodec-release
export BIOCODEC_PATH="$PWD/external/BioCodec-release"

# MERT inference uses the published Hugging Face checkpoint; no MERT clone is needed.
./.venv/bin/python -m pip install "transformers==4.38.2" huggingface-hub safetensors soundfile
export HF_HOME="$PWD/.cache/huggingface"  # optional: keep downloaded MERT weights in the project cache
```

The first run downloads `m-a-p/MERT-v1-95M` from Hugging Face. MERT is loaded
with `trust_remote_code=True`, as required by its official model card, so pin
the dependency version above and inspect the model source before use. The
official MERT repository confirms that Hugging Face checkpoints are the
intended inference interface; its repository need only be cloned to train or
modify MERT itself.

```bash
./.venv/bin/python -m code.classification.biocodec_audio_latent_alignment \
  --checkpoint external/BioCodec-release/biocodec_ckpt.pt --model-name eeg_optimized \
  --train-sessions 0-15 --validation-session 16 --test-session 17 \
  --mert-model m-a-p/MERT-v1-95M --mert-layer 8 --mert-frame-rate 10 \
  --lag-ms 100 --alignment-weight 0.2 --denoise-csv
```

On an Apple-Silicon Mac with a native PyTorch MPS build, add `--device mps`.
The default `--device auto` selects CUDA first, then MPS, then CPU. Confirm
availability with `./.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"`.

Run the diagnostic EEG-only baseline before interpreting MERT alignment. It
does not load MERT or audio files, reports genre and alignment losses
separately, saves a best EEG-only checkpoint immediately on improvement, and
stops after 15 non-improving epochs by default:

```bash
./.venv/bin/python -m code.classification.biocodec_audio_latent_alignment \
  --checkpoint external/BioCodec-release/biocodec_ckpt.pt --model-name eeg_optimized \
  --train-sessions 0-15 --validation-session 16 --test-session 17 \
  --alignment-weight 0 --denoise-csv --device auto \
  --output data/biocodec_eeg_only_baseline_results.json \
  --output-checkpoint data/biocodec_eeg_only_baseline.pt
```

Only if this baseline learns above chance on validation should MERT alignment
be compared, initially with a small weight such as `--alignment-weight 0.02`.
If the baseline itself fails, inspect BioCodec temporal pooling/projection
before tuning MERT.

Furthermore, [svm.py](svm.py), [ranndom_forest.py](random_forest.py) and [mlp.py](mlp.py) realize the classification by initializing their respective model beforehand and perform the hyperparameter tuning.

### Evaluation
Firstly, there is the [deap_playground.ipynb](deap_playground.ipynb), in which the main diagrams are created including accuracy distribution, F1-score distribution and hyperparameter distribution. In [headset_confusion_matrix.py](headset_confusion_matrix.py) and [deap_confusion_matrix.py](deap_confusion_matrix.py), confusion matrices from the best perforiming models are generated and saved in the data directory.

## Key Results

### DEAP
![](../../thesis/figures/chapter06/deap/deap_accuracies.png)
![](../../thesis/figures/chapter06/deap/deap_f1score.png)
![](../../thesis/figures/chapter06/deap/deap_global_confusion.png)

### MuSiCGen
![](../../thesis/figures/chapter06/headset/headset_accuracies.png)
![](../../thesis/figures/chapter06/headset/headset_f1score.png)
![](../../thesis/figures/chapter06/headset/headset_global_confusion.png)
