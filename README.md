# Modeling Genuine Audio-Visual Correspondence for Temporal Forgery Localization and Generalizable Deepfake Detection (AVcorr)

> *Modeling Genuine Audio-Visual Correspondence for Temporal Forgery Localization and Generalizable Deepfake Detection* — **AVcorr**

**AVcorr** is a multimodal deepfake detection system that jointly analyses audio and video streams using learned temporal attention and frequency-aware feature selection. It is designed for the LAV-DF benchmark and is extensible to other audio-visual forgery datasets.

---

## Core Contributions

| Component | What It Does |
|-----------|-------------|
| **GFC** (Global Frequency Consistency) | Extracts 8 FFT annular band energies per frame to detect high-frequency GAN synthesis artifacts that spatial CNNs miss |
| **TDFS** (Temporal Differentiable Feature Selector) | Learns a per-timestep soft feature mask end-to-end via a BiGRU gating network; selects discriminative features at each frame independently |
| **Cross-Modal Attention** | Bidirectional audio↔video multi-head attention that forces the model to measure audio-visual synchrony as a forgery signal |
| **A/V Agreement Loss** | Supervises cross-modal contexts: real streams must align (high cosine sim), fake streams must diverge (low cosine sim) |
| **TFLH** (Temporal Frame-Level Head) | Per-frame forgery localisation head trained with `fake_periods` annotations for interpretability and auxiliary supervision |

---

## Project Structure

```
AVcorr/
├── configs/                    # YAML experiment configs (v5.5 stable, v6 experimental)
├── datasets/                   # Metadata loaders, sequence builders, annotation loader
├── features/                   # MFCC, GFC, ResNet CNN, fusion, caching
├── models/                     # BiGRUSoftplusDetector, attention layers, TDFS
├── training/                   # Trainer, train.py, kfold_train.py
├── evaluation/                 # Metrics, evaluate.py
├── inference/                  # predict.py (single video or batch directory)
├── preprocessing/              # Audio/video extraction pipeline
├── scripts/                    # eval_temporal.py, precompute_gfc.py, gen_dashboard.py
├── utils/                      # Config loader, reproducibility, I/O helpers
├── checkpoints/                # Saved model weights
├── logs/                       # TensorBoard event files + training_status.json
└── visualizations/             # Plots, ROC curves, attention maps
```

---

## Key Modules

| Module | Description |
|--------|-------------|
| `features/audio_features.py` | MFCC (40 coeffs) + Δ + ΔΔ = 120-dim audio features per frame |
| `features/video_cnn.py` | ResNet-18 (512-dim) frozen ImageNet backbone |
| `features/gfc.py` | 2-D FFT → 8 annular frequency band energies per frame |
| `features/fusion.py` | Per-modality normalisation and sequence alignment |
| `features/build_cache.py` | Build and version feature caches |
| `features/parallel_extract.py` | Parallel feature extraction with thread pool |
| `models/bigru_detector.py` | `BiGRUSoftplusDetector` — full AVcorr model |
| `models/attention.py` | `CrossModalAttentionFusion`, `ModalAttentionFusion`, `TemporalAttention` |
| `models/feature_selector.py` | `TemporalDifferentiableFeatureSelector` (TDFS) |
| `training/trainer.py` | Mixed-precision trainer, multi-task loss, TensorBoard, early stopping |
| `training/train.py` | Main training entry point |
| `training/kfold_train.py` | K-fold cross-validation training |
| `evaluation/evaluate.py` | Held-out test set evaluation |
| `evaluation/metrics.py` | Accuracy, Balanced Accuracy, F1, AUC, ROC, Confusion Matrix |
| `inference/predict.py` | Single-video or batch-directory inference |
| `scripts/eval_temporal.py` | Frame-level + segment AP temporal evaluation |
| `scripts/precompute_gfc.py` | Pre-compute GFC features (CPU-only, fast) |
| `datasets/annotation_loader.py` | `fake_periods` loader from LAV-DF metadata JSON |
| `datasets/sequence_dataset.py` | `MultiModalSequenceDataset`, `PackedSequenceDataset` |

---

## Feature Dimensions

| Stream | Extractor | Dimension |
|--------|-----------|-----------|
| Audio | MFCC + Δ + ΔΔ (40 coeffs each) | 120 |
| Visual — spatial | ResNet-18 (ImageNet, frozen) | 512 |
| Visual — frequency | GFC (8 FFT annular bands) | 8 |
| **Total input per frame** | Concatenated | **640** |

Sequences are sampled at **2 FPS** over **24 timesteps** (≈ 12-second clips).

---

## Installation

```powershell
cd AVcorr
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Ensure `ffmpeg` is installed and available in PATH.

**Dependencies:** PyTorch ≥ 2.2, torchvision ≥ 0.17, librosa ≥ 0.10, OpenCV ≥ 4.8, scikit-learn ≥ 1.4

---

## Dataset Setup

### Datasets

| Dataset | Role in AVcorr | Labels / annotations | Manipulation types |
|---------|----------------|----------------------|--------------------|
| **LAV-DF** | Primary training set; temporal forgery localization and in-domain detection (official train/val/test split) | Video-level real/fake + `fake_periods` (segment timestamps) from `metadata.min.json` | Content-driven audio and/or visual manipulation (real, fake-audio, fake-video, both) |
| **AV-Deepfake1M** | Large-scale temporal localization (train + dev split); zero-shot localization / detection | Video-level label, `modify_type`, `fake_segments`, per-modality `audio_fake_segments` / `visual_fake_segments` | Visual-modified, audio-modified, both-modified |
| **FakeAVCeleb** | Zero-shot cross-dataset detection (no refitting on target) | Video-level real/fake; `modify_video` / `modify_audio` flags inferred from folder names | Real/Fake Video × Real/Fake Audio (4 categories) |
| **DFDC** | Zero-shot cross-dataset detection | Video-level REAL/FAKE from per-part `metadata.json`; no modality annotation | Face-swap deepfakes |
| **DeepfakeTIMIT** (+ VidTIMIT reals) | Zero-shot cross-dataset detection | Video-level real/fake; quality tag (HQ / LQ) and subject id | Face-swap (HQ 128x128, LQ 64x64) |
| **VoxCeleb** | Real-only false-positive-rate probe | None (all clips treated as real) | n/a |

**Metadata builders and evaluation entry points**

| Dataset | Metadata builder | Config / evaluation |
|---------|------------------|---------------------|
| **LAV-DF** | `datasets/prepare_metadata.py` | `configs/lavdf_avcorr_*.yaml`, `scripts/eval_localization.py`, `scripts/eval_temporal.py` |
| **AV-Deepfake1M** | `scripts/build_avdeepfake1m_csv.py`, `scripts/build_present_indexes_df1m.py` | `configs/avdeepfake1m_avcorr_l*.yaml`, `scripts/eval_localization_df1m.py` |
| **FakeAVCeleb** | `scripts/build_fakeavceleb_csv.py` | `scripts/eval_crossdataset.py`, `scripts/eval_crossdataset_x.py` |
| **DFDC** | `scripts/build_dfdc_csv.py` | `scripts/eval_crossdataset.py` |
| **DeepfakeTIMIT** | `scripts/build_deepfaketimit_csv.py`, `scripts/fetch_vidtimit_reals.py` | `scripts/eval_crossdataset.py` |
| **VoxCeleb** | CSV with a `video_path` column | `scripts/eval_crossdataset.py --real_only` |

**Notes**

- The AV-Deepfake1M experiments use the locally prepared subset indexed by `data/df1m_present_all.min.json` (50,424 train clips + 1,500 dev clips), not the full multi-TB release.
- Cross-dataset runs are zero-shot: the LAV-DF scaler, feature mask and decision threshold are applied as-is, and nothing is fitted on the target dataset.
- LAV-DF is class-imbalanced (about 2.7:1 fake skew); the training configs use a balanced sampler to compensate.
- Datasets without `fake_periods` (FakeAVCeleb, DFDC, DeepfakeTIMIT, VoxCeleb) are evaluated at video level only; localization is reported on LAV-DF and AV-Deepfake1M.

### Supported Configs

| Dataset | Config | Status |
|---------|--------|--------|
| LAV-DF | `configs/lavdf_avcorr_v5_5.yaml` | Stable |
| LAV-DF | `configs/lavdf_avcorr_v6.yaml` | Experimental |

### Metadata CSV Format

| Column | Description |
|--------|-------------|
| `video_path` | Path to video file |
| `label` | `0` (real) or `1` (fake) |
| `sample_id` | Unique numeric identifier |

Generate from folder structure:

```powershell
python -m datasets.prepare_metadata `
    --root ./datasets/LAV-DF `
    --output ./data/lavdf_compact_metadata.csv `
    --dataset LAV-DF
```

For frame-level supervision (TFLH), place the annotation JSON at:

```
./datasets/LAV-DF/metadata.min.json
```

---

## End-to-End Workflow

### 1. Pre-compute GFC features (CPU-only, run once)

```powershell
python scripts/precompute_gfc.py --config configs/lavdf_avcorr_v5_5.yaml
```

### 2. Build MFCC + CNN feature cache

```powershell
python -m features.build_cache --config configs/lavdf_avcorr_v5_5.yaml
```

### 3. Train AVcorr

```powershell
python -m training.train --config configs/lavdf_avcorr_v5_5.yaml
```

Optional K-fold training:

```powershell
python -m training.kfold_train --config configs/lavdf_avcorr_v5_5.yaml
```

### 4. Evaluate on held-out test set

```powershell
python -m evaluation.evaluate `
    --config configs/lavdf_avcorr_v5_5.yaml `
    --checkpoint checkpoints/avcorr_v5_5/best_model.pt
```

### 5. Temporal localisation evaluation (video + frame + segment AP)

```powershell
python scripts/eval_temporal.py --config configs/lavdf_avcorr_v5_5.yaml
```

With on-demand GFC computation:

```powershell
python scripts/eval_temporal.py --config configs/lavdf_avcorr_v5_5.yaml --compute_gfc
```

### 6. Single-video inference

```powershell
python -m inference.predict `
    --config configs/lavdf_avcorr_v5_5.yaml `
    --checkpoint checkpoints/avcorr_v5_5/best_model.pt `
    --input path/to/video.mp4
```

### 7. Batch inference

```powershell
python -m inference.predict `
    --config configs/lavdf_avcorr_v5_5.yaml `
    --checkpoint checkpoints/avcorr_v5_5/best_model.pt `
    --input path/to/video_dir/
```

---

## Config Versions

| Config | Status | Key Differences |
|--------|--------|-----------------|
| `lavdf_avcorr_v4.yaml` | Baseline | MFCC + ResNet-18, plain BiGRU, no attention, no GFC |
| `lavdf_avcorr_v5.yaml` | Prior | + GFC, cross-modal attention, A/V agreement loss |
| `lavdf_avcorr_v5_5.yaml` | **Stable** | + TDFS, TFLH with fixed frame labels, dropout 0.45 |
| `lavdf_avcorr_v6.yaml` | Experimental | Dropout 0.4, TFLH weight 0.15, patience 20 |

---

## Multi-Task Loss

```
L_total = L_BCE                          (video-level classification, main task)
        + λ_tflh × L_TFLH               (frame-level localisation)
        + λ_tdfs × L_TDFS_L1            (TDFS sparsity regularisation)
        + λ_av   × L_AV_Agreement       (audio-visual sync supervision)
```

| Weight | v5.5 | v6 |
|--------|------|----|
| `λ_tflh` | 0.10 | 0.15 |
| `λ_tdfs` | 0.05 | 0.05 |
| `λ_av` | 0.10 | 0.10 |

---

## TensorBoard

```powershell
tensorboard --logdir logs
```

---

## Visualisation Outputs

`visualizations/plots.py` generates:

- Training / validation loss and accuracy curves
- ROC curves with AUC
- Confusion matrices
- Temporal attention weight heatmaps
- TDFS soft-mask averages (feature importance)

---

## Reproducibility

- `seed: 42` in YAML controls all randomness (NumPy, PyTorch, CUDA)
- Feature caches versioned by directory signature (`cache_v3`, `cache_v5_gfc`)
- Training state persisted to `logs/<experiment>/training_status.json` after every epoch
- Best checkpoint selected by AUC on validation set