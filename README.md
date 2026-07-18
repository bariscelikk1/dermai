# DermAI — Deep Learning Skin Lesion Classifier

7-class dermoscopic image classifier built on **HAM10000** (10,015 images) with
**EfficientNetB0 transfer learning**, optimized for **melanoma recall** as the
primary clinical metric.

Python · TensorFlow/Keras · EfficientNetB0 · Transfer Learning · Kaggle

## Key techniques

- **Two-stage LP-FT pipeline** (Linear Probing → Fine-Tuning, per Kumar et al. 2022):
  train the classifier head on a frozen backbone first, then fine-tune the whole
  network at a low learning rate. **BatchNormalization layers stay frozen during
  fine-tuning** — unfreezing them lets HAM10000 batch statistics overwrite the
  ImageNet running stats and causes validation loss to explode.
- **Custom Keras callbacks** (`ResumableCheckpoint`, `SimpleReduceLROnPlateau`)
  written from scratch to bypass a Keras 2.19 deepcopy incompatibility, while
  keeping full checkpoint/resume support under Kaggle's session-temporary
  storage (`/kaggle/working`).
- **58:1 class imbalance** (`nv` ≈ 6,705 vs `df` ≈ 115) handled via strategic
  oversampling of minority classes on the training split, replacing the naive
  `class_weight` approach that failed to converge.
- **Lesion-level train/val split** — HAM10000 contains multiple images per
  lesion; splitting at image level leaks near-duplicates into validation.

## Project layout

```
src/dermai/
  config.py     # paths, classes, hyperparameters (env-overridable)
  data.py       # metadata loading, lesion-level split, oversampling, tf.data
  model.py      # EfficientNetB0 head + LP-FT freeze/unfreeze logic
  callbacks.py  # deepcopy-free checkpoint + ReduceLROnPlateau with resume
  train.py      # CLI: --stage lp | ft, --resume
  evaluate.py   # classification report, confusion matrix, melanoma recall
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Get the data

Kaggle dataset: [`kmader/skin-cancer-mnist-ham10000`](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

```bash
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p data/ham10000 --unzip
```

On Kaggle notebooks, just attach the dataset and set:

```bash
export DERMAI_DATA_DIR=/kaggle/input/skin-cancer-mnist-ham10000
export DERMAI_WORK_DIR=/kaggle/working/checkpoints
```

## Train

```bash
cd src

# Stage 1 — Linear Probing (frozen backbone, ~10 epochs)
python -m dermai.train --stage lp

# Stage 2 — Fine-Tuning (backbone unfrozen, BatchNorm frozen, ~30 epochs)
python -m dermai.train --stage ft

# Resume after a killed Kaggle session
python -m dermai.train --stage ft --resume
```

## Evaluate

```bash
python -m dermai.evaluate            # uses checkpoints/ft_best.keras
```

Prints per-class precision/recall/F1, the confusion matrix, overall validation
accuracy, and **melanoma recall** (the primary metric).

## Classes

| Code | Diagnosis |
|------|-----------|
| akiec | Actinic keratoses / intraepithelial carcinoma |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis-like lesions |
| df | Dermatofibroma |
| mel | **Melanoma** (primary recall target) |
| nv | Melanocytic nevi |
| vasc | Vascular lesions |
