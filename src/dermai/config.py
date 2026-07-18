"""Central configuration for DermAI.

Paths default to the Kaggle environment layout for the
`kmader/skin-cancer-mnist-ham10000` dataset, but every value can be
overridden via environment variables so the same code runs locally.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
# On Kaggle: /kaggle/input/skin-cancer-mnist-ham10000
DATA_DIR = Path(os.environ.get("DERMAI_DATA_DIR", "data/ham10000"))
METADATA_CSV = DATA_DIR / "HAM10000_metadata.csv"
IMAGE_DIRS = [
    DATA_DIR / "HAM10000_images_part_1",
    DATA_DIR / "HAM10000_images_part_2",
    # Kaggle mirror sometimes nests them one level deeper:
    DATA_DIR / "ham10000_images_part_1",
    DATA_DIR / "ham10000_images_part_2",
]

# Checkpoints must live somewhere writable — on Kaggle that's /kaggle/working,
# which is session-temporary; the resume logic in callbacks.py exists
# precisely because of that constraint.
WORK_DIR = Path(os.environ.get("DERMAI_WORK_DIR", "checkpoints"))
OUTPUT_DIR = Path(os.environ.get("DERMAI_OUTPUT_DIR", "outputs"))

# ---------------------------------------------------------------- classes
# HAM10000 diagnosis codes, alphabetical for a stable label order.
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)
MEL_INDEX = CLASS_NAMES.index("mel")  # melanoma recall is the primary metric

# ---------------------------------------------------------------- training
IMG_SIZE = 224          # EfficientNetB0 native input resolution
BATCH_SIZE = 32
SEED = 42
VAL_FRACTION = 0.15     # split at lesion level to avoid leakage

# Stage 1 — Linear Probing (frozen backbone, train classifier head)
LP_EPOCHS = 10
LP_LR = 1e-3

# Stage 2 — Fine-Tuning (unfreeze backbone, BatchNorm layers stay frozen)
FT_EPOCHS = 30
FT_LR = 1e-4

# Oversampling: cap how far minority classes are duplicated toward the
# majority count so df (n=115, the 58:1 tail) isn't repeated absurdly.
OVERSAMPLE_TARGET_FRACTION = 0.5  # each class raised to >= 50% of majority

# ReduceLROnPlateau
RLROP_FACTOR = 0.5
RLROP_PATIENCE = 3
RLROP_MIN_LR = 1e-6
