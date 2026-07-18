"""Evaluate a trained checkpoint on the validation split.

Reports overall accuracy, the per-class classification report, a confusion
matrix, and melanoma recall — the primary clinical metric (a missed
melanoma is far costlier than a false alarm).

Usage:
    python -m dermai.evaluate                    # uses ft_best.keras
    python -m dermai.evaluate --model checkpoints/lp_best.keras
"""

import argparse

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, recall_score

from . import config, data


def main():
    parser = argparse.ArgumentParser(description="DermAI evaluation")
    parser.add_argument("--model", default=str(config.WORK_DIR / "ft_best.keras"))
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    _, val_ds, val_df = data.get_datasets()

    probs = model.predict(val_ds, verbose=1)
    y_pred = probs.argmax(axis=1)
    y_true = val_df["label"].values

    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4))

    mel_recall = recall_score(y_true, y_pred, labels=[config.MEL_INDEX], average=None)[0]
    acc = (y_true == y_pred).mean()
    print(f"Validation accuracy : {acc:.4f}")
    print(f"Melanoma recall     : {mel_recall:.4f}   <-- primary metric")

    print("\n=== Confusion matrix (rows=true, cols=pred) ===")
    cm = confusion_matrix(y_true, y_pred)
    header = "      " + " ".join(f"{n:>6}" for n in config.CLASS_NAMES)
    print(header)
    for name, row in zip(config.CLASS_NAMES, cm):
        print(f"{name:>5} " + " ".join(f"{v:>6}" for v in row))

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.OUTPUT_DIR / "confusion_matrix.npy", cm)
    print(f"\nSaved confusion matrix to {config.OUTPUT_DIR / 'confusion_matrix.npy'}")


if __name__ == "__main__":
    main()
