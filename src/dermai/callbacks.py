"""Custom Keras callbacks, written from scratch.

Why not the built-ins: Keras 2.19's stock ModelCheckpoint /
ReduceLROnPlateau path performs a deepcopy of callback state that fails on
this model (the wrapped EfficientNet backbone holds non-copyable
references), crashing at the first save. These implementations keep no
deep-copied state: they save the model with `model.save()`, track scalars
only, and persist their own state to a small JSON file so training can
resume after Kaggle kills the session and /kaggle/working is restored
from a saved version.
"""

import json
from pathlib import Path

import tensorflow as tf

from . import config


class ResumableCheckpoint(tf.keras.callbacks.Callback):
    """Save best + last model every epoch, plus a JSON training-state file.

    State file records: last completed epoch, best monitored value, and the
    current learning rate — everything needed to resume mid-run.
    """

    def __init__(self, stage: str, monitor: str = "val_accuracy", mode: str = "max"):
        super().__init__()
        self.stage = stage
        self.monitor = monitor
        self.sign = 1.0 if mode == "max" else -1.0
        self.best = -float("inf")
        config.WORK_DIR.mkdir(parents=True, exist_ok=True)
        self.best_path = config.WORK_DIR / f"{stage}_best.keras"
        self.last_path = config.WORK_DIR / f"{stage}_last.keras"
        self.state_path = config.WORK_DIR / f"{stage}_state.json"

    # ---- resume support -------------------------------------------------
    def load_state(self) -> dict | None:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            self.best = state["best"] * self.sign
            return state
        return None

    # ---- callback hooks -------------------------------------------------
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        self.model.save(self.last_path)
        if current is not None and current * self.sign > self.best:
            self.best = current * self.sign
            self.model.save(self.best_path)
            print(f"\n[checkpoint] new best {self.monitor}={current:.4f} -> {self.best_path}")

        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.state_path.write_text(
            json.dumps(
                {
                    "stage": self.stage,
                    "last_epoch": int(epoch) + 1,  # epochs completed
                    "best": self.best * self.sign,
                    "monitor": self.monitor,
                    "lr": lr,
                    "logs": {k: float(v) for k, v in logs.items()},
                }
            )
        )


class SimpleReduceLROnPlateau(tf.keras.callbacks.Callback):
    """Halve the LR when the monitored metric stops improving.

    Scalar state only — no config deepcopy, safe under Keras 2.19.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        factor: float = config.RLROP_FACTOR,
        patience: int = config.RLROP_PATIENCE,
        min_lr: float = config.RLROP_MIN_LR,
    ):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float("inf")
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current is None:
            return
        if current < self.best - 1e-4:
            self.best = current
            self.wait = 0
            return
        self.wait += 1
        if self.wait >= self.patience:
            old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            new_lr = max(old_lr * self.factor, self.min_lr)
            if new_lr < old_lr:
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                print(f"\n[reduce-lr] {self.monitor} plateaued; lr {old_lr:.2e} -> {new_lr:.2e}")
            self.wait = 0
