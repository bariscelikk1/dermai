"""Two-stage LP-FT training entry point with checkpoint/resume.

Usage:
    python -m dermai.train --stage lp          # stage 1: linear probing
    python -m dermai.train --stage ft          # stage 2: fine-tuning
    python -m dermai.train --stage lp --resume # continue an interrupted run
"""

import argparse

import tensorflow as tf

from . import callbacks as cb
from . import config, data, model as model_lib


def _load_or_build(stage: str, resume: bool, ckpt: cb.ResumableCheckpoint):
    initial_epoch = 0

    if resume:
        state = ckpt.load_state()
        if state is None or not ckpt.last_path.exists():
            raise SystemExit(f"--resume requested but no checkpoint found for stage '{stage}'")
        m = tf.keras.models.load_model(ckpt.last_path)
        initial_epoch = state["last_epoch"]
        tf.keras.backend.set_value(m.optimizer.learning_rate, state["lr"])
        print(f"[resume] stage={stage} from epoch {initial_epoch}, lr={state['lr']:.2e}")
        return m, initial_epoch

    if stage == "lp":
        m = model_lib.build_model()
        model_lib.compile_model(m, config.LP_LR)
    else:  # ft — starts from the best linear-probe checkpoint
        lp_best = config.WORK_DIR / "lp_best.keras"
        if not lp_best.exists():
            raise SystemExit("Fine-tuning requires a completed LP stage (lp_best.keras missing)")
        m = tf.keras.models.load_model(lp_best)
        model_lib.unfreeze_for_finetune(m)
        model_lib.compile_model(m, config.FT_LR)  # recompile so freezing takes effect
    return m, initial_epoch


def main():
    parser = argparse.ArgumentParser(description="DermAI LP-FT training")
    parser.add_argument("--stage", choices=["lp", "ft"], required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    epochs = args.epochs or (config.LP_EPOCHS if args.stage == "lp" else config.FT_EPOCHS)

    train_ds, val_ds, _ = data.get_datasets()
    ckpt = cb.ResumableCheckpoint(stage=args.stage)
    m, initial_epoch = _load_or_build(args.stage, args.resume, ckpt)

    m.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=[ckpt, cb.SimpleReduceLROnPlateau()],
    )
    print(f"\nDone. Best model: {ckpt.best_path}")


if __name__ == "__main__":
    main()
