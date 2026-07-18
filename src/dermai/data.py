"""HAM10000 data pipeline.

- Lesion-level train/val split (multiple images can share a lesion_id;
  splitting at image level would leak near-duplicates into validation).
- Strategic oversampling of minority classes on the *training* split only.
- tf.data pipeline with augmentation for training.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from . import config


def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(config.METADATA_CSV)

    # Resolve each image_id to an actual file path across the part_1/part_2 dirs
    path_map = {}
    for d in config.IMAGE_DIRS:
        if d.is_dir():
            for p in d.glob("*.jpg"):
                path_map[p.stem] = str(p)
    df["path"] = df["image_id"].map(path_map)
    missing = df["path"].isna().sum()
    if missing:
        raise FileNotFoundError(
            f"{missing} images from the metadata were not found under "
            f"{config.DATA_DIR}. Check DERMAI_DATA_DIR."
        )

    df["label"] = df["dx"].map({name: i for i, name in enumerate(config.CLASS_NAMES)})
    return df


def lesion_level_split(df: pd.DataFrame):
    """Split so that all images of one lesion land on the same side."""
    rng = np.random.default_rng(config.SEED)
    lesions = df["lesion_id"].unique()
    rng.shuffle(lesions)
    n_val = int(len(lesions) * config.VAL_FRACTION)
    val_lesions = set(lesions[:n_val])
    val_df = df[df["lesion_id"].isin(val_lesions)].reset_index(drop=True)
    train_df = df[~df["lesion_id"].isin(val_lesions)].reset_index(drop=True)
    return train_df, val_df


def oversample(train_df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate minority-class rows until each class reaches
    OVERSAMPLE_TARGET_FRACTION of the majority class count.

    This replaced class_weight, which failed to converge at 58:1 —
    extreme weights make the df/vasc gradients dominate early updates.
    Duplicated rows get independent augmentation each epoch, so the
    effective inputs differ even though source pixels repeat.
    """
    counts = train_df["label"].value_counts()
    target = int(counts.max() * config.OVERSAMPLE_TARGET_FRACTION)
    rng = np.random.default_rng(config.SEED)

    parts = [train_df]
    for label, count in counts.items():
        if count < target:
            cls = train_df[train_df["label"] == label]
            extra = cls.sample(n=target - count, replace=True, random_state=config.SEED)
            parts.append(extra)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)


# ------------------------------------------------------------ tf.data

_augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ],
    name="augmentation",
)


def _decode(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [config.IMG_SIZE, config.IMG_SIZE])
    # EfficientNet includes its own input scaling layer — keep raw [0, 255]
    return img, label


def make_dataset(df: pd.DataFrame, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (df["path"].values, df["label"].values.astype("int32"))
    )
    if training:
        ds = ds.shuffle(len(df), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(config.BATCH_SIZE)
    if training:
        ds = ds.map(
            lambda x, y: (_augment(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.prefetch(tf.data.AUTOTUNE)


def get_datasets():
    """Returns (train_ds, val_ds, val_df)."""
    df = load_metadata()
    train_df, val_df = lesion_level_split(df)
    train_df = oversample(train_df)
    return make_dataset(train_df, True), make_dataset(val_df, False), val_df
