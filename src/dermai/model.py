"""EfficientNetB0 transfer-learning model with the two-stage LP-FT protocol.

LP-FT (Linear Probing then Fine-Tuning, Kumar et al. 2022): training the
head first with a frozen backbone keeps the pretrained features intact,
then a low-LR fine-tune adapts the backbone without destroying them.

During fine-tuning, all BatchNormalization layers stay frozen. Unfreezing
them lets HAM10000's batch statistics (small batches, oversampled classes)
overwrite the ImageNet running statistics, which manifests as a validation
loss explosion within a few epochs.
"""

import tensorflow as tf

from . import config


def build_model() -> tf.keras.Model:
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    )
    base.trainable = False  # stage 1: linear probing

    inputs = tf.keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3))
    # training=False pins BatchNorm to inference mode permanently for the
    # backbone — required both while frozen and during fine-tuning.
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(config.NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="dermai_effnetb0")
    return model


def compile_model(model: tf.keras.Model, lr: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )


def unfreeze_for_finetune(model: tf.keras.Model):
    """Stage 2: unfreeze the backbone but keep every BatchNorm layer frozen."""
    base = model.get_layer("efficientnetb0")
    base.trainable = True
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
