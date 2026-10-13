"""
train_char_model.py
====================
Train a per-character CNN classifier on EMNIST (byclass) for ASCII62:
  0-9, A-Z, a-z  (62 classes)

Saves:
  - model/alphabet_char_cnn.keras
  - model/vocab_char.json

Default training uses the full shuffled stream with strong augmentation and early
stopping (no tiny 400-batch cap). For a quick smoke test:

  DIGIT_FAST_TRAIN=1 python train_char_model.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers


EMNIST_CLASSES = 62
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

MODEL_PATH = os.path.join("model", "alphabet_char_cnn.keras")
VOCAB_PATH = os.path.join("model", "vocab_char.json")

FAST = os.environ.get("DIGIT_FAST_TRAIN", "").strip() in ("1", "true", "yes")


@dataclass
class TrainConfig:
    batch_size: int = 128
    # Steps per epoch (random stream); higher = more distinct samples per epoch.
    steps_per_epoch: int = 2500 if not FAST else 300
    validation_steps: int = 200 if not FAST else 80
    epochs: int = 40 if not FAST else 4
    early_stop_patience: int = 6 if not FAST else 2


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _apply_env_overrides(cfg: TrainConfig) -> None:
    """Optional: DIGIT_STEPS_PER_EPOCH, DIGIT_VAL_STEPS, DIGIT_EPOCHS, DIGIT_BATCH."""
    cfg.steps_per_epoch = _env_int("DIGIT_STEPS_PER_EPOCH", cfg.steps_per_epoch)
    cfg.validation_steps = _env_int("DIGIT_VAL_STEPS", cfg.validation_steps)
    cfg.epochs = _env_int("DIGIT_EPOCHS", cfg.epochs)
    cfg.batch_size = _env_int("DIGIT_BATCH", cfg.batch_size)


def _preprocess(img, label):
    img = tf.cast(img, tf.float32)
    if img.shape.rank == 2:
        img = img[..., tf.newaxis]
    img = img / 255.0 if tf.reduce_max(img) > 1.5 else img
    label = tf.cast(label, tf.int32)
    return img, label


def build_model(num_classes: int = EMNIST_CLASSES) -> keras.Model:
    """
    Deeper CNN + on-GPU augmentation (inactive at inference when model.predict).
    """
    aug = keras.Sequential(
        [
            layers.RandomRotation(0.08),
            layers.RandomZoom(height_factor=(-0.2, 0.05), width_factor=(-0.2, 0.05)),
            layers.RandomTranslation(0.12, 0.12),
            layers.GaussianNoise(0.04),
        ],
        name="aug",
    )

    inputs = keras.Input(shape=(28, 28, 1))
    x = aug(inputs)

    x = layers.Conv2D(48, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(48, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv2D(96, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(96, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="EMNIST_Char_CNN_v2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_vocab():
    os.makedirs("model", exist_ok=True)
    payload = {"alphabet": ALPHABET}
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    cfg = TrainConfig()
    _apply_env_overrides(cfg)
    os.makedirs("model", exist_ok=True)
    save_vocab()

    print("[INFO] Loading EMNIST byclass...")
    ds_train = tfds.load("emnist/byclass", split="train", as_supervised=True, shuffle_files=True)
    ds_test = tfds.load("emnist/byclass", split="test", as_supervised=True, shuffle_files=False)

    ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.shuffle(16384).repeat().batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)

    print("[INFO] Building model...")
    model = build_model(EMNIST_CLASSES)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    mode = "FAST (DIGIT_FAST_TRAIN)" if FAST else "full"
    print(f"[INFO] Training mode: {mode}")
    print(
        f"[INFO] epochs={cfg.epochs} steps_per_epoch={cfg.steps_per_epoch} "
        f"val_steps={cfg.validation_steps} batch={cfg.batch_size}"
    )

    history = model.fit(
        ds_train,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps_per_epoch,
        validation_data=ds_test,
        validation_steps=cfg.validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    print("[INFO] Saving model...")
    model.save(MODEL_PATH)
    print(f"[INFO] Saved -> {MODEL_PATH}")

    if "val_accuracy" in history.history and history.history["val_accuracy"]:
        best = max(history.history["val_accuracy"])
        print(f"[RESULT] Best val_accuracy: {best:.4f}")


if __name__ == "__main__":
    main()
