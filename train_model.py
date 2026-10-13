"""
train_model.py
==============
Train a 62-class CRNN+CTC model for ASCII62 text-line recognition:
  0-9, A-Z, a-z

Usage:
  python train_model.py

This replaces the old MNIST classifier training.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import backend as K

from ocr_model import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CHARS,
    VOCAB_SIZE,
    DEFAULT_ALPHABET,
    build_crnn,
    save_vocab,
)


# Fixed-size input for the CRNN.
TIME_STEPS = IMAGE_WIDTH // 4  # must match ocr_model.build_crnn()
CHANNELS = 1


@dataclass
class TrainConfig:
    # Defaults tuned for better initial recognition on CPU.
    # (CTC/CRNN training is slow on Windows CPU-only; these values aim for usable results.)
    epochs: int = 1
    batch_size: int = 8
    # Synthetic samples per epoch. Increase for better results (slower).
    samples_per_epoch: int = 2000
    # Generate random lines with 1..max_chars visible characters.
    min_chars: int = 1
    max_chars: int = 10
    # CTC label padding length (>= max_chars).
    max_label_len: int = 12
    # Model artifacts
    model_path: str = "model/alphabet_ctc_crnn.keras"
    vocab_dir: str = "model"


def _as_binary_strokes(img: np.ndarray) -> np.ndarray:
    """
    Convert EMNIST grayscale image into a binary stroke mask.
    Returns float32 mask with strokes=1, background=0.
    """
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[:, :, 0]
    img = img.astype(np.float32)

    # EMNIST from tfds is typically already scaled to [0, 1], but be defensive.
    if img.max() > 1.5:
        img = img / 255.0

    # threshold at mid-point
    binary = img > 0.5
    foreground_ratio = float(np.mean(binary))

    # We want strokes to be True/1. If most pixels are True, invert.
    if foreground_ratio > 0.5:
        binary = ~binary

    return binary.astype(np.float32)


def load_emnist_pools() -> Tuple[Dict[int, List[np.ndarray]], str, Dict[int, str]]:
    """
    Loads EMNIST (byclass; 62 labels) and returns:
      - pools[label_id] = list of 28x28 stroke masks (float32)
      - alphabet string in the fixed ASCII62 order (DEFAULT_ALPHABET)
      - id_to_char mapping for that order
    """
    print("[INFO] Loading EMNIST (byclass; 62 labels) via tensorflow-datasets...")
    builder = tfds.builder("emnist/byclass")
    builder.download_and_prepare()
    label_names = builder.info.features["label"].names
    if len(label_names) != NUM_CHARS:
        raise ValueError(f"Unexpected EMNIST label count: {len(label_names)}; expected {NUM_CHARS}")

    # We assume tfds class ids are ordered as:
    #   0-9 -> digits, 10-35 -> 'A'-'Z', 36-61 -> 'a'-'z'
    alphabet = DEFAULT_ALPHABET
    id_to_char = {i: alphabet[i] for i in range(NUM_CHARS)}

    ds = tfds.load("emnist/byclass", split="train", as_supervised=True, shuffle_files=True)

    pools: Dict[int, List[np.ndarray]] = {i: [] for i in range(NUM_CHARS)}

    # Load all training samples. This is memory-heavy but keeps training simpler.
    # 112,800 images * 28*28 bytes ~ 88MB (float32 would be more); we store float32 masks.
    for img, label in ds:
        img = img.numpy()
        label = int(label.numpy())
        mask = _as_binary_strokes(img)
        pools[label].append(mask)

    total = sum(len(v) for v in pools.values())
    print(f"[INFO] Loaded EMNIST pools: total samples={total}")
    return pools, alphabet, id_to_char


def augment_char(stroke_mask: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Apply small affine transform + random thickness.
    stroke_mask: float32 28x28 with strokes=1.
    """
    img = (stroke_mask * 255).astype(np.uint8)

    # Random rotation and scaling.
    angle = rng.uniform(-15.0, 15.0)
    scale = rng.uniform(0.9, 1.1)
    tx = rng.uniform(-2.0, 2.0)
    ty = rng.uniform(-2.0, 2.0)

    M = cv2.getRotationMatrix2D((14, 14), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    warped = cv2.warpAffine(img, M, (28, 28), flags=cv2.INTER_NEAREST, borderValue=0)

    # Random dilation to simulate pen thickness / merged strokes.
    if rng.random() < 0.7:
        k = rng.choice([2, 3])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        warped = cv2.dilate(warped, kernel, iterations=1)

    out = (warped > 0).astype(np.float32)
    return out


def compose_line_from_pools(
    pools: Dict[int, List[np.ndarray]],
    rng: random.Random,
    cfg: TrainConfig,
) -> Tuple[np.ndarray, List[int]]:
    """
    Create one synthetic line image (32x280) and its label sequence.
    Returns:
      - line_img: float32 (H,W) in [0,1] with strokes=1
      - labels: list of int tokens in [0..61] (CTC blank is handled separately)
    """
    canvas = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)

    num_chars = rng.randint(cfg.min_chars, cfg.max_chars)

    # Start position with some margin.
    x = rng.randint(0, 10)
    y0 = (IMAGE_HEIGHT - 28) // 2 + rng.randint(-2, 2)
    y0 = max(0, min(IMAGE_HEIGHT - 28, y0))

    labels: List[int] = []

    for _ in range(num_chars):
        label_id = rng.randrange(NUM_CHARS)
        pool = pools[label_id]
        if not pool:
            continue
        base_img = pool[rng.randrange(len(pool))]
        aug = augment_char(base_img, rng)

        # If overflow, stop adding more characters.
        if x >= IMAGE_WIDTH:
            break

        x1 = int(x)
        x2 = min(IMAGE_WIDTH, x1 + 28)
        roi_w = x2 - x1
        if roi_w > 0:
            canvas[y0 : y0 + 28, x1:x2] = np.maximum(canvas[y0 : y0 + 28, x1:x2], aug[:, :roi_w])

        labels.append(label_id)

        # Step size < 28 creates intentional touching/merging.
        step = rng.randint(16, 28)
        x += step

    # If we somehow generated nothing, force one char.
    if not labels:
        label_id = rng.randrange(NUM_CHARS)
        base_img = pools[label_id][rng.randrange(len(pools[label_id]))]
        aug = augment_char(base_img, rng)
        canvas[:, :] = 0
        canvas[y0 : y0 + 28, 0:28] = aug
        labels = [label_id]

    return canvas, labels


def make_batch_generator(
    pools: Dict[int, List[np.ndarray]],
    cfg: TrainConfig,
    steps_per_epoch: int,
    seed: int = 1337,
):
    rng = random.Random(seed)

    def gen():
        while True:
            images = np.zeros((cfg.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS), dtype=np.float32)
            # CTC uses label_length to ignore padded positions; padding with 0 is safe.
            labels_padded = np.zeros((cfg.batch_size, cfg.max_label_len), dtype=np.int32)
            label_lengths = np.zeros((cfg.batch_size, 1), dtype=np.int32)
            input_lengths = np.full((cfg.batch_size, 1), TIME_STEPS, dtype=np.int32)

            for b in range(cfg.batch_size):
                line_img, labels = compose_line_from_pools(pools, rng, cfg)
                images[b, :, :, 0] = line_img

                # Truncate labels to max_label_len.
                labels = labels[: cfg.max_label_len]
                label_lengths[b, 0] = len(labels)
                if labels:
                    labels_padded[b, : len(labels)] = np.array(labels, dtype=np.int32)

            # The model output itself is the loss tensor, so y_dummy is ignored.
            y_dummy = np.zeros((cfg.batch_size, 1), dtype=np.float32)
            yield (images, labels_padded, input_lengths, label_lengths), y_dummy

    return gen()


def build_ctc_training_model(crnn: tf.keras.Model, cfg: TrainConfig) -> tf.keras.Model:
    """
    Wrap the CRNN with CTC loss so Keras can train it.
    The returned model outputs the per-batch loss.
    """
    image_in = tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS), name="image", dtype=tf.float32)
    labels_in = tf.keras.Input(shape=(cfg.max_label_len,), name="labels", dtype=tf.int32)
    input_len_in = tf.keras.Input(shape=(1,), name="input_length", dtype=tf.int32)
    label_len_in = tf.keras.Input(shape=(1,), name="label_length", dtype=tf.int32)

    y_pred = crnn(image_in)  # (batch, time, vocab)

    # CTC loss expects:
    #  y_true: labels (batch, max_label_len)
    #  y_pred: probabilities (batch, time, vocab)
    #  input_length: (batch, 1) or (batch,)
    #  label_length: (batch, 1) or (batch,)
    def ctc_loss_lambda(args):
        labels, y_pred_probs, input_len, label_len = args
        # K.ctc_batch_cost returns shape (batch, 1)
        return K.ctc_batch_cost(labels, y_pred_probs, input_len, label_len)

    loss_out = tf.keras.layers.Lambda(ctc_loss_lambda, name="ctc_loss")(
        [labels_in, y_pred, input_len_in, label_len_in]
    )

    model = tf.keras.Model(inputs=[image_in, labels_in, input_len_in, label_len_in], outputs=loss_out)
    return model


if __name__ == "__main__":
    cfg = TrainConfig()
    os.makedirs(cfg.vocab_dir, exist_ok=True)

    pools, alphabet, _id_to_char = load_emnist_pools()
    save_vocab(alphabet, model_dir=cfg.vocab_dir)

    print(f"[INFO] Training vocab saved: {cfg.vocab_dir}/vocab.json")
    print(f"[INFO] Alphabet (len={len(alphabet)}): {alphabet}")

    print("[INFO] Building CRNN...")
    crnn = build_crnn()

    print("[INFO] Building CTC training wrapper...")
    ctc_model = build_ctc_training_model(crnn, cfg)

    # When the model output is the loss tensor, use identity loss.
    ctc_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=lambda y_true, y_pred: y_pred,
    )
    ctc_model.summary()

    steps_per_epoch = max(1, cfg.samples_per_epoch // cfg.batch_size)
    print(f"[INFO] steps_per_epoch={steps_per_epoch}, epochs={cfg.epochs}")

    batch_gen = make_batch_generator(pools, cfg, steps_per_epoch)

    history = ctc_model.fit(
        batch_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=cfg.epochs,
        verbose=1,
    )

    print("[INFO] Training finished. Saving inference model...")
    os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)
    crnn.save(cfg.model_path)
    print(f"[INFO] Saved model -> {cfg.model_path}")

