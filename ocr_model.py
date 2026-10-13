"""
ocr_model.py
============
CRNN+CTC model utilities for ASCII62 (0-9, A-Z, a-z) text-line recognition.

This module provides:
  - Vocabulary mapping persistence (model/vocab.json)
  - CRNN network (image -> per-timestep character probabilities)
  - Greedy CTC decoding (probabilities -> predicted text)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


IMAGE_HEIGHT = 32
IMAGE_WIDTH = 280
NUM_CHARS = 62
VOCAB_SIZE = NUM_CHARS + 1  # +1 for CTC blank at index NUM_CHARS (i.e., last id)
BLANK_INDEX = NUM_CHARS

# Assumed char order for ASCII62.
DEFAULT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _vocab_path(model_dir: str = "model") -> str:
    return os.path.join(model_dir, "vocab.json")


def save_vocab(alphabet: str, model_dir: str = "model") -> None:
    os.makedirs(model_dir, exist_ok=True)
    payload = {
        "alphabet": alphabet,
        "blank_index": BLANK_INDEX,
    }
    with open(_vocab_path(model_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_vocab(model_dir: str = "model") -> Tuple[str, Dict[int, str]]:
    """
    Returns:
      alphabet: string of length 62
      id_to_char: mapping from CTC ids (1..62) to characters
    """
    path = _vocab_path(model_dir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        alphabet = payload.get("alphabet", DEFAULT_ALPHABET)
    else:
        alphabet = DEFAULT_ALPHABET

    if len(alphabet) != NUM_CHARS:
        raise ValueError(f"Invalid alphabet length {len(alphabet)}; expected {NUM_CHARS}")

    # CTC labels are expected in range [0, NUM_CHARS-1].
    # The blank token is index NUM_CHARS (i.e., VOCAB_SIZE-1).
    id_to_char = {i: ch for i, ch in enumerate(alphabet)}
    return alphabet, id_to_char


def build_crnn(width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT, vocab_size: int = VOCAB_SIZE) -> tf.keras.Model:
    """
    Build a simple CRNN suitable for CTC on fixed-size inputs.

    Network:
      (H,W,1) -> Conv -> MaxPool -> Conv -> MaxPool -> Conv
               -> reshape to (time_steps, features)
               -> BiLSTM -> BiLSTM -> Dense(vocab_size, softmax)
    """
    time_steps = width // 4
    feature_h = height // 4  # because we pool (2,2) twice
    if feature_h <= 0:
        raise ValueError("Invalid height for CRNN pooling")

    inputs = tf.keras.Input(shape=(height, width, 1), name="image", dtype=tf.float32)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # -> (H/2, W/2)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # -> (H/4, W/4)

    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)  # (H/4, W/4, 256)

    # Reformat to time major sequence for RNN: (batch, W/4, H/4*256)
    x = tf.keras.layers.Permute((2, 1, 3))(x)  # (batch, W/4, H/4, 256)
    x = tf.keras.layers.Reshape((time_steps, feature_h * 256))(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)

    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax", name="char_probs")(x)
    model = tf.keras.Model(inputs, outputs, name="CRNN_CTC_ASCII62")
    return model


def decode_ctc_greedy(
    y_pred: np.ndarray,
    id_to_char: Dict[int, str],
    time_steps: int,
) -> Tuple[List[str], np.ndarray]:
    """
    y_pred:
      shape (batch, time_steps, vocab_size)

    Returns:
      decoded_texts: list of strings length=batch
      confidences: array shape (batch,) in [0,100]
    """
    if y_pred.ndim != 3:
        raise ValueError(f"Expected y_pred rank=3 (batch,time,vocab), got shape {y_pred.shape}")

    batch_size = y_pred.shape[0]
    input_length = np.full(shape=(batch_size,), fill_value=time_steps, dtype=np.int32)

    decoded, log_prob = tf.keras.backend.ctc_decode(
        y_pred,
        input_length=input_length,
        greedy=True,
        beam_width=None,
        top_paths=1,
    )

    # decoded is a list/tuple; first element corresponds to path 0.
    seqs = decoded[0]  # shape (batch, max_dec_len) padded with -1

    decoded_texts: List[str] = []
    # Confidence heuristic: mean of per-timestep max probabilities.
    per_timestep_max = np.max(y_pred, axis=-1)  # (batch, time_steps)
    conf = np.mean(per_timestep_max, axis=1) * 100.0

    for b in range(batch_size):
        chars: List[str] = []
        for token in seqs[b].numpy().tolist():
            if token == -1:
                continue
            # token is either -1 (padding), blank (VOCAB_SIZE-1), or a class id [0..61]
            if token == BLANK_INDEX:
                continue
            ch = id_to_char.get(int(token), "")
            if ch:
                chars.append(ch)
        decoded_texts.append("".join(chars))

    return decoded_texts, conf

