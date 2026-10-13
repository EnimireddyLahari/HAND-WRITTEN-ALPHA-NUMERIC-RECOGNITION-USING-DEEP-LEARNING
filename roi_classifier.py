"""
roi_classifier.py
==================
ROI-based OCR inference for ASCII62 (0-9A-Za-z).

Approach:
  1) Adaptive threshold (Otsu + fallback) -> binary mask (strokes=1)
  2) Find character contours (connected components)
  3) Group bboxes into text lines (top->bottom)
  4) For merged/touching regions, split using vertical projection valleys
  5) Classify each char ROI with a per-character EMNIST-trained CNN
     (tight ink crop, square padding, optional test-time augmentation)
  6) Concatenate characters left->right per line
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tensorflow import keras

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "model", "alphabet_char_cnn.keras")
VOCAB_PATH = os.path.join(_PROJECT_ROOT, "model", "vocab_char.json")

IMAGE_SIZE = 28
CHAR_CLASSES = 62

DEFAULT_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def strokes_mask_from_gray(gray_u8: np.ndarray) -> np.ndarray:
    """
    Binary mask with strokes=1. Input: bright strokes on dark background (as from Flask invert).
    Uses Gaussian blur + Otsu; falls back to fixed threshold if Otsu is degenerate.
    """
    if gray_u8.ndim != 2:
        raise ValueError("Expected grayscale 2D array")
    g = cv2.GaussianBlur(gray_u8, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = float(np.mean(th > 0))
    if fg > 0.92 or fg < 0.002:
        _, th = cv2.threshold(gray_u8, 50, 255, cv2.THRESH_BINARY)
    return (th > 0).astype(np.uint8)


def _load_vocab() -> Tuple[str, Dict[int, str]]:
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        alphabet = payload.get("alphabet", DEFAULT_ALPHABET)
    else:
        alphabet = DEFAULT_ALPHABET

    if len(alphabet) != CHAR_CLASSES:
        raise ValueError(f"Invalid alphabet length {len(alphabet)}; expected {CHAR_CLASSES}")

    id_to_char = {i: alphabet[i] for i in range(CHAR_CLASSES)}
    return alphabet, id_to_char


def _try_load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        m = keras.models.load_model(MODEL_PATH, compile=False)
        _, id_to_char = _load_vocab()
        return m, id_to_char
    except Exception as e:
        print(f"[WARN] Could not load char ROI model from {MODEL_PATH}: {e}")
        return None, None


model, id_to_char = _try_load_model()


def _ensure_model_loaded():
    global model, id_to_char
    if model is not None and id_to_char is not None:
        return
    model, id_to_char = _try_load_model()


def _find_initial_bboxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Find external contours and return bounding boxes (x1,y1,x2,y2).
    mask: uint8 strokes=1.
    """
    dilated = cv2.dilate(
        mask * 255,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape[:2]

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 6 or h < 10:
            continue
        if w > W * 0.96 or h > H * 0.96:
            continue
        bboxes.append((x, y, x + w, y + h))
    return bboxes


def _group_bboxes_into_lines(bboxes: List[Tuple[int, int, int, int]], y_tol: int = 14) -> List[List[Tuple[int, int, int, int]]]:
    if not bboxes:
        return []

    bboxes_sorted = sorted(bboxes, key=lambda b: (b[1] + b[3]) / 2.0)
    lines: List[List[Tuple[int, int, int, int]]] = []

    current: List[Tuple[int, int, int, int]] = []
    current_y2 = -1

    for bb in bboxes_sorted:
        x1, y1, x2, y2 = bb
        if not current:
            current = [bb]
            current_y2 = y2
            continue

        if y1 <= current_y2 + y_tol:
            current.append(bb)
            current_y2 = max(current_y2, y2)
        else:
            lines.append(current)
            current = [bb]
            current_y2 = y2

    if current:
        lines.append(current)

    lines.sort(key=lambda line: np.mean([(b[1] + b[3]) / 2.0 for b in line]))

    for line in lines:
        line.sort(key=lambda b: b[0])

    return lines


def _split_merged_roi(mask: np.ndarray, roi_bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = roi_bbox
    roi = mask[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if w <= 0 or h <= 0:
        return [roi_bbox]

    col_sum = roi.sum(axis=0).astype(np.float32)
    if col_sum.max() <= 0:
        return [roi_bbox]

    k = 9 if w >= 36 else 5
    kernel = np.ones((k,), dtype=np.float32) / float(k)
    smooth = np.convolve(col_sum, kernel, mode="same")

    valley_thr = 0.12 * float(smooth.max())

    low = smooth < valley_thr
    splits = []
    start = None
    for i in range(w):
        if low[i] and start is None:
            start = i
        elif (not low[i]) and start is not None:
            end = i - 1
            splits.append((start, end))
            start = None
    if start is not None:
        splits.append((start, w - 1))

    min_char_w = 8

    segments: List[Tuple[int, int]] = []
    prev_end = 0
    for (s, e) in splits:
        if s - prev_end >= min_char_w:
            segments.append((prev_end, s))
        prev_end = e + 1
    if w - prev_end >= min_char_w:
        segments.append((prev_end, w))

    if len(segments) < 2:
        return [roi_bbox]

    out_bboxes = []
    for (sx1, sx2) in segments:
        seg = roi[:, sx1:sx2]
        if seg.size == 0:
            continue
        if seg.sum() < 0.025 * seg.shape[0] * seg.shape[1] * 1.0:
            continue
        out_bboxes.append((x1 + sx1, y1, x1 + sx2, y2))

    return out_bboxes if out_bboxes else [roi_bbox]


def _patch_to_emnist_square(
    gray_roi: np.ndarray,
    mask_roi: np.ndarray,
    margin_ratio: float = 0.15,
) -> np.ndarray | None:
    """
    Tight crop around ink, pad to a square, resize to 28x28 float32 [0,1].
    Output matches tfds EMNIST: bright ink on dark background (see train_char_model._preprocess).
    """
    if gray_roi.size == 0 or mask_roi.size == 0:
        return None
    ys, xs = np.where(mask_roi > 0)
    if xs.size == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop_g = gray_roi[y0 : y1 + 1, x0 : x1 + 1]
    crop_m = mask_roi[y0 : y1 + 1, x0 : x1 + 1]

    ch, cw = crop_g.shape[:2]
    side = max(ch, cw)
    margin = max(2, int(round(side * margin_ratio)))
    side_p = side + 2 * margin

    # Center crop in square of side_p
    out = np.zeros((side_p, side_p), dtype=np.uint8)
    ox = (side_p - cw) // 2
    oy = (side_p - ch) // 2
    out[oy : oy + ch, ox : ox + cw] = crop_g

    # Match tfds EMNIST: light ink on dark background (same as train_char_model — divide by 255, no invert).
    # Transpose maps upright canvas strokes to tfds-native orientation (see emnist dataset_info in tfds).
    resized = cv2.resize(out, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    resized = resized.T
    return np.clip(resized.astype(np.float32) / 255.0, 0.0, 1.0)


def _tta_prob_vector(model: keras.Model, square28: np.ndarray) -> np.ndarray:
    """Average softmax over small rotations (test-time augmentation)."""
    base = square28.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
    mats = [base]
    for deg in (-4.0, 4.0):
        M = cv2.getRotationMatrix2D((IMAGE_SIZE / 2.0 - 0.5, IMAGE_SIZE / 2.0 - 0.5), deg, 1.0)
        rot = cv2.warpAffine(
            square28,
            M,
            (IMAGE_SIZE, IMAGE_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        mats.append(rot.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1))
    batch = np.concatenate(mats, axis=0)
    probs = model.predict(batch, verbose=0)
    return np.mean(probs, axis=0)


def _classify_char_roi(gray_u8: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict:
    _ensure_model_loaded()
    if model is None or id_to_char is None:
        return {"char": "", "confidence": 0.0, "error": "OCR char model not loaded"}

    x1, y1, x2, y2 = bbox
    roi_g = gray_u8[y1:y2, x1:x2]
    if roi_g.size == 0:
        return {"char": "", "confidence": 0.0}

    mask_roi = strokes_mask_from_gray(roi_g)

    sq = _patch_to_emnist_square(roi_g, mask_roi)
    if sq is None:
        sq = cv2.resize(roi_g, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        sq = sq.T
        sq = np.clip(sq, 0.0, 1.0)

    probs = _tta_prob_vector(model, sq)
    cls = int(np.argmax(probs))
    conf = float(np.max(probs)) * 100.0
    ch = id_to_char.get(cls, "")
    return {"char": ch, "confidence": round(conf, 2)}


def predict_textlines_roi(gray_u8: np.ndarray) -> Dict:
    try:
        _ensure_model_loaded()
        if model is None or id_to_char is None:
            hint = (
                "Char model not loaded. "
                "If model/alphabet_char_cnn.keras exists, upgrade TensorFlow to >=2.15 "
                "(Keras 3 .keras zips are not loadable on TF 2.10). Otherwise run train_char_model.py."
            )
            return {
                "prediction_lines": [],
                "prediction_text": "",
                "lines": [],
                "error": hint,
            }

        if gray_u8.ndim != 2:
            return {"prediction_lines": [], "prediction_text": "", "lines": [], "error": "Invalid input image"}

        mask = strokes_mask_from_gray(gray_u8)
        bboxes = _find_initial_bboxes(mask)
        if not bboxes:
            return {"prediction_lines": [], "prediction_text": "", "lines": [], "error": "No characters detected"}

        line_groups = _group_bboxes_into_lines(bboxes)
        lines_out = []

        for line_bboxes in line_groups:
            refined: List[Tuple[int, int, int, int]] = []
            for bb in line_bboxes:
                x1, y1, x2, y2 = bb
                w = x2 - x1
                h = y2 - y1
                is_merged = (w >= 38) or (w / max(1, h) >= 1.65)
                if is_merged:
                    refined.extend(_split_merged_roi(mask, bb))
                else:
                    refined.append(bb)

            refined.sort(key=lambda b: b[0])

            chars = []
            confs = []
            for bb in refined:
                pred = _classify_char_roi(gray_u8, bb)
                if pred.get("error"):
                    continue
                if pred["char"] == "":
                    continue
                chars.append(pred["char"])
                confs.append(float(pred["confidence"]))

            if not chars:
                continue

            text = "".join(chars)
            avg_conf = round(float(np.mean(confs)) if confs else 0.0, 2)
            lines_out.append({"text": text, "confidence": avg_conf})

        if not lines_out:
            return {
                "prediction_lines": [],
                "prediction_text": "",
                "lines": [],
                "error": "OCR could not recognize any characters yet.",
            }

        prediction_lines = [l["text"] for l in lines_out]
        prediction_text = "\n".join(prediction_lines)
        return {"prediction_lines": prediction_lines, "prediction_text": prediction_text, "lines": lines_out}
    except Exception as e:
        return {"prediction_lines": [], "prediction_text": "", "lines": [], "error": str(e)}
