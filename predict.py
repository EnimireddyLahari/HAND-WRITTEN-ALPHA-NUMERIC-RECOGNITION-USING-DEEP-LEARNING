"""
predict.py
==========
Standalone prediction helper.

Loads the trained CNN model and exposes two functions:
  • predict_single(img_array)  → single-digit prediction
  • predict_multi(img_array)   → multi-digit prediction via contour segmentation

Both functions accept a NumPy array (H × W, uint8, 0–255 grayscale).
"""

import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from ocr_model import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    decode_ctc_greedy,
    load_vocab,
)

from roi_classifier import predict_textlines_roi, strokes_mask_from_gray

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "model")

# ─────────────────────────────────────────────
# 1. LOAD MODEL (once, at import time)
# ─────────────────────────────────────────────

MODEL_PATH = os.path.join(_MODEL_DIR, "alphabet_ctc_crnn.keras")

model = None
_id_to_char = None


def _try_load_model() -> None:
    """
    Reload OCR model when the training process creates it.
    This prevents you needing to restart the Flask server.
    """
    global model, _id_to_char

    if model is not None and _id_to_char is not None:
        return

    if not os.path.exists(MODEL_PATH):
        return

    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        _, _id_to_char = load_vocab(_MODEL_DIR)
        print(f"[INFO] OCR model loaded from {MODEL_PATH}")
    except Exception as e:
        model = None
        _id_to_char = None
        print(f"[WARN] Could not load OCR model yet: {e}")
        msg = str(e).lower()
        if "signature not found" in msg or "file signature" in msg:
            print(
                "[HINT] These .keras files are Keras 3 (zip) saves. "
                "TensorFlow 2.10 cannot load them. Upgrade:  "
                "pip install -U \"tensorflow>=2.15.0,<2.18.0\"  "
                "(then restart this app)."
            )


# Best-effort initial load (may fail during training).
_try_load_model()


# ─────────────────────────────────────────────
# 2. IMAGE PREPROCESSING HELPERS
# ─────────────────────────────────────────────

def preprocess_digit(roi: np.ndarray) -> np.ndarray:
    """
    Prepare a single-digit ROI (region-of-interest) for the CNN.

    Steps:
      1. Resize to 28 × 28
      2. Normalize pixels to [0, 1]
      3. Reshape to (1, 28, 28, 1)
    """
    resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, 28, 28, 1)


# ─────────────────────────────────────────────
# 3. SINGLE-DIGIT PREDICTION
# ─────────────────────────────────────────────

def predict_single(img_array: np.ndarray) -> dict:
    """
    Predict a single digit from a grayscale image array.

    Parameters
    ----------
    img_array : np.ndarray
        Grayscale image (any size, uint8, 0-255).

    Returns
    -------
    dict : {"digit": int, "confidence": float}
    """
    return {
        "digit": -1,
        "confidence": 0.0,
        "error": "This project now uses ASCII62 line OCR. Use predict_textlines().",
    }


# ─────────────────────────────────────────────
# 4. MULTI-DIGIT PREDICTION
# ─────────────────────────────────────────────

def segment_digits(gray: np.ndarray) -> list:
    """
    Detect individual digit regions using contour analysis.

    Returns a list of (x, y, w, h) bounding boxes sorted left → right.
    """
    # Threshold: white digits on black background (MNIST style)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Morphological dilation helps connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Ignore very small noise contours
        if w > 10 and h > 10:
            bboxes.append((x, y, w, h))

    # Sort left to right by x-coordinate
    bboxes.sort(key=lambda b: b[0])
    return bboxes


def predict_multi(img_array: np.ndarray) -> dict:
    """
    Predict multiple digits from a grayscale image.

    Segments each digit via contours, predicts individually,
    and concatenates the results.

    Parameters
    ----------
    img_array : np.ndarray
        Grayscale image (any size, uint8, 0-255).

    Returns
    -------
    dict : {"prediction": str, "digits": list[dict]}
    """
    # Backwards-compatible wrapper: old name now returns the new OCR contract.
    return predict_textlines(img_array)


def detect_text_line_bboxes(gray_u8: np.ndarray) -> list:
    """
    Detect text lines using horizontal projection.
    Returns bboxes (x1, y1, x2, y2) sorted top->bottom.
    """
    mask = strokes_mask_from_gray(gray_u8)

    row_sum = mask.sum(axis=1)
    max_row = float(row_sum.max()) if row_sum.size else 0.0
    if max_row <= 0:
        return []

    row_thr = max(20.0, 0.2 * max_row)
    active_rows = row_sum > row_thr

    ranges = []
    start = None
    for y, is_active in enumerate(active_rows.tolist()):
        if is_active and start is None:
            start = y
        elif (not is_active) and start is not None:
            ranges.append((start, y - 1))
            start = None
    if start is not None:
        ranges.append((start, len(active_rows) - 1))

    # Merge near ranges.
    merged = []
    gap_allow = 3
    for (y1, y2) in ranges:
        if not merged:
            merged.append((y1, y2))
            continue
        prev_y1, prev_y2 = merged[-1]
        if y1 - prev_y2 <= gap_allow:
            merged[-1] = (prev_y1, y2)
        else:
            merged.append((y1, y2))

    H, W = gray_u8.shape[:2]
    bboxes = []
    pad_y = 4
    pad_x = 6

    for (y1, y2) in merged:
        y1p = max(0, y1 - pad_y)
        y2p = min(H - 1, y2 + pad_y)

        line_mask = mask[y1p : y2p + 1, :]
        col_sum = line_mask.sum(axis=0)
        max_col = float(col_sum.max()) if col_sum.size else 0.0
        if max_col <= 0:
            continue

        col_thr = 0.15 * max_col
        active_cols = col_sum > col_thr
        xs = np.where(active_cols)[0]
        if xs.size == 0:
            continue

        x1p = max(0, int(xs.min()) - pad_x)
        x2p = min(W - 1, int(xs.max()) + pad_x)

        h = y2p - y1p + 1
        w = x2p - x1p + 1
        if h < 10 or w < 15:
            continue

        bboxes.append((x1p, y1p, x2p, y2p))

    bboxes.sort(key=lambda b: b[1])
    return bboxes


def _prepare_line_for_model(gray_u8: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Crop line bbox and convert to a fixed (32x280x1) float32 tensor.
    """
    if model is None:
        raise RuntimeError("OCR model not loaded")

    x1, y1, x2, y2 = bbox
    crop = gray_u8[y1 : y2 + 1, x1 : x2 + 1]
    crop = crop.astype(np.uint8)

    mask = strokes_mask_from_gray(crop).astype(np.float32)  # strokes=1
    ch, cw = mask.shape[:2]
    if ch <= 0 or cw <= 0:
        raise ValueError("Invalid crop")

    # Resize to fixed height, preserving aspect ratio for width.
    scale = IMAGE_HEIGHT / float(ch)
    new_w = max(1, int(round(cw * scale)))

    if new_w > IMAGE_WIDTH:
        # Scale down so width fits.
        scale = IMAGE_WIDTH / float(cw)
        new_h = max(1, int(round(ch * scale)))
        mask = cv2.resize(mask, (IMAGE_WIDTH, new_h), interpolation=cv2.INTER_NEAREST)
        if new_h < IMAGE_HEIGHT:
            top = (IMAGE_HEIGHT - new_h) // 2
            out = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
            out[top : top + new_h, :] = mask
            mask = out
        else:
            mask = mask[:IMAGE_HEIGHT, :]
    else:
        mask = cv2.resize(mask, (new_w, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
        out = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
        out[:, :new_w] = mask
        mask = out

    return mask.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 1).astype(np.float32)


def predict_textlines(gray_u8: np.ndarray) -> dict:
    """
    Predict multi-line ASCII62 text from a canvas grayscale image.
    Input should have bright strokes (app inverts before calling).

    Per-character ROI + EMNIST-style CNN is tried first (best letter accuracy);
    CRNN+CTC is used when ROI finds no usable text.
    """
    try:
        if gray_u8.ndim != 2:
            return {"prediction_lines": [], "prediction_text": "", "lines": [], "error": "Invalid input image"}

        roi_first = predict_textlines_roi(gray_u8)
        if roi_first.get("lines"):
            out = dict(roi_first)
            out["engine"] = "char_roi"
            return out

        _try_load_model()
        if model is None or _id_to_char is None:
            err = roi_first.get("error", "ROI path did not produce lines.")
            return {
                "prediction_lines": [],
                "prediction_text": "",
                "lines": [],
                "error": "OCR CTC model not loaded. ROI: " + err,
            }

        bboxes = detect_text_line_bboxes(gray_u8)
        if not bboxes:
            return {
                "prediction_lines": [],
                "prediction_text": "",
                "lines": [],
                "error": "No text lines for CTC. Earlier ROI: " + roi_first.get("error", "no lines"),
            }

        time_steps = IMAGE_WIDTH // 4
        line_tensors = [_prepare_line_for_model(gray_u8, bb) for bb in bboxes]
        x = np.stack(line_tensors, axis=0)  # (n_lines,32,280,1)

        y_pred = model.predict(x, verbose=0)  # (n_lines,time,vocab)
        decoded_texts, confidences = decode_ctc_greedy(y_pred, _id_to_char, time_steps=time_steps)

        lines = []
        for text, conf in zip(decoded_texts, confidences.tolist()):
            lines.append({"text": text, "confidence": round(float(conf), 2)})

        # If the model decodes only blanks, return a helpful error.
        any_text = any((li.get("text") or "").strip() != "" for li in lines)
        if not any_text:
            return {
                "prediction_lines": [],
                "prediction_text": "",
                "lines": [],
                "error": "CTC decoded blanks. Earlier ROI: " + roi_first.get("error", "no lines"),
            }

        prediction_lines = [li["text"] for li in lines]
        prediction_text = "\n".join(prediction_lines)
        return {
            "prediction_lines": prediction_lines,
            "prediction_text": prediction_text,
            "lines": lines,
            "engine": "ctc_crnn",
        }
    except Exception as e:
        return {"prediction_lines": [], "prediction_text": "", "lines": [], "error": str(e)}


# ─────────────────────────────────────────────
# QUICK CLI TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Could not read image: {path}")
        sys.exit(1)

    result = predict_textlines(img)
    print(f"\nPrediction text:\n{result.get('prediction_text','')}")
    for i, line in enumerate(result.get("lines", [])):
        print(f"  Line {i+1}: {line['text']}  (confidence: {line['confidence']}%)")
