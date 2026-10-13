#!/usr/bin/env python
"""Test loading OCR CTC and Char models"""

import os
import json
import numpy as np
from tensorflow import keras

print("[1] Checking model files...")
model_dir = "model"
files = os.listdir(model_dir)
for f in files:
    path = os.path.join(model_dir, f)
    size = os.path.getsize(path)
    print(f"  - {f}: {size:,} bytes")

print("\n[2] Loading OCR CTC Model...")
try:
    ctc_model = keras.models.load_model('model/alphabet_ctc_crnn.keras')
    print("  ✓ OCR CTC Model loaded successfully")
    print(f"    Input shape: {ctc_model.input_shape}")
    print(f"    Output shape: {ctc_model.output_shape}")
except Exception as e:
    print(f"  ✗ Error loading CTC model: {e}")

print("\n[3] Loading ROI Char Model...")
try:
    char_model = keras.models.load_model('model/alphabet_char_cnn.keras')
    print("  ✓ Char Model loaded successfully")
    print(f"    Input shape: {char_model.input_shape}")
    print(f"    Output shape: {char_model.output_shape}")
except Exception as e:
    print(f"  ✗ Error loading Char model: {e}")

print("\n[4] Loading vocabularies...")
try:
    with open('model/vocab_char.json', 'r') as f:
        char_vocab = json.load(f)['alphabet']
    print(f"  ✓ Char vocabulary loaded: {char_vocab}")
except Exception as e:
    print(f"  ✗ Error loading vocab_char.json: {e}")

try:
    with open('model/vocab.json', 'r') as f:
        vocab = json.load(f)
    print(f"  ✓ CTC vocabulary loaded: {len(vocab)} classes")
except Exception as e:
    print(f"  ✗ Error loading vocab.json: {e}")

print("\n[5] Testing Char Model prediction...")
try:
    test_image = np.random.rand(1, 28, 28, 1).astype(np.float32)
    prediction = char_model.predict(test_image, verbose=0)
    predicted_idx = np.argmax(prediction[0])
    predicted_char = char_vocab[predicted_idx]
    confidence = prediction[0][predicted_idx] * 100
    print(f"  ✓ Prediction successful")
    print(f"    Predicted character: '{predicted_char}'")
    print(f"    Confidence: {confidence:.2f}%")
except Exception as e:
    print(f"  ✗ Error making prediction: {e}")

print("\n✓ Test completed!")
