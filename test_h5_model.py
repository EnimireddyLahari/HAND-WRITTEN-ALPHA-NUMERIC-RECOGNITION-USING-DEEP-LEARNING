#!/usr/bin/env python
"""Test loading H5 model instead"""

import os
import json
import numpy as np
from tensorflow import keras

print("[1] Attempting to load digit_model.h5...")
try:
    model = keras.models.load_model('model/digit_model.h5')
    print("  ✓ H5 Model loaded successfully!")
    print(f"    Input shape: {model.input_shape}")
    print(f"    Output shape: {model.output_shape}")
    model.summary()
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n[2] Loading vocabularies...")
with open('model/vocab_char.json', 'r') as f:
    char_vocab = json.load(f)['alphabet']
print(f"  ✓ Char alphabet: {char_vocab}")

print("\n[3] Testing prediction on random image...")
try:
    test_image = np.random.rand(1, 28, 28, 1).astype(np.float32)
    prediction = model.predict(test_image, verbose=0)
    predicted_idx = np.argmax(prediction[0])
    predicted_char = char_vocab[predicted_idx]
    confidence = prediction[0][predicted_idx] * 100
    print(f"  ✓ Predicted: '{predicted_char}' (confidence: {confidence:.2f}%)")
except Exception as e:
    print(f"  ✗ Error: {e}")
