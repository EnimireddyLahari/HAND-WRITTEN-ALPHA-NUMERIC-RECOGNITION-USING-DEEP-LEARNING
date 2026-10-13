import numpy as np
import cv2
import tensorflow as tf

import predict
import ocr_model


def main():
    arr = 255 * np.ones((280, 280), dtype=np.uint8)
    # Draw a simple synthetic "A"-like shape (black strokes on white),
    # then invert to match the Flask pipeline (white strokes on black).
    cv2.line(arr, (100, 200), (140, 80), 0, 10)
    cv2.line(arr, (180, 200), (140, 80), 0, 10)
    cv2.line(arr, (110, 130), (170, 130), 0, 10)
    arr = cv2.bitwise_not(arr)

    bbs = predict.detect_text_line_bboxes(arr)
    print("bboxes:", bbs)
    if not bbs:
        print("no bboxes - exiting")
        return

    x = np.stack([predict._prepare_line_for_model(arr, bb) for bb in bbs], axis=0)
    y = predict.model.predict(x, verbose=0)
    print("y shape:", y.shape)

    # Inspect which class has the highest probability per timestep.
    max_idx = np.argmax(y[0], axis=-1)
    blank_idx = ocr_model.BLANK_INDEX
    blank_count = int(np.sum(max_idx == blank_idx))
    print("timestep max class counts:", {int(i): int(np.sum(max_idx == i)) for i in np.unique(max_idx)})
    print("blank max timesteps:", blank_count, "of", max_idx.shape[0])

    time_steps = ocr_model.IMAGE_WIDTH // 4
    input_len = np.full((y.shape[0],), time_steps, dtype=np.int32)
    decoded, _ = tf.keras.backend.ctc_decode(
        y,
        input_length=input_len,
        greedy=True,
        beam_width=None,
        top_paths=1,
    )
    seqs = decoded[0].numpy().tolist()
    print("seqs:", seqs)
    print("blank_index:", ocr_model.BLANK_INDEX)
    print("vocab_size:", ocr_model.VOCAB_SIZE)


if __name__ == "__main__":
    main()

