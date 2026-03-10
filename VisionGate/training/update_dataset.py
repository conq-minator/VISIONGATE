"""
training/update_dataset.py
==========================
Appends new face samples from the learning capture directory and
incrementally updates the existing LBPH model (no full re-train needed).

Usage:
    python training/update_dataset.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from utils.logger   import setup_logging
from utils.helpers  import load_face_images
from core.face_recognizer import FaceRecognizer

setup_logging()
import logging
logger = logging.getLogger(__name__)


def update():
    print("\n=== VisionGate – Incremental Dataset Update ===")
    print(f"Scanning for new learning samples in: {config.FACES_DIR}")

    face_images, labels = load_face_images(config.FACES_DIR)

    # Filter to only newly-captured learning images (prefix "learn_")
    new_images = []
    new_labels = []
    user_dirs  = [
        d for d in os.listdir(config.FACES_DIR)
        if os.path.isdir(os.path.join(config.FACES_DIR, d))
    ]

    import cv2

    for user_dir in user_dirs:
        try:
            user_id = int(user_dir.split("_")[-1])
        except ValueError:
            continue

        dir_path = os.path.join(config.FACES_DIR, user_dir)
        for f in os.listdir(dir_path):
            if f.startswith("learn_") and f.endswith(".jpg"):
                img = cv2.imread(os.path.join(dir_path, f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    new_images.append(img)
                    new_labels.append(user_id)

    if not new_images:
        print("No new learning samples found.")
        return

    print(f"Found {len(new_images)} new learning sample(s).")
    FaceRecognizer.update(new_images, new_labels, config.MODEL_PATH)
    print("Model updated successfully.")


if __name__ == "__main__":
    update()
