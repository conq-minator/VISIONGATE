"""
training/train_model.py
=======================
Trains an LBPH face recogniser from collected samples.

Usage:
    python training/train_model.py

Reads all images from data/faces/, trains the model, and saves to data/face_model.yml.
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


def train():
    print("\n=== VisionGate – Model Training ===")
    print(f"Loading face images from: {config.FACES_DIR}")

    face_images, labels = load_face_images(config.FACES_DIR)

    if len(face_images) == 0:
        print(
            "No face images found. Run training/collect_faces.py first."
        )
        return

    unique_users = len(set(labels))
    print(f"Found {len(face_images)} images for {unique_users} user(s).")
    print("Training LBPH model…")

    FaceRecognizer.train(face_images, labels, config.MODEL_PATH)

    print(f"Model saved to: {config.MODEL_PATH}")
    print("Training complete.")


if __name__ == "__main__":
    train()
