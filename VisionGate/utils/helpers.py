"""
utils/helpers.py
================
Shared utility functions used across modules.
"""

import time
import os
import cv2
import numpy as np
import imutils
import logging

logger = logging.getLogger(__name__)


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    """Resize frame to target width, preserving aspect ratio."""
    if frame.shape[1] <= width:
        return frame
    return imutils.resize(frame, width=width)


def get_timestamp() -> str:
    """Return current time as a formatted string."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_confidence(confidence: float) -> float:
    """
    Normalise an LBPH confidence value (lower=better) to a 0–100% match score.
    """
    return max(0.0, min(100.0, 100.0 - confidence))


def load_face_images(faces_dir: str) -> tuple[list[np.ndarray], list[int]]:
    """
    Walk *faces_dir* and load all face images.

    Expected layout:
        faces_dir/
            user_1/  ← folder name determines label
                001.jpg
                002.jpg
            user_2/
                ...

    Returns:
        (face_images, labels)
    """
    face_images: list[np.ndarray] = []
    labels      : list[int]       = []

    if not os.path.isdir(faces_dir):
        logger.warning(f"Faces directory not found: {faces_dir}")
        return face_images, labels

    for user_folder in sorted(os.listdir(faces_dir)):
        folder_path = os.path.join(faces_dir, user_folder)
        if not os.path.isdir(folder_path):
            continue

        # Extract numeric user ID from folder name (e.g. "user_3" → 3)
        try:
            user_id = int(user_folder.split("_")[-1])
        except ValueError:
            logger.warning(f"Cannot parse user ID from folder: {user_folder}")
            continue

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            face_images.append(img)
            labels.append(user_id)

    logger.info(f"Loaded {len(face_images)} face images for {len(set(labels))} users.")
    return face_images, labels


def ensure_dirs(*paths: str):
    """Create directories if they don't exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)
