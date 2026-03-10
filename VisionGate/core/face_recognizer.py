"""
core/face_recognizer.py
========================
Wraps OpenCV LBPH face recogniser.
Confidence is a distance metric (lower = better match).
We expose a normalised match_percent for the rest of the system.
"""

import os
import cv2
import numpy as np
import logging

import config

logger = logging.getLogger(__name__)


def _confidence_to_percent(confidence: float) -> float:
    """
    Convert LBPH distance to a 0–100 % match score.
    LBPH confidence 0   → 100 %   (perfect match)
    LBPH confidence 100 → 0   %   (no match)
    Values above 100 are clamped to 0 %.
    """
    return max(0.0, min(100.0, 100.0 - confidence))


class FaceRecognizer:
    """LBPH face recogniser with load/predict/update capabilities."""

    def __init__(self):
        self._model: cv2.face.LBPHFaceRecognizer | None = None
        self._trained = False
        self._model_path = config.MODEL_PATH

    # ── Public interface ──────────────────────────────────────────────────────

    def load(self) -> bool:
        """
        Load a previously trained model from disk.
        Returns True if successful, False if model file doesn't exist.
        """
        if not os.path.isfile(self._model_path):
            logger.warning(
                f"FaceRecognizer: model not found at '{self._model_path}'. "
                "Run training/train_model.py first."
            )
            return False

        self._model = cv2.face.LBPHFaceRecognizer_create()
        self._model.read(self._model_path)
        self._trained = True
        logger.info(f"FaceRecognizer: model loaded from {self._model_path}")
        return True

    def predict(
        self, face_roi: np.ndarray
    ) -> tuple[int, float]:
        """
        Predict user ID and match percentage from a greyscale face ROI.

        Returns:
            (user_id, match_percent)
            user_id      = -1          if model not trained or prediction failed
            match_percent = 0.0–100.0
        """
        if not self._trained or self._model is None:
            return -1, 0.0

        try:
            label, confidence = self._model.predict(face_roi)
            match_pct = _confidence_to_percent(confidence)
            logger.debug(
                f"FaceRecognizer: label={label} confidence={confidence:.1f} "
                f"match={match_pct:.1f}%"
            )
            return label, match_pct
        except cv2.error as exc:
            logger.error(f"FaceRecognizer.predict error: {exc}")
            return -1, 0.0

    def is_trained(self) -> bool:
        return self._trained

    # ── Training helpers (used by train_model.py) ─────────────────────────────

    @staticmethod
    def train(
        face_images: list[np.ndarray],
        labels: list[int],
        save_path: str = config.MODEL_PATH,
    ) -> None:
        """
        Train a new LBPH model and save it to disk.

        Args:
            face_images : list of greyscale face numpy arrays
            labels      : corresponding integer user labels
            save_path   : path to save the .yml model file
        """
        if len(face_images) == 0:
            raise ValueError("No face images provided for training.")

        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(face_images, np.array(labels))

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        logger.info(f"FaceRecognizer: model trained on {len(face_images)} images, "
                    f"saved to {save_path}")

    @staticmethod
    def update(
        face_images: list[np.ndarray],
        labels: list[int],
        model_path: str = config.MODEL_PATH,
    ) -> None:
        """
        Update an existing LBPH model with new face samples (incremental learning).
        If no model exists, creates a new one.
        """
        if not os.path.isfile(model_path):
            FaceRecognizer.train(face_images, labels, model_path)
            return

        model = cv2.face.LBPHFaceRecognizer_create()
        model.read(model_path)
        model.update(face_images, np.array(labels))
        model.save(model_path)
        logger.info(
            f"FaceRecognizer: model updated with {len(face_images)} new samples."
        )
