"""
core/face_detector.py
=====================
Lightweight face detector using OpenCV Haar Cascade.
Applies grayscale conversion and optional downscaling for speed.
"""

import os
import cv2
import numpy as np
import logging
import imutils

import config

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detects faces in a frame using Haar Cascade classifier."""

    def __init__(self, cascade_path: str | None = None):
        path = cascade_path or config.CASCADE_PATH

        # If the user-supplied path doesn't exist, fall back to OpenCV's bundled file
        if not os.path.isfile(path):
            bundled = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.isfile(bundled):
                path = bundled
                logger.info(f"FaceDetector: using bundled cascade: {bundled}")
            else:
                raise FileNotFoundError(
                    f"Haar cascade not found at '{path}' or bundled location."
                )

        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError("Failed to load Haar cascade classifier.")
        logger.info(f"FaceDetector: cascade loaded from {path}")

    # ── Public interface ──────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect faces in *frame*.

        Returns:
            List of (x, y, w, h) bounding boxes in original frame coordinates.
        """
        gray = self._to_gray(frame)
        small, scale = self._downscale(gray)

        rects = self._cascade.detectMultiScale(
            small,
            scaleFactor=config.SCALE_FACTOR,
            minNeighbors=config.MIN_NEIGHBORS,
            minSize=config.MIN_FACE_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(rects) == 0:
            return []

        # Scale bounding boxes back to original resolution
        faces = []
        for (x, y, w, h) in rects:
            ox = int(x * scale)
            oy = int(y * scale)
            ow = int(w * scale)
            oh = int(h * scale)
            faces.append((ox, oy, ow, oh))

        return faces

    def extract_roi(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract and normalise a face region-of-interest from *frame*.
        Returns a greyscale, resized image ready for the recogniser.
        """
        x, y, w, h = bbox
        # Guard against out-of-bounds slices
        fh, fw = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros(config.FACE_IMG_SIZE, dtype=np.uint8)

        gray_roi = self._to_gray(roi)
        resized  = cv2.resize(gray_roi, config.FACE_IMG_SIZE)
        return resized

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _downscale(gray: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Resize *gray* to DETECTION_RESIZE_WIDTH if it is wider.
        Returns (resized_frame, scale_factor).
        """
        target_w = config.DETECTION_RESIZE_WIDTH
        h, w = gray.shape[:2]
        if w <= target_w:
            return gray, 1.0
        scale = target_w / w
        small = imutils.resize(gray, width=target_w)
        return small, 1.0 / scale
