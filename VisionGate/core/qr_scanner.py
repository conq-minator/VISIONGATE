"""
core/qr_scanner.py
==================
Scans frames for QR codes using pyzbar.
Decodes payload and maps to a user ID string.
"""

import cv2
import numpy as np
import logging

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "pyzbar not installed – QR scanning disabled. "
        "Install with: pip install pyzbar"
    )

import config

logger = logging.getLogger(__name__)


class QRScanner:
    """
    Decodes QR codes from camera frames.

    The QR code payload is expected to be a plain user ID string, e.g. '42'.
    Any other format is treated as invalid.
    """

    def __init__(self):
        self._frame_count = 0
        self._available   = PYZBAR_AVAILABLE

    def scan(self, frame: np.ndarray) -> str | None:
        """
        Attempt to decode a QR code in *frame*.

        Only actually scans every QR_SCAN_INTERVAL frames to reduce CPU load.

        Returns:
            Decoded user-ID string if a valid QR code is found, else None.
        """
        self._frame_count += 1
        if self._frame_count % config.QR_SCAN_INTERVAL != 0:
            return None

        if not self._available:
            return None

        gray = self._to_gray(frame)
        decoded = pyzbar.decode(gray)

        for obj in decoded:
            payload = obj.data.decode("utf-8", errors="ignore").strip()
            if payload.isdigit():
                logger.info(f"QRScanner: QR detected, user_id={payload}")
                return payload
            else:
                logger.debug(f"QRScanner: non-numeric QR payload ignored: '{payload}'")

        return None

    def scan_all(self, frame: np.ndarray) -> list[str]:
        """
        Scan and return ALL valid QR user IDs in the frame (no rate limiting).
        Used by training/verification scripts.
        """
        if not self._available:
            return []

        gray = self._to_gray(frame)
        decoded = pyzbar.decode(gray)
        results = []
        for obj in decoded:
            payload = obj.data.decode("utf-8", errors="ignore").strip()
            if payload.isdigit():
                results.append(payload)
        return results

    def get_barcodes(self, frame: np.ndarray):
        """
        Return raw pyzbar decoded objects for drawing bounding boxes.
        Returns empty list if pyzbar unavailable.
        """
        if not self._available:
            return []
        gray = self._to_gray(frame)
        return pyzbar.decode(gray)

    # ── Helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if len(frame.shape) == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
