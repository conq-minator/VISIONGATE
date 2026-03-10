"""
core/camera_manager.py
======================
Manages detection, initialisation and threaded capture of multiple cameras.
Each camera runs in its own daemon thread.  Frames are pushed into per-camera
queues consumed by the processing pipeline.
"""

import threading
import time
import queue
import cv2
import logging

import config

logger = logging.getLogger(__name__)


class CameraCapture:
    """Represents a single camera stream running in its own thread."""

    def __init__(self, camera_id: int, frame_queue: queue.Queue):
        self.camera_id   = camera_id
        self.frame_queue = frame_queue
        self._cap        = None
        self._thread     = None
        self._running    = threading.Event()
        self.fps         = 0.0

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self) -> bool:
        """Open camera and start capture thread. Returns True on success."""
        self._cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Retry without backend hint (Linux / older OpenCV)
            self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            logger.error(f"Camera {self.camera_id}: failed to open.")
            return False

        # Apply resolution settings
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAPTURE_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          config.CAPTURE_FPS)

        self._running.set()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"CamCapture-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Camera {self.camera_id}: capture thread started.")
        return True

    def stop(self):
        """Signal thread to stop and release camera resource."""
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        logger.info(f"Camera {self.camera_id}: stopped.")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _capture_loop(self):
        prev_time = time.time()
        frame_count = 0

        while self._running.is_set():
            ret, frame = self._cap.read()
            if not ret:
                logger.warning(f"Camera {self.camera_id}: read failure – retrying…")
                time.sleep(0.05)
                continue

            # Frame skipping: discard if queue is nearly full (CPU protection)
            if self.frame_queue.qsize() >= int(
                config.FRAME_QUEUE_MAXSIZE * config.SKIP_THRESHOLD_RATIO
            ):
                continue

            try:
                self.frame_queue.put_nowait((self.camera_id, frame))
            except queue.Full:
                pass  # drop frame silently

            # Rolling FPS calculation
            frame_count += 1
            now = time.time()
            if now - prev_time >= 2.0:
                self.fps = frame_count / (now - prev_time)
                frame_count = 0
                prev_time = now


class CameraManager:
    """Detects and manages up to MAX_CAMERAS camera streams."""

    def __init__(self):
        self._cameras: dict[int, CameraCapture] = {}
        # Shared frame queue consumed by the pipeline workers
        self.frame_queue: queue.Queue = queue.Queue(
            maxsize=config.FRAME_QUEUE_MAXSIZE * config.MAX_CAMERAS
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self, camera_ids: list[int] | None = None) -> list[int]:
        """
        Start cameras for the given IDs (defaults to config.CAMERA_IDS).
        Returns the list of successfully opened camera IDs.
        """
        ids = camera_ids if camera_ids is not None else config.CAMERA_IDS
        started = []

        for cam_id in ids[:config.MAX_CAMERAS]:
            cap = CameraCapture(cam_id, self.frame_queue)
            if cap.start():
                self._cameras[cam_id] = cap
                started.append(cam_id)
            else:
                logger.warning(f"Skipping camera {cam_id} (not available).")

        logger.info(f"CameraManager: {len(started)} camera(s) active: {started}")
        return started

    def stop(self):
        """Stop all camera threads."""
        for cam in self._cameras.values():
            cam.stop()
        self._cameras.clear()
        logger.info("CameraManager: all cameras stopped.")

    def active_cameras(self) -> list[int]:
        return list(self._cameras.keys())

    def get_fps(self, camera_id: int) -> float:
        cam = self._cameras.get(camera_id)
        return cam.fps if cam else 0.0

    # ── Auto-detect helper ────────────────────────────────────────────────────

    @staticmethod
    def detect_available_cameras(max_test: int = config.MAX_CAMERAS) -> list[int]:
        """
        Probe camera indices 0…max_test-1 and return those that open.
        Useful for setup scripts.
        """
        available = []
        for idx in range(max_test):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                available.append(idx)
                cap.release()
        return available
