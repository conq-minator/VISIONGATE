"""
core/frame_pipeline.py
======================
Pulls frames from the shared camera queue and chains them through:
  face detection → recognition → QR scan → decision → overlay → hardware
Uses a pool of worker threads so no single thread blocks the pipeline.
"""

import threading
import queue
import logging
import time
from collections import defaultdict

import numpy as np

import config
from core.face_detector   import FaceDetector
from core.face_recognizer import FaceRecognizer
from core.qr_scanner      import QRScanner
from core.decision_engine import DecisionEngine, Decision, Command
from database.db          import get_connection
from database.face_storage import log_access
from hardware.arduino_comm import ArduinoComm
from interface.status_overlay import draw_overlay

logger = logging.getLogger(__name__)


class ProcessedFrame:
    """Holds a fully-processed frame ready for display."""

    __slots__ = ("camera_id", "frame", "result", "fps")

    def __init__(self, camera_id: int, frame: np.ndarray, result, fps: float = 0.0):
        self.camera_id = camera_id
        self.frame     = frame
        self.result    = result
        self.fps       = fps


class FramePipeline:
    """
    Orchestrates the processing pipeline across multiple worker threads.
    One output queue (display_queue) is filled with ProcessedFrame objects
    for the main thread's display loop.
    """

    def __init__(
        self,
        frame_queue  : queue.Queue,
        camera_manager,                   # CameraManager for FPS data
        db_lookup_fn,                     # callable(user_id) → (name, qr_code)|None
        num_workers  : int = config.NUM_WORKER_THREADS,
    ):
        self._frame_queue   = frame_queue
        self._camera_manager = camera_manager
        self._db_lookup     = db_lookup_fn
        self._num_workers   = num_workers

        # Shared output queue (camera_id → ProcessedFrame)
        self.display_queue: queue.Queue = queue.Queue(maxsize=20)

        # Per-camera state (created lazily)
        self._detectors   : dict[int, FaceDetector]   = {}
        self._recognizers : dict[int, FaceRecognizer]  = {}
        self._qr_scanners : dict[int, QRScanner]       = {}
        self._engines     : dict[int, DecisionEngine]  = {}
        self._lock        = threading.Lock()

        self._running     = threading.Event()
        self._threads     : list[threading.Thread] = []

        # Shared recogniser (LBPH model is read-only after load)
        self._recognizer  = FaceRecognizer()
        self._model_loaded = self._recognizer.load()

        self._hw = ArduinoComm()

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self):
        """Start worker threads."""
        self._running.set()
        for i in range(self._num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"Pipeline-Worker-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
        logger.info(f"FramePipeline: {self._num_workers} worker thread(s) started.")

    def stop(self):
        """Signal all workers to stop."""
        self._running.clear()
        for t in self._threads:
            t.join(timeout=2.0)
        logger.info("FramePipeline: stopped.")

    # ── Worker loop ───────────────────────────────────────────────────────────

    def _worker_loop(self):
        while self._running.is_set():
            try:
                camera_id, frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                processed = self._process(camera_id, frame)
                try:
                    self.display_queue.put_nowait(processed)
                except queue.Full:
                    # Display is slower than processing – drop oldest frame
                    try:
                        self.display_queue.get_nowait()
                        self.display_queue.put_nowait(processed)
                    except queue.Empty:
                        pass
            except Exception as exc:
                logger.exception(f"Pipeline worker error (cam {camera_id}): {exc}")
            finally:
                self._frame_queue.task_done()

    # ── Core processing chain ─────────────────────────────────────────────────

    def _process(self, camera_id: int, frame: np.ndarray) -> "ProcessedFrame":
        detector = self._get_detector(camera_id)
        engine   = self._get_engine(camera_id)
        qr       = self._get_qr(camera_id)

        fps = self._camera_manager.get_fps(camera_id)

        # 1. QR scan (rate-limited internally)
        qr_user_id = qr.scan(frame)

        # 2. Face detection
        faces = detector.detect(frame)

        if not faces:
            result = engine.evaluate_no_face()
            annotated = draw_overlay(
                frame.copy(), camera_id, result, fps,
                faces=[], qr_user_id=qr_user_id,
            )
            return ProcessedFrame(camera_id, annotated, result, fps)

        # 3. Pick largest face for recognition
        largest = max(faces, key=lambda b: b[2] * b[3])
        roi = detector.extract_roi(frame, largest)

        # 4. Recognition
        user_id, match_pct = self._recognizer.predict(roi)

        # 5. DB lookup
        user_name = "Unknown"
        db_has_qr = False
        if user_id != -1:
            user_data = self._db_lookup(user_id)
            if user_data:
                user_name, qr_code = user_data
                db_has_qr = bool(qr_code)

        # 6. Decision
        result = engine.evaluate(user_id, match_pct, qr_user_id, db_has_qr)
        result.user_id = user_id  # may be -1 if model unloaded

        # 7. Hardware command
        if result.command == Command.OPEN_GATE:
            self._hw.open_gate()
        elif result.command == Command.BUZZER_ON:
            self._hw.buzzer_on()

        # 8. Log to DB (non-blocking – use fire-and-forget thread)
        if result.decision not in (Decision.IDLE, Decision.REQUIRE_QR):
            threading.Thread(
                target=self._log_access,
                args=(camera_id, user_id, result.decision.value),
                daemon=True,
            ).start()

        # 9. Learning mode – capture additional samples
        if result.learning:
            threading.Thread(
                target=self._capture_learning_sample,
                args=(user_id, roi),
                daemon=True,
            ).start()

        # 10. Draw overlay
        annotated = draw_overlay(
            frame.copy(), camera_id, result, fps,
            faces=faces, qr_user_id=qr_user_id,
            user_name=user_name,
        )

        return ProcessedFrame(camera_id, annotated, result, fps)

    # ── Per-camera component factories ────────────────────────────────────────

    def _get_detector(self, cam_id: int) -> FaceDetector:
        with self._lock:
            if cam_id not in self._detectors:
                self._detectors[cam_id] = FaceDetector()
            return self._detectors[cam_id]

    def _get_engine(self, cam_id: int) -> DecisionEngine:
        with self._lock:
            if cam_id not in self._engines:
                self._engines[cam_id] = DecisionEngine(cam_id)
            return self._engines[cam_id]

    def _get_qr(self, cam_id: int) -> QRScanner:
        with self._lock:
            if cam_id not in self._qr_scanners:
                self._qr_scanners[cam_id] = QRScanner()
            return self._qr_scanners[cam_id]

    # ── Side-effect helpers ───────────────────────────────────────────────────

    @staticmethod
    def _log_access(camera_id: int, user_id: int, decision: str):
        try:
            log_access(camera_id, user_id, decision)
        except Exception as exc:
            logger.error(f"DB log error: {exc}")

    @staticmethod
    def _capture_learning_sample(user_id: int, roi: np.ndarray):
        """Save a face ROI to disk for future re-training."""
        import os, cv2, time
        user_dir = os.path.join(config.FACES_DIR, f"user_{user_id}")
        os.makedirs(user_dir, exist_ok=True)
        filename = os.path.join(user_dir, f"learn_{int(time.time()*1000)}.jpg")
        cv2.imwrite(filename, roi)
        logger.info(f"Learning sample saved: {filename}")
