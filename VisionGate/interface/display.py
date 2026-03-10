"""
interface/display.py
====================
DisplayManager handles OpenCV windows for each camera.
Must run in the main thread (OpenCV GUI requirement).
"""

import cv2
import logging
import queue
from typing import Callable

import config

logger = logging.getLogger(__name__)


class DisplayManager:
    """
    Manages one named OpenCV window per camera.
    Call run() from the main thread.
    """

    def __init__(self, display_queue: queue.Queue):
        self._display_queue = display_queue
        self._windows       : set[int] = set()

    def run(self, stop_event=None):
        """
        Main display loop. Blocks until 'q' is pressed or stop_event is set.

        Args:
            stop_event: optional threading.Event; loop exits when set.
        """
        logger.info("Display loop started. Press 'q' to quit.")

        while True:
            # Check stop signal
            if stop_event and stop_event.is_set():
                break

            # Drain display queue
            messages_processed = 0
            while not self._display_queue.empty() and messages_processed < 8:
                try:
                    processed = self._display_queue.get_nowait()
                except queue.Empty:
                    break

                cam_id = processed.camera_id
                frame  = processed.frame

                win_name = f"VisionGate | Camera {cam_id}"
                if cam_id not in self._windows:
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(
                        win_name,
                        int(config.CAPTURE_WIDTH  * config.WINDOW_SCALE),
                        int(config.CAPTURE_HEIGHT * config.WINDOW_SCALE),
                    )
                    self._windows.add(cam_id)
                    logger.info(f"Display window created for camera {cam_id}.")

                cv2.imshow(win_name, frame)
                messages_processed += 1

            # Process GUI events
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit key pressed.")
                break

        self._cleanup()

    def _cleanup(self):
        cv2.destroyAllWindows()
        logger.info("All display windows closed.")
