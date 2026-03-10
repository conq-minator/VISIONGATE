"""
main.py
=======
VisionGate – Entry Point
========================
Startup sequence:
  1. Setup logging
  2. Initialise database
  3. Load face recognition model
  4. Start camera manager
  5. Start processing pipeline
  6. Run main display loop
  7. Graceful shutdown
"""

import sys
import os
import threading
import signal
import logging

# ── Setup path ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── Logging first ─────────────────────────────────────────────────────────────
from utils.logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# ── Remaining imports ─────────────────────────────────────────────────────────
import config
from database.db            import initialize as init_db
from database.face_storage  import get_user_name_and_qr
from core.camera_manager    import CameraManager
from core.frame_pipeline    import FramePipeline
from interface.display      import DisplayManager
from utils.helpers          import ensure_dirs


def main():
    logger.info("=" * 60)
    logger.info("VisionGate starting up…")
    logger.info("=" * 60)

    # ── 1. Ensure data directories exist ──────────────────────────────────────
    ensure_dirs(config.FACES_DIR, config.QR_DIR, config.LOG_DIR)

    # ── 2. Initialise database ─────────────────────────────────────────────────
    logger.info("Initialising database…")
    init_db()

    # ── 3. Camera Manager ──────────────────────────────────────────────────────
    logger.info("Starting camera manager…")
    camera_manager = CameraManager()
    active_cams    = camera_manager.start()

    if not active_cams:
        logger.warning(
            "No cameras detected. Check connections and CAMERA_IDS in config.py.\n"
            "Running in headless/test mode – starting with virtual feed."
        )
        # Headless mode: just initialise services without display loop
        _headless_mode()
        return

    logger.info(f"Active cameras: {active_cams}")

    # ── 4. Processing Pipeline ─────────────────────────────────────────────────
    logger.info("Starting frame processing pipeline…")
    pipeline = FramePipeline(
        frame_queue    = camera_manager.frame_queue,
        camera_manager = camera_manager,
        db_lookup_fn   = get_user_name_and_qr,
    )
    pipeline.start()

    # ── 5. Shutdown coordination ───────────────────────────────────────────────
    stop_event = threading.Event()

    def _shutdown(sig=None, frame=None):
        logger.info("Shutdown signal received.")
        stop_event.set()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── 6. Display loop (main thread) ──────────────────────────────────────────
    logger.info("Entering display loop. Press Q in any camera window to quit.")
    display = DisplayManager(pipeline.display_queue)

    try:
        display.run(stop_event=stop_event)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught.")
    finally:
        logger.info("Shutting down…")
        stop_event.set()
        pipeline.stop()
        camera_manager.stop()
        logger.info("VisionGate stopped cleanly.")


def _headless_mode():
    """
    Run in headless mode for environments without cameras.
    Useful for testing the pipeline without physical hardware.
    """
    logger.info("Headless mode active. Run tests/simulation_test.py for pipeline test.")
    print(
        "\nNo cameras found.\n"
        "  • Connect a USB camera and update CAMERA_IDS in config.py\n"
        "  • Or run: python tests/simulation_test.py  to test with synthetic frames\n"
        "  • Or run: python training/collect_faces.py  to register users\n"
    )


if __name__ == "__main__":
    main()
