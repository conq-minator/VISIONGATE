"""
training/collect_faces.py
=========================
Interactive CLI to collect face samples for a new user.

Usage:
    python training/collect_faces.py

Follow the prompts:
  1. Enter user ID (must already exist in the database, or create via prompt)
  2. Press SPACE to capture a sample
  3. Press Q to finish early
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import time

import config
from utils.logger  import setup_logging
from database.db   import initialize
from database.face_storage import add_user, get_user, add_face_record
from core.face_detector import FaceDetector
from utils.helpers import ensure_dirs

setup_logging()
import logging
logger = logging.getLogger(__name__)


def collect_faces():
    initialize()
    detector = FaceDetector()

    # ── Get user info ──────────────────────────────────────────────────────────
    print("\n=== VisionGate – Face Collection ===")
    user_id_input = input("Enter user ID (press ENTER to register a new user): ").strip()

    if user_id_input.isdigit():
        user_id = int(user_id_input)
        user = get_user(user_id)
        if not user:
            print(f"User {user_id} not found. Please register them first.")
            return
        print(f"Collecting faces for: {user.name} (ID: {user_id})")
    else:
        name = input("Enter new user's full name: ").strip()
        qr   = input("Enter QR code for this user (or press ENTER to skip): ").strip()
        user_id = add_user(name, qr or None)
        print(f"Registered new user '{name}' with ID: {user_id}")

    # ── Setup output directory ─────────────────────────────────────────────────
    user_dir = os.path.join(config.FACES_DIR, f"user_{user_id}")
    ensure_dirs(user_dir)

    cam_id = int(input(f"Camera index to use (default 0): ").strip() or "0")

    # ── Camera capture loop ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"Cannot open camera {cam_id}.")
        return

    sample_count = 0
    target       = config.COLLECT_SAMPLES
    print(f"\nPress SPACE to capture (need {target} samples). Press Q to quit.\n")

    while sample_count < target:
        ret, frame = cap.read()
        if not ret:
            continue

        faces = detector.detect(frame)
        display = frame.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        status = f"Samples: {sample_count}/{target}"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Collect Faces – SPACE to capture, Q to quit", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and faces:
            roi = detector.extract_roi(frame, faces[0])
            filename = os.path.join(user_dir, f"{sample_count:04d}.jpg")
            cv2.imwrite(filename, roi)
            sample_count += 1
            print(f"  Captured sample {sample_count}/{target}")
            time.sleep(0.1)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if sample_count > 0:
        add_face_record(user_id, user_id)  # face_label = user_id
        print(f"\nCollected {sample_count} samples for user {user_id}.")
        print("Run: python training/train_model.py  to re-train the model.")
    else:
        print("No samples collected.")


if __name__ == "__main__":
    collect_faces()
