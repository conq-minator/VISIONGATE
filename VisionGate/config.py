"""
VisionGate Configuration
========================
Central configuration file. Edit this file to tune system behaviour.
"""

import os

# ──────────────────────────────────────────────
# Camera Settings
# ──────────────────────────────────────────────
# List of camera indices to use. Add/remove IDs as needed.
CAMERA_IDS = [0]          # e.g. [0, 1, 2, 3]
MAX_CAMERAS = 7

# Frame capture settings
CAPTURE_WIDTH  = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS    = 15       # target FPS per camera; actual FPS depends on hardware

# ──────────────────────────────────────────────
# Processing Pipeline Settings
# ──────────────────────────────────────────────
FRAME_QUEUE_MAXSIZE = 10  # per-camera queue depth; drop frames beyond this
NUM_WORKER_THREADS  = 2   # processing worker threads (increase on multi-core)

# Frame skipping: if queue depth ≥ this fraction of FRAME_QUEUE_MAXSIZE, skip
SKIP_THRESHOLD_RATIO = 0.8

# ──────────────────────────────────────────────
# Face Detection (Haar Cascade)
# ──────────────────────────────────────────────
# Resize frame to this width before detection (speeds up detection)
DETECTION_RESIZE_WIDTH = 320
SCALE_FACTOR   = 1.1
MIN_NEIGHBORS  = 5
MIN_FACE_SIZE  = (60, 60)

# Path to Haar cascade XML (bundled with opencv)
CASCADE_PATH = os.path.join(
    os.path.dirname(__file__),
    "data", "haarcascade_frontalface_default.xml"
)

# ──────────────────────────────────────────────
# Face Recognition (LBPH)
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "face_model.yml")

# LBPH confidence: LOWER value = BETTER match (distance metric)
# Map to 0–100% using:  match% = max(0, 100 - confidence)
CONFIDENCE_GRANT   = 20   # confidence ≤ this → ≥ 80% match → ACCESS_GRANTED
CONFIDENCE_QR      = 60   # confidence ≤ this → ≥ 40% match → REQUIRE_QR
# confidence > CONFIDENCE_QR → < 40% match → ACCESS_DENIED

# Minimum face region size for recognition (pixels)
MIN_FACE_WIDTH  = 80
MIN_FACE_HEIGHT = 80

# ──────────────────────────────────────────────
# QR Code Settings
# ──────────────────────────────────────────────
QR_SCAN_INTERVAL = 5      # scan for QR every N processed frames (saves CPU)

# ──────────────────────────────────────────────
# Decision Engine
# ──────────────────────────────────────────────
QR_VERIFICATION_TIMEOUT = 15  # seconds to wait for QR after partial match

# Number of extra face samples to capture during ACCESS_GRANTED_LEARNING
LEARNING_SAMPLES = 10

# ──────────────────────────────────────────────
# Database
# ──────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "database.db")

# ──────────────────────────────────────────────
# Data Paths
# ──────────────────────────────────────────────
FACES_DIR = os.path.join(os.path.dirname(__file__), "data", "faces")
QR_DIR    = os.path.join(os.path.dirname(__file__), "data", "qr")

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
COLLECT_SAMPLES = 30          # face samples to collect per user
FACE_IMG_SIZE   = (200, 200)  # normalise all face images to this size

# ──────────────────────────────────────────────
# Display / Overlay
# ──────────────────────────────────────────────
WINDOW_SCALE   = 1.0          # scale display window (0.5 = half size)
OVERLAY_ALPHA  = 0.6          # transparency of status panel
FONT_SCALE     = 0.55
FONT_THICKNESS = 1

# Colours (BGR)
COLOR_GRANTED = (50, 205, 50)    # green
COLOR_QR      = (0, 165, 255)    # orange
COLOR_DENIED  = (0, 0, 220)      # red
COLOR_IDLE    = (200, 200, 200)  # grey

# ──────────────────────────────────────────────
# Hardware / Arduino
# ──────────────────────────────────────────────
SIMULATION_MODE   = True        # True = print commands; False = use serial port
ARDUINO_PORT      = "COM3"      # Windows serial port (change as needed)
ARDUINO_BAUD_RATE = 9600
GATE_OPEN_DURATION = 3          # seconds to keep gate open

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_DIR   = os.path.join(os.path.dirname(__file__), "logs")
LOG_LEVEL = "INFO"              # DEBUG | INFO | WARNING | ERROR
LOG_TO_FILE = True
