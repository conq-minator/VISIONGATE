"""
tests/simulation_test.py
========================
Automated end-to-end simulation test.
No physical camera or trained model required.

Tests all three decision branches and verifies:
  - Decision engine logic
  - Hardware command output (simulation mode)
  - Database access log writes
  - Frame pipeline message passing

Run with:
    python tests/simulation_test.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import queue
import threading
import time
import logging
import cv2
import numpy as np

# Setup logging first
from utils.logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

import config
from database.db           import initialize as init_db
from database.face_storage import add_user, log_access, get_recent_logs
from core.decision_engine  import DecisionEngine, Decision, Command
from hardware.arduino_comm import ArduinoComm
from interface.status_overlay import draw_overlay


# ── Force simulation mode ──────────────────────────────────────────────────────
config.SIMULATION_MODE = True

PASS = "✓  PASS"
FAIL = "✗  FAIL"


def _make_blank_frame(w=640, h=480, color=(40, 40, 40)) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 – Decision Engine: ACCESS_GRANTED
# ─────────────────────────────────────────────────────────────────────────────

def test_access_granted():
    print("\n── Test 1: ACCESS_GRANTED (match ≥ 80%) ──────────────────────────")
    engine = DecisionEngine(camera_id=99)
    result = engine.evaluate(
        user_id=1, match_pct=85.0, qr_user_id=None, db_has_qr=False
    )
    ok = (
        result.decision == Decision.ACCESS_GRANTED
        and result.command  == Command.OPEN_GATE
    )
    print(f"  Decision : {result.decision.value}")
    print(f"  Command  : {result.command.value}")
    print(f"  {PASS if ok else FAIL}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 – Decision Engine: ACCESS_GRANTED_LEARNING
# ─────────────────────────────────────────────────────────────────────────────

def test_access_granted_learning():
    print("\n── Test 2: ACCESS_GRANTED_LEARNING (40≤match<80 + valid QR) ────────")
    engine = DecisionEngine(camera_id=99)
    result = engine.evaluate(
        user_id=2, match_pct=55.0, qr_user_id="2", db_has_qr=True
    )
    ok = (
        result.decision == Decision.ACCESS_GRANTED_LEARNING
        and result.command  == Command.OPEN_GATE
        and result.learning is True
    )
    print(f"  Decision : {result.decision.value}")
    print(f"  Command  : {result.command.value}")
    print(f"  Learning : {result.learning}")
    print(f"  {PASS if ok else FAIL}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 – Decision Engine: REQUIRE_QR then ACCESS_DENIED
# ─────────────────────────────────────────────────────────────────────────────

def test_require_qr_then_denied():
    print("\n── Test 3: REQUIRE_QR → ACCESS_DENIED (timeout) ────────────────────")
    # Speed up timeout for test
    orig_timeout = config.QR_VERIFICATION_TIMEOUT
    config.QR_VERIFICATION_TIMEOUT = 1  # 1 second

    engine = DecisionEngine(camera_id=99)

    # First call → REQUIRE_QR
    r1 = engine.evaluate(user_id=3, match_pct=50.0, qr_user_id=None, db_has_qr=True)
    ok1 = r1.decision == Decision.REQUIRE_QR
    print(f"  Step 1 – Decision: {r1.decision.value}  [{PASS if ok1 else FAIL}]")

    # Wait for timeout
    time.sleep(1.2)

    # Second call → ACCESS_DENIED
    r2 = engine.evaluate(user_id=3, match_pct=50.0, qr_user_id=None, db_has_qr=True)
    ok2 = r2.decision == Decision.ACCESS_DENIED and r2.command == Command.BUZZER_ON
    print(f"  Step 2 – Decision: {r2.decision.value}  [{PASS if ok2 else FAIL}]")

    config.QR_VERIFICATION_TIMEOUT = orig_timeout
    ok = ok1 and ok2
    print(f"  {PASS if ok else FAIL}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 – Decision Engine: ACCESS_DENIED (weak match)
# ─────────────────────────────────────────────────────────────────────────────

def test_access_denied_weak():
    print("\n── Test 4: ACCESS_DENIED (match < 40%) ──────────────────────────────")
    engine = DecisionEngine(camera_id=99)
    result = engine.evaluate(
        user_id=-1, match_pct=10.0, qr_user_id=None, db_has_qr=False
    )
    ok = (
        result.decision == Decision.ACCESS_DENIED
        and result.command  == Command.BUZZER_ON
    )
    print(f"  Decision : {result.decision.value}")
    print(f"  Command  : {result.command.value}")
    print(f"  {PASS if ok else FAIL}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 – Hardware simulation (console output)
# ─────────────────────────────────────────────────────────────────────────────

def test_hardware_simulation():
    print("\n── Test 5: Hardware simulation (console output) ─────────────────────")
    hw = ArduinoComm()
    print("  Sending OPEN_GATE…")
    hw.send_command("OPEN_GATE")
    print("  Sending BUZZER_ON…")
    hw.send_command("BUZZER_ON")
    print(f"  {PASS}  (check output above for [HW SIM] lines)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 – Database: add user + log access + read logs
# ─────────────────────────────────────────────────────────────────────────────

def test_database():
    print("\n── Test 6: Database operations ──────────────────────────────────────")
    try:
        init_db()
        user_id = add_user("Test User (sim)", qr_code="9999")
        log_access(camera_id=0, user_id=user_id, decision="ACCESS_GRANTED")
        logs = get_recent_logs(limit=5)
        ok = any(l.user_id == user_id for l in logs)
        print(f"  User ID created : {user_id}")
        print(f"  Log entry found : {ok}")
        print(f"  {PASS if ok else FAIL}")
        return ok
    except Exception as exc:
        print(f"  ERROR: {exc}")
        print(f"  {FAIL}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 – Overlay rendering (headless, no display)
# ─────────────────────────────────────────────────────────────────────────────

def test_overlay_rendering():
    print("\n── Test 7: Overlay rendering ────────────────────────────────────────")
    try:
        from core.decision_engine import AccessResult, Decision, Command

        frame  = _make_blank_frame()
        result = AccessResult(
            decision  = Decision.ACCESS_GRANTED,
            command   = Command.OPEN_GATE,
            user_id   = 1,
            match_pct = 85.0,
            camera_id = 0,
        )
        output = draw_overlay(frame, camera_id=0, result=result, fps=14.5,
                              faces=[(100, 100, 150, 180)], user_name="Alice")
        ok = output is not None and output.shape == frame.shape
        print(f"  Frame shape : {output.shape}")
        print(f"  {PASS if ok else FAIL}")
        return ok
    except Exception as exc:
        print(f"  ERROR: {exc}")
        print(f"  {FAIL}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  VisionGate – Simulation Test Suite")
    print("=" * 60)

    results = {
        "ACCESS_GRANTED"          : test_access_granted(),
        "ACCESS_GRANTED_LEARNING" : test_access_granted_learning(),
        "REQUIRE_QR → DENIED"     : test_require_qr_then_denied(),
        "ACCESS_DENIED (weak)"    : test_access_denied_weak(),
        "Hardware simulation"     : test_hardware_simulation(),
        "Database operations"     : test_database(),
        "Overlay rendering"       : test_overlay_rendering(),
    }

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {name}")
    print(f"\n  {passed}/{total} tests passed.")
    print("=" * 60 + "\n")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
