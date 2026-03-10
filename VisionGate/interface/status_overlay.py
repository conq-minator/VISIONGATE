"""
interface/status_overlay.py
============================
Draws bounding boxes and status panels on camera frames.
Colour-coded by decision outcome.
"""

import cv2
import numpy as np
import logging

import config
from core.decision_engine import AccessResult, Decision, Command

logger = logging.getLogger(__name__)


# Decision → colour mapping (BGR)
_DECISION_COLOR = {
    Decision.ACCESS_GRANTED          : config.COLOR_GRANTED,
    Decision.ACCESS_GRANTED_LEARNING : config.COLOR_GRANTED,
    Decision.REQUIRE_QR              : config.COLOR_QR,
    Decision.ACCESS_DENIED           : config.COLOR_DENIED,
    Decision.IDLE                    : config.COLOR_IDLE,
}


def draw_overlay(
    frame       : np.ndarray,
    camera_id   : int,
    result      : AccessResult,
    fps         : float = 0.0,
    faces       : list  = None,
    qr_user_id  : str | None = None,
    user_name   : str = "Unknown",
) -> np.ndarray:
    """
    Draw full overlay on *frame* in-place (copy is passed in).

    Elements:
      - Face bounding boxes
      - Status panel (top-left corner)
      - Match percentage bar
    """
    if faces is None:
        faces = []

    color = _DECISION_COLOR.get(result.decision, config.COLOR_IDLE)

    # ── 1. Face bounding boxes ─────────────────────────────────────────────────
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # ── 2. Status panel ────────────────────────────────────────────────────────
    _draw_panel(
        frame,
        camera_id  = camera_id,
        user_name  = user_name,
        result     = result,
        fps        = fps,
        qr_visible = qr_user_id is not None,
        color      = color,
    )

    return frame


def _draw_panel(
    frame      : np.ndarray,
    camera_id  : int,
    user_name  : str,
    result     : AccessResult,
    fps        : float,
    qr_visible : bool,
    color      : tuple,
):
    """Draw the translucent status panel in the top-left corner."""
    lines = _build_lines(camera_id, user_name, result, fps, qr_visible)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    fs         = config.FONT_SCALE
    thickness  = config.FONT_THICKNESS
    padding    = 8
    line_h     = int(cv2.getTextSize("A", font, fs, thickness)[0][1] * 1.9)

    panel_w = 280
    panel_h = padding * 2 + line_h * len(lines)

    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, config.OVERLAY_ALPHA, frame, 1 - config.OVERLAY_ALPHA, 0, frame)

    # Draw coloured left border
    cv2.rectangle(frame, (0, 0), (4, panel_h), color, cv2.FILLED)

    # Draw text lines
    for i, (label, value, line_color) in enumerate(lines):
        y = padding + (i + 1) * line_h - 4
        if label:
            cv2.putText(
                frame, f"{label}: ", (10, y), font, fs, (160, 160, 160), thickness, cv2.LINE_AA
            )
            # Measure label width to position value
            lw = cv2.getTextSize(f"{label}: ", font, fs, thickness)[0][0]
            cv2.putText(
                frame, str(value), (10 + lw, y), font, fs, line_color, thickness, cv2.LINE_AA
            )
        else:
            cv2.putText(
                frame, str(value), (10, y), font, fs, line_color, thickness, cv2.LINE_AA
            )

    # ── Match percentage bar ──────────────────────────────────────────────────
    if result.match_pct > 0:
        match_pct = int(result.match_pct)
        bar_y = panel_h + 6
        bar_h = 6
        bar_w = panel_w - 20
        filled = int(bar_w * match_pct / 100)

        cv2.rectangle(frame, (10, bar_y), (10 + bar_w, bar_y + bar_h), (60, 60, 60), cv2.FILLED)
        cv2.rectangle(frame, (10, bar_y), (10 + filled, bar_y + bar_h), color, cv2.FILLED)


def _build_lines(
    camera_id  : int,
    user_name  : str,
    result     : AccessResult,
    fps        : float,
    qr_visible : bool,
) -> list[tuple[str, str, tuple]]:
    """
    Returns a list of (label, value, colour) tuples for each overlay line.
    """
    white  = (240, 240, 240)
    yellow = (0, 220, 255)
    grey   = (160, 160, 160)
    color  = _DECISION_COLOR.get(result.decision, config.COLOR_IDLE)

    lines = [
        ("", f"VisionGate  |  Cam {camera_id}", grey),
        ("User", user_name if result.user_id != -1 else "Unknown", white),
    ]

    if result.match_pct > 0:
        lines.append(("Match", f"{result.match_pct:.1f}%", white))

    if qr_visible:
        lines.append(("QR", "Detected", yellow))

    lines.append(("Status", result.decision.value, color))

    if result.command and result.command.value:
        lines.append(("CMD", result.command.value, color))

    if fps > 0:
        lines.append(("FPS", f"{fps:.1f}", grey))

    return lines
