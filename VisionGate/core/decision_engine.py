"""
core/decision_engine.py
========================
Implements the 3-tier access decision logic:

  match% ≥ 80  →  ACCESS_GRANTED        → OPEN_GATE
  40 ≤ match < 80 + valid QR  →  ACCESS_GRANTED_LEARNING  → OPEN_GATE
  40 ≤ match < 80 + no/bad QR →  REQUIRE_QR
  match < 40   →  REQUIRE_QR  → if still bad → ACCESS_DENIED → BUZZER_ON

Thread-safe: each camera has its own DecisionEngine instance.
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum

import config

logger = logging.getLogger(__name__)


class Decision(str, Enum):
    IDLE                    = "IDLE"
    ACCESS_GRANTED          = "ACCESS GRANTED"
    ACCESS_GRANTED_LEARNING = "ACCESS GRANTED (LEARNING)"
    REQUIRE_QR              = "QR VERIFICATION REQUIRED"
    ACCESS_DENIED           = "ACCESS DENIED"


class Command(str, Enum):
    NONE      = ""
    OPEN_GATE = "OPEN_GATE"
    BUZZER_ON = "BUZZER_ON"


@dataclass
class AccessResult:
    decision    : Decision = Decision.IDLE
    command     : Command  = Command.NONE
    user_id     : int      = -1
    match_pct   : float    = 0.0
    qr_user_id  : str | None = None
    learning    : bool     = False
    camera_id   : int      = -1


class DecisionEngine:
    """
    Per-camera stateful decision engine.
    Maintains a QR-pending window so the system can wait for a QR scan
    after a partial face match.
    """

    def __init__(self, camera_id: int):
        self.camera_id        = camera_id
        self._pending_user_id : int | None = None
        self._pending_since   : float      = 0.0
        self._last_result     : AccessResult = AccessResult(camera_id=camera_id)

    # ── Public interface ──────────────────────────────────────────────────────

    def evaluate(
        self,
        user_id     : int,
        match_pct   : float,
        qr_user_id  : str | None,
        db_has_qr   : bool,       # True if the user's QR is registered in DB
    ) -> AccessResult:
        """
        Core decision logic.

        Args:
            user_id    : predicted user ID from face recogniser (-1 = unknown)
            match_pct  : 0–100 % face match
            qr_user_id : decoded QR user-ID string, or None
            db_has_qr  : whether this user has a registered QR code

        Returns:
            AccessResult with decision, command, and learning flag.
        """
        result = AccessResult(
            user_id=user_id,
            match_pct=match_pct,
            qr_user_id=qr_user_id,
            camera_id=self.camera_id,
        )

        # ── Tier 1: Strong face match ─────────────────────────────────────────
        if match_pct >= 80.0 and user_id != -1:
            result.decision = Decision.ACCESS_GRANTED
            result.command  = Command.OPEN_GATE
            self._clear_pending()
            logger.info(
                f"[Cam {self.camera_id}] ACCESS_GRANTED  user={user_id} "
                f"match={match_pct:.1f}%"
            )

        # ── Tier 2: Partial match – QR fallback ──────────────────────────────
        elif 40.0 <= match_pct < 80.0:
            if qr_user_id is not None and str(user_id) == qr_user_id:
                result.decision = Decision.ACCESS_GRANTED_LEARNING
                result.command  = Command.OPEN_GATE
                result.learning = True
                self._clear_pending()
                logger.info(
                    f"[Cam {self.camera_id}] ACCESS_GRANTED_LEARNING  "
                    f"user={user_id} match={match_pct:.1f}% QR={qr_user_id}"
                )
            else:
                # Set/maintain pending window
                self._set_pending(user_id)
                if self._pending_expired():
                    result.decision = Decision.ACCESS_DENIED
                    result.command  = Command.BUZZER_ON
                    self._clear_pending()
                    logger.info(
                        f"[Cam {self.camera_id}] ACCESS_DENIED (QR timeout) "
                        f"user={user_id} match={match_pct:.1f}%"
                    )
                else:
                    result.decision = Decision.REQUIRE_QR
                    result.command  = Command.NONE
                    logger.debug(
                        f"[Cam {self.camera_id}] REQUIRE_QR  "
                        f"user={user_id} match={match_pct:.1f}%"
                    )

        # ── Tier 3: Weak / no match ───────────────────────────────────────────
        else:
            if qr_user_id is not None:
                # QR provided but face is unknown → still deny (anti-spoofing)
                result.decision = Decision.ACCESS_DENIED
                result.command  = Command.BUZZER_ON
                logger.info(
                    f"[Cam {self.camera_id}] ACCESS_DENIED (face too weak + QR) "
                    f"match={match_pct:.1f}%"
                )
            else:
                result.decision = Decision.ACCESS_DENIED
                result.command  = Command.BUZZER_ON
                logger.info(
                    f"[Cam {self.camera_id}] ACCESS_DENIED  match={match_pct:.1f}%"
                )
            self._clear_pending()

        self._last_result = result
        return result

    def evaluate_no_face(self) -> AccessResult:
        """Called when no face is detected in the frame."""
        # Keep REQUIRE_QR state alive within timeout; otherwise go IDLE
        if (self._pending_user_id is not None
                and not self._pending_expired()):
            return AccessResult(
                decision  = Decision.REQUIRE_QR,
                camera_id = self.camera_id,
            )
        self._clear_pending()
        return AccessResult(
            decision  = Decision.IDLE,
            camera_id = self.camera_id,
        )

    def last_result(self) -> AccessResult:
        return self._last_result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _set_pending(self, user_id: int):
        if self._pending_user_id != user_id:
            self._pending_user_id = user_id
            self._pending_since   = time.time()

    def _clear_pending(self):
        self._pending_user_id = None
        self._pending_since   = 0.0

    def _pending_expired(self) -> bool:
        if self._pending_since == 0.0:
            return False
        return (time.time() - self._pending_since) > config.QR_VERIFICATION_TIMEOUT
