"""
database/face_storage.py
========================
CRUD helpers for Users, Faces, and AccessLogs tables.
Each function opens and closes its own connection for thread safety.
"""

import logging
from datetime import datetime
from typing import Optional

from database.db import get_connection
from database.models import User, AccessLog

logger = logging.getLogger(__name__)


# ── Users ──────────────────────────────────────────────────────────────────────

def add_user(name: str, qr_code: str | None = None) -> int:
    """Insert a new user. Returns the new user_id."""
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO Users (name, qr_code) VALUES (?, ?)",
                (name, qr_code),
            )
            user_id = cur.lastrowid
        logger.info(f"User added: id={user_id} name='{name}'")
        return user_id
    finally:
        conn.close()


def get_user(user_id: int) -> Optional[User]:
    """Return User for given user_id, or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT user_id, name, qr_code FROM Users WHERE user_id=?", (user_id,)
        ).fetchone()
        return User(**dict(row)) if row else None
    finally:
        conn.close()


def get_user_by_qr(qr_code: str) -> Optional[User]:
    """Look up a user by their QR code value."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT user_id, name, qr_code FROM Users WHERE qr_code=?", (qr_code,)
        ).fetchone()
        return User(**dict(row)) if row else None
    finally:
        conn.close()


def get_user_name_and_qr(user_id: int) -> Optional[tuple[str, str | None]]:
    """Return (name, qr_code) tuple for use in the pipeline, or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT name, qr_code FROM Users WHERE user_id=?", (user_id,)
        ).fetchone()
        return (row["name"], row["qr_code"]) if row else None
    finally:
        conn.close()


def update_user_qr(user_id: int, qr_code: str):
    """Set or update the QR code for a user."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE Users SET qr_code=? WHERE user_id=?", (qr_code, user_id)
            )
        logger.info(f"User {user_id} QR updated.")
    finally:
        conn.close()


def list_users() -> list[User]:
    """Return all registered users."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT user_id, name, qr_code FROM Users").fetchall()
        return [User(**dict(r)) for r in rows]
    finally:
        conn.close()


def search_users(query: str) -> list[User]:
    """
    Search users by name (case-insensitive substring) or exact user_id.
    Returns matching User objects.
    """
    conn = get_connection()
    try:
        rows = []
        # Try numeric ID match first
        if query.strip().isdigit():
            rows = conn.execute(
                "SELECT user_id, name, qr_code FROM Users WHERE user_id=?",
                (int(query.strip()),),
            ).fetchall()
        if not rows:
            rows = conn.execute(
                "SELECT user_id, name, qr_code FROM Users "
                "WHERE LOWER(name) LIKE ?",
                (f"%{query.lower()}%",),
            ).fetchall()
        return [User(**dict(r)) for r in rows]
    finally:
        conn.close()


def count_face_samples(user_id: int) -> int:
    """Count the number of face image files stored for a user."""
    import os, config
    user_dir = os.path.join(config.FACES_DIR, f"user_{user_id}")
    if not os.path.isdir(user_dir):
        return 0
    return sum(
        1 for f in os.listdir(user_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def get_last_access(user_id: int) -> Optional[str]:
    """Return the most recent access timestamp for a user, or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT timestamp FROM AccessLogs WHERE user_id=? ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        return row["timestamp"] if row else None
    finally:
        conn.close()


def delete_user(user_id: int):
    """Delete a user and their face/log records."""
    conn = get_connection()
    try:
        with conn:
            conn.execute("DELETE FROM Faces WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM Users WHERE user_id=?", (user_id,))
        logger.info(f"User {user_id} deleted.")
    finally:
        conn.close()


# ── Faces ──────────────────────────────────────────────────────────────────────

def add_face_record(user_id: int, face_label: int):
    """Record that face_label images were collected for user_id."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO Faces (user_id, face_label) VALUES (?, ?)",
                (user_id, face_label),
            )
    finally:
        conn.close()


def get_face_label(user_id: int) -> Optional[int]:
    """Return the LBPH training label for a user, or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT face_label FROM Faces WHERE user_id=? LIMIT 1", (user_id,)
        ).fetchone()
        return row["face_label"] if row else None
    finally:
        conn.close()


# ── Access Logs ────────────────────────────────────────────────────────────────

def log_access(camera_id: int, user_id: int | None, decision: str):
    """Insert an access log entry."""
    conn = get_connection()
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with conn:
            conn.execute(
                "INSERT INTO AccessLogs (timestamp, user_id, camera_id, decision) "
                "VALUES (?, ?, ?, ?)",
                (ts, user_id, camera_id, decision),
            )
        logger.debug(f"AccessLog: cam={camera_id} user={user_id} decision={decision}")
    finally:
        conn.close()


def get_recent_logs(limit: int = 50) -> list[AccessLog]:
    """Return the most recent access log entries."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, timestamp, user_id, camera_id, decision "
            "FROM AccessLogs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [AccessLog(**dict(r)) for r in rows]
    finally:
        conn.close()
