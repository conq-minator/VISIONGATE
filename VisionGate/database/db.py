"""
database/db.py
==============
SQLite connection helper and database initialisation.
"""

import sqlite3
import os
import logging

import config

logger = logging.getLogger(__name__)

# Thread-local storage isn't needed for SQLite in Python because each call
# creates its own connection from the pool below.  We use a simple helper that
# returns a new connection with row_factory set.

def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory enabled."""
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")   # allows concurrent readers
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def initialize():
    """
    Create all tables if they do not already exist.
    Called once at startup from main.py.
    """
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = get_connection()
    try:
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS Users (
                    user_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    name      TEXT    NOT NULL,
                    qr_code   TEXT
                );

                CREATE TABLE IF NOT EXISTS Faces (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id   INTEGER NOT NULL REFERENCES Users(user_id),
                    face_label INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS AccessLogs (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp  TEXT    NOT NULL,
                    user_id    INTEGER,
                    camera_id  INTEGER NOT NULL,
                    decision   TEXT    NOT NULL
                );
            """)
        logger.info(f"Database initialised at {config.DB_PATH}")
    finally:
        conn.close()
