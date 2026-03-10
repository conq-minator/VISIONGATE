"""
utils/logger.py
===============
Timestamped, rotating file + console logger.
Call setup_logging() once at startup from main.py.
"""

import logging
import logging.handlers
import os
import sys

import config


def setup_logging():
    """
    Configure root logger with:
      - Console handler (INFO by default)
      - Rotating file handler (10 MB max, 5 backups)
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)

    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    root  = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler
    if config.LOG_TO_FILE:
        log_file = os.path.join(config.LOG_DIR, "visiongate.log")
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    logging.info("Logging initialised.")
