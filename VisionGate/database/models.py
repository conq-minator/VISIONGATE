"""
database/models.py
==================
Simple data classes mirroring database tables.
No ORM overhead – just plain Python dataclasses.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    user_id  : int
    name     : str
    qr_code  : Optional[str] = None


@dataclass
class Face:
    id         : int
    user_id    : int
    face_label : int


@dataclass
class AccessLog:
    id        : int
    timestamp : str
    user_id   : Optional[int]
    camera_id : int
    decision  : str
