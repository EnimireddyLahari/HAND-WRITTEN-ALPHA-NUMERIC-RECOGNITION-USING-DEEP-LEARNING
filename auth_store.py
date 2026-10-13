"""
auth_store.py — minimal SQLite user store for Flask sessions.
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Tuple

from werkzeug.security import check_password_hash, generate_password_hash

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,32}$")


def db_path(app) -> str:
    return os.path.join(app.instance_path, "users.sqlite")


def init_db(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def normalize_username(raw: str) -> str:
    return raw.strip().lower()


def validate_username(raw: str) -> Optional[str]:
    u = normalize_username(raw)
    if not USERNAME_RE.match(u):
        return "Username must be 3–32 characters (letters, digits, underscore)."
    return None


def validate_password(pw: str) -> Optional[str]:
    if len(pw) < 8:
        return "Password must be at least 8 characters."
    return None


def create_user(path: str, username: str, password: str) -> Optional[str]:
    err = validate_username(username) or validate_password(password)
    if err:
        return err
    u = normalize_username(username)
    h = generate_password_hash(password)
    now = datetime.now(timezone.utc).isoformat()
    try:
        with sqlite3.connect(path) as con:
            con.execute(
                "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                (u, h, now),
            )
        return None
    except sqlite3.IntegrityError:
        return "That username is already taken."


def verify_login(path: str, username: str, password: str) -> Tuple[Optional[int], Optional[str]]:
    err = validate_username(username)
    if err:
        return None, err
    u = normalize_username(username)
    with sqlite3.connect(path) as con:
        row = con.execute(
            "SELECT id, password_hash FROM users WHERE username = ?",
            (u,),
        ).fetchone()
    if not row:
        return None, "Invalid username or password."
    uid, ph = row
    if not check_password_hash(ph, password):
        return None, "Invalid username or password."
    return uid, None
