"""Shared helpers for long-running experiment entry points."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import time


JST = timezone(timedelta(hours=9))


def now_jst() -> str:
    """Return a human-readable timestamp in Japan Standard Time."""
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S %Z")


def skip_if_done(path: str | Path, max_age_hours: float = 24.0) -> bool:
    """Return true when a nonempty result file was updated recently."""
    result_path = Path(path)
    if not result_path.is_file() or result_path.stat().st_size == 0:
        return False
    age_seconds = max(0.0, time.time() - result_path.stat().st_mtime)
    return age_seconds < max_age_hours * 60 * 60
