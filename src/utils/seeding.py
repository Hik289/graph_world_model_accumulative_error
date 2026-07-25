"""Deterministic seed derivation for reproducible experiments."""
from __future__ import annotations

import hashlib
import json


def stable_seed(*parts: object) -> int:
    """Derive a process-independent NumPy-compatible seed from primitive values."""
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"), default=repr)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)
