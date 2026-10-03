from __future__ import annotations

import hashlib
import json


def stable_seed(*parts: object) -> int:
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"), default=repr)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)
