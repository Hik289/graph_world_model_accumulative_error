#!/usr/bin/env python3
"""verify_manifest.py — round-trip sha256 check for gwmerror data subdirs.

Usage:
import os as _os; PROJECT_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    python scripts/verify_manifest.py <manifest_or_subdir> [...]
    python scripts/verify_manifest.py --all      # all 5 subdirs under data/

Per-file: re-compute sha256(file) and compare to manifest entry.
Per-manifest: also re-compute sha256(manifest.json) and emit first-16 fingerprint.
Exit code 0 iff all match; 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

DATA_ROOT = "PROJECT_ROOT/data"
SUBDIRS = [
    "synthetic_graphs",
    "synthetic_rollouts",
    "injection_data",
    "agent_calling_tree",
    "platform_skill_graph",
]


def sha256_file(path: str, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_one(manifest_path: Path) -> Tuple[bool, dict]:
    """Verify one manifest. Returns (pass, report_dict)."""
    t0 = time.time()
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    subdir = manifest_path.parent
    files = manifest.get("files", [])

    # Flat list of {path, sha256, ...} entries — path may already include split prefix.
    flat: List[dict] = []
    for e in files:
        if isinstance(e, dict) and "path" in e and "sha256" in e:
            flat.append({**e, "_rel": e["path"]})
    # Fallback: dict-of-lists layout
    if not flat and isinstance(manifest.get("files"), dict):
        for split_name, entries in manifest["files"].items():
            for e in entries:
                if "path" in e and "sha256" in e:
                    flat.append({**e, "_rel": os.path.join(split_name, e["path"])})

    n_total = len(flat)
    n_pass = 0
    failures = []
    for entry in flat:
        rel = entry.get("_rel") or entry.get("path", "?")
        full = subdir / rel
        expected = entry.get("sha256")
        if not expected:
            failures.append({"file": str(rel), "reason": "no sha256 in manifest"})
            continue
        if not full.exists():
            failures.append({"file": str(rel), "reason": "missing file on disk"})
            continue
        got = sha256_file(str(full))
        if got != expected:
            failures.append({"file": str(rel), "expected": expected[:16], "got": got[:16],
                             "reason": "sha256 mismatch"})
        else:
            n_pass += 1

    # Manifest-of-manifest sha
    with open(manifest_path, "rb") as f:
        mhash = hashlib.sha256(f.read()).hexdigest()

    report = {
        "subdir": subdir.name,
        "manifest_path": str(manifest_path),
        "n_files_in_manifest": n_total,
        "n_files_passed": n_pass,
        "n_files_failed": len(failures),
        "manifest_sha256_full": mhash,
        "manifest_sha256_first16": mhash[:16],
        "wall_time_sec": round(time.time() - t0, 2),
        "failures": failures[:50],  # cap to keep report small
    }
    return (len(failures) == 0), report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="manifest.json paths or subdir names")
    ap.add_argument("--all", action="store_true", help="verify all 5 standard subdirs")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    targets: List[Path] = []
    if args.all:
        for sd in SUBDIRS:
            targets.append(Path(args.data_root) / sd / "manifest.json")
    for t in args.targets:
        p = Path(t)
        if p.is_dir():
            p = p / "manifest.json"
        elif not p.exists():
            # try as subdir name
            cand = Path(args.data_root) / t / "manifest.json"
            if cand.exists():
                p = cand
        targets.append(p)

    if not targets:
        print("nothing to verify", file=sys.stderr)
        return 2

    all_ok = True
    reports = []
    for m in targets:
        if not m.exists():
            print(f"❌  {m} not found")
            all_ok = False
            reports.append({"manifest_path": str(m), "error": "not found"})
            continue
        ok, rep = verify_one(m)
        flag = "✅" if ok else "❌"
        print(f"{flag}  {rep['subdir']:24s}  {rep['n_files_passed']:>4d}/{rep['n_files_in_manifest']:>4d} files OK  "
              f"sha16={rep['manifest_sha256_first16']}  ({rep['wall_time_sec']:.1f}s)")
        if not ok:
            all_ok = False
            for fail in rep["failures"][:10]:
                print(f"    └─ FAIL {fail.get('file')}  reason={fail.get('reason')}")
        reports.append(rep)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"all_ok": all_ok, "reports": reports}, f, indent=2)
        print(f"report written to {args.out}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
