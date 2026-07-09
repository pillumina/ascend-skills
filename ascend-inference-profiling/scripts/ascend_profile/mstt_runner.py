#!/usr/bin/env python3
"""Run msprof-analyze slow_rank detection against a profiling root.

This module is an optional upstream signal source for the diagnostics stage.
When msprof-analyze is installed on the remote host, it runs ``msprof-analyze
cluster -m slow_rank`` against the same profiling root the pipeline already
consumes, parses the resulting SQLite DB, and writes ``mstt_slow_rank.csv``.

When msprof-analyze is not available, the stage silently produces a status
manifest and downstream stages fall back to the existing heuristics.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


MSTT_OUTPUT_CSV = "mstt_slow_rank.csv"
MSTT_MANIFEST_JSON = "mstt_manifest.json"


def _check_msprof_installed() -> bool:
    """Return True if ``msprof-analyze`` CLI is on PATH."""
    try:
        result = subprocess.run(
            ["msprof-analyze", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _install_msprof() -> bool:
    """Attempt ``pip install msprof-analyze``. Return True on success."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-U", "msprof-analyze"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(
                f"[mstt] pip install msprof-analyze failed (rc={result.returncode})",
                flush=True,
            )
            return False
        return _check_msprof_installed()
    except subprocess.TimeoutExpired:
        print("[mstt] pip install msprof-analyze timed out", flush=True)
        return False


def _run_slow_rank(profile_root: Path, output_dir: str) -> str | None:
    """Run msprof-analyze slow_rank. Return path to cluster_analysis.db or None."""
    cmd = [
        "msprof-analyze", "cluster",
        "-d", str(profile_root),
        "-m", "slow_rank",
        "-o", output_dir,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        out = Path(output_dir)
        db_path = out / "cluster_analysis.db"
        if db_path.is_file():
            return str(db_path)
        nested = out / "cluster_analysis_output" / "cluster_analysis.db"
        if nested.is_file():
            return str(nested)
        print(
            f"[mstt] msprof-analyze completed but no cluster_analysis.db found "
            f"(rc={result.returncode})",
            flush=True,
        )
        return None
    except subprocess.TimeoutExpired:
        print("[mstt] msprof-analyze timed out", flush=True)
        return None
    except FileNotFoundError:
        return None


def _parse_slow_rank_db(db_path: str) -> list[dict[str, Any]]:
    """Extract SlowRank table rows from the msprof-analyze output DB."""
    if not Path(db_path).is_file():
        return []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT rankId, slowAffectCount FROM SlowRank")
        rows = [
            {"rank_id": str(row[0]), "slow_affect_count": int(row[1])}
            for row in cursor.fetchall()
        ]
        conn.close()
        return rows
    except sqlite3.Error as exc:
        print(f"[mstt] failed to read SlowRank table: {exc}", flush=True)
        return []


def _discover_rank_ids(profile_root: Path) -> list[str]:
    """Walk the profiling root for device_* / rank_* directories."""
    rank_ids: set[str] = set()
    if not profile_root.is_dir():
        return []
    for entry in profile_root.iterdir():
        if not entry.is_dir():
            continue
        for prefix in ("device_", "rank_"):
            if entry.name.startswith(prefix):
                rank_id = entry.name[len(prefix):].split("_")[0]
                rank_ids.add(rank_id)
    return sorted(rank_ids)


def run_mstt_slow_rank(profile_root: Path, output_dir: Path, *, verbose: bool = False) -> dict[str, Any]:
    """Entry point. Run msprof-analyze slow_rank, write CSV + manifest.

    Uses ``common.write_json`` and ``common.write_csv`` when they are
    importable (i.e. inside the pipeline); falls back to bare json / csv
    for standalone invocation or testing.
    """
    profile_root = Path(profile_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    installed = _check_msprof_installed()
    if not installed:
        if verbose:
            print("[mstt] msprof-analyze not found, attempting pip install...", flush=True)
        installed = _install_msprof()
        if not installed:
            manifest = {
                "status": "unavailable",
                "reason": "msprof-analyze not installed and auto-install failed",
            }
            _write_manifest_fw(output_dir, manifest)
            return manifest

    if verbose:
        print("[mstt] running msprof-analyze cluster -m slow_rank ...", flush=True)

    with tempfile.TemporaryDirectory(prefix="mstt_slow_rank_") as tmpdir:
        db_path = _run_slow_rank(profile_root, tmpdir)
        if db_path is None:
            manifest = {
                "status": "failed",
                "reason": "msprof-analyze did not produce cluster_analysis.db",
            }
            _write_manifest_fw(output_dir, manifest)
            return manifest

        rows = _parse_slow_rank_db(db_path)
        if not rows:
            manifest = {
                "status": "no_data",
                "reason": "SlowRank table is empty — no slow ranks detected or data unsupported",
            }
            _write_manifest_fw(output_dir, manifest)
            return manifest

    # Fill in rank_ids that mstt didn't report (slow_affect_count = 0).
    rank_ids = _discover_rank_ids(profile_root)
    reported = {row["rank_id"] for row in rows}
    for rid in rank_ids:
        if rid not in reported:
            rows.append({"rank_id": rid, "slow_affect_count": 0})

    rows.sort(key=lambda r: r["rank_id"])
    _write_csv_fw(output_dir / MSTT_OUTPUT_CSV, rows, fieldnames=["rank_id", "slow_affect_count"])

    slow_ranks = [r for r in rows if r["slow_affect_count"] > 0]
    manifest = {
        "status": "ok",
        "rank_count": len(rows),
        "slow_rank_count": len(slow_ranks),
        "max_slow_affect_count": max((r["slow_affect_count"] for r in rows), default=0),
        "slow_ranks": [
            {"rank_id": r["rank_id"], "slow_affect_count": r["slow_affect_count"]}
            for r in sorted(slow_ranks, key=lambda x: -x["slow_affect_count"])
        ],
    }
    _write_manifest_fw(output_dir, manifest)

    if verbose:
        print(
            f"[mstt] done — {len(slow_ranks)} slow rank(s) detected, "
            f"max slow_affect_count={manifest['max_slow_affect_count']}",
            flush=True,
        )
    return manifest


def _write_json_fw(path: Path, data: dict[str, Any]) -> None:
    """Write JSON via common.write_json when available; fall back to stdlib."""
    try:
        from .common import write_json
        write_json(path, data)
    except ImportError:
        import json
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def _write_manifest_fw(out: Path, data: dict[str, Any]) -> None:
    _write_json_fw(out / MSTT_MANIFEST_JSON, data)


def _write_csv_fw(out: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    """Write CSV via common.write_csv when available."""
    try:
        from .common import write_csv
        write_csv(out, rows)
    except ImportError:
        import csv
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def load_mstt_slow_rank(output_dir: Path) -> dict[str, int] | None:
    """Read ``mstt_slow_rank.csv`` into {rank_id: slow_affect_count}.

    Returns None when the file does not exist (mstt stage was not run or failed).
    Uses ``common.csv_rows`` when available for consistency.
    """
    csv_path = output_dir / MSTT_OUTPUT_CSV
    if not csv_path.is_file():
        return None
    result: dict[str, int] = {}
    try:
        from .common import csv_rows
        for row in csv_rows(csv_path):
            result[row["rank_id"]] = int(row["slow_affect_count"])
    except ImportError:
        import csv as _csv
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                result[row["rank_id"]] = int(row["slow_affect_count"])
    return result
