#!/usr/bin/env python3
"""Triage: scan step_trace_time.csv for bottleneck direction.

This stage always runs before the full pipeline and provides a low-confidence
bottleneck hint based on ``step_trace_time.csv`` from the profiling root.
It completes in <1s and never skips downstream stages — its output is
informational, referenced in the report Executive Summary.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


TRIAGE_JSON = "triage.json"
TRIAGE_MANIFEST = "triage_manifest.json"

# Classification thresholds (aligned with CANN profiler conventions).
FREE_HOSTBOUND_PCT = 20.0
COMPUTE_COMPUTE_PCT = 85.0
COMM_COMM_PCT = 10.0


def _find_step_trace_files(profile_root: Path) -> list[Path]:
    """Recursively find all step_trace_time.csv files under a profiling root."""
    if not profile_root.is_dir():
        return []
    return sorted(profile_root.rglob("step_trace_time.csv"))


def _classify_bottleneck(
    avg_computing: float,
    avg_communication: float,
    avg_free: float,
) -> str:
    """Return a bottleneck label based on CANN step_trace_time conventions."""
    if avg_free > FREE_HOSTBOUND_PCT:
        return "hostbound"
    if avg_computing > COMPUTE_COMPUTE_PCT:
        return "computing"
    if avg_communication > COMM_COMM_PCT:
        return "communication"
    return "none_obvious"


def _parse_steps(csv_path: Path) -> dict[str, float] | None:
    """Parse one step_trace_time.csv, returning {computing, communication, free} pct.

    Returns None if the file is missing required columns or has zero totals.
    """
    required = {"Computing", "Communication(Not Overlapped)", "Free"}
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not required.issubset(reader.fieldnames or set()):
                return None
            compute_sum = 0.0
            comm_sum = 0.0
            free_sum = 0.0
            for row in reader:
                try:
                    compute_sum += float(row["Computing"])
                    comm_sum += float(row["Communication(Not Overlapped)"])
                    free_sum += float(row["Free"])
                except (ValueError, KeyError):
                    continue
    except (OSError, UnicodeDecodeError):
        return None

    total = compute_sum + comm_sum + free_sum
    if total <= 0:
        return None

    return {
        "computing_pct": round(compute_sum / total * 100, 2),
        "communication_pct": round(comm_sum / total * 100, 2),
        "free_pct": round(free_sum / total * 100, 2),
    }


def run_triage(profile_root: Path, output_dir: Path, *, verbose: bool = False) -> dict[str, Any]:
    """Scan step_trace_time.csv files and write triage.json.

    Returns a summary dict. The triage is always lightweight (<1s for
    typical roots) and informational only — it never controls whether
    downstream stages execute.
    """
    profile_root = Path(profile_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _find_step_trace_files(profile_root)

    if not files:
        manifest: dict[str, Any] = {
            "status": "no_step_trace_files",
            "reason": "No step_trace_time.csv found in profiling root",
        }
        _write_manifest_fw(output_dir, manifest)
        return manifest

    all_pcts: list[dict[str, Any]] = []
    for path in files:
        pcts = _parse_steps(path)
        if pcts is not None:
            pcts["source"] = str(path)
            all_pcts.append(pcts)

    if not all_pcts:
        manifest = {
            "status": "parse_error",
            "reason": "Found step_trace_time.csv files but none were parseable",
        }
        _write_manifest_fw(output_dir, manifest)
        return manifest

    rank_count = len(all_pcts)
    avg_comp = round(sum(r["computing_pct"] for r in all_pcts) / rank_count, 2)
    avg_comm = round(sum(r["communication_pct"] for r in all_pcts) / rank_count, 2)
    avg_free = round(sum(r["free_pct"] for r in all_pcts) / rank_count, 2)
    bottleneck = _classify_bottleneck(avg_comp, avg_comm, avg_free)

    # Per-rank extremes
    max_free = max(all_pcts, key=lambda r: r["free_pct"])
    max_comp = max(all_pcts, key=lambda r: r["computing_pct"])
    max_comm = max(all_pcts, key=lambda r: r["communication_pct"])

    result = {
        "status": "ok",
        "rank_count": rank_count,
        "avg_computing_pct": avg_comp,
        "avg_communication_pct": avg_comm,
        "avg_free_pct": avg_free,
        "primary_bottleneck": bottleneck,
        "confidence": "low",
        "max_free_pct": {"value": max_free["free_pct"], "source": max_free["source"]},
        "max_comp_pct": {"value": max_comp["computing_pct"], "source": max_comp["source"]},
        "max_comm_pct": {"value": max_comm["communication_pct"], "source": max_comm["source"]},
        "per_rank": all_pcts,
        "note": (
            "Based on step_trace_time.csv only. This is a low-confidence "
            "pre-scan. The full pipeline (segment -> classify -> summarize -> "
            "cross_rank -> diagnostics) may identify different or additional "
            "bottlenecks with evidence-chain backing."
        ),
    }

    _write_json_fw(output_dir / TRIAGE_JSON, result)
    _write_manifest_fw(output_dir, result)

    if verbose:
        print(
            f"[triage] bottleneck={bottleneck} "
            f"(comp={avg_comp}% comm={avg_comm}% free={avg_free}%)",
            flush=True,
        )

    return result


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
    _write_json_fw(out / TRIAGE_MANIFEST, data)


def load_triage(output_dir: Path) -> dict[str, Any] | None:
    """Read triage.json if it exists, else None."""
    triage_path = Path(output_dir) / TRIAGE_JSON
    if not triage_path.is_file():
        return None
    try:
        from .common import read_json
        return read_json(triage_path)
    except ImportError:
        import json
        return json.loads(triage_path.read_text(encoding="utf-8"))
