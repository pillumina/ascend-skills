#!/usr/bin/env python3
"""Collect run observations for skill calibration.

This stage produces ``run_observations.json`` (per-run snapshot) and
appends a line to ``~/.cache/ascend-inference-profiling/observations.jsonl``
(persistent aggregate across runs). Both are factual — no suggestions.

The persistent aggregate is the mechanism that makes self-calibration
work: observations from run #1 are available when the user is looking
at run #50. Without it, each run's observations die when the run dir
is cleaned up.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .common import csv_rows, read_json, read_jsonl, SCHEMA_VERSION, TOOL_VERSION, utc_now, write_json
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import csv_rows, read_json, read_jsonl, SCHEMA_VERSION, TOOL_VERSION, utc_now, write_json  # type: ignore[no-redef]


OUTPUT_JSON = "run_observations.json"
OUTPUT_MANIFEST = "observations_manifest.json"

# Per-run persistent copy — written to the output dir and pulled back.
# The agent (LLM) is responsible for accumulating across runs into a
# local persistent log (see SKILL.md § Observations and Skill Calibration).
PERSISTENT_LOCAL_CACHE = Path.home() / ".cache" / "ascend-inference-profiling" / "observations.jsonl"

# Minimum repeat count for an unknown kernel to be worth flagging.
MIN_UNKNOWN_KERNEL_OCCURRENCES = 3


def _collect_unknown_kernels(output_dir: Path) -> list[dict[str, Any]]:
    """Find kernel names that received no op_categories.

    Sorted by occurrence_count descending — the most frequent unknown
    kernel is the most urgent gap to fill.
    """
    events_path = output_dir / "normalized_event_index.jsonl"
    if not events_path.is_file():
        return []

    counter: Counter = Counter()
    total_events = 0
    for event in read_jsonl(events_path):
        total_events += 1
        categories = event.get("op_categories") or []
        if categories:
            continue
        name = event.get("name_raw")
        if name:
            counter[str(name)] += 1

    unknown = []
    for name, count in counter.most_common(50):
        if count < MIN_UNKNOWN_KERNEL_OCCURRENCES:
            continue
        unknown.append({
            "kernel_name": name,
            "occurrence_count": count,
            "pct_of_total_events": round(count / total_events * 100, 2) if total_events else 0,
        })
    return unknown


def _collect_segmentation_issues(output_dir: Path) -> list[dict[str, Any]]:
    """Extract segmentation issues from segment_manifest.json."""
    seg_manifest = read_json(output_dir / "segment_manifest.json", default={}) or {}
    issues: list[dict[str, Any]] = []

    hard_errors = seg_manifest.get("hard_errors") or []
    for err in hard_errors[:10]:
        issues.append({
            "type": "hard_error",
            "rank_id": err.get("rank_id"),
            "error_type": err.get("error_type"),
            "detail": err.get("detail") or err.get("error") or "",
            "row_range": err.get("row_range"),
        })

    interior_total = seg_manifest.get("interior_island_total", 0)
    if interior_total > 0:
        issues.append({
            "type": "interior_islands",
            "total_count": interior_total,
        })

    return issues


def _collect_config_observations(
    characterize_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flag config detections that need human review."""
    config = characterize_data.get("config_signatures") or {}
    observations: list[dict[str, Any]] = []

    gm = config.get("graph_mode") or {}
    if gm.get("detected") in ("unclear",) or gm.get("confidence") == "low":
        observations.append({
            "signal": "graph_mode",
            "detected": gm.get("detected"),
            "confidence": gm.get("confidence"),
        })

    ab = config.get("attention_backend") or {}
    if ab.get("confidence") == "low" or ab.get("detected") == ["unknown"]:
        observations.append({
            "signal": "attention_backend",
            "detected": ab.get("detected"),
        })

    # MoE family: check if moe dispatch is detected but family is unknown
    moe = config.get("moe_dispatch") or {}
    if moe.get("detected") not in (None, "not_applicable", "unknown"):
        # MoE present — check if we have a family classification
        op_chars = characterize_data.get("operator_characterizations") or []
        has_moe_categories = any(
            "moe." in str(c.get("roles") or "")
            for c in op_chars
        )
        if has_moe_categories and not any(
            "moe." in str(o.get("signal") or "")
            for o in observations
        ):
            observations.append({
                "signal": "moe_present",
                "note": "MoE dispatch detected but family not classified — check moe_families.yaml coverage.",
            })

    return observations


def _check_version_gap(user_version: str) -> dict[str, Any] | None:
    """Check if user's vLLM-Ascend version is covered by known guides."""
    if not user_version:
        return None

    nearest = _find_nearest_known_version(user_version)
    if nearest == user_version:
        return None

    return {
        "type": "version_gap",
        "user_version": user_version,
        "nearest_known": nearest,
    }


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    """Parse a version string into an integer tuple for comparison.

    Handles common vLLM-Ascend version formats:
      "0.18.0" → (0, 18, 0)
      "0.22.1rc1" → (0, 22, 1)  (rc/beta/post suffixes stripped for comparison)
    """
    import re

    # Strip pre-release / build suffixes for comparison purposes.
    # "0.22.1rc1" → "0.22.1", "0.17.0rc1" → "0.17.0"
    clean = re.sub(r"(rc|beta|post|dev|a|b)\d+.*$", "", version)
    parts = clean.replace("-", ".").split(".")
    result: list[int] = []
    for p in parts:
        try:
            result.append(int(p))
        except ValueError:
            break
    return tuple(result) if result else (0,)


def _find_nearest_known_version(user_version: str) -> str:
    """Find the closest known version in the knowledge base.

    Uses semver-aware comparison — ``0.17.0rc1`` sorts before ``0.17.0``.
    Prefers an exact match; otherwise returns the highest known version
    that is ≤ the user's version. Falls back to the lowest known version
    if no lower version exists.
    """
    knowledge_dir = Path(__file__).resolve().parent / "knowledge" / "vllm-ascend"
    if not knowledge_dir.is_dir():
        return "unknown"

    known = sorted(f.stem.lstrip("v") for f in knowledge_dir.glob("v*.md"))
    if not known:
        return "unknown"
    if user_version in known:
        return user_version

    user_tuple = _parse_version_tuple(user_version)
    # Sort known versions by their parsed tuples.
    known_tuples = [(v, _parse_version_tuple(v)) for v in known]
    known_tuples.sort(key=lambda x: x[1])

    best = None
    for ver, ver_tuple in known_tuples:
        if ver_tuple <= user_tuple:
            best = ver
    return best if best is not None else known_tuples[0][0]


def _append_to_persistent_log(payload: dict[str, Any], output_dir: Path) -> None:
    """Write a summary line to the per-run persistent log.

    This file is pulled back with other artifacts. The agent accumulates
    lines from multiple run dirs into a single cross-run persistent log
    (see SKILL.md for the procedure).
    """
    summary = {
        "created_at": payload["created_at"],
        "profile_root": str(output_dir),
        "statistics": payload["statistics"],
        "top_unknown_kernels": [
            {"name": k["kernel_name"], "count": k["occurrence_count"]}
            for k in payload["observations"]["unknown_kernels"][:5]
        ],
        "has_segmentation_issues": len(payload["observations"]["segmentation_issues"]) > 0,
        "version_gap": payload["observations"].get("version_gap"),
    }
    summary_line = json.dumps(summary, ensure_ascii=False) + "\n"

    history_path = output_dir / "observations_history.jsonl"
    try:
        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write(summary_line)
    except (OSError, PermissionError):
        pass


def _load_persistent_history(*, output_dir: Path | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """Load recent entries from the persistent aggregate log.

    Tries in order: per-run file (survives pull-back) → local agent cache.
    Returns up to ``limit`` most recent entries, or empty list.
    """
    candidates = []
    if output_dir:
        candidates.append(output_dir / "observations_history.jsonl")
    candidates.append(PERSISTENT_LOCAL_CACHE)

    for path in candidates:
        if not path.is_file():
            continue
        lines: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass
        if lines:
            return lines[-limit:]
    return []


def _aggregate_historical_stats(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize historical observations for the agent to present context.

    This tells the agent "across your last N runs, these kernels have been
    consistently unrecognized" — much more useful than per-run snapshots.
    """
    if not history:
        return {"total_runs": 0}

    kernel_counter: Counter = Counter()
    runs_with_seg_issues = 0
    version_gaps: set[str] = set()
    total_runs = len(history)

    for entry in history:
        for k in entry.get("top_unknown_kernels") or []:
            kernel_counter[k["name"]] += k.get("count", 0)
        if entry.get("has_segmentation_issues"):
            runs_with_seg_issues += 1
        vg = entry.get("version_gap")
        if vg:
            version_gaps.add(vg.get("user_version", ""))

    return {
        "total_runs": total_runs,
        "runs_with_seg_issues": runs_with_seg_issues,
        "persistent_unknown_kernels": [
            {"name": name, "total_count": count}
            for name, count in kernel_counter.most_common(10)
        ],
        "version_gaps_seen": sorted(version_gaps),
    }


def collect_observations(
    output_dir: Path,
    *,
    user_vllm_ascend_version: str = "",
) -> dict[str, Any]:
    """Entry point. Collect observations, write per-run + persistent."""
    output_dir.mkdir(parents=True, exist_ok=True)

    unknown_kernels = _collect_unknown_kernels(output_dir)
    segmentation_issues = _collect_segmentation_issues(output_dir)

    characterize_data = read_json(output_dir / "characterizations.json", default={}) or {}
    config_observations = _collect_config_observations(characterize_data)

    version_gap = None
    if user_vllm_ascend_version:
        version_gap = _check_version_gap(user_vllm_ascend_version)

    statistics = {
        "unknown_kernel_names": len(unknown_kernels),
        "segmentation_issues": len(segmentation_issues),
        "config_observations": len(config_observations),
        "has_version_gap": version_gap is not None,
        "total_observations": (
            len(unknown_kernels) + len(segmentation_issues) + len(config_observations) + (1 if version_gap else 0)
        ),
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "analysis_stage": "observations",
        "created_at": utc_now(),
        "statistics": statistics,
        "observations": {
            "unknown_kernels": unknown_kernels,
            "segmentation_issues": segmentation_issues,
            "config_observations": config_observations,
            "version_gap": version_gap,
            # Agent writes corrections here during interactive follow-up.
            # Each correction has: signal, script_detected, user_confirmed,
            # user_explanation.
            "user_corrections": [],
        },
        # Historical context — helps the agent prioritize across runs.
        "historical_aggregate": _aggregate_historical_stats(_load_persistent_history(output_dir=output_dir)),
    }

    write_json(output_dir / OUTPUT_JSON, payload)
    write_json(output_dir / OUTPUT_MANIFEST, {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "analysis_stage": "observations",
        "created_at": utc_now(),
        "output_dir": str(output_dir),
        "counts": statistics,
    })

    # Append to persistent log AFTER writing the per-run file.
    # The per-run file is the primary artifact; the persistent log is
    # a convenience for cross-run aggregation.
    _append_to_persistent_log(payload, output_dir)

    return payload
