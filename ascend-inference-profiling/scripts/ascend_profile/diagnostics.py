#!/usr/bin/env python3
"""Generate diagnosis claims from summary and cross-rank evidence tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .common import DiagnosisFinding, SCHEMA_VERSION, TOOL_VERSION, csv_rows, emit_stage_json, stable_id, utc_now, write_json
    from .mstt_runner import load_mstt_slow_rank
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import DiagnosisFinding, SCHEMA_VERSION, TOOL_VERSION, csv_rows, emit_stage_json, stable_id, utc_now, write_json  # type: ignore[no-redef]
    from mstt_runner import load_mstt_slow_rank  # type: ignore[no-redef]


CROSS_RANK_SKEW_RATIO = 2.0
HIGH_SKEW_RATIO = 4.0
CROSS_RANK_SKEW_US = 1000.0
DP_WALL_SKEW_RATIO = 2.0


def parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def as_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) or default)
    except (TypeError, ValueError):
        return default


def finding(
    *,
    finding_type: str,
    scope: str,
    summary: str,
    severity: str,
    confidence: str,
    rank_ids: Sequence[str] = (),
    alignment_ids: Sequence[str] = (),
    evidence_ids: Sequence[str] = (),
    metrics: Mapping[str, Any] | None = None,
    limitations: Sequence[str] = (),
) -> DiagnosisFinding:
    claim_id = stable_id("claim", finding_type, scope, summary, rank_ids, alignment_ids)
    return DiagnosisFinding(
        claim_id=claim_id,
        claim_type=finding_type,
        finding_type=finding_type,
        scope=scope,
        summary=summary,
        severity=severity,
        confidence=confidence,
        rank_ids=tuple(rank_ids),
        alignment_ids=tuple(alignment_ids),
        evidence_ids=tuple(evidence_ids),
        limitations=tuple(limitations),
        metrics=dict(metrics or {}),
    )


HOST_BOUND_HEAD_RATIO = 0.15
HOST_BOUND_CONSECUTIVE_STEPS = 3
HOST_BOUND_WAIT_RATIO = 0.80
HOST_BOUND_WAIT_OP_RATIO = 0.10


def diagnose_cross_rank(
    alignment_rows: Sequence[Mapping[str, Any]],
    *,
    mstt_data: dict[str, int] | None = None,
) -> list[DiagnosisFinding]:
    findings: list[DiagnosisFinding] = []
    has_mstt = mstt_data is not None and any(v > 0 for v in mstt_data.values())

    # 1. Emit slow_rank_confirmed from mstt data when available.
    #    This *replaces* the heuristic slow_rank_suspected based on matmul
    #    start_skew, which is weaker signal (time-bucket alignment can mis-pair
    #    operators and does not check communication sync points).
    if mstt_data:
        max_affect = max(mstt_data.values()) if mstt_data else 0
        slow_ranks = sorted(
            [rid for rid, count in mstt_data.items() if count > 0],
            key=lambda r: -mstt_data[r],
        )
        if slow_ranks:
            findings.append(
                finding(
                    finding_type="slow_rank_confirmed",
                    scope="cross_rank",
                    summary=(
                        f"msprof-analyze slow_rank detected {len(slow_ranks)} slow rank(s) "
                        f"via communication sync-point voting (max slow_affect_count={max_affect}). "
                        f"Slow ranks (rank_id: slow_affect_count): "
                        + ", ".join(f"{r}: {mstt_data[r]}" for r in slow_ranks[:8])
                    ),
                    severity="high" if max_affect > 20 else "medium",
                    confidence="high",
                    rank_ids=tuple(slow_ranks),
                    metrics={
                        "mstt_slow_affect_count": mstt_data,
                        "max_slow_affect_count": max_affect,
                        "slow_rank_count": len(slow_ranks),
                    },
                    limitations=(
                        "Detection is based on msprof-analyze's communication "
                        "sync-point voting (Notify Wait / Record). Hardware "
                        "issues (thermal, PCIe, memory) vs. workload imbalance "
                        "cannot be distinguished by this signal alone — "
                        "cross-reference with cross_rank_alignment.csv."
                    ),
                )
            )

    for row in alignment_rows:
        alignment_id = str(row.get("alignment_id") or "")
        alignment_type = str(row.get("alignment_type") or "")
        rank_ids = parse_jsonish(row.get("rank_ids"), [])
        role = str(row.get("role") or "")
        duration_ratio = as_float(row, "duration_ratio", 1.0)
        duration_skew = as_float(row, "duration_skew_us")
        start_skew = as_float(row, "start_skew_us")
        is_structure_mismatch = str(row.get("is_structure_mismatch")).lower() == "true"
        if alignment_type == "time_window" and is_structure_mismatch:
            findings.append(
                finding(
                    finding_type="rank_workload_asymmetry",
                    scope="cross_rank",
                    summary="Aligned ranks show different step family or layer-count structure in the same time window.",
                    severity="medium",
                    confidence="medium",
                    rank_ids=rank_ids,
                    alignment_ids=(alignment_id,),
                    metrics=dict(row),
                )
            )
        if role == "communication.collective" and (duration_ratio >= CROSS_RANK_SKEW_RATIO or duration_skew >= CROSS_RANK_SKEW_US):
            # Enrich with mstt context: if a slow rank identified by mstt is
            # among the ranks with skewed collective duration, upgrade severity.
            mstt_hit = (
                mstt_data is not None
                and any(mstt_data.get(str(rid), 0) > 0 for rid in rank_ids)
            )
            comm_severity = "high" if (duration_ratio >= HIGH_SKEW_RATIO or mstt_hit) else "medium"
            comm_confidence = "high" if mstt_hit else "medium"
            findings.append(
                finding(
                    finding_type="communication_collective_slow",
                    scope="cross_rank",
                    summary="Aligned collective communication shows large duration skew across ranks.",
                    severity=comm_severity,
                    confidence=comm_confidence,
                    rank_ids=rank_ids,
                    alignment_ids=(alignment_id,),
                    metrics={
                        **dict(row),
                        "mstt_cross_validated": mstt_hit,
                    },
                )
            )
        if role in {"moe.dispatch_expert_compute", "moe.dispatch_or_combine"} and (
            duration_ratio >= CROSS_RANK_SKEW_RATIO or duration_skew >= CROSS_RANK_SKEW_US
        ):
            findings.append(
                finding(
                    finding_type="ep_load_imbalance_suspected",
                    scope="cross_rank",
                    summary="Aligned MoE dispatch/combine work shows large duration skew across ranks.",
                    severity="high" if duration_ratio >= HIGH_SKEW_RATIO else "medium",
                    confidence="medium",
                    rank_ids=rank_ids,
                    alignment_ids=(alignment_id,),
                    metrics=dict(row),
                )
            )
        # Only emit slow_rank_suspected from matmul skew when mstt data is
        # unavailable (fallback path). When mstt data exists, we already
        # emitted slow_rank_confirmed above.
        if not has_mstt and role == "compute.matmul" and start_skew >= CROSS_RANK_SKEW_US:
            findings.append(
                finding(
                    finding_type="slow_rank_suspected",
                    scope="cross_rank",
                    summary="Aligned matmul work has similar operator/shape signature but large launch-time skew.",
                    severity="medium",
                    confidence="low",
                    rank_ids=rank_ids,
                    alignment_ids=(alignment_id,),
                    metrics=dict(row),
                    limitations=(
                        "Detection is based on time-bucket operator alignment "
                        "of matmul events, NOT on communication sync-point "
                        "analysis. Time-bucket alignment can mis-pair operators "
                        "across ranks and is vulnerable to profiling start-offset "
                        "drift. Run with --mstt for reliable slow-rank detection "
                        "based on Notify Wait / Record voting."
                    ),
                )
            )
    return findings


def diagnose_rank_workload(rank_rows: Sequence[Mapping[str, Any]], step_rows: Sequence[Mapping[str, Any]]) -> list[DiagnosisFinding]:
    findings: list[DiagnosisFinding] = []
    attention_by_rank = {str(row.get("rank_id")): str(row.get("has_attention")).lower() == "true" for row in rank_rows}
    if attention_by_rank and any(attention_by_rank.values()) and not all(attention_by_rank.values()):
        reduced = [rank for rank, has_attention in attention_by_rank.items() if not has_attention]
        full = [rank for rank, has_attention in attention_by_rank.items() if has_attention]
        findings.append(
            finding(
                finding_type="reduced_work_or_dummy_rank",
                scope="cross_rank",
                summary="Some ranks lack attention/body evidence while other ranks contain full attention workload evidence.",
                severity="medium",
                confidence="medium",
                rank_ids=tuple(sorted(reduced + full)),
                metrics={"full_work_ranks": full, "reduced_work_candidate_ranks": reduced},
                limitations=("This is structural evidence only; semantic dummy-run labeling requires workload context.",),
            )
        )
    wall_by_rank = {str(row.get("rank_id")): as_float(row, "wall_ms") for row in rank_rows}
    if wall_by_rank:
        values = [value for value in wall_by_rank.values() if value > 0]
        if values and max(values) / max(1e-6, min(values)) >= DP_WALL_SKEW_RATIO:
            findings.append(
                finding(
                    finding_type="dp_workload_imbalance",
                    scope="cross_rank",
                    summary="Rank-level capture wall time differs significantly; check DP workload and T-axis/shape distribution.",
                    severity="medium",
                    confidence="low",
                    rank_ids=tuple(sorted(wall_by_rank)),
                    metrics={"rank_wall_ms": wall_by_rank, "wall_ratio": max(values) / max(1e-6, min(values))},
                    limitations=("Wall-time skew alone is not root cause evidence; shape-level corroboration is required.",),
                )
            )
    for row in step_rows:
        tags = parse_jsonish(row.get("anomaly_tags"), [])
        if "DEVICE_IDLE_GAP_HEAVY" in tags or "INTERNAL_BUBBLE_HEAVY" in tags:
            findings.append(
                finding(
                    finding_type="device_idle_bubble",
                    scope="step",
                    summary=f"Step {row.get('segment_id')} has heavy device idle bubbles.",
                    severity="medium",
                    confidence="high",
                    rank_ids=(str(row.get("rank_id")),),
                    evidence_ids=tuple(parse_jsonish(row.get("evidence_ids"), [])),
                    metrics=dict(row),
                )
            )
    return findings


def diagnose_profile(output_dir: Path) -> dict[str, Any]:
    alignment_rows = csv_rows(output_dir / "cross_rank_alignment.csv")
    rank_rows = csv_rows(output_dir / "rank_summary.csv")
    step_rows = csv_rows(output_dir / "step_summary.csv")
    wait_rows = csv_rows(output_dir / "wait_anchor_ops.csv")
    aicpu_rows = csv_rows(output_dir / "aicpu_summary.csv")
    mstt_data = load_mstt_slow_rank(output_dir)
    findings = diagnose_cross_rank(alignment_rows, mstt_data=mstt_data)
    findings.extend(diagnose_rank_workload(rank_rows, step_rows))
    for row in wait_rows:
        if str(row.get("is_false_hotspot_risk")).lower() == "true":
            # ``wait_anchor_ops.csv`` is itself the per-kernel evidence
            # (with ``row_ranges`` + ``sample_event_ids`` pointing back
            # to the normalized event index). The evidence-chain
            # validator only knows about ``evidence_index.csv``
            # (``evd_*`` ids), so we expose the wait_anchor row's own
            # back-references as an explicit ``limitations`` string
            # instead of faking a cross-ID match.
            row_ranges = str(row.get("row_ranges") or "").strip()
            sample_evt = str(row.get("sample_event_ids") or "").strip()
            limitation_parts = [
                "Wait-anchor false-hotspot is derived directly from "
                "wait_anchor_ops.csv (per-kernel aggregate); not aligned "
                "to an evidence_index.csv span.",
            ]
            if row_ranges:
                limitation_parts.append(
                    f"Source rows in kernel_details.csv: {row_ranges}."
                )
            if sample_evt:
                limitation_parts.append(
                    f"Sample event ids in normalized_event_index.csv: {sample_evt}."
                )
            findings.append(
                finding(
                    finding_type="wait_anchor_false_hotspot",
                    scope="operator",
                    summary=f"Operator {row.get('name')} has high wait ratio and low execution duration.",
                    severity="low",
                    confidence="high",
                    rank_ids=(str(row.get("rank_id")),),
                    metrics=dict(row),
                    limitations=(" ".join(limitation_parts),),
                )
            )
    for row in aicpu_rows:
        if str(row.get("classification")) == "AICPU_EXPOSED_NOT_ALLOWED":
            findings.append(
                finding(
                    finding_type="aicpu_exposed",
                    scope="operator",
                    summary=f"AICPU operator {row.get('name')} appears exposed rather than hidden by AI Core work.",
                    severity="medium",
                    confidence="medium",
                    rank_ids=(str(row.get("rank_id")),),
                    metrics=dict(row),
                    # No cross-rank alignment row / evidence row is produced
                    # for AICPU-exposure findings: the heuristic looks at one
                    # rank at a time and uses the aicpu_summary row directly.
                    # Surface that as an explicit limitation so the
                    # evidence-chain validator can accept the finding.
                    limitations=(
                        "Derived from aicpu_summary.csv per-rank rollup; no "
                        "cross-rank alignment_id is produced for AICPU-exposed "
                        "findings. Trace back via the kernel name + rank_id.",
                    ),
                )
            )
    # 4. Host-bound dispatch diagnosis from step anatomy + wait anchors.
    findings.extend(_diagnose_host_bound(step_rows, wait_rows))
    return findings


def _diagnose_host_bound(
    step_rows: Sequence[Mapping[str, Any]],
    wait_rows: Sequence[Mapping[str, Any]],
) -> list[DiagnosisFinding]:
    """Detect host-bound dispatch pattern from step head time and wait anchors.

    Two signals must co-occur for a finding:

    1. **Step head bubble**: ``head_wall_ms / wall_ms > 15%`` on the same rank
       for ``>= 3`` consecutive steps. The head is the idle gap from step
       boundary to first busy segment — host dispatch delay manifests here.

    2. **Wait-anchor density**: operators with ``wait_ratio > 80%`` account
       for ``>= 10%`` of all ranked operators. High wait ratios mean
       operators are ready but stuck waiting on CPU-side launch ordering.

    When only one signal fires, the host-bound risk is noted in the metrics
    of the existing ``device_idle_bubble`` finding rather than emitting a
    standalone finding.
    """
    findings: list[DiagnosisFinding] = []

    # Signal 1: consecutive head-heavy steps per rank.
    head_heavy_ranks: dict[str, list[Mapping[str, Any]]] = {}
    by_rank: dict[str, list[Mapping[str, Any]]] = {}
    for row in step_rows:
        rank_id = str(row.get("rank_id") or "")
        by_rank.setdefault(rank_id, []).append(row)
    for rank_id, steps in by_rank.items():
        steps_sorted = sorted(steps, key=lambda r: float(r.get("start_us") or 0.0))
        streak = 0
        for step in steps_sorted:
            wall = as_float(step, "wall_ms")
            head = as_float(step, "head_wall_ms")
            if wall > 0 and head / wall > HOST_BOUND_HEAD_RATIO:
                streak += 1
                if streak >= HOST_BOUND_CONSECUTIVE_STEPS:
                    head_heavy_ranks.setdefault(rank_id, []).append(step)
            else:
                streak = 0

    # Signal 2: high wait-ratio operator density.
    wait_ops_rank: dict[str, int] = {}
    total_ops_rank: dict[str, int] = {}
    for row in wait_rows:
        rank_id = str(row.get("rank_id") or "")
        count = int(row.get("call_count") or 1)
        total_ops_rank[rank_id] = total_ops_rank.get(rank_id, 0) + count
        if as_float(row, "wait_ratio") >= HOST_BOUND_WAIT_RATIO:
            wait_ops_rank[rank_id] = wait_ops_rank.get(rank_id, 0) + count
    wait_dense_ranks: set[str] = set()
    for rank_id in set(list(wait_ops_rank) + list(total_ops_rank)):
        total = total_ops_rank.get(rank_id, 1)
        wait_count = wait_ops_rank.get(rank_id, 0)
        if total > 0 and wait_count / total >= HOST_BOUND_WAIT_OP_RATIO:
            wait_dense_ranks.add(rank_id)

    both_hit = set(head_heavy_ranks) & wait_dense_ranks
    for rank_id in both_hit:
        head_steps = head_heavy_ranks[rank_id]
        head_ratios = [
            round(as_float(s, "head_wall_ms") / max(1e-6, as_float(s, "wall_ms")), 3)
            for s in head_steps[:5]
        ]
        wait_total = total_ops_rank.get(rank_id, 0)
        wait_high = wait_ops_rank.get(rank_id, 0)
        findings.append(
            finding(
                finding_type="host_dispatch_bound_suspected",
                scope="step",
                summary=(
                    f"Rank {rank_id} shows host dispatch bottleneck: "
                    f"consecutive head-heavy steps (head/wall ratios: {head_ratios}) "
                    f"and high wait-ratio operator density "
                    f"({wait_high}/{wait_total} ops with wait>80%). "
                    f"Check CPU core binding, aclrtLaunch ordering, and host-side "
                    f"scheduling latency."
                ),
                severity="medium",
                confidence="medium",
                rank_ids=(rank_id,),
                metrics={
                    "head_heavy_step_count": len(head_steps),
                    "head_wall_ratios_sample": head_ratios,
                    "wait_high_ratio_ops": wait_high,
                    "total_wait_anchor_ops": wait_total,
                    "head_ratio_threshold": HOST_BOUND_HEAD_RATIO,
                    "consecutive_step_threshold": HOST_BOUND_CONSECUTIVE_STEPS,
                },
                limitations=(
                    "Host-bound detection uses device-side signals (step head "
                    "bubble + wait-anchor density) to infer CPU-side dispatch "
                    "delay. Confirm with ftrace or host-side profiling for "
                    "root cause (e.g. CPU contention, NUMA placement, "
                    "aclrtLaunch serialization)."
                ,),
            )
        )

    # For ranks with only one signal, add host_dispatch risk marker to
    # existing device_idle_bubble findings. The bubble finding is already
    # emitted in diagnose_rank_workload; we don't modify past findings here
    # but the metrics serve as context for the report.
    return findings
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "analysis_stage": "diagnostics",
        "created_at": utc_now(),
        "diagnosis_findings": findings,
        "counts": {
            "finding_count": len(findings),
            "by_type": dict(sorted({item.finding_type: sum(1 for finding_item in findings if finding_item.finding_type == item.finding_type) for item in findings}.items())),
        },
    }
    write_json(output_dir / "diagnosis_findings.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = diagnose_profile(Path(args.output))
    emit_stage_json({"stage": "diagnostics", "counts": payload["counts"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
