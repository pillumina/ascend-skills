#!/usr/bin/env python3
"""Characterize operators and blocks using quantitative metrics.

This stage does NOT emit optimisation advice. It outputs measurable
observations keyed to specific operator classes so the report can
surface them in a traceable table.

Two tiers of observation:

  L0 (measured, high confidence):
    - bound_family from pipeline stage measurements (hardware-profiled)
    - operator duration and call count
    - parsed M/K/N dimensions where shape extraction succeeds

  L1 (derived, medium confidence):
    - arithmetic intensity (AI) from parsed shapes: a pure mathematical
      value computed as FLOPs / Bytes for the FP16 matmul workload.
      No hardware bandwidth is needed for this computation.
    - shape-size patterns: M=1 (single-token decode), K≪M,N etc.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

try:
    from .common import PIPELINE_FIELDS, csv_rows, read_json, read_jsonl, SCHEMA_VERSION, TOOL_VERSION, utc_now, write_json
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import PIPELINE_FIELDS, csv_rows, read_json, read_jsonl, SCHEMA_VERSION, TOOL_VERSION, utc_now, write_json  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_JSON = "characterizations.json"
OUTPUT_MANIFEST = "characterize_manifest.json"

HARDWARE_YAML = Path(__file__).resolve().parent / "knowledge" / "hardware_capabilities.yaml"

DEFAULT_DEVICE = "ascend_910b2"

# Bound-family sets (from semantic_conventions.yaml / bound_classification.md).
#
# Memory-bound families: the dominant pipeline cost is data movement
# (MTE = Memory Transfer Engine), not compute.
_MEMORY_BOUND_FAMILIES = frozenset({
    "mte1", "mte2", "aic_mte", "aiv_mte",
})

# Compute-bound families: the dominant cost is MAC or vector ALU.
_COMPUTE_BOUND_FAMILIES = frozenset({
    "mac", "vec",
})

# Operators whose input shapes carry interpretable M/K/N dimensions.
_MNK_OP_PATTERNS = frozenset({
    "matmul", "batchmatmul", "groupedmatmul",
    "fusedinferattentionscore", "unpadflashattention",
})


# ---------------------------------------------------------------------------
# Hardware identification
# ---------------------------------------------------------------------------

def _load_hardware() -> dict[str, Any] | None:
    if not HARDWARE_YAML.is_file():
        return None
    import yaml as _yaml
    return _yaml.safe_load(HARDWARE_YAML.read_text(encoding="utf-8"))


def _identify_device(output_dir: Path) -> dict[str, Any]:
    """Identify the device from the normalize manifest.

    Falls back to DEFAULT_DEVICE when the manifest contains no device info.
    """
    hw_doc = _load_hardware()
    norm_manifest = read_json(output_dir / "normalize_manifest.json", default={}) or {}
    # Future: normalize.py may record the detected device name.
    device_name = norm_manifest.get("device") or DEFAULT_DEVICE
    device_info: dict[str, Any] = {}
    if hw_doc:
        devices = hw_doc.get("devices") or {}
        device_info = devices.get(device_name, {})
    device_info.setdefault("display_name", device_name)
    device_info.setdefault("architecture", "unknown")
    device_info.setdefault("peak_fp16_tflops", None)
    return device_info


def _compute_roofline_ridge(peak_tflops: float | None, device_info: dict[str, Any]) -> float | None:
    """Compute the roofline ridge point (FLOP/byte) from peak FLOPs and HBM bandwidth.

    Ridge = peak_FP16_FLOPs / peak_HBM_bandwidth

    Operators with arithmetic intensity > ridge are compute-bound;
    operators with AI < ridge are memory-bound.

    Returns None when either peak FLOPs or bandwidth is unavailable.
    """
    if peak_tflops is None:
        return None
    bw = device_info.get("memory_bandwidth_gb_s")
    if bw is None:
        return None
    # peak_tflops in TFLOPS (10^12 FLOP/s), bw in GB/s (10^9 bytes/s).
    # ridge (FLOP/byte) = TFLOPS * 1e12 / (GB/s * 1e9) = TFLOPS * 1000 / GB_s
    return round(peak_tflops * 1000.0 / bw, 1)


# ---------------------------------------------------------------------------
# Shape parsing (M / K / N)
# ---------------------------------------------------------------------------

def _parse_matmul_shape(raw_text: str) -> dict[str, Any] | None:
    """Parse a CANN Input-Shapes string into {M, K, N, rule}.

    Returns None when the shape is unrecognised or unparseable.

    Supported patterns (from awesome-ascend-skills extract_op_shapes.py):

      basic-2x2-auto:  M,K ; K,N  or  M,K ; N,K    (K matched adaptively)
      packed-2x4-bc:   M,K ; A,B,C,D  where K = B*C
      packed-2x4-ad:   M,K ; A,B,C,D  where K = A*D
      batched-3x2:     B,M,K ; K,N  or  B,M,K ; N,K
    """
    text = (
        raw_text.strip()
        .replace("\"", "").replace("'", "")
        .replace("，", ",").replace("；", ";")
        .replace("[", "").replace("]", "")
        .replace("(", "").replace(")", "")
    )
    pieces = [p.strip() for p in text.split(";") if p.strip()]
    if len(pieces) < 2:
        return None
    pieces = pieces[:2]

    def _dims(part: str) -> list[int]:
        out: list[int] = []
        for token in part.split(","):
            token = token.strip()
            if token:
                try:
                    out.append(int(token))
                except ValueError:
                    return []
        return out

    left = _dims(pieces[0])
    right = _dims(pieces[1])
    if not left or not right:
        return None

    # ── 2×2: M,K ; K,N  or  M,K ; N,K ──
    if len(left) == 2 and len(right) == 2:
        m, k_left = left
        if right[0] == k_left:
            return {"M": m, "K": k_left, "N": right[1], "rule": "basic-2x2-auto"}
        if right[1] == k_left:
            return {"M": m, "K": k_left, "N": right[0], "rule": "basic-2x2-auto"}
        return {"M": m, "K": k_left, "N": right[1], "rule": "basic-2x2-positional"}

    # ── 2×4 packed: M,K ; A,B,C,D ──
    if len(left) == 2 and len(right) == 4:
        m, k = left
        a, b_val, c, d = right
        if b_val * c == k:
            return {"M": m, "K": k, "N": a * d, "rule": "packed-2x4-bc"}
        if a * d == k:
            return {"M": m, "K": k, "N": b_val * c, "rule": "packed-2x4-ad"}
        return None

    # ── 3×2 batched: B,M,K ; K,N ──
    if len(left) == 3 and len(right) == 2:
        b, m, k_left = left
        if right[0] == k_left:
            return {"M": m, "K": k_left, "N": right[1], "batch": b, "rule": "batched-3x2"}
        if right[1] == k_left:
            return {"M": m, "K": k_left, "N": right[0], "batch": b, "rule": "batched-3x2"}
        return None

    return None


def _op_matches_keywords(name: str) -> bool:
    lowered = name.lower()
    for kw in _MNK_OP_PATTERNS:
        if kw in lowered:
            return True
    return False


# ---------------------------------------------------------------------------
# Arithmetic intensity (FP16 matmul, pure math — no hardware bandwidth needed)
# ---------------------------------------------------------------------------

def _arithmetic_intensity(m: int, k: int, n: int) -> float:
    """FLOPs/Byte for an FP16 M×K × K×N matmul.

    FLOPs = 2 × M × K × N  (one multiply + one add per output element)
    Bytes = 2 × (M×K + K×N + M×N)  (FP16 = 2 bytes/element, read A+B, write C)

    Returns FLOPs/Byte.  This is a pure function of the shape — no
    hardware bandwidth is assumed or required.
    """
    if m <= 0 or k <= 0 or n <= 0:
        return 0.0
    flops = 2.0 * m * k * n
    bytes_total = 2.0 * (m * k + k * n + m * n)
    if bytes_total <= 0:
        return 0.0
    return flops / bytes_total


def _observed_bandwidth_gb_s(m: int, k: int, n: int, call_count: int, mte_time_us: float) -> float:
    """Observed data-movement bandwidth through the MTE pipeline stages.

    Denominator is the **sum of MTE pipeline stage times** (aic_mte1,
    aic_mte2, aic_fixpipe, aiv_mte2, aiv_mte3), NOT the total operator
    duration.  This measures how fast data actually moved through the
    memory transfer engines, independent of how much time was spent on
    MAC compute or scalar ops.

    bytes_per_call = 2 × (M×K + K×N + M×N)  # FP16 reads + write
    total_bytes   = bytes_per_call × call_count
    total_time_s  = mte_time_us / 1e6

    Returns GB/s.  Uses measured pipeline stage times — no hardware
    bandwidth assumptions.
    """
    if m <= 0 or k <= 0 or n <= 0 or mte_time_us <= 0 or call_count <= 0:
        return 0.0
    bytes_per_call = 2.0 * (m * k + k * n + m * n)  # FP16
    total_bytes = bytes_per_call * call_count
    total_time_s = mte_time_us / 1e6
    return total_bytes / total_time_s / 1e9


_MTE_FIELDS = ("aic_mte1_time", "aic_mte2_time", "aic_fixpipe_time", "aiv_mte2_time", "aiv_mte3_time")


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _stage_pct(op: Mapping[str, Any]) -> float:
    """What fraction of total pipeline time the bound_stage accounts for."""
    total = 0.0
    target_stage = str(op.get("bound_stage") or "")
    for field in PIPELINE_FIELDS:
        total += _as_float(op.get(field))
    target_val = _as_float(op.get(target_stage))
    return round(target_val / total * 100, 1) if total > 0 else 0.0


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Characterization
# ---------------------------------------------------------------------------

def _characterize_operator(
    op: Mapping[str, Any],
    mnk: dict[str, Any] | None,
    architecture: str,
) -> dict[str, Any]:
    """Produce a characterization dict for one operator class."""
    bound_family = str(op.get("bound_family") or "unknown")
    bound_stage = str(op.get("bound_stage") or "unknown")
    name = str(op.get("name") or "")
    stage_pct_val = _stage_pct(op)

    result: dict[str, Any] = {
        "operator_name": name,
        "op_type": op.get("op_type"),
        "roles": op.get("roles"),
        "bound_family": bound_family,
        "bound_stage": bound_stage,
        "bound_stage_pct": stage_pct_val,
    }

    # L0: memory-bound vs compute-bound from MEASURED pipeline data.
    # This is the single most reliable signal — it comes from hardware
    # profiling, not theoretical estimation.
    if bound_family in _MEMORY_BOUND_FAMILIES:
        mb_label = "memory-bound"
        mb_confidence = "high"
    elif bound_family in _COMPUTE_BOUND_FAMILIES:
        mb_label = "compute-bound"
        mb_confidence = "high"
    elif bound_family in ("mixed",):
        mb_label = "mixed-bound"
        mb_confidence = "medium"
    else:
        mb_label = bound_family
        mb_confidence = "medium"

    result["bound_classification"] = mb_label
    result["bound_confidence"] = mb_confidence

    # Build the human-readable characterization.
    parts: list[str] = []

    if mnk:
        m, k_val, n_val = mnk.get("M"), mnk.get("K"), mnk.get("N")
        if all(v is not None and v > 0 for v in (m, k_val, n_val)):
            ai = _arithmetic_intensity(int(m), int(k_val), int(n_val))
            result["shape"] = {"M": m, "K": k_val, "N": n_val, "rule": mnk.get("rule")}
            result["arithmetic_intensity"] = round(ai, 2)

            # MTE bandwidth: total bytes / MTE pipeline stage time.
            # Uses measured MTE stage durations from the profiler — no
            # hardware bandwidth assumptions. Works for both memory-bound
            # (low BW, saturated MTEs) and compute-bound (high BW, MTEs
            # idle while MAC computes) operators.
            call_count = max(1, int(_as_float(op.get("call_count"), default=1.0)))
            mte_us = sum(_as_float(op.get(f)) for f in _MTE_FIELDS)
            obs_bw = _observed_bandwidth_gb_s(int(m), int(k_val), int(n_val), call_count, mte_us)
            if obs_bw > 0:
                result["observed_bandwidth_gb_s"] = round(obs_bw, 2)

            # L0: What the measured data tells us.
            if bound_family in _MEMORY_BOUND_FAMILIES:
                parts.append(
                    f"Memory-bound ({bound_family}): "
                    f"{stage_pct_val:.0f}% of pipeline in {bound_stage}. "
                )
            elif bound_family in _COMPUTE_BOUND_FAMILIES:
                parts.append(
                    f"Compute-bound ({bound_family}): "
                    f"{stage_pct_val:.0f}% of pipeline in {bound_stage}. "
                )
            else:
                parts.append(
                    f"Bound family={bound_family}: "
                    f"{stage_pct_val:.0f}% in {bound_stage}. "
                )

            # L1: What the shape tells us about WHY.
            if m == 1:
                result["decode_like"] = True
                parts.append(
                    f"M=1 (single-token decode): "
                    f"no batch dimension to amortise weight loads. "
                    f"AI={ai:.1f} FLOPs/Byte. "
                )
            else:
                parts.append(
                    f"Shape: M={m}, K={k_val}, N={n_val}, "
                    f"AI={ai:.1f} FLOPs/Byte. "
                )

            if bound_family in _MEMORY_BOUND_FAMILIES and k_val is not None and int(k_val) < 256:
                parts.append(
                    f"Small K={k_val}: limited data reuse per weight load — "
                    f"each weight byte is used only O(M×N / K) times. "
                )
            if obs_bw > 0:
                parts.append(
                    f"MTE BW: {obs_bw:.1f} GB/s (bytes moved / MTE pipeline time). "
                )
        else:
            parts.append(
                f"Shape parse returned invalid dims; "
                f"bound classification from measurement: {mb_label}. "
            )
    else:
        # No shape — purely measurement-based.
        if bound_family in _MEMORY_BOUND_FAMILIES:
            parts.append(
                f"Memory-bound ({bound_family}): "
                f"{stage_pct_val:.0f}% of pipeline in {bound_stage}. "
                f"No shape available — AI not computed. "
            )
        elif bound_family in _COMPUTE_BOUND_FAMILIES:
            parts.append(
                f"Compute-bound ({bound_family}): "
                f"{stage_pct_val:.0f}% of pipeline in {bound_stage}. "
                f"No shape available — AI not computed. "
            )
        else:
            parts.append(
                f"Bound family={bound_family} ({stage_pct_val:.0f}% in {bound_stage}). "
                f"No shape available. "
            )

    # A3 architecture note: dual-die means Cube/Vector can truly overlap.
    if architecture == "A3" and bound_family in ("mixed",):
        parts.append(
            "A3 dual-die: Cube and Vector may run on separate dies. "
            "Check per-die aicore_time / aiv_time in kernel_details.csv."
        )

    result["characterization"] = "".join(parts)
    # Bound classification is always from measured pipeline data → high.
    # Shape-derived characterisation text is L1 but the bound label itself
    # does not depend on shape availability.
    result["confidence"] = "high"
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def characterize_profile(output_dir: Path) -> dict[str, Any]:
    op_class_rows = csv_rows(output_dir / "operator_class_summary.csv")
    block_class_rows = csv_rows(output_dir / "block_class_summary.csv")
    hccl_class_rows = csv_rows(output_dir / "hccl_class_summary.csv")

    device_info = _identify_device(output_dir)
    architecture = str(device_info.get("architecture") or "unknown")
    device_display = str(device_info.get("display_name") or "unknown")
    peak_tflops = device_info.get("peak_fp16_tflops")
    peak_binning = device_info.get("peak_fp16_tflops_binning")

    # Collect shape data for matmul operators from the event index.
    op_shapes: dict[str, dict[str, Any]] = {}
    events_path = output_dir / "normalized_event_index.jsonl"
    if events_path.is_file():
        shape_counter: dict[str, Counter] = {}
        for event in read_jsonl(events_path):
            name = str(event.get("name_raw") or "")
            if not _op_matches_keywords(name):
                continue
            sf = event.get("shape_features") or {}
            raw_text = sf.get("raw_text")
            if not raw_text:
                continue
            mnk = _parse_matmul_shape(str(raw_text))
            if mnk is None:
                continue
            shape_key = str(mnk.get("M")) + "," + str(mnk.get("K")) + "," + str(mnk.get("N"))
            if name not in shape_counter:
                shape_counter[name] = Counter()
            shape_counter[name][shape_key] += 1

        for name, counter in shape_counter.items():
            top_key = counter.most_common(1)[0][0]
            parts = top_key.split(",")
            if len(parts) == 3:
                op_shapes[name] = {
                    "M": int(parts[0]), "K": int(parts[1]), "N": int(parts[2]),
                    "rule": "event-majority",
                }

    # Characterize operators (rank-merged).
    op_chars: list[dict[str, Any]] = []
    for op in op_class_rows:
        name = str(op.get("name") or "")
        if not _has_pipeline_signal(op):
            continue
        # Match shape by operator name (case-insensitive).
        mnk = None
        name_lower = name.lower()
        for shape_name, shape_data in op_shapes.items():
            if shape_name.lower() == name_lower:
                mnk = shape_data
                break

        ch = _characterize_operator(op, mnk, architecture)
        if ch.get("characterization"):
            op_chars.append(ch)

    # Sort: memory-bound first, then compute-bound, by duration descending.
    _op_by_name = {str(r.get("name") or "").lower(): r for r in op_class_rows}
    op_chars.sort(key=lambda c: (
        0 if c.get("bound_classification") in ("memory-bound", "mixed-bound") else 1,
        -_as_float(_op_by_name.get(str(c.get("operator_name") or "").lower(), {}).get("duration_sum_us")),
    ))

    # Block-level characterization.
    block_chars: list[dict[str, Any]] = []
    for block in block_class_rows:
        kind = str(block.get("block_kind") or "")
        comm_share = _as_float(block.get("comm_share_mean"))
        if kind in ("attention", "moe") and comm_share > 0.20:
            block_chars.append({
                "block_kind": kind,
                "comm_share_mean": round(comm_share, 3),
                "member_count": block.get("member_count"),
                "bound_family": block.get("bound_family"),
                "characterization": (
                    f"Communication accounts for {comm_share*100:.1f}% of "
                    f"{kind} block wall time. "
                    f"Check whether collectives (allreduce/alltoallv) can be "
                    f"overlapped with compute or if TP/EP topology can be adjusted."
                ),
                "confidence": "medium",
            })

    # HCCL skew characterization.
    for hccl in hccl_class_rows[:8]:
        skew = _as_float(hccl.get("rank_skew_ratio"))
        if skew > 1.5:
            block_chars.append({
                "hccl_op_kind": hccl.get("hccl_op_kind"),
                "rank_skew_ratio": round(skew, 3),
                "characterization": (
                    f"HCCL {hccl.get('hccl_op_kind')} has rank-skew ratio "
                    f"{skew:.2f}: ranks are not balanced in this collective. "
                    f"Cross-reference with slow_rank detection for root cause "
                    f"(hardware vs. workload imbalance)."
                ),
                "confidence": "medium",
            })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "analysis_stage": "characterize",
        "created_at": utc_now(),
        "hardware_device": device_display,
        "architecture": architecture,
        "peak_fp16_tflops": peak_tflops,
        "peak_fp16_tflops_binning": peak_binning,
        "roofline_ridge": _compute_roofline_ridge(peak_tflops, device_info),
        "roofline_ridge_note": (
            "Roofline ridge computed from hardware_capabilities.yaml peak FLOPs "
            "and official HBM bandwidth. Bandwidth values are theoretical maxima "
            "from device datasheets; achievable bandwidth may be lower. "
            "Bound classification uses measured pipeline data "
            "(hardware-profiled mte/mac/vec stage breakdown) as primary source."
        ),
        "config_signatures": _detect_config_signatures(output_dir),
        "operator_characterizations": op_chars,
        "block_characterizations": block_chars,
        "counts": {
            "operator_characterizations": len(op_chars),
            "block_characterizations": len(block_chars),
            "memory_bound_ops": sum(1 for c in op_chars if c.get("bound_classification") == "memory-bound"),
            "compute_bound_ops": sum(1 for c in op_chars if c.get("bound_classification") == "compute-bound"),
            "mixed_bound_ops": sum(1 for c in op_chars if c.get("bound_classification") == "mixed-bound"),
            "decode_like_ops": sum(1 for c in op_chars if c.get("decode_like")),
        },
    }
    write_json(output_dir / OUTPUT_JSON, payload)
    write_json(output_dir / OUTPUT_MANIFEST, {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "analysis_stage": "characterize",
        "created_at": utc_now(),
        "output_dir": str(output_dir),
        "counts": payload["counts"],
    })
    return payload


# ---------------------------------------------------------------------------
# Config signature detection — kernel fingerprints → feature state
# ---------------------------------------------------------------------------

def _detect_config_signatures(output_dir: Path) -> dict[str, Any]:
    """Detect vLLM-Ascend configuration from kernel fingerprints.

    Reads the normalized event index, step/rank/hccl summaries.
    Returns a dict of detected features with evidence and confidence.
    No external inputs needed — all evidence is in the profiling data.
    """
    result: dict[str, Any] = {}

    # Collect all categories and names from the event index.
    events_path = output_dir / "normalized_event_index.jsonl"
    if not events_path.is_file():
        return {"status": "no_event_index", "note": "Event index not found; config detection skipped"}

    all_categories: set[str] = set()
    all_names: set[str] = set()
    for event in read_jsonl(events_path):
        for cat in (event.get("op_categories") or []):
            all_categories.add(str(cat))
        name = event.get("name_raw")
        if name:
            all_names.add(str(name).lower())

    # Load supporting CSVs.
    rank_rows = csv_rows(output_dir / "rank_summary.csv")
    hccl_rows = csv_rows(output_dir / "hccl_op_summary.csv")
    step_rows = csv_rows(output_dir / "step_summary.csv")

    # ── Attention backend ──
    result["attention_backend"] = _detect_attention_backend(all_categories, all_names)

    # ── KV cache compression ──
    signpack = any("kvcomp" in c for c in all_categories)
    result["kv_cache_compression"] = {
        "detected": "enabled" if signpack else "not_detected",
        "evidence": "NpuSignBitsPack/NpuHammingDistTopK kernels present" if signpack else "No KVComp kernels found",
        "confidence": "high",
    }

    # ── MoE dispatch fusion ──
    result["moe_dispatch"] = _detect_moe_dispatch(all_categories, all_names)

    # ── Graph mode ──
    result["graph_mode"] = _detect_graph_mode(step_rows)

    # ── Parallelism: TP / EP ──
    has_attention_comm = _has_comm_in_role(all_categories, "attention")
    has_moe_comm = _has_comm_in_role(all_categories, "moe")
    result["parallelism"] = _detect_parallelism(rank_rows, hccl_rows, has_attention_comm, has_moe_comm)

    # ── Reduced-work / dummy ranks ──
    result["reduced_work_ranks"] = _detect_reduced_work(rank_rows)

    # ── Context parallelism (PCP/DCP) ──
    result["context_parallelism"] = _detect_cp(hccl_rows, step_rows, all_categories)

    return result


def _detect_attention_backend(categories: set[str], names: set[str]) -> dict[str, Any]:
    """Infer attention backend from kernel signatures."""
    has_flash = "attention.flash_score" in categories
    has_sparse = any(c.startswith("attention.sparse_sharedkv") for c in categories)
    has_indexer = "attention.lightning_indexer" in categories
    has_compressor = "attention.kv_compressor" in categories
    has_atb = any(c.startswith("attention.kv_cache_io") for c in categories)
    is_mla = "attention.mla" in categories or "attention.mla.preprocess" in categories

    detected: list[str] = []
    confidence = "high"

    if has_sparse and has_indexer and has_compressor:
        detected.append("csa")
    elif has_sparse and has_indexer:
        detected.append("dsa")
    elif has_sparse and has_compressor:
        detected.append("hca")
    elif is_mla:
        detected.append("mla")
    elif has_flash:
        detected.append("fia")
    elif has_atb:
        detected.append("atb_paged")
    else:
        # Check raw names for coarse fallback.
        name_set = {str(n).lower() for n in names}
        if any("unpadflash" in n for n in name_set):
            detected.append("unpadfa")
        elif any("fusedinfer" in n for n in name_set):
            detected.append("fia")
        else:
            detected.append("unknown")
            confidence = "low"

    return {
        "detected": detected,
        "evidence": _backend_evidence(has_flash, has_sparse, has_indexer, has_compressor, has_atb, is_mla),
        "confidence": confidence,
    }


def _backend_evidence(flash: bool, sparse: bool, indexer: bool, compressor: bool,
                      atb: bool, mla: bool) -> str:
    parts = []
    if flash: parts.append("flash_score kernels")
    if sparse: parts.append("sparse_sharedkv kernels")
    if indexer: parts.append("lightning_indexer kernels")
    if compressor: parts.append("kv_compressor kernels")
    if atb: parts.append("kv_cache_io kernels")
    if mla: parts.append("mla kernels")
    return ", ".join(parts) if parts else "no recognizable attention kernel found"


def _detect_moe_dispatch(categories: set[str], names: set[str]) -> dict[str, Any]:
    """Detect fused vs unfused MoE dispatch."""
    has_fused = any("dispatchffncombine" in str(n).lower() for n in names)
    has_fused_decode = any("dispatchgmmcombinedecode" in str(n).lower() for n in names)
    has_unfused_dispatch = "moe.dispatch" in categories
    has_unfused_combine = "moe.combine" in categories

    if has_fused or has_fused_decode:
        mode = "fused"
        if has_fused_decode and not has_fused:
            mode = "fused_decode"
        elif has_fused and not has_fused_decode:
            mode = "fused_prefill"
        else:
            mode = "fused"
        return {
            "detected": mode,
            "evidence": "DispatchFFNCombine/GmmCombineDecode kernels present",
            "confidence": "high",
        }
    if has_unfused_dispatch and has_unfused_combine:
        return {
            "detected": "unfused",
            "evidence": "Separate dispatch + combine kernels (no fusion)",
            "confidence": "high",
        }
    return {"detected": "not_applicable", "evidence": "No MoE dispatch kernels found", "confidence": "high"}


def _has_comm_in_role(categories: set[str], role: str) -> bool:
    return any(
        cat.startswith(f"communication.{prefix}")
        for prefix in ("allreduce", "reducescatter", "allgather", "alltoallv", "broadcast")
        for cat in categories
    )


def _detect_parallelism(rank_rows: list[dict[str, Any]],
                        hccl_rows: list[dict[str, Any]],
                        has_attn_comm: bool, has_moe_comm: bool) -> dict[str, Any]:
    """Infer TP/EP from collective patterns."""
    rank_count = len([r for r in rank_rows if r.get("rank_id")])

    # Count distinct HCCL kinds.
    hccl_kinds: set[str] = set()
    for row in hccl_rows:
        kind = str(row.get("hccl_op_kind") or "")
        if kind:
            hccl_kinds.add(kind)

    tp_inferred = has_attn_comm and rank_count >= 2
    ep_inferred = has_moe_comm and rank_count >= 2

    result: dict[str, Any] = {"rank_count": rank_count}

    if "allreduce" in hccl_kinds:
        result["has_allreduce"] = True
    if "reducescatter" in hccl_kinds:
        result["has_reducescatter"] = True
    if "alltoallv" in hccl_kinds:
        result["has_alltoallv"] = True

    if tp_inferred and ep_inferred:
        result["tp"] = "≥2"
        result["ep"] = "≥2"
        result["confidence"] = "high"
        result["note"] = f"TP≥2 & EP≥2 inferred from allreduce in attention + alltoallv in MoE across {rank_count} ranks"
    elif tp_inferred:
        result["tp"] = f"≈{rank_count}" if not ep_inferred else "≥2"
        result["confidence"] = "high"
        result["note"] = f"TP ≥ 2 inferred from allreduce/reducescatter in attention across {rank_count} ranks"
    elif ep_inferred:
        result["ep"] = f"≈{rank_count}" if not tp_inferred else "≥2"
        result["confidence"] = "high"
        result["note"] = f"EP ≥ 2 inferred from alltoallv in MoE across {rank_count} ranks"
    else:
        result["confidence"] = "medium"
        result["note"] = f"No TP/EP collective pattern detected among {rank_count} ranks"

    return result


def _detect_reduced_work(rank_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect ranks with reduced workload (dummy / pipeline-parallel)."""
    has_attention = []
    no_attention = []
    for row in rank_rows:
        rank_id = str(row.get("rank_id") or "")
        if str(row.get("has_attention", "")).lower() in ("true", "1", "yes"):
            has_attention.append(rank_id)
        else:
            no_attention.append(rank_id)

    if no_attention and has_attention:
        return {
            "detected": True,
            "full_work_ranks": has_attention,
            "reduced_work_ranks": no_attention,
            "note": (
                f"Ranks {','.join(no_attention)} lack attention kernels while "
                f"ranks {','.join(has_attention[:3])}… have full workloads. "
                "Possible pipeline parallelism or reduced-work setup."
            ),
            "confidence": "high",
        }
    return {"detected": False, "confidence": "high"}


def _detect_cp(hccl_rows: list[dict[str, Any]],
               step_rows: list[dict[str, Any]],
               categories: set[str]) -> dict[str, Any]:
    """Detect PCP and/or DCP from HCCL collective patterns.

    Context Parallelism creates sub-groups within the TP group for attention.
    PCP (prefill) and DCP (decode) each produce distinct fingerprints:

    - **allgather in attention**: CP exchanges KV chunks via allgather across
      CP ranks. Standard TP uses only allreduce/reducescatter — allgather in
      attention blocks is a strong CP signal.
    - **collective rank_count < total rank_count**: CP groups are smaller than
      the full TP group. When the same collective kind appears with different
      rank counts (e.g., allreduce on 4 ranks vs allreduce on 2 ranks), the
      smaller group is likely a CP subgroup.
    - **step_type differentiation**: CP in prefill-only steps → PCP. CP in
      decode-only steps → DCP. Both → PCP+DCP active.

    Confidence: allgather signal is high (strong CP fingerprint). Rank-count
    asymmetry is medium on its own (could be DP or other grouping) but high
    when combined with allgather.
    """
    # Collect all HCCL kinds and their per-kind rank counts.
    kind_rank_counts: dict[str, set[int]] = {}
    for row in hccl_rows:
        kind = str(row.get("hccl_op_kind") or "")
        if not kind:
            continue
        rank_count = int(row.get("rank_count") or 0)
        if rank_count > 0:
            kind_rank_counts.setdefault(kind, set()).add(rank_count)

    # Determine total rank count from step rows.
    total_ranks: set[str] = {str(s.get("rank_id") or "") for s in step_rows}
    total_rank_count = len(total_ranks)

    has_allgather = "allgather" in kind_rank_counts
    has_allreduce = "allreduce" in kind_rank_counts or "reducescatter" in kind_rank_counts

    # Check for CP-like collective patterns: same collective kind with
    # different rank counts suggests sub-grouping (CP creates smaller
    # attention collectives while FFN collectives stay at full TP).
    cp_rank_count_evidence = False
    for kind, counts in kind_rank_counts.items():
        if len(counts) > 1 and any(c < total_rank_count for c in counts):
            cp_rank_count_evidence = True
            break

    # Check step types for prefill/decode CP differentiation.
    step_types: set[str] = {str(s.get("step_type") or "") for s in step_rows}
    has_prefill = "prefill" in step_types
    has_decode = "decode" in step_types

    # Detection logic.
    pcp_detected = False
    dcp_detected = False
    evidence_parts: list[str] = []

    if has_allgather:
        evidence_parts.append("allgather detected in HCCL ops")
        # allgather is the KV-exchange primitive for CP — strong signal.
        # Both PCP and DCP use allgather, so we need step-type differentiation.
        if has_prefill and has_decode:
            pcp_detected = True
            dcp_detected = True
            evidence_parts.append("both prefill and decode steps present")
        elif has_prefill:
            pcp_detected = True
            evidence_parts.append("prefill-only steps")
        elif has_decode:
            dcp_detected = True
            evidence_parts.append("decode-only steps")
        confidence = "high"
    elif cp_rank_count_evidence and has_allreduce:
        # Rank-count asymmetry is weaker signal — could be finegrained TP or DP.
        evidence_parts.append(f"collective rank counts vary within HCCL kinds (total ranks={total_rank_count})")
        confidence = "medium"
        # Can't distinguish PCP vs DCP from rank counts alone.
        if has_prefill and has_decode:
            pcp_detected = True
            dcp_detected = True
        elif has_prefill:
            pcp_detected = True
        elif has_decode:
            dcp_detected = True
    else:
        return {"detected": "none", "confidence": "medium",
                "evidence": f"No allgather or rank-count asymmetry among {total_rank_count} ranks"}

    result: dict[str, Any] = {
        "detected": [],
        "confidence": confidence,
        "evidence": ", ".join(evidence_parts),
        "total_ranks": total_rank_count,
    }
    if pcp_detected:
        result["detected"].append({"type": "pcp", "note": "Prefill Context Parallelism likely active. Check prefill_context_parallel_size."})
    if dcp_detected:
        result["detected"].append({"type": "dcp", "note": "Decode Context Parallelism likely active. Check decode_context_parallel_size (upstream vLLM)."})
    if not result["detected"]:
        result["detected"].append({"type": "unknown", "note": "CP signal present but cannot distinguish PCP vs DCP. Ask user."})

    return result


def _detect_graph_mode(step_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Detect graph capture state from host dispatch patterns.

    In vLLM-Ascend, graph mode is enabled by ``enforce_eager=False``
    (vLLM standard) and compiled by the ``AscendCompiler`` which uses
    ``npugraph_ex`` (default, via ``enable_npugraph_ex=True`` in
    ``ascend_compilation_config``) or ``torchair`` fallback.

    Eager mode (``enforce_eager=True``): each operator is individually
      launched by the host → large head_ms, high wait_us, many tiny
      kernels with gaps between them.

    Graph mode: operators are fused into a compiled ACL graph → head_ms
      near zero, wait_us near zero, large contiguous kernel blocks.

    Partial capture: when ``cudagraph_capture_sizes`` does not cover all
      batch sizes in the profile window, some steps show graph-mode
      patterns while others show eager-mode patterns.
    """
    if not step_rows:
        return {"detected": "unknown", "confidence": "low"}

    head_ratios = []
    for row in step_rows:
        wall = _as_float(row.get("wall_ms"))
        head = _as_float(row.get("head_wall_ms"))
        if wall > 0:
            head_ratios.append(head / wall)

    if not head_ratios:
        return {"detected": "unknown", "confidence": "low"}

    avg_head = sum(head_ratios) / len(head_ratios)
    high_count = sum(1 for r in head_ratios if r > 0.10)
    low_count = sum(1 for r in head_ratios if r < 0.05)
    mid_count = len(head_ratios) - high_count - low_count

    # ── Warmup check: first few steps may show high head/wall even with
    #    graph mode active. If only the first ~3 steps are high and the
    #    rest are low, this is warmup, not eager mode.
    warmup_count = 3
    if len(head_ratios) > warmup_count:
        early_high = sum(1 for r in head_ratios[:warmup_count] if r > 0.10)
        later_low = sum(1 for r in head_ratios[warmup_count:] if r < 0.05)
        later_total = len(head_ratios) - warmup_count
        if early_high >= warmup_count * 0.5 and later_low >= later_total * 0.8:
            return {
                "detected": "graph_mode",
                "evidence": (
                    f"Mean head/wall = {avg_head:.3f}. "
                    f"First {warmup_count} steps show warmup-like patterns "
                    f"(likely graph compilation), "
                    f"subsequent {later_total} steps show graph-mode "
                    f"({later_low}/{later_total} head<5%)."
                ),
                "warmup_steps": warmup_count,
                "confidence": "high",
            }

    # ── Partial capture: some steps graph-like (head < 5%), others
    #    eager-like (head > 10%). The gap between low and high counts
    #    reveals capture-list coverage gaps. This is the highest-value
    #    detection because the fix is specific (add missing batch size).
    if low_count > 0 and high_count > 0:
        return {
            "detected": "partial_capture",
            "evidence": (
                f"{low_count} steps graph-like (head<5%), "
                f"{high_count} steps eager-like (head>10%)"
                + (f", {mid_count} steps unclear (5-10% head/wall)" if mid_count else "")
                + ". Some batch sizes are captured, others are not."
            ),
            "confidence": "high",
        }

    # ── Graph mode: consistent low head/wall. Operators fused into
    #    compiled ACL graphs via enforce_eager=False + enable_npugraph_ex.
    if avg_head < 0.05:
        return {
            "detected": "graph_mode",
            "evidence": (
                f"Mean step head/wall = {avg_head:.3f} "
                f"({low_count}/{len(head_ratios)} steps head<5%)"
            ),
            "confidence": "high",
        }

    # ── Eager mode: consistent high head/wall. Operators individually
    #    launched — host dispatch overhead dominates each step.
    if avg_head > 0.10:
        return {
            "detected": "eager_mode",
            "evidence": (
                f"Mean step head/wall = {avg_head:.3f} "
                f"({high_count}/{len(head_ratios)} steps head>10%)"
            ),
            "confidence": "high",
        }

    # ── Unclear: head/wall in ambiguous range (5-10%). Not graph-mode,
    #    not clearly eager. May indicate partial warmup, hybrid execution,
    #    or a model with inherently moderate host dispatch overhead.
    return {
        "detected": "unclear",
        "evidence": (
            f"Mean step head/wall = {avg_head:.3f} "
            f"(low<5%: {low_count}, mid 5-10%: {mid_count}, high>10%: {high_count}). "
            "Head/wall in ambiguous range — cannot distinguish graph from eager."
        ),
        "confidence": "low",
    }


def _has_pipeline_signal(op: Mapping[str, Any]) -> bool:
    for field in PIPELINE_FIELDS:
        if _as_float(op.get(field)) > 0:
            return True
    return False
