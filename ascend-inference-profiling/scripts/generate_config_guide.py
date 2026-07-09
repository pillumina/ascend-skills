#!/usr/bin/env python3
"""Generate a versioned vLLM-Ascend config guide from a source checkout.

Usage:
    python3 generate_config_guide.py --src /path/to/vllm-ascend [--output v0.11.0.md]

The script extracts:
  - Version from ``git describe --tags`` (or ``version.info``)
  - ``AscendCompilationConfig`` defaults from ``ascend_config.py``
  - Environment variable defaults from ``envs.py``
  - Attention backend kernel names from attention module files

Output is a structured Markdown file following ``knowledge/vllm-ascend/_template.md``.
Fields that cannot be auto-extracted are marked with ``TODO(review)`` and must be
filled in manually by an expert reviewer.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def _get_version(src: Path) -> dict[str, str]:
    """Extract version info from a vllm-ascend checkout."""
    result = {"tag": "unknown", "commit": "unknown"}

    # Try git describe
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--always"],
            cwd=str(src), text=True, stderr=subprocess.DEVNULL,
        ).strip()
        result["tag"] = tag
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try version.info
        vi = src / "version.info"
        if vi.is_file():
            result["tag"] = vi.read_text().strip()

    # Try git rev-parse
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(src), text=True, stderr=subprocess.DEVNULL,
        ).strip()
        result["commit"] = commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return result


def _parse_class_defaults(source: str, class_name: str) -> dict[str, Any]:
    """Extract __init__ parameter defaults from a class definition.

    Parses Python source with ast, finds the named class, and reads its
    __init__ method's keyword argument defaults.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
                continue
            defaults: dict[str, Any] = {}
            args = item.args
            # args with defaults: names from the end of args.args
            n_defaults = len(args.defaults)
            if n_defaults == 0:
                return defaults
            arg_names = [a.arg for a in args.args[-n_defaults:]]
            for name, default_node in zip(arg_names, args.defaults):
                if name == "self":
                    continue
                if isinstance(default_node, ast.Constant):
                    defaults[name] = default_node.value
                elif isinstance(default_node, ast.Name) and default_node.id in ("True", "False"):
                    defaults[name] = default_node.id == "True"
                elif isinstance(default_node, ast.UnaryOp):
                    defaults[name] = f"TODO(review): unary op"
                else:
                    defaults[name] = f"TODO(review): complex expression"
            return defaults
    return {}


def _parse_env_vars(source: str) -> dict[str, dict[str, str]]:
    """Extract env var name, default, and description from envs.py."""
    result: dict[str, dict[str, str]] = {}
    pattern = re.compile(
        r'"([A-Z_][A-Z0-9_]*)"\s*:\s*lambda:\s*(.*?)(?:,\s*$|(?=\s*"}))',
        re.MULTILINE | re.DOTALL,
    )
    # Simpler approach: find all env var blocks
    for match in re.finditer(
        r'#\s*(.*?)\n\s*"([A-Z_][A-Z0-9_]*)"\s*:\s*lambda:\s*os\.getenv\("([^"]*)",\s*([^)]*)\)',
        source,
    ):
        desc, name, env_name, default_raw = match.groups()
        default = default_raw.strip().strip('"').strip("'")
        result[name] = {"env": env_name, "default": default, "desc": desc.strip()}
    return result


def _find_attention_backends(src: Path) -> list[dict[str, str]]:
    """Scan attention/*.py for backend class names and kernel references."""
    backends = []
    attn_dir = src / "vllm_ascend" / "attention"
    if not attn_dir.is_dir():
        return backends

    for py_file in sorted(attn_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        source = py_file.read_text(encoding="utf-8")
        # Find class definitions
        for match in re.finditer(r"class\s+(\w+)\s*(?:\([^)]*\))?\s*:", source):
            class_name = match.group(1)
            if "Backend" in class_name or "Impl" in class_name or "Attention" in class_name:
                backends.append({
                    "file": py_file.name,
                    "class": class_name,
                    "note": "TODO(review): list detectable kernel names and model families",
                })
    return backends


def generate(src_path: str, output_path: str) -> None:
    src = Path(src_path).resolve()
    if not src.is_dir():
        print(f"ERROR: source path is not a directory: {src}", file=sys.stderr)
        sys.exit(1)

    version = _get_version(src)
    tag = version["tag"]
    commit = version["commit"]

    # Parse config defaults
    ascend_config_src = (src / "vllm_ascend" / "ascend_config.py").read_text(encoding="utf-8")
    compilation_defaults = _parse_class_defaults(ascend_config_src, "AscendCompilationConfig")

    # Parse env vars
    envs_src = (src / "vllm_ascend" / "envs.py").read_text(encoding="utf-8")
    env_vars = _parse_env_vars(envs_src)

    # Find attention backends
    backends = _find_attention_backends(src)

    # Build output
    lines: list[str] = [
        f"# vLLM-Ascend Config Guide — {tag}",
        "",
        "<!--",
        f"  Canonical source: https://github.com/vllm-project/vllm-ascend",
        f"  Version:          {tag}",
        f"  Commit:           {commit}",
        f"  Generated:        {__import__('datetime').datetime.now().strftime('%Y-%m')}",
        "-->",
        "",
        "## Graph compilation & execution mode",
        "",
        "### `enforce_eager`",
        "",
        "- **vLLM key**: `model_config.enforce_eager`",
        "- **Default**: `False` (graph mode by default)",
        "- **When `True` (eager) → profiling fingerprint**:",
        "  - `step_anatomy.head_wall_ms / step_summary.wall_ms > 10%`",
        "  - Many small kernels with high `wait_us`, gaps between sequential ops",
        "- **When `False` (graph) → profiling fingerprint**:",
        "  - `head_wall_ms / wall_ms < 5%`",
        "  - Large fused kernel blocks, no inter-kernel gaps",
        "- **Interaction**: vLLM-Ascend graph mode uses `enable_npugraph_ex`",
        "  for compilation. Without it, graph mode falls back to torchair.",
        "- **Source**: `vllm_ascend/compilation/compiler_interface.py` `AscendCompiler`",
        "- **Agent note**: if `config_signatures.graph_mode = \"eager_mode\"`, ask whether",
        "  `enforce_eager` was explicitly set to True (default is False).",
        "",
        "### `enable_npugraph_ex`",
        "",
        f"- **vLLM-Ascend key**: `ascend_compilation_config.enable_npugraph_ex`",
        f"- **Default**: `{compilation_defaults.get('enable_npugraph_ex', 'TODO(review)')}`",
        f"- **Source**: `vllm_ascend/ascend_config.py` `AscendCompilationConfig`",
        "",
        "### `enable_static_kernel`",
        "",
        f"- **vLLM-Ascend key**: `ascend_compilation_config.enable_static_kernel`",
        f"- **Default**: `{compilation_defaults.get('enable_static_kernel', 'TODO(review)')}`",
        f"- **Requires**: `enable_npugraph_ex=True`",
        "",
        "### `cudagraph_capture_sizes`",
        "",
        "- **vLLM key**: `compilation_config.cudagraph_capture_sizes`",
        "- **Default**: vLLM dynamic default",
        "- **Profiling fingerprint when incomplete**:",
        "  - Mixed pattern: some steps graph-like, others eager-like",
        "  - `config_signatures.graph_mode = \"partial_capture\"`",
        "- **Agent note**: if partial_capture detected, ask for cudagraph_capture_sizes",
        "  and num_speculative_tokens. Must cover num_spec_tokens + 1 for decode.",
        "",
        "## Compilation fusion passes",
        "",
        "### `fuse_qknorm_rope`",
        f"- **Default**: `{compilation_defaults.get('fuse_qknorm_rope', 'TODO(review)')}`",
        "- **When True → fingerprint**: No separate RoPE kernel in attention block",
        "- **When False → fingerprint**: Explicit `attention.rope` kernel appears separately",
        "",
        "### `fuse_allreduce_rms`",
        f"- **Default**: `{compilation_defaults.get('fuse_allreduce_rms', 'TODO(review)')}`",
        "- **When True → fingerprint**: Allreduce and rmsnorm fused, no gap",
        "- **When False → fingerprint**: Separate allreduce then rmsnorm with visible gap",
        "- **Agent note**: off by default but can significantly reduce TP overhead.",
        "  Only works in graph mode with npugraph_ex.",
        "",
        "### `fuse_norm_quant`",
        f"- **Default**: `{compilation_defaults.get('fuse_norm_quant', 'TODO(review)')}`",
        "- **Profiling visibility**: Cannot reliably confirm from kernel names alone.",
        "",
        "## Attention backends",
        "",
    ]

    if backends:
        for b in backends:
            lines.append(f"### {b['class']}")
            lines.append(f"- **Source**: `vllm_ascend/attention/{b['file']}`")
            lines.append(f"- {b['note']}")
            lines.append("")
    else:
        lines.append("TODO(review): attention backend kernel mapping")
        lines.append("")

    lines.extend([
        "## KV cache compression",
        "",
        "### `hamming_sparse`",
        "",
        "- **vLLM-Ascend key**: `additional_config.hamming_sparse.enabled`",
        "- **Default**: `False`",
        "- **When True → fingerprint**: `NpuHammingDistTopK`, `NpuSignBitsPack` kernels present",
        "- **Source**: `vllm_ascend/ascend_config.py` `AscendConfig`",
        "",
        "## MoE dispatch fusion",
        "",
        "### `enable_fused_mc2`",
        "",
    ])

    mc2_default = env_vars.get("VLLM_ASCEND_ENABLE_FUSED_MC2", {}).get("default", "TODO(review)")
    lines.extend([
        f"- **vLLM-Ascend key**: `additional_config.enable_fused_mc2` or env `VLLM_ASCEND_ENABLE_FUSED_MC2`",
        f"- **Default**: `{mc2_default}`",
        "- **Modes**:",
        f"  - 0: standard alltoall + MC2 (no fusion)",
        f"  - 1: fused `DispatchFFNCombine` for prefill (W8A8 only, EP≤32, no MTP, no dynamic EPLB)",
        f"  - 2: fused `DispatchGmmCombineDecode` for decode (W8A8 only, MTP must be W8A8)",
        "",
        "| Mode | Kernel pattern |",
        "|------|---------------|",
        "| 0 (unfused) | Separate `MoeDistributeDispatchV2` + `MoeDistributeCombineV2` |",
        "| 1 (prefill fused) | `DispatchFFNCombine` replaces dispatch + combine |",
        "| 2 (decode fused) | `DispatchGmmCombineDecode` replaces dispatch + combine |",
        "",
        "- **Agent note**: if unfused, check: W8A8 quant active? EP ≤ 32? MTP active?",
        "",
        "## Speculative decode",
        "",
        "- **vLLM key**: `speculative_config`",
        "- **Profiling fingerprint**: `step_type = \"speculative\"` in `step_summary.csv`",
        "- **Interaction with graph**: cudagraph_capture_sizes must cover num_spec_tokens + 1",
        "",
        "## Parallelism",
        "",
        "### Tensor parallelism (TP)",
        "- **vLLM key**: `parallel_config.tensor_parallel_size`",
        "- **Profiling fingerprint**: allreduce/reducescatter in attention/FFN blocks",
        "- **TP = 1 fingerprint**: No HCCL collectives in compute blocks",
        "",
        "### Expert parallelism (EP)",
        "- **vLLM key**: `parallel_config.enable_expert_parallel`",
        "- **Profiling fingerprint**: alltoallv before/after MoE expert blocks",
        "",
        "### Pipeline parallelism (PP)",
        "- **Profiling fingerprint**: Some ranks show `has_attention = False`",
        "",
        "### Fine-grained TP",
        "- **vLLM-Ascend key**: `additional_config.finegrained_tp_config`",
        "- **Agent note**: not detectable from profiling. Ask if TP overhead varies by module.",
        "",
        "## Communication optimization",
        "",
    ])

    # Env var entries
    env_entries = [
        ("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", "MatMul + AllReduce fusion"),
        ("VLLM_ASCEND_ENABLE_FLASHCOMM1", "FlashComm1 (TP communication optimization)"),
        ("VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", "FlashComm2 (O-matrix TP group)"),
    ]
    for name, desc in env_entries:
        info = env_vars.get(name, {})
        default = info.get("default", "TODO(review)")
        lines.append(f"### `{name}` — {desc}")
        lines.append(f"- **Default**: `{default}`")
        lines.append("")

    lines.extend([
        "## Other configurable features",
        "",
    ])

    other_env = [
        ("VLLM_ASCEND_ENABLE_MLAPO", "MLA prefill optimization for DeepSeek W8A8"),
        ("VLLM_ASCEND_ENABLE_NZ", "Weight fractal NZ layout (0=off, 1=quant, 2=all)"),
    ]
    for name, desc in other_env:
        info = env_vars.get(name, {})
        default = info.get("default", "TODO(review)")
        lines.append(f"### `{name}` — {desc}")
        lines.append(f"- **Default**: `{default}`")
        lines.append("")

    lines.extend([
        "### Continuous batching (chunked prefill)",
        "- **Profiling fingerprint**: mix of prefill and decode steps in same window",
        "- **Agent note**: check `step_summary.csv` for both step_type values",
        "",
        "### Multi-stream DSA overlap",
        "- **vLLM-Ascend key**: `additional_config.multistream_dsv4_dsa_overlap`",
        "- **Default**: TODO(review)",
        "- **When True → fingerprint**: attention compute overlaps with HCCL communication",
        "",
        "## Agent usage pattern",
        "",
        "1. Read `characterizations.json` → `config_signatures` for detected states",
        "2. Look up each detection in this file, compare with documented default",
        "3. Formulate specific follow-up questions, referencing exact config keys",
        "4. Ask only relevant questions — do not dump the entire checklist",
        "5. If user's answer reveals intentional config, acknowledge and move on",
    ])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated: {output} ({len(lines)} lines)")
    print(f"Version: {tag} ({commit})")
    print(f"Compilation defaults extracted: {len(compilation_defaults)}")
    print(f"Env vars extracted: {len(env_vars)}")
    print(f"Attention backends found: {len(backends)}")
    print("")
    print("NEXT: review the generated file. Search for TODO(review) and fill in:")
    print("  - Profiling fingerprint descriptions (ON vs OFF patterns)")
    print("  - Agent notes (context-specific follow-up questions)")
    print("  - Any config keys the script couldn't auto-extract")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="path to vllm-ascend checkout")
    parser.add_argument("--output", required=True, help="output markdown file path")
    args = parser.parse_args()
    generate(args.src, args.output)
