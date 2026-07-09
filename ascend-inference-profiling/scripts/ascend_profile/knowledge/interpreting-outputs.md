# Interpreting Analysis Outputs

<!--
  Lazily-loaded reference. The agent reads this file on demand after a
  pipeline run completes. SKILL.md contains only workflow + interaction
  patterns (~400 lines). This file contains the detailed interpretation
  guide for output files, config signatures, characterization data, and
  follow-up question patterns.

  Read order: this file first → diagnostic-playbook.md for specific signals
  → v{version}.md for version-specific defaults → changelog/v{version}.md
  for version-specific caveats.
-->


## Agent Interpretation Guide

Scripts produce **deterministic, fact-based data** — measurements, classifications, pattern detections. They do NOT generate suggestions, recommendations, or follow-up questions. Those are your responsibility.

### How to read the analysis outputs

Read the local run dir in this order:

1. **`triage.json`** — bottleneck direction (computing / communication / hostbound). Low-confidence pre-scan.
2. **`report/report.md`** — full narrative report. Start here for a human-readable overview.
3. **`characterizations.json`** — per-operator metrics + `config_signatures` (see below).
4. **`diagnosis_findings.json`** — structured evidence-backed claims.
5. **`mstt_slow_rank.csv`** — per-rank slow_affect_count (when `--mstt` was used).
6. **CSV summaries** — `step_summary.csv`, `step_anatomy.csv`, `operator_summary.csv`, `hccl_op_summary.csv`.
7. **`step_type_stats.csv`** — per-step-type aggregate statistics (see below).
8. **`run_observations.json`** — calibration observations.

### Interpreting step type statistics

`step_type_stats.csv` provides per-step-type aggregate metrics (count, median/max/avg wall_ms, median head/bubble ratios). Use it to understand the workload composition.

**Step type inference** is heuristic:
- `decode`: `main_layer_count == 1` — a single forward pass through model layers
- `prefill`: `main_layer_count > 1` — full model forward on a batch of prompt tokens
- `speculative`: `speculative_layer_count > 0` — includes draft model layers
- `unknown`: `main_layer_count is None` — step fragments (head/tail/dummy) without identifiable layers

**How to use this data:**

1. **Check the step-type mix.** A healthy decode-heavy serving workload typically has 95%+ decode steps with occasional prefill bursts. A training or prefill-only capture will be 100% prefill.

2. **Compare median wall across types.** The ratio `median_decode_wall : median_prefill_wall` indicates the latency gap between token generation and prompt processing. In PD-disaggregated scenarios, these numbers come from separate capture runs (P and D nodes profiled independently).

3. **Check bubble ratio by type.** High decode bubble ratios combined with low prefill bubble ratios may signal that prefill steps consume resources (compute, bandwidth, HCCL channels) that starve subsequent decode steps. This is a **structural observation, not a definitive diagnosis** — multiple factors affect decode bubble (batch size, CPU dispatch, HCCL contention). The pipeline reports the data; the agent decides what it means.

4. **When a type has fewer than 3 samples**: all computed metrics are marked as indicative. Do not draw conclusions from these groups.

**PD-disaggregation hints:**

When 100% of steps are a single type within a capture window:
- **All decode, zero prefill**: may be a PD-disaggregated D (decode) node, or a mixed deployment captured during a decode-only period. The report adds a note to this effect.
- **All prefill, zero decode**: may be a PD-disaggregated P (prefill) node.

These are **hints, not detections**. The pipeline cannot distinguish a PD-separated D node from a mixed deployment profiled during a decode-only window. To confirm, ask the user about their deployment topology (`--disaggregation-mode` or equivalent). The report includes a note making the ambiguity explicit.

**When analyzing PD-disaggregated captures:**
- P and D node captures must be analyzed separately — their step structures, bottlenecks, and optimization strategies are fundamentally different
- P nodes: prefill-heavy, compute-bound, sensitive to graph capture and batch sizing
- D nodes: decode-heavy, memory-bound, sensitive to host dispatch latency and KV cache efficiency
- Do NOT compare prefill metrics from one capture with decode metrics from another as if they were from the same system — they represent different hardware/process configurations

### Version lookup

Find the nearest known config guide version with a one-liner:
```bash
python3 -c "import sys; sys.path.insert(0,'ascend_profile'); from observations import _find_nearest_known_version; print(_find_nearest_known_version('0.19.0'))"
```
→ prints `0.18.0` (the nearest known version with a config guide).

### Knowledge lookup order

For each detected signal, use the playbook's signal index at the top of the file:

1. Read the signal index at the top of `diagnostic-playbook.md` — find your signal by name
2. Read the full file (limit=550, covers all signals with margin) — locate the
   matching `### Signal:` heading. Note its line number from the Read tool output
3. Re-read that line with limit=30 — diagnosis, version info, and actions all inline
4. Only check `v{version}.md` and `changelog.md` if the user asks for exact
   defaults or version-gap confirmation

### Config Signatures: what they mean and what to ask

The `config_signatures` section in `characterizations.json` contains deterministic detections.
Use the **three-layer knowledge base** to interpret them:

1. **`diagnostic-playbook.md`** — profiling signal → diagnosis → action. Read this FIRST when you see a known signal.
2. **`v{version}.md`** — check version-specific config defaults referenced by the playbook.
3. **`changelog.md`** — ALL profiling-relevant version changes in a single file. Read once, scan `## v{version}` headings for entries between the config guide version and the user's version.

**When you detect a profiling signal:**
1. Look it up in `diagnostic-playbook.md`
2. Cross-reference the user's version against changelog entries
3. Check the version config guide for exact default values
4. Present diagnosis + suggested action
5. Ask ONE follow-up question

The diagnostic playbook covers: graph compilation, attention & KV cache, MoE dispatch
& parallelism sizing, speculative decode, communication optimization, and upstream vLLM
configuration. Performance impact estimates are approximate (±10%).

**attention_backend**: Which vLLM-Ascend attention backend is active. See `diagnostic-playbook.md` §"Attention & KV Cache" for interpretation. Possible values: `fia`, `csa`, `hca`, `dsa`, `mla`, `unknown`.

**kv_cache_compression**: Whether Hamming-distance KV pruning is active. See playbook § "Attention & KV Cache".

**moe_dispatch**: MoE dispatch/combine kernel fusion state. See playbook § "MoE Dispatch & Parallelism".

**graph_mode**: Graph capture state. See playbook § "Graph Compilation & Dispatch".

**parallelism**: Inferred TP/EP from HCCL collective patterns. See playbook § "MoE Dispatch & Parallelism" for sizing guidelines.

**reduced_work_ranks**: Ranks with asymmetric workloads. Possible pipeline-parallel dummy ranks.

**context_parallelism**: PCP/DCP detection from HCCL allgather patterns. See playbook § "MoE Dispatch & Parallelism".

### How to use characterization data

**Version-aware config lookup.** When `config_signatures` reports a detected state
that needs follow-up, first ask the user which vLLM-Ascend version they are running.
Then:
- If `knowledge/vllm-ascend/v{version}.md` exists → use it
- If no exact match → use the **closest lower version** and warn the user:
  "No config guide for v0.19.0 — using v0.18.0 as nearest reference. Some defaults may have changed."
- If no lower version exists → use the closest available, warn prominently
- If the user provides a vLLM-Ascend source path → run `scripts/generate_config_guide.py --src <path> --output knowledge/vllm-ascend/v{version}.md` to auto-generate a draft, then review `TODO(review)` markers

**Operator bound classification** (from `operator_characterizations`):
- `memory-bound` (mte1/mte2/aic_mte/aiv_mte): data movement dominates. The operator is waiting on memory bandwidth.
  - Check the MTE BW value: if it's well below the hardware peak (~1.5 TB/s for 910B2), there may be an alignment or tiling issue.
  - For decode (M=1): this is expected — small M means no batch reuse of weights.
  - For prefill with small K (K < 256): limited data reuse. This is a model architecture constraint, not a configuration issue.
- `compute-bound` (mac/vec): the compute unit is saturated. This is the desired state for throughput.
- `mixed-bound`: no single pipeline stage dominates. On A3 (910C) dual-die, Cube and Vector may run on separate dies.

**Arithmetic intensity (AI)**: pure shape-derived metric. High AI means good data reuse. Low AI means the kernel touches more memory per FLOP — it will tend to be memory-bound.

**MTE bandwidth (BW)**: measured bytes moved / MTE pipeline time. This is a *measurement*, not an estimate.
- Prefill matmul ~1000+ GB/s is near hardware peak → good
- Decode matmul ~200-400 GB/s is typical for small M
- If prefill BW is unexpectedly low (< 400 GB/s), check for: alignment issues, small K dimension, or suboptimal tiling

**Block/HCCL characterization**:
- comm_share > 20% in attention: TP communication is significant. For TP > 2, consider whether reducing TP would help.
- comm_share > 20% in MoE: EP alltoallv overhead. Check rank_skew_ratio — if > 2x, expert load is imbalanced.
- HCCL rank_skew > 1.5: ranks are not balanced in this collective. Cross-reference with `mstt_slow_rank.csv` to distinguish hardware issues from workload imbalance.

### How to ask follow-up questions

**The version question is the single most important follow-up.** Without knowing
the vLLM-Ascend version, you cannot distinguish between "enable this feature"
(config change, immediate) and "upgrade to get this feature" (version change,
may require planning). Ask this FIRST when you detect 2+ optimization signals.

Other information to request when specifically relevant:
- Model architecture: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim` (from HuggingFace `config.json`) — needed for TP/EP sizing recommendations
- Parallelism config: `tensor_parallel_size`, `expert_parallel_size`, `data_parallel_size`, `pipeline_parallel_size` — needed when TP×EP < rank_count
- Compilation config: `enforce_eager`, `cudagraph_capture_sizes`, `max_num_seqs` — needed for graph-mode diagnosis
- Quantization config: whether W8A8 is active — needed for fused MC2 prerequisite check

**Ask ONE at a time.** Never list multiple questions.

### How to synthesize findings

**Do not present findings one by one.** Group them by root cause. Multiple signals
often trace back to the same underlying issue. Use these patterns to connect dots:

**Missing graph capture** (1 root signal, multiple effects):
- Root: `graph_mode = "eager_mode"` or `"partial_capture"` — operators not running through compiled graphs
- Cascade: `head_ms / wall_ms > 15%` consistently (host dispatch visible)
- Cascade: `host_dispatch_bound_suspected` finding (head + wait-anchor pattern)
- Cascade: `device_idle_bubble` finding (gaps between un-fused kernels)
- These are NOT independent signals — they're downstream symptoms of the same cause.
  Present the root: "Your model is running without full graph capture. Enabling
  graph mode is the single fix for all visible symptoms (high dispatch overhead,
  variable step timing, device idle gaps)."

**Outdated vLLM-Ascend version** (2+ signals):
- `moe_dispatch = "unfused"` + `ep_load_imbalance_suspected`
- `moe_dispatch = "unfused"` + missing FlashComm1/2
- `attention_backend = unknown` + `graph_mode = "eager_mode"`
- If multiple features that appeared in v0.12+ or v0.17+ are absent, the version
  is likely old. Ask for the version BEFORE suggesting individual fixes: "I see
  multiple optimization opportunities that were introduced in different versions.
  What vLLM-Ascend version are you running? The fix may be a single upgrade rather
  than individual config changes."

**EP is the bottleneck** (2+ signals):
- `comm_share > 20%` in MoE blocks + alltoallv `rank_skew_ratio > 1.5`
- `ep_load_imbalance_suspected` finding
- `hcom_alltoallV` dominates block timeline
- These point to EP as the dominant issue. Don't suggest tweaking individual
  operators — the problem is the alltoallv communication pattern. Focus on:
  fused MC2 (if unfused), EPLB (if imbalance), MC2 hierarchy comm (if multi-node).

**Host CPU is the bottleneck** (2+ signals):
- `head_ms / wall_ms > 15%` + `wait_anchor_false_hotspot` findings
- `aicpu_exposed` finding
- High `wait_us` on small kernels
- These point to CPU-side dispatch. Don't focus on device operators — the fix
  is graph capture + CPU binding.

**Decode is inherently memory-bound** (normal, not a problem):
- `MTE BW < 400 GB/s` on operators with M=1
- `bound_classification = memory-bound` on decode-step operators
- These are expected for single-token decode. Do NOT suggest "fixing" memory
  bandwidth. Only flag if prefill operators also show low BW.

**How to present the synthesis:**

1. **Root cause** (one sentence): the underlying issue
2. **Observed symptoms** (bullet list, 2-3 items): what the profile shows
3. **Why this matters** (one sentence): the performance impact
4. **Fix priority** (ordered list): what to do, in order of impact
5. **Version check** (one question): if version determines whether fix is config or upgrade

Example:
```
Your DeepSeek V4 model has two independent issues:

1. Missing graph capture — your steps show eager-mode dispatch patterns,
   causing 20-35% excess decode latency. The fix is enforce_eager=False
   + verifying cudagraph_capture_sizes covers your decode batch size.

2. Unfused MoE dispatch + EP load imbalance — these share a common cause:
   likely an older vLLM-Ascend version. Fused dispatch (v0.12+) would
   reduce MoE layer latency by 15-30%, and dynamic EPLB (v0.17+) would
   balance expert load. Which version are you running? If v0.18+, these
   are config changes. If v0.16 or earlier, a version upgrade is the
   single most impactful change you can make.
```

### How to present findings

Follow the two-tier approach from the [Agent Interaction Guide](#agent-interaction-guide):

1. **Bottom line first** (2-3 sentences, always before anything else)
2. **Offer structure, let user explore** — don't dump all findings at once

When the user asks about specific metrics, reference the exact source:
"characterizations.json shows this MatMul is memory-bound (MTE BW = 288 GB/s, M=1 decode)."

Default to being concise. Users who want deep detail will ask for it.
The full `report.md` is always available as reference.

### Follow-up question reference table

When `config_signatures` triggers a follow-up, use the version-matched
`knowledge/vllm-ascend/v{version}.md` to find the exact config key.
This table is a quick lookup — not a script to follow blindly.

### Skill Calibration（技能校准）
