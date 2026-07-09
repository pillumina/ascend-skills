# Ascend Profiling Analysis

End-to-end Ascend NPU profiling analysis framework for PyTorch (torch_npu) inference workloads. Consumes raw profiling output (`kernel_details.csv`, `trace_view.json`, `op_summary`, `communication.json`) and produces traceable reports with evidence chains back to source row ranges.

**Target workload**: vLLM-Ascend inference.
**Target hardware**: Ascend 910B2/B3 (A2), Ascend 910C (A3).

## Capabilities

- 11-stage pipeline: triage → normalize → segment → classify → summarize → mstt → cross_rank → diagnostics → characterize → observations → report
- Step / layer / block / operator decomposition with shape-strict class grouping
- AIC/AIV pipeline stage breakdown for Cube/Vector decoupled architectures
- Bound-family classification from hardware-profiled MTE/MAC/Vector data
- Cross-rank alignment with msprof-analyze slow_rank integration
- Host-bound dispatch diagnosis (head bubble + wait-anchor density pattern)
- Quantitative operator characterization: parsed M/K/N dimensions, arithmetic intensity, MTE bandwidth
- vLLM-Ascend config signature detection: attention backend (FIA/SFA/MLA/DSA), graph mode (eager/graph/partial), KV cache compression, MoE dispatch fusion, TP/EP parallelism, context parallelism (PCP/DCP)
- 24-signal diagnostic playbook with version-annotated vLLM-Ascend configuration mappings
- Single-file interactive HTML report with zoomable timeline, per-operator cards, and bubble tracing
- Participatory skill calibration: pipeline collects observation data, agent proposes knowledge base updates

## Setup

### Requirements

- Python 3.8+ with PyYAML
- Optional: `pandas`, `numpy` (for CSV processing; fallback to stdlib csv module)
- Optional: `msprof-analyze` (for slow-rank detection; auto-installed via pip when `--mstt` is passed)

### Installation

**Via `npx skills` (recommended):**

```bash
# Install both Ascend skills (inference + training)
npx skills add pillumina/ascend-skills

# Install inference skill only
npx skills add pillumina/ascend-skills --skill ascend-inference-profiling
```

After installation, the skill is available under `.agents/skills/ascend-inference-profiling/` in your project. Claude Code detects it automatically — no additional configuration is needed.

**Manual installation:**

```bash
mkdir -p .agents/skills
cp -r ascend-inference-profiling .agents/skills/
# Or symlink for development:
ln -s /path/to/ascend-inference-profiling .agents/skills/ascend-inference-profiling
```

No Python packages need to be installed locally — the analysis framework runs remotely on the Ascend host. The only local execution is the thin wrapper that handles SSH connectivity, tar synchronization, and artifact pulling.

### Remote vs local execution

The pipeline can run in two modes:

| Mode | Entry point | When to use |
|------|------------|-------------|
| **Remote** (default) | `profile_analyze.py` | Profiling data is on the Ascend host (typical: tens of GB). The wrapper handles SSH, tar-sync of the framework, remote execution, and pulling lightweight artifacts back. |
| **Local** | `python3 -m ascend_profile.analyze` | Profiling data is already on your local machine or has been pulled from the server. The pipeline runs directly against a local `profile_root`. |

Local execution requires profiling data accessible on the local filesystem. The pipeline itself has no SSH dependency — only `profile_analyze.py` (the wrapper) does. To run locally:

```bash
cd .agents/skills/ascend-inference-profiling/scripts
python3 -m ascend_profile.analyze /path/to/profiling_root --output /path/to/output --verbose --mstt
```

All pipeline stages, knowledge files, and report generation work identically in both modes.

### Remote host access

Remote mode requires SSH access to the Ascend NPU host. The skill reads machine inventory from the `machine-management` skill or accepts explicit session identifiers. The framework subtree (`scripts/ascend_profile/`) is synchronized to a configurable remote work directory (default `/tmp/ascend_profile_framework`). Local mode does not require SSH.

## Usage

**Remote mode** (profiling data on Ascend host) uses `profile_analyze.py`. **Local mode** (data already on your machine) calls `python3 -m ascend_profile.analyze` directly. All flags except `--machine` are shared between both modes.

### Single profiling root (remote)

```bash
python3 .agents/skills/ascend-inference-profiling/scripts/profile_analyze.py \
  --machine <alias-or-ip> \
  --manifest <local-run-dir>/manifest.json \
  --tag my_analysis \
  --mstt
```

### Single profiling root (local)

```bash
cd .agents/skills/ascend-inference-profiling/scripts
python3 -m ascend_profile.analyze /path/to/profiling_root \
  --output /path/to/output \
  --mstt \
  --verbose
```

Flags:

| Flag | Purpose |
|------|---------|
| `--manifest` | Consume output from `ascend-profiling-collection` (reads `analysis_status` and `remote_profile_root`) |
| `--remote-profile-root` | Direct remote path to profiling root (for roots not managed by the collection skill) |
| `--mstt` | Run `msprof-analyze cluster -m slow_rank` for authoritative slow-rank detection |
| `--user-vllm-ascend-version` | Specify vLLM-Ascend version (e.g. `0.18.0`) for version-matched config lookups |
| `--report-mode` | `full-raw` (default, complete HTML) or `summary` (md+xlsx only, faster iteration) |
| `--from-stage` / `--to-stage` / `--only-stage` | Resume or restrict pipeline to specific stages |
| `--keep-remote-output` | Pull full remote output (default pulls lightweight subset only) |
| `--verbose` | Stream remote stage progress to stderr |

### Multi-root sweep (remote)

```bash
python3 .agents/skills/ascend-inference-profiling/scripts/profile_sweep.py \
  --machine <alias-or-ip> \
  --search-root <remote-path> \
  --limit 50 --jobs 4
```

Sweep discovers all directories under `--search-root` that contain `kernel_details.csv`, runs the full pipeline on each, and produces `sweep_summary.json` and `sweep_class_rollup.csv` for cross-capture comparison. For local sweeps, use `python3 -m ascend_profile.sweep --search-root <local-path> --output <out>`.

### Re-running a specific stage

```bash
python3 .agents/skills/ascend-inference-profiling/scripts/profile_analyze.py \
  --machine <alias-or-ip> \
  --remote-output-dir /tmp/ascend_profile_framework/runs/20260701_my_analysis \
  --only-stage report
```

Useful for iterating on report formatting or diagnosis rules without re-running the compute-intensive normalization and segmentation stages. Locally, pass `--only-stage report` directly to `python3 -m ascend_profile.analyze`.

## Pipeline stages

| Stage | Input | Output | Description |
|-------|-------|--------|-------------|
| `triage` | `step_trace_time.csv` | `triage.json` | Low-confidence bottleneck direction from step-level timing aggregates. Always runs. |
| `normalize` | `kernel_details.csv` (per rank) | `normalized_event_index.jsonl` | Unified event stream with pipeline stage decomposition, shape signature extraction, and kernel taxonomy classification. |
| `segment` | Normalized events | `step_segments.json`, `layer_segments.json` | Deterministic step/layer boundary detection from structural anchors. Hard errors and interior islands are fatal. |
| `classify` | Layers | `block_segments.json`, `class_signatures.json` | Layer-to-block decomposition (attention/ffn/moe/aicpu). Shape-strict class grouping. |
| `summarize` | Events + segments | `step_summary.csv`, `step_type_stats.csv`, `layer_summary.csv`, `block_summary.csv`, `operator_summary.csv`, `hccl_op_summary.csv`, +18 additional CSVs | Per-rank and rank-merged aggregates with pipeline breakdown, step-type distribution, bound classification, HCCL collective taxonomy, and evidence indexing. |
| `mstt` | Profiling root | `mstt_slow_rank.csv` | Optional. Runs `msprof-analyze cluster -m slow_rank`. Auto-installs if missing. |
| `cross_rank` | Events + segments | `cross_rank_alignment.csv` | Step-level time-window alignment and operator-level time-bucket alignment across ranks. |
| `diagnostics` | Summaries + alignment | `diagnosis_findings.json` | 11 finding types with confidence, severity, evidence ids, and limitations. Evidence-chain validation. |
| `characterize` | Summaries + events | `characterizations.json` | Operator-level bound/AI/BW metrics. 7 config signature detections from kernel fingerprints. PCP/DCP detection from HCCL patterns. |
| `observations` | All preceding | `run_observations.json` | Unknown kernel collection, segmentation issue logging, config detection edge cases, version gap detection. Persistent aggregate via `observations_history.jsonl`. |
| `report` | All preceding | `report/report.md`, `report/report.xlsx`, `report/report.html` | 13-chapter narrative markdown, multi-sheet Excel, single-file interactive HTML with timeline, operator cards, and bubble tracing. |

![11-stage analysis pipeline](https://github.com/pillumina/ascend-skills/raw/main/docs/ascend-inference-profiling/pipeline-stages.png)

*MSTT is optional (`--mstt` flag). All stages are independently re-runnable via `--from-stage` / `--to-stage` / `--only-stage`.*

## Outputs

### Primary reports

| File | Format | Contents |
|------|--------|----------|
| `report/report.md` | Markdown | 13-section narrative: Executive Summary, Capture & Segmentation, Macro Step Timeline, Pipeline Coverage, Step Type Distribution, Step Class View, Layer & Block View, Operator View, Step Inventory, Cross-Rank & Anomaly Findings, Characterization, Config Signatures, Finding Inventory, Evidence Chain, Limitations |
| `report/report.xlsx` | Excel | Multi-sheet workbook with all summary tables, evidence index, and raw kernel index |
| `report/report.html` | HTML | Single-file zero-dependency interactive report with zoomable Chrome-tracing-style timeline, per-operator cards (46 fields), bubble tracing axes, and L1/L2/L3 navigation |

### Diagnostic outputs

| File | Purpose |
|------|---------|
| `diagnosis_findings.json` | Structured, evidence-backed claims. 11 finding types: `slow_rank_confirmed`, `slow_rank_suspected`, `communication_collective_slow`, `ep_load_imbalance_suspected`, `dp_workload_imbalance`, `rank_workload_asymmetry`, `reduced_work_or_dummy_rank`, `device_idle_bubble`, `host_dispatch_bound_suspected`, `aicpu_exposed`, `wait_anchor_false_hotspot` |
| `characterizations.json` | Quantitative per-operator metrics with config signatures |
| `run_observations.json` | Calibration observations: unknown kernels, segmentation issues, config edge cases |
| `step_type_stats.csv` | Per-step-type aggregate statistics (count, median/max/avg wall_ms, head/bubble ratios). Includes PD-disaggregation hints when capture is 100% single type. |

### Config signatures (automatically detected)

| Detection | Source | Values |
|-----------|--------|--------|
| Attention backend | Kernel category patterns | `fia`, `csa`, `hca`, `dsa`, `mla`, `unknown` |
| KV cache compression | `NpuHammingDistTopK` presence | `enabled`, `not_detected` |
| MoE dispatch fusion | Dispatch kernel names | `fused`, `unfused`, `not_applicable` |
| Graph mode | Step head/wall ratios | `graph_mode`, `eager_mode`, `partial_capture`, `unclear` |
| Parallelism | HCCL collective patterns | TP ≥ 2 / EP ≥ 2 / rank count |
| Context parallelism | HCCL allgather + step type | PCP / DCP / both / none |
| Reduced-work ranks | `has_attention` asymmetry | detected / not detected |

![7 config signature detections](https://github.com/pillumina/ascend-skills/raw/main/docs/ascend-inference-profiling/config-signatures.png)

*Detection engine from `characterize.py`. Seven independent detectors normalize heterogeneous signals into a unified `config_signatures` block. Each has an annotated confidence level.*

## Architecture

### Execution model

In **remote mode**, the analysis framework executes on the Ascend host — profiling roots are typically tens of gigabytes, so the framework reads them in place and only lightweight artifacts are pulled back.

In **local mode**, the pipeline runs directly on your machine against data you already have locally. The `profile_analyze.py` / `profile_sweep.py` wrappers are remote-only; the `ascend_profile` package (`python3 -m ascend_profile.analyze`) runs identically in both modes.

![Remote SSH vs local direct execution](https://github.com/pillumina/ascend-skills/raw/main/docs/ascend-inference-profiling/execution-model.png)

*Both paths produce identical analysis artifacts. Remote mode is for profiling data on Ascend hosts; local mode is for data already pulled to your machine.*

### Agent interaction

The agent interprets pipeline outputs, not the pipeline itself. Scripts produce deterministic data — the agent synthesizes findings, cross-references the diagnostic playbook, and presents root causes with one follow-up question at a time.

![Agent interaction flow](https://github.com/pillumina/ascend-skills/raw/main/docs/ascend-inference-profiling/agent-interaction.png)

*Participatory calibration loop (rightmost branch): the pipeline collects observation data, the agent proposes knowledge base updates, the user confirms, and pytest validates.*

### Knowledge architecture

Three-layer knowledge system for vLLM-Ascend configuration context:

```
knowledge/vllm-ascend/
├── diagnostic-playbook.md        # 24 signal → diagnosis → action entries
├── changelog.md                  # 8 version entries, profiling-relevant changes
├── v0.11.0.md .. v0.22.1rc1.md  # 8 per-version config guides (auto-generated)
├── README.md                     # Knowledge base index and version matching logic
└── _template.md                  # Quality specification for new version entries
```

The playbook is the primary reference. Per-version config guides and changelog are consulted only when exact default values or version-gap confirmation is needed.

![Three-layer knowledge architecture](https://github.com/pillumina/ascend-skills/raw/main/docs/ascend-inference-profiling/knowledge-architecture.png)

### Hardware architecture

The pipeline's bound classification and pipeline stage decomposition target the Ascend DaVinci architecture. A2 (910B2/B3) is a single-die Cube/Vector decoupled design. A3 (910C) is a dual-die chiplet — two A2-equivalent dies with independent Cube/Vector units that can truly overlap.

![A2 vs A3 hardware architecture](https://github.com/pillumina/ascend-skills/raw/main/docs/ascend-inference-profiling/hardware-architecture.png)

*All hardware capacity values are sourced from official Huawei documentation with annotated binning ranges. See `hardware_capabilities.yaml` for source references.*

## vLLM-Ascend knowledge coverage

| Layer | Content | Versions |
|-------|---------|----------|
| Diagnostic playbook | 24 profiling signals covering graph compilation, cross-rank & cluster, attention & KV cache, MoE dispatch & parallelism, speculative decode, communication optimization, host & cluster optimization, upstream vLLM configuration, and workload estimation | Version-independent with inline `*(vX.Y.Z+)*` annotations |
| Config guides | Per-version config defaults, profiling fingerprints, agent notes | v0.11.0, v0.12.0rc1, v0.13.0, v0.17.0rc1, v0.18.0, v0.20.2rc1, v0.21.0rc1, v0.22.1rc1 |
| Changelog | Profiling-relevant changes (new backends, default changes, removed config keys) | Same 8 versions |

New config guides can be generated from a vLLM-Ascend source checkout:

```bash
python3 scripts/generate_config_guide.py \
  --src /path/to/vllm-ascend \
  --output scripts/ascend_profile/knowledge/vllm-ascend/v{version}.md
```

## Tests

```bash
python3 -m pytest tests/ -q \
  --ignore=tests/test_skill_contract.py \
  --ignore=tests/test_stage_validation.py \
  --ignore=tests/test_timeout.py \
  --ignore=tests/e2e \
  --ignore=tests/ut
```

255 tests covering: semantic convention enumeration sync (5), manifest schema validation (4), HTML diagnosis key resolution (5), kernel signature classification (66), attention family resolution (52), MoE family resolution (6), segment validation (15), triage (11), mstt_runner (11), host-bound diagnosis (8), mstt-enriched cross_rank (6), characterize (28), observations (7), graph mode detection (6), CP detection (4), step-type stats (6).

Tests that require external dependencies (`modelscope`, `torch_npu`, `inventory`) are excluded from the local test run. They are exercised in integration environments with full hardware access.

## Project structure

```
ascend-inference-profiling/
├── SKILL.md                          # Agent instructions (workflow, interaction, calibration)
├── .gitignore
├── references/
│   ├── behavior.md                   # Input/output contract, stage definitions
│   ├── acceptance.md                 # Acceptance checklist
│   ├── command-recipes.md            # Usage examples
│   └── deferred-work.md              # Roadmap items
├── scripts/
│   ├── profile_analyze.py            # Single-root entry point (wrapper)
│   ├── profile_sweep.py              # Multi-root entry point (wrapper)
│   ├── _common.py                    # SSH, tar-sync, inventory, manifest helpers
│   ├── generate_config_guide.py      # Version config guide generator
│   └── ascend_profile/               # Analysis framework (runs remotely)
│       ├── analyze.py                # Pipeline orchestrator
│       ├── normalize.py              # Event normalization and taxonomy classification
│       ├── segment.py                # Step/layer boundary detection
│       ├── classify.py               # Block decomposition and class grouping
│       ├── summarize.py              # Per-rank and rank-merged aggregation
│       ├── cross_rank.py             # Cross-rank time-window alignment
│       ├── diagnostics.py            # Evidence-backed diagnosis claims
│       ├── characterize.py           # Quantitative metrics and config detection
│       ├── observations.py           # Calibration data collection
│       ├── report.py                 # Markdown, Excel, and HTML report generation
│       ├── html_report.py            # Interactive single-file HTML report
│       ├── sweep.py                  # Multi-root sweep logic
│       ├── triage.py                 # step_trace_time.csv pre-scan
│       ├── mstt_runner.py            # msprof-analyze slow_rank wrapper
│       ├── common.py                 # Shared dataclasses, I/O, classification
│       ├── knowledge/                # Domain knowledge (YAML contracts + reference docs)
│       │   ├── semantic_conventions.yaml
│       │   ├── kernel_signatures.yaml
│       │   ├── attention_families.yaml
│       │   ├── moe_families.yaml
│       │   ├── model_architectures.yaml
│       │   ├── hardware_capabilities.yaml
│       │   ├── pipeline_taxonomy.md
│       │   ├── bound_classification.md
│       │   ├── block_taxonomy.md
│       │   ├── step_anatomy.md
│       │   ├── step_class_grouping.md
│       │   ├── communication_taxonomy.md
│       │   ├── known_counterexamples.md
│       │   ├── interpreting-outputs.md
│       │   ├── index.md
│       │   └── vllm-ascend/          # Versioned config knowledge
│       │       ├── diagnostic-playbook.md
│       │       ├── changelog.md
│       │       └── v{version}.md × 8
│       └── schemas/
│           └── analysis_bundle.schema.json
└── tests/
    ├── test_new_stages.py             # 81 tests for triage, mstt, characterize, observations
    ├── test_semantic_conventions.py   # Enum sync between Python and YAML
    ├── test_kernel_signatures.py      # Kernel → category classification
    ├── test_attention_families.py     # Attention family resolution + shape refinement
    ├── test_moe_families.py           # MoE family resolution
    ├── test_segment_validator.py      # Segmentation exact-cover validation
    ├── test_manifest_schema.py        # Manifest schema regression
    ├── test_html_diagnosis_key.py     # HTML finding key resolution
    ├── test_skill_contract.py         # Wrapper CLI surface validation
    ├── test_stage_validation.py       # Stage-aware artifact validation
    └── test_timeout.py                # Remote command timeout regression
```

## Limitations

- Communication level-1 analysis (Notify Wait / RDMASend / Memcpy task breakdown) is not implemented. This requires profiling data captured at level ≥ 1. Documented in `communication_taxonomy.md`.
- Triage thresholds are static (free > 20% → hostbound, compute > 85% → computing, comm > 10% → communication). No adaptive thresholding based on model scale or parallelism configuration.
- Attention backend detection for non-DeepSeek architectures falls through to the umbrella `gqa_or_mha` label. Shape-based refinement to `mha`/`gqa`/`mqa` is best-effort (depends on `Input Shapes` being present in `kernel_details.csv`).
- Roofline ridge is computed from hardware_capabilities.yaml (peak FP16 FLOPs / HBM bandwidth from official Huawei documentation). The ridge is a theoretical upper bound — achievable bandwidth is typically lower. Bound classification from measured MTE/MAC/Vector pipeline data remains the primary source.
- Per-version config guides (8 versions) are a mix of manual curation (v0.17.0rc1) and auto-generation with review (all others). Fingerprint stability across v0.11–v0.22 makes this acceptable, but version-specific config defaults should be re-verified against source when adding new versions.
