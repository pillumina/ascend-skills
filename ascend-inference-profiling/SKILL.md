---
name: ascend-inference-profiling
description: Analyze Ascend NPU torch profiler output (kernel_details.csv / trace_view.json / op_summary / communication.json) for one or many profiling roots and produce a traceable report (rank/step/layer/operator summary, cross-rank alignment, diagnosis findings, report.md / report.xlsx / report.html with single-step inspectors, bubble tracing axes, and zoomable Chrome-tracing-style timelines). Supports --mstt (msprof-analyze slow_rank integration) and --user-vllm-ascend-version (version-matched config knowledge). Triage scan of step_trace_time.csv always runs. Use for requests like "分析 profiling", "解析这份 kernel_details", "看 step/layer 切分", "跨 rank 对齐", "通信慢/EP 不均/快慢卡", "生成 profiling 报告". Do not use for HBM/显存归因 (use ascend-memory-profiling), service lifecycle (use vllm-ascend-serving), benchmarks (use vllm-ascend-benchmark), or采集 profiling 数据 (use ascend-profiling-collection).
---

# Ascend Profiling Analysis

> Status: **active**. 远端 pipeline + evidence-chained report + HTML 三级聚焦视图 + stage selector + 自动 config signature 检测 + participatory calibration。YAML 化 knowledge 已起步（见 [Knowledge map](#knowledge-map-for-agents)），新模型 / 新算子族碰到问题时，把 counterexample 落到 `knowledge/known_counterexamples.md`。

Remote substrate rule: use `.remote-dev` remote tools for ad hoc remote
read/edit/bash/search/patch work around profiling roots and generated reports.
Use this skill for the domain analysis workflow and keep its scripts as the
compatibility backend for managed VAWS sessions.

读取 Ascend NPU torch profiler 的产物 (`kernel_details.csv`, `trace_view.json`, `op_summary`, `communication.json` 等)，做 **triage → normalize → segment → classify → summarize → mstt → cross_rank → diagnostics → characterize → observations → report** 的端到端分析，产物全部可追溯到原始 row range。

本 skill 只消费已经采集好的 profiling root，**不负责采集**，不负责服务生命周期，不负责 benchmark。

## Use this skill when

- 用户提供一个 profiling root 路径（远端或工作区路径），或者 `ascend-profiling-collection` 写出的 `manifest.json`，要求分析。
- 用户问 step / layer / operator 统计、跨 rank 对齐、bubble、AICPU、wait anchor。
- 用户怀疑通信慢、EP 负载不均、快慢卡、陪跑/dummy rank、workload 非对称。**传递 `--mstt`** 以获得可靠的慢卡检测。
- 用户需要带 evidence 链的 `report.md` / `report.xlsx` / `report.html`（HTML 报告是单文件零依赖，含交互式 Single-step Inspector、bubble tracing axis、可缩放多流时间轴、46 字段算子卡）。
- 用户要在多个 profiling root 之间扫一遍 (sweep) 并对比。
- 涉及 ≥4 rank 的多卡 profiling 分析。**传递 `--mstt`**——rank 数量越多，慢卡检测价值越大。

## Do not use this skill when

- 任务是 HBM / 显存归因 → 用 `ascend-memory-profiling`。
- 任务是启停服务 → 用 `vllm-ascend-serving`。
- 任务是吞吐/性能 benchmark → 用 `vllm-ascend-benchmark`。
- 任务是采集新的 torch profiler 数据（起服务、控 profile 窗口、跑 workload、analyse） → 用 `ascend-profiling-collection`。
- profiling root 还没采到 `kernel_details.csv`（采集阶段失败） → 先回到 collection skill 排查；本 skill 不做补救。

## Critical rules

- **准确性优先于覆盖率**：宁可报错或留 `low confidence`，也不输出无法追溯的结论。
- **远端解析**：profiling root 通常几十 GB，禁止全量拉回本地解析。本地只做静态检查、schema 校验、产物 manifest 阅读。真实 analyze 在远端容器里跑，必要时把 `report/` 目录拉回本地。
- **入口稳定**：agent 调用 `profile_analyze.py` / `profile_sweep.py`，不要绕过去手写 `python3 -m ascend_profile.analyze` 命令。
- **manifest-aware**：当 `ascend-profiling-collection` 产物可用时，优先把 `--manifest <run_dir>/manifest.json` 喂给 `profile_analyze.py`，让本 skill 自己从 manifest 里读 `remote_profile_root` / `analysis_status`。`analysis_status != "ok"` 直接拒绝，不要静默跳过。
- **进度协议**：进度走 `stderr`，前缀 `__VAWS_PROFILE_ANALYSIS_PROGRESS__=<json>`。最终结果走 `stdout`，单个 JSON 对象。
- **本地状态**：本 skill 的本地状态全部放在 `.vaws-local/profiling-analysis/runs/<timestamp>_<tag>/`（untracked）。远端工作目录默认 `/tmp/ascend_profile_framework`。
- **不写死层数 / 模型语义**：层数 (24/27/36/40/48 …) 是观测结果不是规则；模型名 (LLM / VIT / dummy) 不在 skill 文案里下结论，除非用户/上下文明确给出。

## Cross-platform launcher rule

- macOS / Linux / WSL: `python3 ...`
- Windows: `py -3 ...`

## Public entry points

### Single-root analysis

```bash
python3 .agents/skills/ascend-inference-profiling/scripts/profile_analyze.py \
  (--machine <alias-or-ip> | --session-id <id> | --session-file <session.json>) \
  ( --manifest <local-run-dir>/manifest.json
   | --remote-profile-root <remote-path> ) \
  [--tag <name>] \
  [--local-output-dir <local-dir-to-pull-report>] [--overwrite] \
  [--remote-work-dir /tmp/ascend_profile_framework] \
  [--remote-output-dir <absolute-remote-output-dir>] \
  [--remote-timeout 3600] \
  [--keep-remote-output] \
  [--skip-html] [--report-mode summary|full-raw] \
  [--mstt] [--user-vllm-ascend-version <version>] \
  [--from-stage <stage>] [--to-stage <stage>] [--only-stage <stage>] \
  [--verbose]
```

Flag notes:

- `--local-output-dir`: explicit local dir to write pulled artifacts into. If omitted, defaults to `.vaws-local/profiling-analysis/runs/<timestamp>_<tag>/`. Pass `--overwrite` to allow a non-empty target.
- `--remote-output-dir`: explicit **absolute** remote output dir. Useful with `--from-stage` / `--only-stage` to **reuse a previous run's normalize/segment artifacts** when iterating on classify / diagnostics / report. Default: `<remote-work-dir>/runs/<local-run-dir-name>`.
- `--skip-html` / `--report-mode`: forwarded to the remote analyze stage. `full-raw` (default) renders the complete L1/L2/L3 HTML with operator cards backed by raw `kernel_details` rows. `summary` writes an HTML stub instead — use it for first-stage pipeline debugging when md+xlsx are enough and you don't want to wait for HTML rendering. `--skip-html` is the explicit kill-switch and overrides `--report-mode`.
- `--from-stage` / `--to-stage` / `--only-stage`: resume / partial re-runs; require the prior stages' manifest files already exist in the remote output dir. The wrapper validates only the artifacts the chosen stage *should* produce, so `--only-stage normalize` no longer demands `report/report.md`.
- `--mstt`: run `msprof-analyze cluster -m slow_rank` against the profiling root before cross_rank. Auto-installs msprof-analyze via pip if missing. When available, replaces the heuristic `slow_rank_suspected` with `slow_rank_confirmed` (Dixon's Q / 3-sigma voting on communication sync points) and enriches cross-rank alignment with per-alignment slow-rank markers. **Enable when:** user mentions 快慢卡 / 通信慢 / rank 间不均 / 负载不均衡, OR the profiling root has ≥4 ranks. **Skip when:** the request is "just give me a basic report" or the remote host has no pip network access.

行为：

1. 解析 machine inventory 或 session state，得到目标容器 SSH endpoint。若 `--manifest` 来自 session-scoped collection 且未显式传 target，则优先使用 manifest 里的 `session_file` / `session_id`，确保分析在采集同一个 session 容器内运行。
2. 解析输入：
   - `--manifest`：读取 `analysis_status`、`remote_profile_root`、`schema_version`；若不是 `ok` 直接失败。
   - `--remote-profile-root`：直接走原始路径（用于历史 profiling）。
3. 通过 tar-over-ssh 把当前 `scripts/ascend_profile/` 同步到远端 `<remote-work-dir>/ascend_profile/`（仅这一个子目录，去掉 `__pycache__`/`*.pyc`）。
4. 远端跑 `python3 -m ascend_profile.analyze <REMOTE_ROOT> --output <REMOTE_OUT> --verbose`。
5. 校验远端产物：`manifest.json`、`segment_manifest.json`、`diagnosis_findings.json`、`report/report.md`、`report/report.xlsx`、`report/report.html` 必须存在（HTML 生成失败时仍会留下带错误说明的占位 html，`report/manifest.json` 中的 `html_status` 字段会标 `error`）。
6. 拉回轻量产物（`report/`、所有 `*_manifest.json`、`diagnosis_findings.json`、`evidence_index.csv`、`raw_kernel_index.csv`、CSV 摘要），不拉 `normalized_event_index.csv` / `evidence/bubble_windows.jsonl` 这种大文件，除非给了 `--keep-remote-output` 才整目录拉回。
7. 把摘要、diagnosis 计数、stage timing 整理成 stdout JSON。

### Multi-root sweep

```bash
python3 .agents/skills/ascend-inference-profiling/scripts/profile_sweep.py \
  --machine <alias-or-ip> \
  --search-root <remote-path> [--search-root <remote-path> ...] \
  [--tag <name>] \
  [--limit <N>] \
  [--jobs <N>] [--reuse-existing] \
  [--render-html [--report-mode summary|full-raw]] \
  [--pull-html] \
  [--local-output-dir <local-dir>] [--overwrite] \
  [--remote-work-dir /tmp/ascend_profile_framework] \
  [--verbose]
```

行为：

- 通过 `python3 -m ascend_profile.sweep` 在远端发现所有含 `kernel_details.csv` 的 root，逐个 analyze，产 `sweep_summary.json`。
- 拉回 `sweep_summary.json` 和每个 root 的 lightweight 产物。HTML 报告默认 **不** 拉回，因为 sweep 跑很多 root 时 HTML 累计可能上 GB；要拉就显式加 `--pull-html`。
- sweep 默认在远端跑 `--skip-html` 以节省时间和磁盘；要为每个 root 都渲染 HTML，传 `--render-html` 并可选 `--report-mode`。
- `--jobs N` 在远端用 N 个线程并行分析 root（thread pool；GIL 限制下 N=2~4 通常是最佳收益）。
- `--reuse-existing` 让 sweep 跳过已有 `manifest.json` 的 root，用于断点续跑。
- stdout JSON 给出 `root_count`、`status_counts`、`config`（实际使用的 jobs/report mode 等）、失败 root 列表、`union_layers` inventory 分布。

## Workflow

1. **确认输入来源**
   - 优先 `--manifest`（来自 collection skill）。如果 `manifest.analysis_status == "missing_kernel_details"` 立即停止，把这个状态原样回给用户，不试图分析空 root。
   - 其次 `--remote-profile-root`，要求是远端绝对路径。
2. **远端就绪**
   - 通过 `machine-management` 确认机器 ready；本 skill 不重复实现 ready 检查，但调用前会 ping 一下 `which python3`。
   - tar-sync 只 `scripts/ascend_profile/` 这一个子目录到 `<remote-work-dir>/ascend_profile/`，避免污染 `.vaws-runtime`。
3. **执行分析**
   - 单 root：`analyze.py`；多 root：`sweep.py`。
   - Pipeline 阶段顺序：`triage`（默认执行）→ `normalize` → `segment` → `classify` → `summarize` → `mstt`（可选）→ `cross_rank` → `diagnostics` → `characterize` → `observations` → `report`。
   - Triage 阶段始终执行：扫描 `step_trace_time.csv` 快速判定瓶颈方向，不跳过后续阶段。
   - `--mstt` 开启 msprof-analyze 慢卡检测：在通信同步点上做 Dixon's Q / 3-sigma 投票，替代 matmul start_skew 启发式判定。
   - 远端 `--verbose` 默认开，stage timing 会回到 stdout。
4. **校验产物**
   - 必备文件清单见 `references/behavior.md`「Required artifacts」一节，一个都不能缺。
   - `segment_manifest.json` 里有 `hard_errors > 0`、`interior_island_total > 0` 之类必须显式回报，不当成成功。
5. **拉回报告**
   - 默认只拉轻量摘要 + `report/`。`--keep-remote-output` 才整目录拉回。
   - 大文件（`normalized_event_index.csv`, `evidence/bubble_windows.jsonl`, `*.xlsx`）按需选择性拉。
6. **回答用户**
   - 引用 `report.md` 中的 finding，附带 `evidence_id` / `row range` / `source path`。
   - 不能追溯到 row range 的结论必须标注为 limitation。
   - **需要推理的判断由你（agent）做出，不要依赖脚本预生成的文本。** 脚本只产出确定性的事实数据（JSON/CSV），你是阅读这些数据并给出专业判断的昇腾性能工程师。

## Agent Interaction Guide

This section covers **how to interact with the user** throughout the analysis
lifecycle. The "Agent Interpretation Guide" below covers how to read the data.

### Before running: decide what to run

1. **Check if analysis already exists.** If the user says "show me the last report"
   or references a previous run, check `.vaws-local/profiling-analysis/runs/` for
   existing outputs. Do NOT re-run the pipeline unnecessarily.

2. **Decide flags.** Default: always pass `--mstt` when the profile has ≥ 4 ranks
   or the user mentions communication/slow-rank issues. Skip `--mstt` only when
   the user explicitly asks for minimal analysis or the remote host has no network.
   `--report-mode summary` is enough for a quick first look.

3. **Set expectations.** Tell the user approximately how long it will take before
   starting. The pipeline runs remotely and can take 2-15 minutes depending on
   profile size.

### During the pipeline run: relay progress

The remote pipeline emits stage timing lines on stderr (`[ascend_profile] start/done`).
Relay key milestones to the user:

- "Normalizing profiling data..." (stage 1, reads raw CSVs)
- "Segmenting into step/layer boundaries..." (stage 2)
- "Computing summaries and cross-rank alignment..." (stages 3-6)
- "Running diagnostics and characterization..." (stages 7-8)
- "Generating report..." (stage 9)

Do NOT relay every stage — only the ones that represent meaningful progress.
If `triage.json` is available early (it always is), you can give a preliminary
bottleneck hint while the pipeline continues: "初步判定瓶颈方向为通信，完整分析正在进行中..."

### After pipeline completion: present findings

**Two-tier presentation.** Never dump the full report on first response.

**Tier 1 — Bottom line (always first):**
Two to three sentences maximum. Answer "what did we profile and what's the main finding?"
```
分析完成。8-rank DeepSeek V4 推理 profiling，主要瓶颈为 MoE alltoallv 通信（占 step 时间 23%）。
Rank 3 被 mstt 标记为慢卡（slow_affect_count=47）。Graph mode 已正常开启。
```

**Tier 2 — Offer structure, not data dump:**
After the bottom line, tell the user what's available and ask what they want to see:
```
完整报告已生成，包含：
- report.md：13 章节叙事报告（时间线、step/layer/block/operator 视图、diagnosis findings）
- 交互式 HTML 报告：可缩放时序图、per-operator 卡片
- characterize：算子级 AI/BW/MTE 带宽分析
- Config signatures：检测到的 vLLM-Ascend 配置指纹

你想先看哪个部分？
```

Let the user drive the exploration. They may want the characterization table,
or the cross-rank findings, or the config signatures — don't guess.

### Handling follow-up questions interactively

After the user has seen the initial results, they may ask follow-up questions.
**Do NOT re-run the pipeline.** All data is already in the local run dir.

Common follow-ups and how to answer them (without re-running):

| User asks | Answer from |
|-----------|------------|
| "这个 MatMul 为什么这么慢？" | `characterizations.json` → find operator by name → read bound_classification + M/K/N + AI + MTE BW |
| "哪个 rank 是慢卡？" | `mstt_slow_rank.csv` + `diagnosis_findings.json` → slow_rank_confirmed |
| "allreduce 分布均匀吗？" | `hccl_op_summary.csv` / `hccl_class_summary.csv` → rank_skew_ratio |
| "step 0 和 step 5 的时间线有什么区别？" | `step_anatomy.csv` → look up by segment_id |
| "prefill 和 decode 的 head/wall 比是多少？" | `step_summary.csv` → filter by step_type |
| "这个算子是什么 shape？" | `characterizations.json` → operator_characterizations → shape.M/K/N |

### Follow-up questions about config: ask ONE at a time

When `config_signatures` reveals something that needs user input:

1. **Ask for version first if 2+ config signals detected.** The single most
   important question: "Which vLLM-Ascend version are you running?" If the
   version is old (≤ v0.16), an upgrade replaces multiple individual changes.
   If ≥ v0.18, each signal is likely a simple config toggle. See
   `interpreting-outputs.md` § "How to synthesize findings" for more context.

2. **Pick the most actionable finding next.** Priority: `graph_mode = "eager_mode"`
   or `"partial_capture"` (biggest single impact) → parallelism ambiguity →
   `moe_dispatch = "unfused"` (simple fix when version is known).

3. **Ask ONE question, wait for the answer, then ask the next.**

4. **Accept intentional config without argument.** If the user says
   `enforce_eager=True` is for debugging, acknowledge and move on.

### Error handling: explain what happened and what to do

When the pipeline fails, the stdout JSON has `status: "failed"`, `phase`, and
`error` fields. Tell the user what went wrong in plain language:

| Phase | Plain-language explanation |
|-------|---------------------------|
| `manifest_validation` | The profiling collection was incomplete. The data may be missing or corrupted. Re-run the collection step. |
| `parity_sync` | Could not sync the analysis tools to the remote machine. Check SSH connectivity. |
| `remote_analyze` | The analysis pipeline itself failed. This usually means the profiling data is in an unexpected format (new CANN version, new kernel names) or the remote machine ran out of memory. |
| `artifact_validation` | The pipeline completed but some expected output files are missing or the segmentation reported hard errors. The profiling data may have an unusual structure the segmenter cannot handle. |
| `artifact_pull` | Results were generated but could not be pulled back to the local workspace. Check network connectivity. |

Always include the specific error message from the failed phase so the user
can debug further.

### Task description for the user

When the user provides a profiling root and asks for analysis, they are
effectively delegating a task to you. Keep them in the loop:

1. Confirm what you're about to do before starting
2. Relay progress during the long-running pipeline
3. Present the bottom line first, then let them explore
4. Answer follow-ups from already-pulled data — never re-run unnecessarily
5. Ask ONE config question at a time, explain why, accept the answer

## Agent Interpretation Guide

After each pipeline run, read `knowledge/interpreting-outputs.md`. It covers:
- How to read the output files
- How to interpret config signatures (including version & knowledge lookup)
- How to use characterization data
- How to synthesize findings across multiple signals
- How to ask follow-up questions

For specific profiling signals, use **targeted lookup** on the playbook
(grep for the signal heading, read only that section). For calibration
procedures, see below.

### Skill Calibration（技能校准）

This skill calibrates its knowledge base through participatory calibration.
Pipeline detects gaps → agent identifies patterns → user confirms → agent
applies changes via Edit tool → tests validate → next run benefits.

#### Calibration coverage

| # | Calibratable artifact | Trigger signal | Agent action | Test safeguard |
|---|----------------------|---------------|-------------|----------------|
| A | `kernel_signatures.yaml` | Unknown kernel in 3+ runs, 1000+ events | Append entry before `# Deprecations` | `test_kernel_signatures.py` |
| B | `attention_families.yaml` | `config_signatures.attention_backend` = unknown | Add attention family entry | `test_attention_families.py` |
| C | `moe_families.yaml` | MoE dispatch detected, no moe family classified | Add moe family entry | `test_moe_families.py` |
| D | `semantic_conventions.yaml` | New category needed → test fails with missing values | Add enum value, re-run test | `test_semantic_conventions.py` |
| E | `vllm-ascend/v{version}.md` | User version has no matching config guide | Run `generate_config_guide.py` | Manual review |
| F | `known_counterexamples.md` | Segmentation hard error in any run | Append counterexample entry | Manual review |
| G | Python detection thresholds | Same signal corrected 3+ times by user | Edit threshold constant in source | `test_new_stages.py` |

Items not listed require pipeline code changes and cannot be calibrated interactively. Flag those for the skill maintainer.

#### Calibration workflow

```
Pipeline run → observations files
    ↓ (artifact pull)
Agent appends to ~/.cache/ascend-inference-profiling/observations.jsonl
    ↓
Agent checks persistent log for triggers A—G
    ↓
Trigger fires → ask user → apply action → test validates
    ↓
Next run benefits from updated knowledge.
```

#### Persistent log

After each successful run: append `observations_history.jsonl`'s single line
from the local run dir to `~/.cache/ascend-inference-profiling/observations.jsonl`.

#### Calibration triggers

| Priority | Trigger | Threshold |
|----------|---------|-----------|
| **High** | Unknown kernel recurring | Same kernel in 3+ runs, 1000+ total events |
| **High** | Segmentation hard error | Any occurrence |
| **Medium** | Version gap | User version not in knowledge base |
| **Medium** | Attention backend unknown | Low-confidence or `unknown` detection |
| **Low** | Config detection corrected 3+ times | Persistent false positive |

#### Action A: Add kernel to kernel_signatures.yaml

Ask: "What type of kernel? (attention/moe/communication/norm/quantization/sampling/unsure)"

Append BEFORE `# Deprecations`. Use valid categories from `semantic_conventions.yaml`.

| User says | Insert after section | Safe category |
|-----------|---------------------|---------------|
| attention | `# Attention kernels` | `attention.generic` |
| moe | `# MoE kernels` | `moe.gating` |
| communication | `# Communication kernels (HCCL)` | `communication.collective` |
| norm | `# Block-head structural prefix kernels` | `normalization` |
| quantization | `# Quantisation kernels` | `quant.dynamic` |
| sampling | `# Sampling / control` | `sampling.top_k_top_p` |
| unsure | `# Sampling / control` | `compute.aux` |

Run `pytest tests/test_kernel_signatures.py` after. If it fails with missing category names, go to Action D.

#### Action B: Add attention family to attention_families.yaml

Trigger: `config_signatures.attention_backend = unknown`. Ask user for model architecture and attention kernel names. Add family entry. Run `test_attention_families.py`.

#### Action C: Add moe family to moe_families.yaml

Trigger: MoE dispatch detected but moe family unclassified. Ask: dense FFN, MC2, or fused MC2? Unique kernels? Add family. Run `test_moe_families.py`.

#### Action D: Update semantic_conventions.yaml

When `test_kernel_signatures.py` fails with missing category names, add reported values to the relevant enum. Re-run the test.

#### Action E: Generate version config guide

```bash
python3 scripts/generate_config_guide.py --src <path> --output scripts/ascend_profile/knowledge/vllm-ascend/v{version}.md
```

#### Action F: Log segmentation counterexample

Ask for rank and step number. Append to `known_counterexamples.md` following template.

#### Action G: Tune detection threshold

When same signal corrected 3+ times with consistent explanations:

| Drift | File | Constant |
|-------|------|----------|
| graph_mode false positive (warmup) | `characterize.py` | `_detect_graph_mode` thresholds |
| host_bound false positive | `diagnostics.py` | `HOST_BOUND_HEAD_RATIO` |
| wait_anchor false hotspot | `summarize.py` | `WAIT_ANCHOR_RATIO` |

Edit constant, run `pytest tests/test_new_stages.py` to verify.

#### Presenting calibration status

- **No triggers**: Don't mention calibration.
- **1 trigger**: "`NewOp_0` appeared unrecognized (1200 events). If you know its type, I can add it now."
- **3+ triggers**: "3 calibration opportunities. Top priority: `NewOp_0` (12000 events across 3 runs). Address any?"
- **Version gap**: "Your version has no config guide. I can generate one if you have the source."

Pick 1-2 items max. Accept "no" without argument.

### profile_analyze.py 单 root

```json
{
  "status": "ok",
  "machine": "173.131.1.2",
  "remote_profile_root": "/tmp/prof_35b_tp4/s1",
  "remote_output_dir": "/tmp/ascend_profile_framework/runs/20260507_xxx",
  "local_output_dir": ".vaws-local/profiling-analysis/runs/20260507_xxx",
  "stage_timings": [{"stage": "normalize", "elapsed_s": 12.3}, ...],
  "rank_count": 4,
  "event_count": 1234567,
  "segment_count": 87,
  "layer_count": 27,
  "diagnosis_counts": {"high": 1, "medium": 3, "low": 5},
  "report_md": ".vaws-local/profiling-analysis/runs/20260507_xxx/report/report.md",
  "report_xlsx": ".vaws-local/profiling-analysis/runs/20260507_xxx/report/report.xlsx",
  "report_html": ".vaws-local/profiling-analysis/runs/20260507_xxx/report/report.html"
}
```

### Per-step / per-operator pipeline artifacts

`summarize` 阶段额外产出：

- `step_anatomy.csv`: 每个 step 的 head / main / tail / bubble 拆分（行号 + start_us / end_us + wall/busy/bubble 毫秒），由 `layer_segments.json` 推导。规则见 `scripts/ascend_profile/knowledge/step_anatomy.md`。
- `operator_summary.csv` 现包含原始 CANN pipeline 字段（`aicore_time / aiv_time / aic_mac_time / aic_fixpipe_time / aic_mte1_time / aic_mte2_time / aic_scalar_time / aiv_vec_time / aiv_mte2_time / aiv_mte3_time / aiv_scalar_time`，单位 us），以及四列分类：
  - `op_type ∈ {aic, aiv, mix_cv, mix_comm_aiv, communication, aicpu, dsa, unknown}` — 来源是 `kernel_details.csv` 的 `Accelerator Core` 列，CV 解耦架构下 FIA / GroupedMatmul 等真正同时跑 Cube + Vector 的算子归 `mix_cv`；`DispatchFFNCombine` 等 comm + AIV 融合算子归 `mix_comm_aiv`。
  - `bound_stage` — 9 个 sub-stage 中累计耗时最大的那个（`aic_mac_time` / `aic_mte2_time` / `aiv_vec_time` …），`mix_comm_aiv` 只在 AIV 4 个 stage 里取最大。
  - `bound_family ∈ {cube, vector, aic_mte, aiv_mte, scalar, mixed, aicpu, communication, comm_aiv_mix, dsa, unknown}` — Atlas A2/A3 是 Cube/Vector 解耦架构，AIC mte2 与 AIV mte2 **严禁合并**。
  - `dominant_core ∈ {aic, aiv, mix, none}` — 由 stage-time 推算（不是 wall-time）。
  规则见 `scripts/ascend_profile/knowledge/pipeline_taxonomy.md` 与 `bound_classification.md`。
- `normalized_event_index.csv` 每条 event 也带 `op_type` 列（per-event 粒度），下游可按 op_type 切片（例如某 step 内 `mix_cv` 占多少 ms）。
- `summary_manifest.json` 增补 `pipeline_coverage`（events / operators 两级覆盖率）和 `pipeline_fields`（schema），便于报告侧报告「哪些 events / operators 没有 pipeline 数据」。

### Block decomposition + Step / Layer / Block class artifacts

新增 `classify` 阶段（在 `segment` 与 `summarize` 之间）产出：

- `block_segments.json` — 每个 layer 切成 1~2 个 block，类型 `attention | ffn | moe | aicpu | other`；layer 没有 attention 时 `companion_layer=true`，规则见 `scripts/ascend_profile/knowledge/block_taxonomy.md`。
- `class_signatures.json` — `step_class_by_id` / `layer_class_by_id` / `block_class_by_id` 映射 + 每个 class 的成员列表与元信息。class 签名走 **shape 严格相等**（顺序敏感，缺 shape 不合并），具体规则见 `scripts/ascend_profile/knowledge/step_class_grouping.md`。
- `classify_manifest.json` — block_kind 直方图、companion_layers 计数、shape coverage（多少 class 有 shape）。

`summarize` 阶段消费分类产物，新增四张 CSV：

- `block_summary.csv` — 每个 block 一行；含 `block_kind` / `companion_layer` / `bound_family` / `dominant_core` / `comm_share`（HCCL + `mix_comm_aiv` 占 wall 的比例）+ 11 个 CANN pipeline 字段 + `top_ops`（block 内 top-5）。Bound 分类只看 AI-Core stage（compute-first lens），不会因为 block 里 alltoall_v 重就被短路成 `communication`。
- `block_class_summary.csv` — 每个 block class 一行；聚合 wall_ms_sum/mean/p50/p90、pipeline 求和后的 bound 分类、`comm_share_mean`、`bound_family_member_histogram`、top-10 contributors。
- `layer_class_summary.csv` — 每个 layer class 一行；含 `block_kinds` 序列、`block_kind_wall_ms_share_mean`（attention=38%, moe=62% 这种）、companion 标记、top-10 ops。
- `step_class_summary.csv` — 每个 step class 一行；含 head/main/tail/bubble 比例的 mean、`top_layer_classes`（class 内 top-5 layer class 贡献）、top-10 ops。

同时 `step_summary.csv:step_class_id`、`layer_summary.csv:{layer_class_id, companion_layer, block_kinds}` 增补，便于 SQL-style join。

### Operator view + HCCL artifacts

`summarize` 阶段在 `operator_summary.csv` 之外再生成三张 CSV，用于报告 § 7 Operator View：

- `operator_class_summary.csv` — 把 `operator_summary.csv` 按 `(name, task_type, op_type, roles)` 跨 rank 合并；每行包含 `rank_count`、`call_count`、`duration_sum_us`、11 个 pipeline 字段求和、`bound_family` / `dominant_core`，以及 `rank_duration_min/max/p50_us` 与 `rank_duration_skew_ratio`，便于一眼看出 rank 间的不均。
- `hccl_op_summary.csv` — 仅 HCCL（`op_type ∈ {communication, mix_comm_aiv}`）算子，按 `(hccl_op_kind, comm_aiv_fused, rank_id)` 聚合；`hccl_op_kind ∈ {allreduce, allgather, reducescatter, alltoallv, broadcast, send_recv, barrier, other}`，规则与 CANN HCCL 文档术语对齐，详见 `scripts/ascend_profile/knowledge/communication_taxonomy.md`。
- `hccl_class_summary.csv` — 在 `hccl_op_summary.csv` 基础上再跨 rank 汇总；含 `rank_skew_ratio = (max_rank_avg - min_rank_avg) / mean_rank_avg`，可直接用于 `communication_collective_slow` 类诊断。

`mix_comm_aiv` 融合算子（`DispatchFFNCombine` / `MoeDistributeDispatch` / `MoeDistributeCombine` 等）同时出现在 `comm_aiv_fused=true` 行里，pipeline 字段反映 AIV 侧的工作；纯 HCCL 行的 pipeline 字段为空。

> Level-1 `communication.json` 里的 `Notify Wait` / `Notify Record` / `RDMASend` / `Memcpy` / `Reduce_Inline` 任务级数据在本 skill 当前版本不展开（只在 level-1 profile 上才有意义）；后续若需启用，参考 `communication_taxonomy.md` § 3。

### Report 输出

`report.md` 章节布局（v0.3）：

1. Executive Summary
2. Capture And Segmentation
3. Macro Step Timeline — per-rank step 时长分位数 + head/main/tail/bubble + Top 8 重 step
4. Pipeline Coverage And Bound Families — 覆盖率 + op_type 直方图（aicore Σms / aiv Σms 双侧）+ bound_family 直方图
5. **Step Class View** — Top step classes（按 members × wall_mean 总贡献排序）+ 最重 class 的 top layer classes + 最重 class 的 top operators
6. **Layer And Block View** — Top layer classes（含 block_kind 占比）+ Top block classes（按 kind 分组，含 bound_family / dominant_core / comm_share）
7. **Operator View** — Top compute 算子（rank-merged，含 AIC/AIV/MTE2 流水线分解）+ HCCL collective summary（含 `rank_skew_ratio`）+ 最重 HCCL kind 的 per-rank 分布
8. Step Inventory（按 step_family + main_layer_count 聚合，传统视图）
9. Cross-Rank And Anomaly Findings
10. **Characterization** — quantitative per-operator metrics (bound classification from measured pipeline data, arithmetic intensity from parsed shapes, decode-like M=1 pattern detection), block/HCCL comm-share and rank-skew characterization, **config signatures** (attention backend, KV cache compression, MoE dispatch fusion, graph mode, TP/EP parallelism, context parallelism, reduced-work ranks)
11. Finding Inventory
12. Evidence Chain
13. Limitations

XLSX 包新增 sheet：`step_anatomy`、`step_class_summary`、`layer_class_summary`、`block_summary`、`block_class_summary`、`operator_class_summary`、`hccl_op_summary`、`hccl_class_summary`。

### Sweep 级横向对比

`profile_sweep.py` 现额外产出 `sweep_class_rollup.csv`（每个 root 一行），列包括：

- `rank_count` / `event_count` / `step_count` / `wall_ms_sum`：capture 规模。
- `top_step_class_id` / `top_step_wall_ms_mean` / `top_step_wall_ms_p90` / `top_step_bubble_ratio_mean`：贡献最大的 step class。
- `block_kind_wall_share` / `block_kind_wall_ms_sum`：整个 root 的 attention/ffn/moe wall 占比。
- `hccl_total_ms` / `hccl_share_of_wall` / `hccl_top_kind` / `hccl_top_rank_skew_ratio` / `hccl_max_rank_skew_ratio`：通信总开销与最严重的 rank 偏斜。

可直接用作"模型 × 配置"对比表：把多个不同 TP/DP/EP 的 root 排进同一个 sweep 即可看到这些维度的横向走向。

失败时：

```json
{
  "status": "failed",
  "phase": "remote_analyze | parity_sync | manifest_validation | artifact_pull",
  "error": "...",
  "remote_profile_root": "...",
  "manifest_status": "missing_kernel_details | ok | ..."
}
```

### profile_sweep.py 多 root

```json
{
  "status": "ok",
  "machine": "173.131.1.2",
  "root_count": 61,
  "status_counts": {"ok": 61},
  "elapsed_s": 542.1,
  "summary_path": ".vaws-local/profiling-analysis/runs/20260507_xxx/sweep_summary.json",
  "layer_inventory": {"(27, 40)": 17, "(24,)": 9, ...},
  "failed_roots": []
}
```

## Failure policy

必须报错（hard fail，`status != "ok"`）的情况：

- `manifest.analysis_status` 不是 `ok`。
- 远端 `analyze.py` 退出码非 0。
- 必备产物（`manifest.json`、`segment_manifest.json`、`diagnosis_findings.json`、`report/report.md`、`report/report.xlsx`、`report/report.html`）缺一。
- `segment_manifest.json` 里有 `hard_errors`、`interior_island_total > 0`，或者切分后无法按行号无损覆盖原始事件。
- 报告里的 claim 无法追溯到 evidence id + 原始 row range（report 阶段会自己 raise）。

可以低置信度输出的情况（不算失败）：

- 跨 rank 结构不对称但缺少业务输入信息。
- 怀疑通信慢但缺少 shape 佐证。
- AICPU 命中但 op_summary 不完整。

## Interaction with other skills

| Skill | 互动 |
|-------|------|
| `machine-management` | 提供 SSH endpoint；本 skill 只读 inventory，不改 inventory |
| `remote-code-parity` | 本 skill 不依赖 parity skill；用自带的 tar-over-ssh 同步 `scripts/ascend_profile/`，不动 `.vaws-runtime` |
| `ascend-profiling-collection` | 上游：消费它的 `manifest.json`（`analysis_status`、`remote_profile_root`） |
| `ascend-memory-profiling` | 不交叉，专管 HBM |
| `vllm-ascend-serving` / `vllm-ascend-benchmark` | 不交叉，本 skill 不启停服务 |

## Knowledge map for agents

When extending this skill (new model family, new operator subtype, new
diagnosis heuristic), **read knowledge first, change Python only if
knowledge can't express it**. Suggested reading order:

1. `scripts/ascend_profile/knowledge/index.md` — entry to the rest.
2. `scripts/ascend_profile/knowledge/semantic_conventions.yaml` — enums for
   `op_type` / `block_kind` / `finding_type` / `alignment_method`. New
   values must be added here first so downstream schema tests stay green.
3. `scripts/ascend_profile/knowledge/kernel_signatures.yaml` — kernel name →
   `(op_categories, op_roles)` mapping. This is the contract that `common.py:categories_and_roles()`
   mirrors. Add new kernel signatures here first, then update Python.
4. `scripts/ascend_profile/knowledge/communication_taxonomy.md` — HCCL /
   dispatch / combine semantics.
5. `scripts/ascend_profile/knowledge/block_taxonomy.md` — how
   `classify.decompose_layer_into_blocks` cuts layer → attention / ffn /
   moe / aicpu.
6. `scripts/ascend_profile/knowledge/step_anatomy.md` — head / main / tail
   / bubble definition; consumed by `summarize`.
7. `scripts/ascend_profile/knowledge/known_counterexamples.md` — cases
   that previously broke segmentation / classification. **Add new cases
   here before patching Python.**
8. `scripts/ascend_profile/knowledge/vllm-ascend/` — **three-layer agent
   references**: `diagnostic-playbook.md` (signal → diagnosis → action),
   `v{version}.md` (per-version config defaults, auto-generated), and
   `changelog.md` (all profiling-relevant version changes in one file).
   Agent selects the right layer for each diagnostic task. New versions
   via `scripts/generate_config_guide.py`.

Rule-change invalidation (which stage to rerun via `--from-stage`):

| Change | Re-run from |
|---|---|
| operator taxonomy / new kernel naming | `normalize` |
| segmentation strategy / new anchor / new repair | `segment` |
| block taxonomy / new attention or moe variant | `classify` |
| summary metric definition | `summarize` |
| diagnosis rules / new finding type | `diagnostics` |
| hardware capabilities / shape parsing / AI calc | `characterize` |
| observation thresholds / unknown kernel filter | `observations` |
| report template / HTML widget only | `report` |

When the same remote root must be rerun multiple times while iterating on
a downstream stage, pass `--remote-output-dir <abs-path>` so prior stages'
artifacts are reused.

## Layout note

```
.agents/skills/ascend-inference-profiling/
  SKILL.md
  references/                  # behavior / acceptance / command-recipes
  scripts/
    _common.py                 # SSH / tar-sync / inventory / manifest helpers
    profile_analyze.py         # single-root entry point
    profile_sweep.py           # multi-root entry point
    ascend_profile/            # analysis framework, runs remotely as a package
      analyze.py normalize.py segment.py classify.py summarize.py
      cross_rank.py diagnostics.py report.py html_report.py sweep.py
      triage.py mstt_runner.py characterize.py observations.py common.py
      knowledge/               # taxonomy / pipeline / step-anatomy docs
      schemas/                 # analysis_bundle.schema.json
      README.md                # framework data contract
```

本 skill 的 wrapper（`profile_analyze.py` / `profile_sweep.py`）只做远端编排和产物搬运，**不复制分析逻辑**。框架本身的数据契约见 `scripts/ascend_profile/README.md`。

## References

- `references/behavior.md` — 输入/产物契约、阶段定义、远端目录布局。
- `references/command-recipes.md` — 单 root / sweep / 仅拉报告 / 历史 root 追分析的命令样例。
- `references/acceptance.md` — 验收清单（用于 reviewer 和回归测试）。
