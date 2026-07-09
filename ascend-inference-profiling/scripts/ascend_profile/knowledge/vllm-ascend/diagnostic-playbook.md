
## Signal index

| Pipeline detection | Section |
|---|---|
| graph_mode = "eager_mode" (head/wall > 10%) | |
| graph_mode = "partial_capture" (mix of graph-like and eager-like steps) | |
| head_ms / wall_ms > 15% consistently | |
| aicpu_exposed finding in diagnosis_findings.json | |
| slow_rank_confirmed or communication_collective_slow finding | |
| device_idle_bubble + host_dispatch_bound_suspected findings co-occurring | |
| rank_workload_asymmetry or reduced_work_or_dummy_rank finding, or reduced_work_r... | |
| dp_workload_imbalance finding | |
| wait_anchor_false_hotspot finding | |
| config_signatures.kv_cache_compression = enabled but compression kernels > 15% of a... | |
| config_signatures.attention_backend = unknown | |
| attention collective group sizes differ from TP size — possible PCP or DCP active | |
| moe_dispatch = "unfused" (separate dispatch + combine) | |
| alltoallv rank_skew_ratio > 1.5 + ep_load_imbalance_suspected | |
| hcom_allReduce in attention block + comm_share > 20% in attention | |
| hcom_alltoallV in MoE block + comm_share > 20% in MoE | |
| step_type = "speculative" in step_summary.csv | |
| allreduce showing as separate op from adjacent compute | |
| decode matmul showing MTE BW < 400 GB/s | |
| high head_ms with enable_cpu_binding uncertain | |
| alltoallv in MoE with multi-node EP | |
| DSA attention with visible communication gaps | |
| allreduce pattern suggesting FlashComm2 vs FlashComm1 | |
| alltoallv rank_skew_ratio decreasing over profiling captures | |

### How to use this index

1. Find your signal in the index above by name
2. Read the entire file with `limit=550` — all 24 signal headings appear as
   `### Signal:` lines. Find the matching heading, note its line number.
   (The agent Read tool provides line numbers in output.)
3. Re-read that exact line with `limit=30` — each signal section is ~20-30
   lines and fully self-contained. Version annotations, diagnosis, and
   actions are all inline.
4. Only check `v{version}.md` and `changelog.md` if the user asks for exact
   defaults or version-gap confirmation.

**No hardcoded line numbers in these instructions.** The `limit=550` covers
the whole file (currently ~530 lines) with margin for future signals. The
`limit=30` is generous — signal sections vary from 15 to 30 lines.

# Diagnostic Playbook — vLLM-Ascend Profiling

## Graph Compilation & Dispatch

### Signal: `graph_mode = "eager_mode"` (head/wall > 10%)

**Diagnosis**: Operators are individually launched by the host. Every kernel carries host-side dispatch overhead (stream launch, ACL graph dispatch). This is the dominant cause of poor small-batch throughput.

**Performance impact**: Enabling graph mode reduces step latency by 30-50% (prefill) and 20-35% (decode) relative to eager execution on vLLM-Ascend.

**Common causes**:
1. `enforce_eager = True` — user explicitly disabled graph mode (often for debugging)
2. Profiling captured during warmup — first 3-5 iterations are pre-capture. Re-profile after warmup completes.
3. Dynamic shapes preventing capture — model has variable sequence lengths or batch sizes that torch.compile cannot specialize

**Action**: 
- Check `enforce_eager` setting → should be `False` for production
- Verify `enable_npugraph_ex = True` in `ascend_compilation_config` *(v0.12.0+, default True)*
- If profiling during warmup: re-capture after 10+ iterations
- If dynamic shapes: consider `VLLM_BATCH_INVARIANT` or fixed batch sizing

**Caveats**: 
- Some models (custom architectures, dynamic control flow) genuinely cannot be graph-captured. Accept eager mode if unavoidable.
- **v0.11.0 only**: `AscendCompilationConfig` does not exist. Graph mode is hardcoded — `enforce_eager` is the only toggle. The `enable_npugraph_ex`, fusion passes, and static_kernel features are NOT available.

### Signal: `graph_mode = "partial_capture"` (mix of graph-like and eager-like steps)

**Diagnosis**: Some batch sizes are captured by torch.compile / cuda graphs, others are not. This is the single most common profiling pattern in vLLM-Ascend production deployments.

**Root cause in 90% of cases**: `cudagraph_capture_sizes` does not cover all batch sizes present in the profiling window.

**Common scenarios**:
1. **Speculative decode active**: decode uniform query length = `num_spec_tokens + 1`. If this value is not in `cudagraph_capture_sizes`, every decode step falls back to eager → partial capture.
2. **Prefill with varying chunk sizes**: chunked prefill produces many different batch sizes. Only sizes in the capture list get graph-optimized.
3. **Mixed prefill/decode in same window**: prefill batch sizes are rarely captured (they vary). This is expected and not a problem.

**Action**:
- Ask for `cudagraph_capture_sizes` and `num_speculative_tokens`
- Verify `num_spec_tokens + 1` is in the capture list
- If not, add it
- For prefill: partial capture is expected — prefill batch sizes vary by design

**Performance impact**: Adding the missing decode size typically eliminates eager fallback, improving decode throughput by 15-25%.

**MTP graph capture (speculative decode)**: If `step_type = "speculative"` co-occurs with `partial_capture`, the most likely cause is `cudagraph_capture_sizes` missing `num_spec_tokens + 1`. Speculative decode uses a uniform decode query length of `num_spec_tokens + 1` — if this value is not in the capture list, every speculative step falls back to eager. Ask for `num_speculative_tokens` and verify the capture list covers it. **The pipeline does NOT produce a separate `mtp_graph` detection** — it's the agent's job to combine `graph_mode = partial_capture` + `step_type = speculative` to reach this diagnosis.

### Signal: `head_ms / wall_ms > 15%` consistently

**Diagnosis**: Host-side dispatch overhead is high even in graph mode. Possible causes:
1. **CPU core binding not configured**: vLLM-Ascend worker threads competing with system processes
2. **Heavy sampling/logits processing**: large vocabulary models spend significant CPU time in logits → sampling
3. **Dispatch thread contention**: multiple NPU streams competing for a single host dispatch thread

**Action**:
- Check CPU binding: `enable_cpu_binding = True` in `additional_config` (default since v0.11.0)
- For large-vocab models (vocab_size > 100k): sampling is inherently CPU-heavy. Consider speculative decode to reduce sampling frequency.
- If `wait_anchor_ops.csv` shows many operators with wait_ratio > 80%: host dispatch is the bottleneck

### Signal: `aicpu_exposed` finding in diagnosis_findings.json

**Diagnosis**: An AICPU operator appears on the critical path (not hidden by concurrent AI Core/Vector work). AICPU cores are slower than AI Core/Vector — exposed AICPU ops add latency directly to the step.

**Common causes**:
1. **Custom sampling/logits ops**: Some model architectures use AICPU for sampling
2. **Non-fused companion ops**: RoPE, KV cache write, or normalization running on AICPU instead of being fused
3. **Debug/profiling instrumentation**: Some CANN debug features run on AICPU

**Action**:
- Check if the AICPU op can be offloaded to AI Core (model config)
- Check if fusion passes (`fuse_qknorm_rope`, `fuse_norm_quant` *(v0.12.0+)*) are enabled
- If sampling-related: this is expected for large vocabulary models

## Cross-Rank & Cluster

### Signal: `slow_rank_confirmed` or `communication_collective_slow` finding

**Diagnosis**: mstt has identified one or more slow ranks via communication sync-point voting (Notify Wait/Record analysis). This is a **confirmed** detection — not a heuristic guess. The slow rank causes other ranks to wait at collective barriers.

**How to read mstt data**:
- Look at `mstt_slow_rank.csv`: `slow_affect_count` = number of times this rank was the last to arrive at a collective sync point
- Any rank with `slow_affect_count > 0` is slower than peers at some communication barrier
- Count > 20: severe slow card, likely hardware issue
- Count 1-5: mild, may be workload-dependent

**When `communication_collective_slow` co-occurs**:
- Look at `hccl_op_summary.csv` for the specific collective with high `rank_skew_ratio`
- If the slow rank from mstt is the same rank showing high duration in the skewed collective → **this is a hardware issue** (check thermal, PCIe, memory)
- If the slow rank from mstt is NOT the one with high collective duration → **this is a workload imbalance** (the rank with high duration is doing more work, not waiting at barriers)

**Action**:
- Check hardware health on slow rank: `npu-smi info -t temp -i <rank_id>`, `npu-smi info -t pcie -i <rank_id>`
- Compare npu frequency across ranks: `npu-smi info -t freq`
- Check CPU binding on slow rank's node: `enable_cpu_binding = True`
- If hardware is normal: check workload distribution (EP routing, DP load balance)

**Caveats**:
- A single slow rank drags down the entire cluster. Even mild slow-card issues compound over many collective operations.
- `slow_rank_suspected` (from matmul start_skew) is a weaker signal. `slow_rank_confirmed` (from mstt) is authoritative.

### Signal: `device_idle_bubble` + `host_dispatch_bound_suspected` findings co-occurring

**Diagnosis**: Steps show significant idle time (bubble) AND the host dispatch pattern suggests CPU-side bottleneck. This is the most actionable form of "host-bound" — the profiling evidence points to host dispatch, not device-side waiting.

**Distinguishing host-bound from device-waiting**:
- `host_dispatch_bound_suspected`: head_ms is high (step starts with idle gap), wait_anchor density is high (operators ready but waiting on host launch). CPU is the bottleneck.
- `device_idle_bubble` alone: bubble could be from any source (host dispatch, rank synchronization, data dependency). Not specific enough to act on.
- Both co-occurring: high confidence that host dispatch is the specific cause.

**Action**:
- Enable graph mode if not already (`enforce_eager=False`). Graph mode eliminates individual kernel launches — the single strongest intervention for host dispatch overhead.
- Check CPU binding: `enable_cpu_binding = True` in `additional_config`
- Check for CPU contention: `top -H -p <vllm_pid>` on the host to see competing processes
- For A2: consider `HCCL_OP_EXPANSION_MODE=AIV` to offload HCCL dispatch to AIV
- Collect ftrace for definitive host-side diagnosis: `trace-cmd record -e sched:* -p <vllm_pid>`

**Performance impact**: Fixing host dispatch bottleneck typically reduces step latency by 10-30%, with the largest gains on small-batch decode steps.

### Signal: `rank_workload_asymmetry` or `reduced_work_or_dummy_rank` finding, or `reduced_work_ranks` config signature

**Diagnosis**: Some ranks have significantly different workloads from others — fewer events, no attention kernels, or different step structure. This is a structural property of the deployment, not a performance problem per se.

**Common causes**:
1. **Pipeline parallelism (PP)**: ranks handle different layer subsets. Ranks with fewer layers or no attention layers show reduced workload.
2. **Encoder-decoder models**: encoder ranks process different sequence lengths or have different layer counts.
3. **DP with unequal data distribution**: unbalanced batch sizes across DP replicas.
4. **Disaggregated prefill/decode (PD)**: prefill nodes and decode nodes have fundamentally different step structures.

**Action**:
- If expected (PP, PD, encoder-decoder): no action needed. This is normal.
- If unexpected: check `pipeline_parallel_size` and `data_parallel_size` configuration
- Verify all ranks are on the same hardware type and CANN version

**Caveats**: Asymmetric workload is a structural observation, not a performance bug. It becomes a concern only when combined with other signals — e.g., if the asymmetry is unexpected, or if the reduced-work ranks are also identified as slow cards.

### Signal: `dp_workload_imbalance` finding

**Diagnosis**: Data-parallel replicas show significantly different total wall time. This usually means unequal batch sizes or sequence lengths across DP groups.

**This is a structural observation, not directly actionable from profiling alone.** DP workload distribution is controlled by the serving scheduler (batch formation) or training data loader (sample distribution). The profiling data cannot tell you WHY the batches are unbalanced.

**Action**:
- If serving: check scheduler configuration (`max_num_batched_tokens`, `max_num_seqs`, scheduling policy)
- If training: check data loader distribution and global batch size computation
- Flag for further investigation — this finding alone does not point to a specific config change

### Signal: `wait_anchor_false_hotspot` finding

**Diagnosis**: An operator shows high wait_ratio (> 95%) but very low execution duration (< 10us). The operator appears as a "hotspot" in the timeline because of its wait time, but the actual execution is tiny. This operator is waiting on something — typically a synchronization point (stream sync, HCCL barrier) or a data dependency — not executing.

**This is almost always a profiling artifact, not a performance problem.** The wait time is real (the NPU is idle during this period), but the cause is an upstream synchronization, not the operator itself. Removing or optimizing this operator will not improve performance.

**Action**:
- Look at what comes BEFORE this operator in the event stream — that's where the real bottleneck is
- If the operator follows a collective: the collective's duration (including notify wait) is the real bottleneck
- If the operator follows a data dependency: check the producing kernel's duration
- **Do NOT report this as a performance issue to optimize.** It's a symptom, not a cause.

## Attention & KV Cache

### Signal: `config_signatures.kv_cache_compression = enabled` but compression kernels > 15% of attention block

**Diagnosis**: KV cache compression (Hamming-distance KV pruning) is active but its overhead exceeds the memory savings. The `NpuHammingDistTopK` and `NpuSignBitsPack` kernels are consuming significant attention block time.

**Performance trade-off**: KVComp reduces KV cache memory by 2-4x but adds 5-15% compute overhead per attention operation. When the overhead exceeds 15%, the memory savings may not justify the latency cost.

**Action**:
- Check if memory pressure is the primary concern. If yes, accept the overhead.
- If latency is the primary concern, disable KVComp: `additional_config.hamming_sparse.enabled = False`
- Tune the sparsity level via `sparse_json_location`

### Signal: `config_signatures.attention_backend = unknown`

**Diagnosis**: The attention backend cannot be determined from kernel signatures. This means the model uses either a custom attention implementation, a very new backend (added in a version newer than the knowledge base), or a backend that was missed during curation.

**Common scenarios**:
1. **v0.20.2+ with DSA or FA3 backend**: `dsa_v1.py` and `fa3_v1.py` were added in v0.20.2. If running v0.20.2+, these are new backends — `attention_families.yaml` may need a calibration entry.
2. **v0.22.1rc1+ with DSA context parallel**: `dsa_cp.py` added.
3. **Custom/experimental backend**: Some research models use custom attention kernels
4. **Calibration gap**: The backend exists but was missed in `attention_families.yaml`

**Action**:
- Check vLLM-Ascend version. If v0.20.2+: check if DSA or FA3 kernels are present.
- If custom backend: ask user for kernel names → propose calibration (Action B)

## MoE Dispatch & Parallelism

### Signal: attention collective group sizes differ from TP size — possible PCP or DCP active

**Diagnosis**: vLLM-Ascend supports two independent Context Parallelism features:
- **PCP (Prefill Context Parallelism)**: splits prefill attention computation across `parallel_config.prefill_context_parallel_size` ranks *(v0.13.0+)*. Env var `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` removed in v0.21.0.
- **DCP (Decode Context Parallelism)**: splits decode attention computation across `parallel_config.decode_context_parallel_size` ranks *(v0.13.0+)*.

Both create sub-groups within the TP group, making attention collectives (allreduce/allgather) use group sizes smaller than the full TP size.

**Profiling fingerprint**:
- **PCP active, DCP inactive**: prefill steps show allreduce/allgather with groups smaller than full TP. Decode steps show standard TP collective sizes.
- **DCP active, PCP inactive**: decode steps show allreduce/allgather with groups smaller than full TP. Prefill steps show standard TP collective sizes.
- **Both active**: both prefill and decode attention collectives show reduced group sizes. The effective group dimensions are `PCP × TP` for prefill, `DCP × TP` for decode.
- **Neither active**: all attention + FFN allreduce show the same group size (TP only).

**Key interactions**:
- PCP + FlashComm1: `max_num_batched_tokens` must be divisible by `tp_size × pcp_size`. vLLM-Ascend auto-adjusts if not.
- PCP + DCP in PD-disaggregated scenarios: P and D nodes may have different CP sizes. `platform.py` enforces consistency for KV pool sharing scenarios.
- SFA backend: the PCP&DCP implementation has specific constraints on communication group layouts.
- Both consume additional HCCL streams (~100 additional streams for PCP+DCP combined).

**Action**:
- Check `parallel_config.prefill_context_parallel_size` — if > 1, PCP is active
- Check `parallel_config.decode_context_parallel_size` — if > 1, DCP is active
- Verify `max_num_batched_tokens % (tp_size × pcp_size) == 0` when PCP is used with FlashComm1
- **v0.21.0+**: env var removed, use `parallel_config` keys exclusively
- **v0.12.0 and earlier**: PCP/DCP not available

**Performance**: PCP reduces per-rank memory for long-context prefill (> 32k tokens). DCP reduces per-rank memory for large-batch decode. Both add HCCL communication within CP groups — the benefit depends on sequence length (PCP) and batch size (DCP).

### Signal: `moe_dispatch = "unfused"` (separate dispatch + combine)

**Diagnosis**: MoE dispatch and combine are running as separate kernels with an alltoallv in between. This means three kernel launches per MoE layer instead of one — triple the launch overhead.

**Performance impact**: Fused dispatch (mode 2 for decode, mode 1 for prefill) reduces MoE layer latency by 15-30% relative to unfused.

**Prerequisites for fusion** *(v0.12.0+)*:
- W8A8 quantization on EP layers (required for both mode 1 and 2)
- EP ≤ 32 (mode 1 only)
- No dynamic EPLB active (mode 1 only) *(EPLB added v0.17.0+)*
- MTP must also be W8A8 (mode 2 only)
- **v0.11.0**: MC2 fusion not available.

**Action**:
- Verify W8A8 quantization is active
- If decode workload: set `VLLM_ASCEND_ENABLE_FUSED_MC2=2`
- If prefill workload: set `VLLM_ASCEND_ENABLE_FUSED_MC2=1`
- If mixed: mode 2 covers decode; prefill will remain unfused unless mode 1 prereqs are also met

### Signal: alltoallv `rank_skew_ratio > 1.5` + `ep_load_imbalance_suspected`

**Diagnosis**: Expert-to-rank distribution is unbalanced — some ranks handle significantly more expert computation than others. This is the #1 cause of EP performance degradation.

**Common causes**:
1. **Static expert placement**: experts assigned to ranks statically, routing skew causes imbalance
2. **Dynamic EPLB not active** *(v0.17.0+)*: `additional_config.eplb_config.dynamic_eplb = False` (default)
3. **Expert heat collection interval too long**: EPLB can't converge fast enough for short-lived workloads
4. **v0.16.0 and earlier**: EPLB not available — static expert placement is the only option.

**Action** *(v0.17.0+ for EPLB)*:
- Enable dynamic EPLB: `DYNAMIC_EPLB=true` + `additional_config.eplb_config.dynamic_eplb = True`
- Reduce `expert_heat_collection_interval` for faster convergence (default 600, try 100-200)
- Verify `num_redundant_experts > 0` if using redundant expert placement
- **v0.16.0 and earlier**: EPLB not available. Static expert placement only.

### Signal: `hcom_allReduce` in attention block + `comm_share > 20%` in attention

**Diagnosis**: Tensor-parallel allreduce in attention is a significant portion of step time. The communication overhead may exceed the compute benefit of TP.

**TP sizing guideline** (heuristic, not rule):
- Target ≥ 8 attention heads per rank. Below this, allreduce overhead tends to dominate.
- GQA constraint: TP is bounded by KV head count. 4 KV heads → TP ≤ 4 for attention.
- For MoE models: TP in attention is the primary communication overhead. FFN allreduce is secondary.
- For dense models: both attention and FFN allreduce contribute. TP > 4 rarely pays off for models < 30B parameters.

**Action**:
- Check `parallel_config.tensor_parallel_size` against model head count
- If `comm_share > 30%` in attention blocks: consider reducing TP
- If allreduce sizes vary across modules: `finegrained_tp_config` *(v0.17.0+)* may be active, enabling different TP sizes per module
- Evaluate `fuse_allreduce_rms` *(v0.12.0+, default False before v0.18.0, True from v0.18.0+)*: fuses allreduce with rmsnorm, reducing visible allreduce time
- Evaluate `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` *(A2 only, default 0)*: fuses matmul with allreduce

### Signal: `hcom_alltoallV` in MoE block + `comm_share > 20%` in MoE

**Diagnosis**: Expert-parallel alltoallv dominates MoE block time. This is inherent to MoE architectures — alltoallv communication scales with the number of EP ranks.

**EP sizing guideline**:
- EP increases expert capacity (more experts in aggregate) but adds alltoallv overhead
- Rule of thumb: EP > 4 starts to show diminishing returns for latency; EP > 8 shows significant overhead
- Fused MC2 (mode 1/2) mitigates but does not eliminate EP overhead

**Action**:
- Check expert load balance (rank_skew_ratio) — imbalance amplifies EP overhead
- Consider reducing EP if expert capacity is sufficient
- Enable fused MC2 *(v0.12.0+)* if not already active
- For inter-node EP: check `enable_mc2_hierarchy_comm` *(v0.17.0+)* — hierarchical alltoallv reduces cross-node traffic

## Speculative Decode

### Signal: `step_type = "speculative"` in step_summary.csv

**Diagnosis**: Speculative decode is active. The draft model generates `num_speculative_tokens` per step, and the target model verifies them.

**Key profiling implications**:
1. Decode `cudagraph_capture_sizes` must cover `num_spec_tokens + 1`
2. Draft model runs on the same NPU — its kernels appear in the profile
3. Acceptance rate matters: low acceptance → wasted draft compute. Acceptance is NOT visible in profiling.

**Performance impact**: 
- High acceptance (> 80%): 2-4x throughput improvement
- Low acceptance (< 50%): draft model overhead may negate benefits

**Action**:
- Verify `cudagraph_capture_sizes` covers `num_spec_tokens + 1` (see partial_capture above)
- If acceptance rate is unknown: ask user to check vLLM metrics (`spec_decode_acceptance_rate`)
- For MTP (multi-token prediction): draft model must also be graph-captured. The pipeline does NOT produce a separate `mtp_graph` detection. Combine `graph_mode = partial_capture` + `step_type = speculative` to diagnose this. Performance impact: adding the missing size typically improves speculative decode latency by 20-40%.

## Communication Optimization

### Signal: allreduce showing as separate op from adjacent compute

**Diagnosis**: Communication and compute are not overlapped. Two optimization paths exist:
1. **`fuse_allreduce_rms`** (AscendCompilationConfig, default False → True in v0.18.0+): fuses the allreduce with the subsequent rmsnorm into a single kernel. Allreduce wall time appears shorter because rmsnorm is absorbed.
2. **`VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE`** (env, default 0): fuses matmul with allreduce. Matmul wall time appears shorter. A2 only.

**Performance impact**: Fusing allreduce with rmsnorm reduces visible communication time by 5-15% (the rmsnorm portion is hidden). Fusing matmul with allreduce is more impactful (10-20%) but only works on A2.

**Action**:
- Enable `fuse_allreduce_rms=True` in `ascend_compilation_config` *(v0.12.0+; requires npugraph_ex)*
- For A2: enable `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE=1` *(A2 only)*
- FlashComm1/2 are alternative paths for communication optimization — evaluate if the above are insufficient

### Signal: decode matmul showing `MTE BW < 400 GB/s`

**Diagnosis**: For decode (M=1), low MTE bandwidth is expected — the matmul is too small to saturate the memory bus. This is a model architecture constraint (M=1 means single-token decode, no batch dimension).

**When this is a problem**: Only when prefill matmuls also show low BW. Prefill MTE BW should be 800+ GB/s for well-tiled matmuls.

**Action**:
- For decode: this is expected. No action needed.
- For prefill with low BW: check K dimension (head_dim). K=64 or K=96 leads to poor tiling. This is a model architecture constraint — cannot be fixed via config.
- For prefill with small M (M < 32): batch size is too small to amortize weight loads. Consider increasing prefill batch size.

## Host & Cluster Optimization

### Signal: high `head_ms` with `enable_cpu_binding` uncertain

**Diagnosis**: vLLM-Ascend worker threads may be competing with system processes or migrating across CPU cores. This causes inconsistent host dispatch latency — visible as variable `head_ms` across steps.

**CPU binding mechanism**: `enable_cpu_binding = True` in `additional_config` (default since v0.11.0) pins vLLM worker threads to specific CPU cores, reducing NUMA cross-talk and cache migration. When disabled, workers float across cores under Linux CFS scheduling.

**Profiling fingerprint**: 
- `head_ms` varies significantly across steps (standard deviation > 20% of mean)
- Individual steps show large `head_wall_ms` with good `main_wall_ms` — the step body runs fast, but dispatch is slow
- Not to be confused with graph-mode → eager fallback (which shows consistently high head_ms, not variable)

**Action**:
- Verify `enable_cpu_binding` is not explicitly disabled: check `additional_config`
- If on a shared host (multiple processes on same NUMA node): CPU binding prevents migration but doesn't solve contention. Check `top -H` on the host.
- For NUMA-aware deployment: verify vLLM processes are on the correct NUMA node (`numactl --cpunodebind`)

### Signal: alltoallv in MoE with multi-node EP

**Diagnosis**: When EP spans multiple nodes, alltoallv communication crosses the inter-node network (RoCE). The `enable_mc2_hierarchy_comm` option controls whether intra-node alltoallv uses HCCS (fast) with RoCE only for cross-node (hierarchical), or all ranks use RoCE uniformly (flat).

**Profiling fingerprint**:
- Hierarchical (enabled): alltoallv shows two distinct latency tiers — intra-node alltoallv (low latency, ~10-50us) and cross-node alltoallv (higher latency, ~100-500us depending on data size)
- Flat (disabled): all alltoallv pairs show similar latency, all crossing RoCE

**Action** *(v0.17.0+)*:
- Check `enable_mc2_hierarchy_comm` in `additional_config` (default False)
- If deploying multi-node EP: enabling hierarchical alltoallv typically reduces alltoallv wall time by 20-40% for inter-node EP
- Not applicable for single-node EP (all ranks on same node — HCCS is always used)
- **v0.16.0 and earlier**: `enable_mc2_hierarchy_comm` not available.

### Signal: DSA attention with visible communication gaps

**Diagnosis**: The `multistream_dsv4_dsa_overlap` option (default True since available) controls whether attention computation and HCCL communication run on separate NPU streams in the DSA backend. When enabled, allreduce appears to overlap with attention compute — allreduce wall time looks shorter because it's hidden by concurrent computation.

**Profiling fingerprint**:
- Enabled: allreduce in DSA attention blocks shows wall time that is shorter than expected for the data size. The allreduce appears to start before attention compute finishes — there's no gap between the end of attention compute and the start of allreduce because they overlap.
- Disabled: allreduce follows attention compute sequentially — a visible gap exists, and allreduce wall time reflects the full communication cost.

**Action** *(v0.17.0+ for multistream_dsv4_dsa_overlap)*:
- Verify `multistream_dsv4_dsa_overlap` in `additional_config` (default True)
- If disabled and attention blocks show high `comm_share`: re-enable. This is a key DeepSeek V3.2/V4 performance optimization.
- Only applies to SFA/DSA backends (DeepSeek models). No effect on FIA (dense models) or MLA backends.

### Signal: allreduce pattern suggesting FlashComm2 vs FlashComm1

**Diagnosis**: vLLM-Ascend has two FlashComm variants for TP communication. FlashComm1 (`VLLM_ASCEND_ENABLE_FLASHCOMM1`, default 0), FlashComm2 (`VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE`, default 0). *(v0.12.0+ for both)*.

**Profiling fingerprint**:
- FlashComm1: reduces allreduce time uniformly. Better at high concurrency.
- FlashComm2: allreduce time reduction is proportional to the O-matrix TP group size (the value set for `FLASHCOMM2_PARALLEL_SIZE`). When set to N > 0, the O-projection allreduce uses N-rank groups instead of full TP-size groups → smaller allreduce → lower latency.
- Neither enabled: standard HCCL allreduce with full group size. `hccl_op_summary.csv` shows allreduce for all TP ranks.

**Action**:
- For TP=8: try `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=4` (two groups of 4)
- For TP=4: try `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE=2`
- FlashComm2 is generally preferred over FlashComm1 for newer deployments
- Both require graph mode (`enforce_eager=False`)

### Signal: alltoallv rank_skew_ratio decreasing over profiling captures

**Diagnosis**: Dynamic EPLB (Expert Load Balancing) is active and converging. The alltoallv skew across EP ranks decreases as the balancer re-assigns experts to balance load.

**How EPLB works**: `dynamic_eplb` collects expert heat over `expert_heat_collection_interval` steps, runs a balancing algorithm every `algorithm_execution_interval` steps, and redistributes expert-to-rank mapping. The balancer adds `num_redundant_experts` for load flexibility.

**Profiling fingerprint**:
- EPLB converging: `rank_skew_ratio` for alltoallv decreases across consecutive profiling captures (need 2+ captures at different times)
- EPLB active but not converging: `rank_skew_ratio` stays high — may indicate `expert_heat_collection_interval` too long for the workload
- EPLB inactive: `rank_skew_ratio` is static — expert routing is fixed

**Action** *(v0.17.0+)*:
- Enable: `DYNAMIC_EPLB=true` env + `additional_config.eplb_config.dynamic_eplb = True`
- For faster convergence: reduce `algorithm_execution_interval` (default 50, try 10-20 for short-lived workloads)
- For better balance: increase `num_redundant_experts` (default 0, try 1-2)
- Cannot be used with MC2 fusion mode 1 (prefill fused dispatch)
- **v0.16.0 and earlier**: EPLB not available.

## Upstream vLLM Configuration

These are vLLM settings (not vLLM-Ascend specific) that directly affect what
appears in the profiling output. They determine **batch formation** and **graph
compilation scope** — the two upstream factors that shape every profile.

### `max_num_batched_tokens` / `max_num_seqs`

- `max_num_batched_tokens`: hard cap on total tokens per scheduler step. This is
  the primary constraint on prefill batch size — the scheduler packs as many
  requests as fit within this budget. If the cap is low relative to sequence
  lengths, prefill steps show consistently small M dimensions.
- `max_num_seqs`: hard cap on concurrent sequences. This is the primary constraint on
  decode batch size — the scheduler runs at most this many sequences per step.
- **Why this matters for profiling**: if every prefill step shows M roughly equal
  to `max_num_batched_tokens / avg_seq_len`, and every decode step shows `max_num_seqs`
  sequences, the scheduler is saturated — the bottleneck shifts from hardware
  (compute/communication) to scheduling policy.
- **Interaction with graph mode**: `cudagraph_capture_sizes` must cover the range
  of batch sizes the scheduler produces. If `max_num_seqs` is 64 and
  `cudagraph_capture_sizes` only covers up to 32, half the decode steps will be
  uncaptured.

### `enable_chunked_prefill`

- When enabled: prefill is split into chunks → multiple prefill steps of varying size
- Profiling impact: prefill steps show varying `main_layer_count` and varying M
  dimensions. This is expected — chunked prefill intentionally trades per-step
  latency for scheduling fairness.
- **Profiling caveat**: if chunked prefill is active AND `graph_mode = partial_capture`,
  the prefill chunk sizes may be the uncaptured batch sizes. Prefill batch sizes
  vary by design with chunking, but if the capture list is small, many prefill
  sizes will fall outside it.

### `block_size` (KV cache block size)

- Default varies (16 or 128 depending on vLLM version). Affects KV cache
  read/write kernel granularity — smaller blocks mean more kernel launches →
  higher dispatch overhead.
- xlite graph mode *(v0.11.0+)* recommends `block_size = 128` for optimal performance.
- **Profiling visibility**: smaller blocks produce more `PagedCacheLoadNdKernel` /
  `ScatterPaKvCache` events per step. Count these events in attention blocks to
  verify if block size is causing excessive kernel launch overhead.

### `VLLM_BATCH_INVARIANT` (v1 engine default)

- When enabled: operators compiled to be invariant to batch size changes.
  Profiling impact: graph-mode kernels show fewer shape-dependent
  recompilations. Disable if debugging shape-specific performance issues.

### `compilation_config.cudagraph_capture_sizes`

- This is the **single most impactful upstream config for profiling**.
  Controls which concrete batch sizes get full graph capture. Any batch size
  NOT in this list runs with higher dispatch overhead.
- **How to detect an incomplete capture list**: `graph_mode = partial_capture` —
  some steps are graph-like (head < 5%), others are eager-like (head > 10%).
- **Most common gap**: speculative decode adds `num_spec_tokens + 1` as a batch
  size. If this value is not in the capture list, every speculative decode step
  is uncaptured.
- **vLLM-Ascend override**: `_compute_decode_cudagraph_batch_sizes()` in
  `compiler_interface.py` computes valid decode batch sizes accounting for
  speculative tokens. Ask the user for both `cudagraph_capture_sizes` and
  `num_speculative_tokens` to verify coverage.
## Workload Estimation (what-if analysis)

When the user provides model architecture info (`config.json`), you can estimate
the impact of configuration changes. These are approximations (±20%) and should be
presented as rough estimates, not guarantees.

### Estimating TP scaling

**Given**: current TP, current allreduce time in attention blocks (from `hccl_op_summary.csv`),
target TP size.

```
allreduce_data_per_step ≈ B × H × 2 bytes  (FP16, each attention layer)
allreduce_time_tp_N ≈ current_time × log₂(N) / log₂(current_TP)
```

Rough heuristic: doubling TP roughly doubles allreduce time per layer (more ranks
= more communication). Halving TP roughly halves communication. The compute benefit
of TP is approximately linear in the number of attention heads per rank.

**What the agent should say**: "If you reduce TP from 8 to 4, attention allreduce
time would roughly halve (from ~{current_ms}ms to ~{estimated}ms per step), but
your attention heads per rank would double. Given your {num_heads} attention heads,
TP=4 gives {heads_per_rank} heads/rank, which is within the recommended ≥8 range."

### Estimating EP scaling

**Given**: current EP, current alltoallv time in MoE blocks, target EP size.

EP alltoallv time scales with the EP group size: larger groups = more ranks to
exchange data with = higher alltoallv latency. The total data volume is approximately
constant (all expert tokens must be routed), but the per-pair communication increases.

Caveat: EP scaling depends heavily on expert routing patterns. Uneven routing
amplifies alltoallv overhead — this is a major source of estimation error.

### Estimating graph mode benefit

**Given**: current step head_wall_ms and wall_ms.

Enabling graph mode (`enforce_eager=False`) typically:
- Reduces prefill step latency by 30-50%
- Reduces decode step latency by 20-35%
- Largest gains at small batch sizes (head_ms is a larger fraction of total)

If `head_ms = X` ms and `wall_ms = Y` ms, the graph-mode estimate is roughly
`wall_ms - (X * 0.7)` ms (about 70% of host dispatch overhead is eliminated by
graph compilation).

**What the agent should say**: "Based on your current head_ms={X}ms and total
step time of {Y}ms, enabling graph mode would reduce step time to approximately
{estimated}ms (about {pct}% improvement). The actual improvement depends on how
many operators can be fused — operators with dynamic shapes or control flow may
still run in eager mode even with enforce_eager=False."

