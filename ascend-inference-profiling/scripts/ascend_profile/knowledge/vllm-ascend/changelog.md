# vLLM-Ascend Changelog

Profiling-relevant changes per version. One read — all version history.
The agent loads this file ONCE, then scans for entries between the
nearest known config guide version and the user's version.

Each entry is a ``## v{version}`` section, sorted newest first.
Per-entry headings use ``###`` to stay under the version heading.

## v0.22.1rc1

# v0.22.1rc1 Changelog (Profiling-Relevant)

### DSA backend refactored

- `dsa_v1.py`: 244 lines removed — significant refactoring of DSA backend
- `sfa_v1.py`: 171 lines added — SFA backend enhancements
- **Profiling impact**: DSA kernel names and patterns may have changed. If upgrading from v0.21, check whether previously-detected `dsa` attention backend still resolves correctly.

### ASC compile interface changes

- `compiler_interface.py`: +24 lines — minor compilation path updates
- **Profiling impact**: None expected. Compilation behavior unchanged for standard graph capture.

### Context parallel DSA added

- `dsa_cp.py`: 269 lines changed — DSA context parallel support
- **Profiling impact**: When CP is active with DSA backend, attention computation patterns differ from non-CP DSA

### Summary

v0.22.1rc1 is primarily a refinement release. No new attention backends, no config defaults changed. The main profiling impact is DSA backend refactoring — verify that `config_signatures.attention_backend` still correctly detects `dsa` after upgrade.

## v0.21.0

# v0.21.0 Changelog (Profiling-Relevant)

### Removed: `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` env var
- **Impact**: Context parallelism configuration moved to `parallel_config.prefill_context_parallel_size`. The env var no longer exists.
- **Profiling impact**: If user was previously setting this env var, CP may be disabled after upgrade. Check `prefill_context_parallel_size` instead.

### Removed: `VLLM_ASCEND_APPLY_DSV4_PATCH` env var
- **Impact**: Was added in v0.20.2, removed in v0.21.0. Short-lived feature.

### Changed: `enable_sparse_c8` now requires sparse model detection
- **Before**: `enable_sparse_c8` applied unconditionally if enabled
- **After**: `enable_sparse_c8` only activates when `use_sparse` is True (model has `index_topk` in HF config)
- **Profiling impact**: Dense models with `enable_sparse_c8` enabled will no longer show sparse C8 kernels after upgrade to v0.21.0.

### Added: `c8_enable_reshape_optim`
- **Config key**: `additional_config.c8_enable_reshape_optim`
- **Default**: `False`
- **Requires**: `enable_sparse_c8 = True` + sparse model
- **Profiling impact**: When active, optimizes reshape operations in sparse C8 quantized layers.

### Added: `enable_kv_nz` validation
- **Config key**: `additional_config.enable_kv_nz`
- **Default**: `False`
- **Profiling impact**: Now requires explicit model_config validation (must be DeepSeek MLA, non-sparse, PD decode node only). Previously could be set but silently do nothing.

### Added: RejectionSamplerConfig (Block Verify + Entropy Verify)
- **Config keys**: `additional_config.rejection_sampler_config.enable_block_verify`, `.enable_entropy_verify`, `.posterior_threshold`, `.posterior_alpha`
- **Defaults**: `False`, `False`, `0.95`, `0.4`
- **Profiling impact**: When enabled, speculative decode acceptance may improve (fewer rejected tokens). Visible in profiling as higher `step_type = "speculative"` step count or higher acceptance rate (acceptance rate NOT visible in profiling directly — check vLLM metrics).

### Removed: `enable_context_parallel` from AscendCompilationConfig
- **Impact**: The `enable_context_parallel` config key in `ascend_compilation_config` no longer exists. CP configuration moved entirely to `parallel_config`.

## v0.20.2

# v0.20.2 Changelog (Profiling-Relevant)

### New attention backends (major profiling impact)

#### `dsa_v1.py` — DSA (DeepSeek Sparse Attention) Backend
- **Profiling impact**: When active, `QuantLightningIndexer` and `KVQuantSparseAttnSharedKV` kernels appear. The backend auto-selects for DeepSeek V3.2 models with sparse attention configuration.
- **Detection**: `config_signatures.attention_backend` reports `dsa` when indexer + sparse kernels are present without compressor kernels.
- **Note**: This is a separate backend class from SFA. Both produce sparse-attention kernel signatures, but DSA lacks compressor kernels.

#### `fa3_v1.py` — FlashAttention3 Backend
- **Profiling impact**: FlashAttention3 kernel variants instead of standard FIA. Kernel names may differ from `FusedInferAttentionScoreV*`.
- **Detection**: Check for `fa3_*` prefixed kernels in attention blocks. Currently not in `attention_families.yaml` — may show as `unknown`.
- **Calibration needed**: Add `fa3_*` kernel patterns to `attention_families.yaml`.

#### `abstract.py` — Attention Backend Base Class
- **Profiling impact**: None. Internal refactoring. No new kernels.

### New env var: `VLLM_ASCEND_APPLY_DSV4_PATCH`
- **Default**: `0` (disabled)
- **Profiling impact**: When enabled, applies DeepSeek V4-specific kernel patches. Affects kernel selection in SFA/DSA backends.
- **Note**: Removed in v0.21.0.

### Deprecation: env vars → additional_config
- `VLLM_ASCEND_BALANCE_SCHEDULING` and other env vars deprecated. Use `additional_config` equivalents.
- **Profiling impact**: None — config values unchanged, only the mechanism differs.

## v0.18.0

# v0.18.0 Changelog (Profiling-Relevant)

### fuse_allreduce_rms default changed to True

- **Config key**: `ascend_compilation_config.fuse_allreduce_rms`
- **Before v0.18.0**: Default `False`
- **v0.18.0+**: Default `True`
- **Profiling impact**: When enabled, allreduce and subsequent rmsnorm appear fused — allreduce wall time is shorter because the rmsnorm is absorbed. The allreduce and rmsnorm no longer appear as separate ops with a gap between them.
- **Version-aware detection**: If the user is on v0.18.0+ and profiling shows separate allreduce + rmsnorm, the fusion pass may be disabled despite the new default. Check their `ascend_compilation_config`.
- **Performance**: Fusing allreduce with rmsnorm reduces visible communication time by 5-15% (the rmsnorm portion is hidden). Only works in graph mode with npugraph_ex.

### enable_static_kernel added to AscendCompilationConfig

- **Config key**: `ascend_compilation_config.enable_static_kernel`
- **Default**: `False`
- **Requires**: `enable_npugraph_ex=True`
- **Profiling impact**: When enabled, even more aggressive kernel fusion. Head/wall ratio approaches zero. Not directly distinguishable from standard npugraph_ex in the profile.
- **Performance**: Higher first-run latency (static compilation), lower steady-state latency.

## v0.17.0rc1



### Finegrained TP added

#### `finegrained_tp_config`

- **Config keys**: `oproj_tensor_parallel_size`, `lmhead_tensor_parallel_size`, `embedding_tensor_parallel_size`, `mlp_tensor_parallel_size`, `olora_tensor_parallel_size`
- **Profiling impact**: Different modules use different TP sizes. Attention oproj may use a smaller TP group than other layers → varied allreduce group sizes in HCCL traces.
- **Constraint**: `oproj_tp` and `olora_tp` only work in graph mode + PD decode node.
- **Detection**: Multiple allreduce collective sizes in the same profile. Standard TP shows uniform allreduce patterns.

### Expert Load Balancing added

#### `eplb_config`

- **Config keys**: `eplb_config.dynamic_eplb` (default False), `expert_heat_collection_interval` (default 600), `algorithm_execution_interval` (default 50), `num_redundant_experts` (default 0)
- **Requires**: env `DYNAMIC_EPLB=true`
- **Profiling impact**: When enabled, alltoallv rank_skew_ratio should decrease over time as EPLB converges. Static EP typically shows persistent skew.
- **Detection**: Check `hccl_class_summary.csv` rank_skew_ratio for alltoallv over multiple profiling windows. Decreasing skew → EPLB converging.

### Weight Prefetch added

#### `weight_prefetch_config`

- **Config key**: `weight_prefetch_config.enabled` (default False), `prefetch_ratio` (per-module ratios)
- **Profiling impact**: When enabled, weight loading shifts before step boundary → reduced `head_ms`. Not directly visible in kernel patterns — manifests as smaller head/wall ratio.
- **Note**: Hard to confirm from profiling alone. Ask user.

### Shared Expert DP added

#### `enable_shared_expert_dp`

- **Config key**: `enable_shared_expert_dp` (default False)
- **Requires**: EP enabled + TP >= 2
- **Profiling impact**: Shared expert layers can overlap across DP replicas. Reduced shared expert layer wall time.

### Async Exponential added

#### `enable_async_exponential`

- **Config key**: `enable_async_exponential` (default False)
- **Constraint**: Disabled when `VLLM_BATCH_INVARIANT` is active
- **Profiling impact**: Changes sampler behavior — sampling step may show different kernel patterns.

### AscendFusionConfig added

#### `ascend_fusion_config`

- **Config key**: `fusion_ops_gmmswigluquant` (default True)
- **Profiling impact**: Controls whether GMM + SwiGLU + Quant operations are fused. When enabled, MoE expert computation shows fused kernels instead of separate GMM/SwiGLU/Quant.

## v0.13.0

# v0.13.0 Changelog (Profiling-Relevant)

### Context parallel attention added (PCP + DCP)

#### `context_parallel/` module (attention_cp.py, common_cp.py, mla_cp.py)

- **Profiling impact**: Two independent CP features: **PCP** (`prefill_context_parallel_size`) for prefill and **DCP** (`decode_context_parallel_size`) for decode. Both create CP subgroups within the TP group, making attention collectives use groups smaller than full TP size.
- **Config keys**: `parallel_config.prefill_context_parallel_size`, `parallel_config.decode_context_parallel_size`
- **Detection**: Prefill allreduce/allgather groups smaller than TP → PCP active. Decode allreduce/allgather groups smaller than TP → DCP active.
- **Note**: PCP only affects prefill steps. DCP only affects decode steps. Both consume additional HCCL streams.

### AscendCompilationConfig stabilized (from v0.12.0)

- All compilation keys from v0.12.0 are now stable defaults in v0.13.0

### Attention refactoring (non-breaking, profiling-neutral)

- `attention_v1.py`, `mla_v1.py`, `sfa_v1.py` received significant refactoring
- **Profiling impact**: Kernel names unchanged. No new or removed kernels.

## v0.12.0

# v0.12.0 Changelog (Profiling-Relevant)

### AscendCompilationConfig added

- **Config keys added**: `enable_npugraph_ex` (default `True`), `fuse_norm_quant` (default `True`), `fuse_qknorm_rope` (default `True`), `fuse_allreduce_rms` (default `False`)
- **Before v0.12.0**: Compilation configuration was hardcoded in `compiler_interface.py`, not user-configurable. v0.11.0 profiling may show different fusion patterns that are not toggleable.
- **Profiling impact**: Starting from v0.12.0, fusion passes can be toggled, and their profiling fingerprints (presence/absence of separate RoPE kernels, separate allreduce+rmsnorm ops) become actionable — the user can change the config to affect what appears in the profile.
- **Version boundary**: This is the single largest profiling-relevant change in vLLM-Ascend history. All config guide entries for v0.11.0 and earlier are fundamentally different from v0.12.0+.

### MC2 Fusion added

- **Config key**: `VLLM_ASCEND_ENABLE_FUSED_MC2` (env, default `0`)
- **Profiling impact**: When mode 1 or 2 is active, `DispatchFFNCombine` or `DispatchGmmCombineDecode` kernels replace separate `MoeDistributeDispatchV2` + `MoeDistributeCombineV2`. Without this feature, all MoE dispatch is unfused.
- **Prerequisites**: W8A8 quantization, EP ≤ 32 (mode 1), MTP must also be W8A8 (mode 2)

### FlashComm2 added

- **Config key**: `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE` (env, default `0`)
- **Profiling impact**: When enabled, uses a different communication pattern for output projection allreduce. Allreduce wall time reduced.

## v0.11.0

# v0.11.0 Changelog (Profiling-Relevant)

### Baseline version

v0.11.0 is the first stable release with profiling guides. No AscendCompilationConfig
(added in v0.12.0), no MC2 fusion, no FlashComm2. Configuration defaults are
hardcoded in compiler_interface.py — users cannot toggle fusion passes.

### Profiling impact

- All operators run through the hardcoded compilation path
- No `AscendCompilationConfig` means no `enable_npugraph_ex`, `fuse_qknorm_rope`, etc.
- Attention backends: `attention_v1.py` (FIA), `mla_v1.py` (MLA), `sfa_v1.py` (SFA)
- No `dsa_v1.py` or `fa3_v1.py` (these arrive in v0.20.2)
