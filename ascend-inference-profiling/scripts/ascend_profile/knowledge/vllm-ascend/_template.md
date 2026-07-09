# vLLM-Ascend Config Guide Template

<!--
  This is the quality specification for versioned config guides.
  Every v{version}.md file in this directory MUST follow this structure.

  Generation: run `generate_config_guide.py --src <vllm-ascend-path> --output v{version}.md`
  The script extracts defaults from source, but the profiling fingerprints
  and agent notes require expert review after generation.

  Quality checklist for each version file:
  - [ ] Version header has canonical tag/commit + generation date
  - [ ] Every config entry has: key, default, ON fingerprint, OFF fingerprint
  - [ ] Every default value is verified against the actual source code
  - [ ] Every attention backend entry lists the detectable kernel names
  - [ ] Every env var entry lists the env var name AND the additional_config key
  - [ ] Agent notes are concise and actionable (1-2 sentences max)
  - [ ] No line-number references (use module/class names)
  - [ ] No generic "consider enabling" advice without specific conditions
-->

## Version

- **Canonical tag**: {from `git describe --tags` in vllm-ascend repo}
- **Upstream commit**: {from `git rev-parse HEAD`}
- **Generated**: YYYY-MM

## Graph compilation & execution mode

### `enforce_eager`

- **vLLM key**: `model_config.enforce_eager`
- **Default**: {extracted from vllm source or known default}
- **When `True` (eager) → profiling fingerprint**:
  - `step_anatomy.head_wall_ms / step_summary.wall_ms > 10%`
  - Many small kernels with high `wait_us`, gaps between sequential ops
- **When `False` (graph) → profiling fingerprint**:
  - `head_wall_ms / wall_ms < 5%`
  - Large fused kernel blocks, no inter-kernel gaps
- **Interaction**: {list other configs that affect or are affected}
- **Source**: `vllm_ascend/compilation/compiler_interface.py` `AscendCompiler`
- **Agent note**: {1-2 sentences on when and how to ask about this}

### `enable_npugraph_ex`

- **vLLM-Ascend key**: `ascend_compilation_config.enable_npugraph_ex`
- **Default**: {True/False}
- **Source**: `vllm_ascend/ascend_config.py` `AscendCompilationConfig.__init__`

### `cudagraph_capture_sizes`

- **vLLM key**: `compilation_config.cudagraph_capture_sizes`
- **Default**: {vLLM dynamic default}
- **Profiling fingerprint when incomplete**:
  - Mixed pattern: some steps graph-like, others eager-like
  - `config_signatures.graph_mode = "partial_capture"`
- **Source**: `vllm_ascend/compilation/compiler_interface.py` `_compute_decode_cudagraph_batch_sizes()`
- **Agent note**: if partial_capture detected, ask for cudagraph_capture_sizes and num_speculative_tokens

## Compilation fusion passes

### `fuse_qknorm_rope`

- **vLLM-Ascend key**: `ascend_compilation_config.fuse_qknorm_rope`
- **Default**: {True/False}
- **When True → profiling fingerprint**: No separate RoPE kernel in attention block
- **When False → profiling fingerprint**: Explicit `attention.rope` kernel appears separately
- **Source**: `vllm_ascend/ascend_config.py` `AscendCompilationConfig.__init__`

### `fuse_allreduce_rms`

- **vLLM-Ascend key**: `ascend_compilation_config.fuse_allreduce_rms`
- **Default**: {True/False}
- **When True → fingerprint**: Allreduce and rmsnorm appear fused, no gap
- **When False → fingerprint**: Separate allreduce then rmsnorm with visible gap
- **Source**: `vllm_ascend/ascend_config.py` `AscendCompilationConfig.__init__`

### `fuse_norm_quant`

- **vLLM-Ascend key**: `ascend_compilation_config.fuse_norm_quant`
- **Default**: {True/False}
- **Source**: `vllm_ascend/ascend_config.py` `AscendCompilationConfig.__init__`

## Attention backends

### {backend name}

- **Detected when**: {kernel names that indicate this backend}
- **Used for**: {model families}
- **Source**: `vllm_ascend/attention/{file}.py` `{class name}`

## KV cache compression

### `hamming_sparse`

- **vLLM-Ascend key**: `additional_config.hamming_sparse.enabled`
- **Default**: {True/False}
- **When True → fingerprint**: `NpuHammingDistTopK`, `NpuSignBitsPack` kernels present
- **Source**: `vllm_ascend/ascend_config.py` `AscendConfig._check_enable_hamming_sparse()`

## MoE dispatch fusion

### `enable_fused_mc2`

- **vLLM-Ascend key**: `additional_config.enable_fused_mc2` or env `VLLM_ASCEND_ENABLE_FUSED_MC2`
- **Default**: {0/1/2}
- **Modes and fingerprints**:

| Mode | Kernel pattern |
|------|---------------|
| {mode_value} | {kernel names} |

- **Prerequisites**: {quant/EP/MTP constraints}
- **Source**: `vllm_ascend/envs.py`

## Parallelism

### Tensor parallelism (TP)

- **Profiling fingerprint**: allreduce/reducescatter in attention/FFN blocks
- **TP = 1 fingerprint**: No HCCL collectives in compute blocks

### Expert parallelism (EP)

- **Profiling fingerprint**: alltoallv before/after MoE expert blocks

### Pipeline parallelism (PP)

- **Profiling fingerprint**: Some ranks show `has_attention = False`

## Speculative decode

- **vLLM key**: `speculative_config`
- **Profiling fingerprint**: `step_type = "speculative"` in `step_summary.csv`
- **Interaction with graph**: `cudagraph_capture_sizes` must cover `num_spec_tokens + 1`

## Communication optimization

### `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE`

- **Default**: {0/1}
- **When True → fingerprint**: allreduce overlaps with MatMul, no gap

### `VLLM_ASCEND_ENABLE_FLASHCOMM1`

- **Default**: {0/1}

### `VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE`

- **Default**: {0}

## Other configurable features

### `VLLM_ASCEND_ENABLE_MLAPO`

- **Default**: {0/1}
- **Effect**: MLA prefill optimization for DeepSeek W8A8
- **Agent note**: only relevant for MLA-backend models

### `VLLM_ASCEND_ENABLE_NZ`

- **Default**: {1}
- **Modes**: 0 = disabled, 1 = quant only, 2 = BF16/FP16 also

### Continuous batching

- **Profiling fingerprint**: mix of prefill and decode steps in same window
- **Agent note**: check `step_summary.csv` for both `step_type = "prefill"` and `step_type = "decode"`

### `additional_config.multistream_dsv4_dsa_overlap`

- **Default**: {True/False}
- **When True → fingerprint**: attention compute overlaps with HCCL communication

## Agent usage pattern

1. Read `characterizations.json` → `config_signatures` for detected states
2. For each non-trivial `detected` value, find the matching entry in the version-matched config guide
3. Compare detected state with documented default; formulate specific follow-up if they differ
4. Ask only relevant questions — do not dump the entire checklist
5. If user provides their vLLM-Ascend version but it doesn't match this file, warn and use closest match
6. If user's answer reveals intentional config (e.g., `enforce_eager=True` for debugging), acknowledge and move on
