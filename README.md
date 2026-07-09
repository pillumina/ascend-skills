# Ascend Skills

Ascend NPU profiling analysis skills for AI workloads.

## Skills

| Skill | Purpose | Status |
|-------|---------|--------|
| [`ascend-inference-profiling`](./ascend-inference-profiling/) | End-to-end Ascend NPU inference profiling analysis for vLLM-Ascend workloads | Active |
| [`ascend-training-profiling`](./ascend-training-profiling/) | Ascend NPU training profiling analysis for distributed training workloads | Planned |

## Installation

```bash
# Install both skills
npx skills add pillumina/ascend-skills

# Install inference only
npx skills add pillumina/ascend-skills --skill ascend-inference-profiling

# Install training only
npx skills add pillumina/ascend-skills --skill ascend-training-profiling
```

## Requirements

- Python 3.8+ with PyYAML
- Remote mode: SSH access to Ascend NPU host
- Local mode: profiling data accessible on local filesystem

## Related Skills

| Skill | Relationship |
|-------|-------------|
| `ascend-profiling-collection` | Upstream — collects torch profiler data on Ascend hosts |
| `ascend-memory-profiling` | Complementary — HBM memory attribution analysis |
| `vllm-ascend-serving` | Complementary — vLLM-Ascend service lifecycle management |
| `vllm-ascend-benchmark` | Complementary — benchmark execution and result aggregation |
