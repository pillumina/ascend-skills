# Ascend Skills

Ascend NPU skills for vLLM-Ascend workloads — profiling, serving, benchmarking, and infrastructure management.

## Skills

### Analysis

| Skill | Purpose | Status |
|-------|---------|--------|
| [`ascend-profiling-collection`](./ascend-profiling-collection/) | Collect one Ascend torch-profiler case end-to-end on a remote NPU container | Active |
| [`ascend-inference-profiling`](./ascend-inference-profiling/) | End-to-end profiling analysis for vLLM-Ascend inference | Active |
| [`ascend-training-profiling`](./ascend-training-profiling/) | Profiling analysis for distributed training workloads | Planned |

### Infrastructure

| Skill | Purpose | Status |
|-------|---------|--------|
| [`machine-management`](./machine-management/) | Add, verify, repair, or remove managed remote NPU hosts | Active |
| [`session-management`](./session-management/) | Create and manage isolated VAWS agent sessions for parallel execution | Active |

### Runtime

| Skill | Purpose | Status |
|-------|---------|--------|
| [`vllm-ascend-serving`](./vllm-ascend-serving/) | Start, check, or stop a vLLM Ascend online service on a remote container | Active |
| [`vllm-ascend-benchmark`](./vllm-ascend-benchmark/) | Run vLLM online-serving benchmarks against a managed remote container | Active |

### Dependency Chain

```
machine-management → session-management → ascend-profiling-collection → ascend-inference-profiling
                                         → vllm-ascend-serving → vllm-ascend-benchmark
```

Shared library at `lib/` — imported by all infra/runtime/analysis skills.

## Installation

```bash
# Install everything
npx skills add pillumina/ascend-skills

# Specific skills
npx skills add pillumina/ascend-skills --skill ascend-inference-profiling
npx skills add pillumina/ascend-skills --skill machine-management
```

## External Dependencies

| Skill | Status | Location |
|-------|--------|----------|
| `ascend-memory-profiling` | Complementary — HBM memory attribution analysis | `maoxx241/vllm-ascend-workspace` |
| `remote-code-parity` | Used by vllm-ascend-serving for code sync gate | `maoxx241/vllm-ascend-workspace` |
