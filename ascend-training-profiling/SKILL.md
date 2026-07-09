---
name: ascend-training-profiling
description: Analyze Ascend NPU training profiling data (msprof / torch_npu profiler output) for distributed training workloads. Covers computation-communication overlap, HCCL collective efficiency, gradient synchronization bottlenecks, data loading stalls, and MFU estimation. Supports multi-rank alignment, slow-rank detection, and training-specific diagnosis (DP/TP/PP/EP/CP parallelism analysis). Use for requests like "分析训练 profiling", "训练 MFU 计算", "通信瓶颈分析", "梯度同步性能", "慢卡检测". Do not use for inference profiling (use ascend-inference-profiling), HBM memory attribution (use ascend-memory-profiling), or profiling data collection.
---

# Ascend Training Profiling Analysis

> **Status**: Work in progress. This skill is under development.

Analyze Ascend NPU training profiling output for distributed training workloads.

## Overview

Targets:
- **Frameworks**: PyTorch (torch_npu), MindSpore
- **Workloads**: LLM pre-training / fine-tuning (Megatron-style, DeepSpeed-style)
- **Hardware**: Ascend 910B2/B3 (A2), Ascend 910C (A3)

Key analysis areas:
- Computation-communication overlap (HCCL collective hiding)
- Gradient synchronization bottlenecks (allreduce/reducescatter efficiency)
- Data loading stalls (host-to-device transfer gaps)
- MFU (Model FLOPs Utilization) estimation
- DP/TP/PP/EP/CP parallelism efficiency
- Slow-rank detection (msprof-analyze integration)

## Status

This skill is a placeholder. Implementation is planned to follow the architecture patterns established by `ascend-inference-profiling`:

- Pipeline-based analysis stages (normalize → segment → classify → summarize → diagnostics → report)
- Cross-rank alignment with msprof-analyze slow_rank integration
- Knowledge base with versioned Ascend-specific training configurations
- Evidence-chain traceability from report to raw profiling data

Feature contributions and design discussions are welcome.
