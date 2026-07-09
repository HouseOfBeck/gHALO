# Architecture

gHALO will be organized around a small portable core and isolated optional
backends. The repository is intentionally scaffolded before benchmark code is
introduced so that HIP, MPI, RCCL, and UCX can be added without entangling
platform-specific assumptions with the core project.

## Architectural Principles

- Keep host-side coordination portable C++20.
- Treat GPU support as optional at configure time, required only for GPU builds.
- Keep backend-specific code behind narrow interfaces.
- Prefer feature detection over hard-coded platform assumptions.
- Preserve diagnostic metadata alongside benchmark results.

## Planned Layers

```text
Command line / configuration
          |
Benchmark orchestration
          |
Exchange pattern model
          |
Memory and buffer management
          |
Communication backend interface
          |
MPI / GPU-aware MPI / RCCL / UCX
          |
HIP kernels and device-resident buffers
```

## Core Layer

The future core layer should define concepts shared across all benchmark modes:

- domain decomposition metadata
- halo region descriptions
- exchange pattern definitions
- timing and measurement policies
- result records
- diagnostics and topology annotations

This layer should remain buildable on macOS without ROCm, HIP, MPI, RCCL, or
UCX.

## Backend Layer

Backends should be added as optional components. Each backend should own its
library-specific setup, error handling, synchronization semantics, and data
movement behavior.

Planned backend families:

- MPI host-buffer backend
- GPU-aware MPI backend
- RCCL backend
- UCX backend

Backends should report enough metadata for results to identify exactly which
communication path was used.

## GPU Layer

HIP support should be introduced only inside GPU-specific targets and source
trees. HIP language enablement must be gated by CMake options so macOS
development machines can configure the project without ROCm.

Planned GPU responsibilities:

- device buffer allocation
- halo packing and unpacking kernels
- stream and event timing support
- synchronization policies
- validation utilities for device-resident data

## Output and Diagnostics

gHALO should produce structured output suitable for both humans and automation.
The long-term output model should include:

- JSON records for detailed diagnostics
- CSV summaries for quick analysis
- topology and rank-pair metadata
- per-exchange latency and bandwidth measurements
- warnings for asymmetric or degraded communication paths

Diagnostic reporting should eventually support communication heat maps and
cluster health summaries.
