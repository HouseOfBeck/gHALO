# Project Vision

gHALO is a GPU-native halo exchange benchmark and diagnostic suite for modern
HPC systems.

The project is inspired by the spirit of Alan Wallcraft's HALO benchmark:
measure the communication behavior that matters to stencil-like scientific
applications, expose scaling limits, and provide results that help system
operators and application developers reason about real communication behavior.

gHALO is not a source translation of the original HALO benchmark. It is a new
project that carries forward the benchmark philosophy into an environment where
GPU memory, GPU-aware MPI, accelerator interconnects, and communication library
selection are central performance concerns.

## Goals

- Measure halo exchange latency, bandwidth, and scaling behavior.
- Support GPU-resident data paths as first-class benchmark modes.
- Make backend selection explicit and inspectable.
- Produce machine-readable results suitable for regression tracking.
- Grow into a diagnostic suite that can identify communication topology
  problems and cluster health issues.

## Non-Goals for the Initial Phase

- No benchmark implementation code.
- No translated legacy HALO source.
- No assumption that development machines have ROCm, HIP, MPI, RCCL, UCX, or
  GPU hardware installed.
- No vendor-specific code paths outside isolated backend boundaries.

## Intended Users

- HPC application developers evaluating halo exchange behavior.
- System administrators validating GPU cluster communication health.
- Performance engineers comparing communication stacks and topology effects.
- Researchers studying distributed GPU communication patterns.

## Design Values

- GPU-native execution model.
- Portable C++20 host-side foundations.
- Explicit feature detection through CMake.
- Clean separation between core benchmark orchestration and backend-specific
  implementations.
- Reproducible, structured output for long-term analysis.
