# Roadmap

This roadmap is intentionally staged. gHALO should grow from a portable
CPU/MPI reference path into a GPU-native benchmark, then into a diagnostic
suite.

## Phase 0: Project Scaffold

- Create repository structure.
- Define project mission and scope.
- Add CMake feature options for future HIP, MPI, RCCL, and UCX support.
- Document the development model for macOS plus remote Linux clusters.
- Add open-source project hygiene: contribution guide, conduct policy, security
  policy, changelog, issue templates, PR template, CI, and formatting config.

## Phase 1: Portable Core

- Define benchmark configuration structures.
- Define halo geometry and exchange pattern metadata.
- Define timing and result record types.
- Add unit tests for pure C++20 core logic.
- Add JSON and CSV schema drafts.

## Phase 2: MPI Baseline

- Add optional MPI discovery through CMake.
- Implement a CPU-buffer MPI halo exchange baseline.
- Add correctness validation for exchanged halo regions.
- Add latency and bandwidth measurement modes.
- Support structured output for baseline runs.

## Phase 3: HIP Foundations

- Add HIP targets gated by `GHALO_ENABLE_HIP`.
- Introduce GPU buffer management.
- Add HIP packing and unpacking kernels.
- Add device-side validation utilities.
- Keep non-HIP builds fully functional.

## Phase 4: GPU-Aware MPI

- Add GPU-resident exchange paths through GPU-aware MPI.
- Compare host-staged and device-resident communication modes.
- Record MPI and GPU transport metadata in benchmark output.
- Add synchronization and timing policy controls.

## Phase 5: RCCL and UCX Backends

- Add RCCL backend support for supported exchange patterns.
- Add UCX backend support for lower-level communication diagnostics.
- Normalize backend result reporting.
- Compare backend performance under shared benchmark scenarios.

## Phase 6: Diagnostics

- Add communication heat map generation.
- Detect asymmetric rank-pair performance.
- Identify topology-related bottlenecks.
- Summarize cluster health from repeated measurements.
- Support regression tracking across runs.

## Phase 7: Operational Tooling

- Add scripts for remote build and run workflows.
- Add example batch scripts for common schedulers.
- Add result aggregation tools.
- Add documentation for CI, release, and contribution workflows.

See [Versioning](VERSIONING.md) for the proposed release milestones that map
these phases onto `0.x` releases.
