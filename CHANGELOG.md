# Changelog

All notable changes to gHALO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows semantic versioning while it is in active early
development.

## [Unreleased]

### Added

- No unreleased changes yet.

## [0.3.0] - 2026-07-10

### Added

- CPU MPI reference backend preserving the HALO benchmark exchange order,
  periodic Cartesian topology, steady-state timing, and maximum-rank average
  wall-clock reporting.
- AMD HIP device-memory backend exposed as `mpi-hip`.
- GPU-aware MPI path through Cray MPICH with HIP-allocated device buffers passed
  directly to MPI.
- Correctness validation for rank- and direction-specific halo patterns before
  timed measurement.
- Optional MPI-HIP phase timing for device-copy, MPI, and synchronization
  phases.
- Portable system/backend build directories under `builds/<system>/<backend>/`.
- Frontier system profile for MPI and MPI-HIP builds and runs.
- Borg system profile with Frontier build-tree reuse for the Frontier hot-spare
  cabinet.
- Interactive build and execution scripts for repeatable local and HPC runs.
- Slurm batch submission wrapper that preserves the same `scripts/run.sh`
  execution path and result metadata.
- Human-readable console output plus JSON and CSV result capture.
- Result metadata and provenance capture, including commands, environment, Git
  state, system resolution, Slurm metadata, binary metadata, and exit status.
- Portable result analysis tooling for summaries, comparisons, repeated-run
  aggregation, fixed-message-size scaling, and optional plotting.
- GitHub CI for shell validation, CPU-only CMake builds, and Linux OpenMPI
  builds.

### Validated

- Successful MPI-HIP validation on Frontier at 64 nodes and 512 GPU ranks.

### Known Limitations

- MPI-HIP currently uses correctness-first synchronization at GPU-aware
  MPI/HIP handoff points.
- Phase timing perturbs very small messages and should be treated as a
  diagnostic mode rather than the authoritative benchmark mode.
- Independently reduced phase maxima may sum to more than the independently
  reduced total exchange maximum.
- RCCL is not yet implemented.
- GitHub-hosted CI does not test HIP, Cray MPICH, Slurm, GPU execution, or
  performance.
- Large-system performance results remain system- and placement-dependent.

## [0.2.0] - Historical Development Tags

### Added

- GPU-aware MPI bring-up and device-resident exchange path during development.

## [0.1.0] - Initial Development

### Added

- Initial project documentation and repository scaffold.
- HALO source-tree analysis.
- Version 0 MPI backend architecture and CMake feature gates.
- Open-source repository infrastructure, including contribution, security,
  conduct, issue, pull request, CI, and formatting files.
