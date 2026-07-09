# gHALO

[![Build Status](https://github.com/gHALO/gHALO/actions/workflows/ci.yml/badge.svg)](https://github.com/gHALO/gHALO/actions/workflows/ci.yml)

gHALO is a GPU-native halo exchange benchmark and diagnostic suite for modern
high-performance computing systems.

The project is inspired by Alan Wallcraft's HALO benchmark, developed for the
Naval Research Laboratory and described in the Fall 1999 NAVO MSRC Navigator.
gHALO is not a source translation. It preserves the original benchmark
philosophy while modernizing the implementation for GPU-based systems,
GPU-aware communication stacks, structured output, and cluster-scale
diagnostics.

## Project Overview

gHALO evaluates communication latency, bandwidth, scalability, and topology
health for halo exchange patterns on modern HPC platforms. The long-term goal
is to become a diagnostic tool for communication health, not only a bandwidth
benchmark.

Planned capabilities include:

- CPU reference implementation
- GPU-aware MPI
- HIP and AMD GPU support
- RCCL backend
- UCX backend
- JSON and CSV output
- diagnostic reporting
- communication heat maps
- cluster health analysis

## History and Motivation

Many scientific applications decompose a domain across ranks and repeatedly
exchange halo regions with neighboring ranks. Generic bandwidth tests do not
always reveal the latency, synchronization, topology, and straggler behavior
that those applications pay in practice.

Alan Wallcraft's HALO benchmark captured that practical concern for an earlier
generation of HPC systems. gHALO asks the same kind of question for modern
systems: how healthy, scalable, and predictable is halo exchange communication
on the machine where real GPU-accelerated applications will run?

See [Historical Context](docs/HISTORY.md) and
[HALO Analysis](docs/HALO_ANALYSIS.md).

## Current Status

gHALO is in early `0.1.x` development. The repository contains the project
infrastructure, documentation, CMake feature gates, and an MPI-oriented version
0 path. HIP, RCCL, UCX, and GPU-aware MPI are planned but are not required for
the current CPU/MPI development workflow.

The development model assumes:

- development happens on macOS or another workstation;
- HIP, ROCm, MPI, GPU compilation, and GPU execution happen on remote Linux HPC
  systems when those features are enabled;
- local macOS builds must not require ROCm, HIP, MPI, RCCL, UCX, or GPU
  hardware unless the developer explicitly enables those features;
- platform-specific code remains isolated behind CMake feature detection and
  backend boundaries.

## Roadmap Summary

- `0.1.x`: CPU reference implementation and MPI baseline.
- `0.2.x`: GPU-aware MPI and device-resident exchange paths.
- `0.3.x`: HIP foundations, kernels, and GPU buffer management.
- `0.4.x`: RCCL backend experiments.
- `0.5.x`: UCX backend experiments and transport diagnostics.
- `0.6.x`: diagnostic reporting, heat maps, and topology health analysis.
- `1.0.0`: first production release with stable CLI, output schemas,
  compatibility mode, and validation expectations.

See [Roadmap](docs/ROADMAP.md) and [Versioning](docs/VERSIONING.md).

## Repository Layout

```text
.
├── .github/              issue templates, PR template, and CI
├── CMakeLists.txt
├── docs/                 project, architecture, roadmap, and analysis docs
├── examples/             runnable examples and scheduler templates
├── include/ghalo/        public C++ interfaces
├── scripts/              helper scripts
├── src/                  implementation sources
│   ├── app/              command-line entry points
│   ├── backends/         communication backends
│   └── core/             portable benchmark orchestration
└── tests/                tests and test CMake configuration
```

## Build System

gHALO uses CMake and C++20. Optional dependencies are disabled by default so the
project remains usable from development workstations without GPU tooling.

Portable configuration:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

MPI build:

```sh
cmake -S . -B build-mpi -DGHALO_ENABLE_MPI=ON
cmake --build build-mpi
ctest --test-dir build-mpi
```

Example MPI run:

```sh
srun -n 16 ./build-mpi/ghalo --csv ghalo.csv --json ghalo.json
```

Future feature gates:

```sh
cmake -S . -B build -DGHALO_ENABLE_HIP=ON
cmake -S . -B build -DGHALO_ENABLE_RCCL=ON
cmake -S . -B build -DGHALO_ENABLE_UCX=ON
```

Those options are intended for remote Linux HPC environments where the required
toolchains and libraries are available.

## Development Philosophy

gHALO values:

- preservation of HALO benchmark semantics over source compatibility;
- steady-state timing rather than initialization timing;
- maximum-rank timing as the primary application-visible metric;
- backend isolation for MPI, GPU-aware MPI, HIP, RCCL, UCX, and future systems;
- portable C++20 core logic;
- structured output for reproducibility and diagnostics;
- documentation of architectural decisions before large implementation changes.

## Documentation

- [Project Vision](docs/PROJECT.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Design Principles](docs/DESIGN_PRINCIPLES.md)
- [HALO Analysis](docs/HALO_ANALYSIS.md)
- [Roadmap](docs/ROADMAP.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Versioning](docs/VERSIONING.md)
- [Historical Context](docs/HISTORY.md)
- [Version 0 Design](docs/VERSION_0_DESIGN.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md),
and [SECURITY.md](SECURITY.md).

## License

gHALO is released under the MIT License. See [LICENSE](LICENSE).
