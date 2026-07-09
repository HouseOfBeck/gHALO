# gHALO

gHALO is a GPU-native halo exchange benchmark and diagnostic suite for modern
high-performance computing systems.

The project is inspired by Alan Wallcraft's HALO benchmark, developed for the
Naval Research Laboratory and described in the Fall 1999 NAVO MSRC Navigator.
gHALO does not aim to translate the original code. Its purpose is to preserve
the benchmark philosophy while building a modern implementation for GPU-based
systems, GPU-aware communication stacks, and cluster-scale diagnostics.

## Mission

gHALO evaluates communication latency, bandwidth, scalability, and topology
health for halo exchange patterns on modern HPC platforms. Over time, the
project should become a diagnostic tool for understanding communication health,
not simply another bandwidth benchmark.

Planned capabilities include:

- HIP support
- AMD GPU support
- GPU-aware MPI
- RCCL backend
- UCX backend
- JSON and CSV output
- Diagnostic reporting
- Communication heat maps
- Cluster health analysis

## Current Status

This repository is in the initial project-structure phase. It intentionally does
not contain benchmark implementation code yet.

The development model assumes:

- Development happens on macOS.
- HIP, ROCm, MPI, GPU compilation, and GPU execution happen on a remote Linux
  cluster.
- Local macOS builds must not require ROCm, HIP, MPI, RCCL, or UCX.
- Platform-specific code will be isolated behind CMake feature detection and
  backend boundaries.

## Repository Layout

```text
.
├── CMakeLists.txt
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT.md
│   ├── HISTORY.md
│   ├── PROJECT.md
│   └── ROADMAP.md
├── examples/
├── scripts/
└── tests/
```

Future source directories will be added when the first implementation milestone
begins.

## Documentation

- [Project Vision](docs/PROJECT.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Historical Context](docs/HISTORY.md)

## Build System

gHALO uses CMake and C++20. The root build currently validates project
configuration and exposes feature options without requiring GPU tooling on local
macOS development machines.

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

The version 0 executable is MPI-based and is built when MPI is enabled:

```sh
cmake -S . -B build -DGHALO_ENABLE_MPI=ON
cmake --build build
srun -n 16 ./build/ghalo
```

Optional future features remain disabled by default so configuration stays
portable on macOS workstations:

```sh
cmake -S . -B build -DGHALO_ENABLE_HIP=ON
cmake -S . -B build -DGHALO_ENABLE_RCCL=ON
cmake -S . -B build -DGHALO_ENABLE_UCX=ON
```

Those options are intended for remote Linux HPC environments where the required
toolchains and libraries are available.

## Version 0

Version 0 implements the original HALO steady-state methodology using an MPI
backend:

- periodic 2-D Cartesian communicator
- north, south, east, and west neighbors
- halo lengths from 2 through 1024 words
- local communication buffers
- unmeasured warmup exchange per halo size
- five-exchange calibration
- approximately three seconds of steady-state timing per halo size
- maximum average wall-clock time across all ranks
- console, CSV, and JSON output

Example:

```sh
srun -n 16 ./build/ghalo --csv ghalo.csv --json ghalo.json
```

See [Version 0 Design](docs/VERSION_0_DESIGN.md) for implementation notes.

## License

gHALO is released under the MIT License. See [LICENSE](LICENSE).
