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

gHALO v0.3.0 is an early research release with two implemented benchmark
backends and one experimental backend scaffold:

- `mpi`: CPU/host-memory MPI reference backend.
- `mpi-hip`: AMD HIP device-memory backend using GPU-aware MPI.
- `rccl`: experimental Stage B backend with RCCL communicator initialization
  and north/south correctness validation only. Full halo exchange timing is not
  implemented yet.

The `mpi-hip` backend has been validated on Frontier with Cray MPICH, AMD GPUs,
correctness validation, JSON/CSV output, metadata capture, phase timing, and
post-run analysis tooling. RCCL east/west communication, UCX, heat maps, and
automated cluster-health diagnostics remain planned work.

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
- `0.2.x`: GPU-aware MPI bring-up and device-resident exchange path.
- `0.3.x`: release-ready MPI-HIP workflow, phase timing, system scripts, batch
  submission, and result analysis tools.
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

## Build And Run

gHALO uses CMake and C++20. Optional dependencies are disabled by default so the
project remains usable from development workstations without GPU tooling.

### CPU-Only Configuration

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

### CPU MPI Backend

```sh
cmake -S . -B build-mpi -DGHALO_ENABLE_MPI=ON
cmake --build build-mpi
ctest --test-dir build-mpi
```

Example MPI run:

```sh
srun -n 16 ./build-mpi/ghalo --csv ghalo.csv --json ghalo.json
```

Short MPI smoke test:

```sh
cmake -S . -B build -DGHALO_ENABLE_MPI=ON
cmake --build build
mpirun -np 4 ./build/ghalo --target-seconds 0.1
```

### AMD HIP GPU-Aware MPI Backend

The `mpi-hip` backend requires MPI, HIP, and GPU-aware MPI support on the
remote Linux HPC system:

```sh
cmake -S . -B build-mpi-hip \
  -DGHALO_ENABLE_MPI=ON \
  -DGHALO_ENABLE_HIP=ON \
  -DGHALO_ENABLE_MPI_HIP=ON
cmake --build build-mpi-hip
```

On Frontier, GPU-aware MPI runs require:

```sh
export MPICH_GPU_SUPPORT_ENABLED=1
```

Additional future feature gates remain available but are not implemented yet:

```sh
cmake -S . -B build -DGHALO_ENABLE_MPI=ON -DGHALO_ENABLE_HIP=ON -DGHALO_ENABLE_RCCL=ON
cmake -S . -B build -DGHALO_ENABLE_UCX=ON
```

Those options are intended for remote Linux HPC environments where the required
toolchains and libraries are available.

The RCCL path is experimental. It currently requires the explicit
`--rccl-stage-b` flag and validates only north/south halo communication:

```sh
GHALO_SYSTEM_NAME=borg scripts/build.sh --backend rccl --clean
GHALO_SYSTEM_NAME=borg scripts/run.sh \
  --backend rccl \
  --nodes 1 \
  --ranks 8 \
  --ranks-per-node 8 \
  --target-seconds 0.1 \
  --validate \
  --rccl-stage-b
```

Expected paths follow the existing system/backend layout:

```text
builds/frontier/rccl/
results/borg/<timestamp>_rccl_<label>/
```

To inspect RCCL availability on Frontier or Borg:

```sh
module avail rccl
module spider rccl
find "${ROCM_PATH:-/opt/rocm}" -name 'librccl.so*' 2>/dev/null
find "${ROCM_PATH:-/opt/rocm}" -path '*include*' \( -name rccl.h -o -name nccl.h \) 2>/dev/null
```

For repeatable HPC builds and result capture across systems, use the portable
workflow documented in [Build and Run Workflow](docs/BUILD_AND_RUN.md):

```sh
GHALO_SYSTEM_NAME=frontier scripts/build.sh --backend mpi
GHALO_SYSTEM_NAME=frontier scripts/run.sh --backend mpi --nodes 1 --ranks 4
```

Interactive Frontier MPI-HIP run:

```sh
GHALO_SYSTEM_NAME=frontier scripts/build.sh --backend mpi-hip
GHALO_SYSTEM_NAME=frontier scripts/run.sh \
  --backend mpi-hip \
  --nodes 1 \
  --ranks 8 \
  --ranks-per-node 8 \
  --target-seconds 3 \
  --validate
```

Borg, Frontier's hot-spare cabinet, is supported as a first-class system
profile. Borg runs write results under `results/borg/` while reusing compatible
Frontier build trees by default:

```sh
GHALO_SYSTEM_NAME=borg scripts/run.sh \
  --backend mpi-hip \
  --nodes 1 \
  --ranks 8 \
  --ranks-per-node 8 \
  --validate
```

Batch submission is available through Slurm `sbatch` while preserving the same
run workflow and result metadata:

```sh
GHALO_SYSTEM_NAME=frontier scripts/submit.sh \
  --backend mpi-hip \
  --account <project> \
  --nodes 1 \
  --ranks 8 \
  --ranks-per-node 8 \
  --time 00:10:00 \
  --validate
```

GitHub Actions validate portable CPU-only, OpenMPI, and shell workflow paths.
See [Continuous Integration](docs/CI.md).

The `mpi-hip` backend also supports optional diagnostic phase timing:

```sh
ghalo --backend mpi-hip --phase-timing
```

See [Phase Timing](docs/PHASE_TIMING.md).

## Result Analysis

Portable result analysis is available through `tools/ghalo_analyze.py` and the
small `scripts/analyze.sh` wrapper. Core analysis uses only the Python 3
standard library; plots are optional and require `matplotlib`.

Examples:

```sh
python3 tools/ghalo_analyze.py summarize results/frontier/<run-directory>
python3 tools/ghalo_analyze.py compare RUN_A RUN_B
python3 tools/ghalo_analyze.py aggregate results/frontier/*frontier-64node*
python3 tools/ghalo_analyze.py scaling results/borg/*mpi-hip* results/frontier/*mpi-hip*
python3 tools/ghalo_analyze.py plot --kind latency --output latency.png RUN_A RUN_B
```

The analyzer reports measured timing fields separately from derived metrics
such as the per-rank effective transferred-byte rate. See
[Result Analysis](docs/ANALYSIS.md).

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
- [Build and Run Workflow](docs/BUILD_AND_RUN.md)
- [Continuous Integration](docs/CI.md)
- [Phase Timing](docs/PHASE_TIMING.md)
- [Result Analysis](docs/ANALYSIS.md)
- [v0.3.0 Release Notes](docs/RELEASE_NOTES_v0.3.0.md)
- [Versioning](docs/VERSIONING.md)
- [Historical Context](docs/HISTORY.md)
- [Version 0 Design](docs/VERSION_0_DESIGN.md)
- [GPU-Aware MPI Design](docs/GPU_AWARE_MPI_DESIGN.md)
- [RCCL Backend Design](docs/RCCL_BACKEND_DESIGN.md)
- [RCCL Smoke And Stage B Validation](docs/RCCL_SMOKE_TEST.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md),
and [SECURITY.md](SECURITY.md).

## License

gHALO is released under the MIT License. See [LICENSE](LICENSE).
