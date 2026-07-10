# Development Guide

gHALO development is designed around a split environment:

- macOS workstation for editing, Git, documentation, and portable checks.
- Remote Linux HPC cluster for HIP compilation, MPI builds, and GPU execution.

Do not assume that the local macOS machine has ROCm, HIP, MPI, RCCL, UCX, or
GPU hardware available.

## Local Development

Local development should remain useful without accelerator toolchains:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Feature options are disabled by default. This keeps local configuration
portable while preserving clear entry points for remote builds. The MPI
executable is built only when `GHALO_ENABLE_MPI=ON`.

## Remote Development

HIP, GPU-aware MPI, RCCL, UCX, and GPU execution should happen on a remote Linux
cluster with the required modules, compilers, and runtime libraries loaded.

Examples:

```sh
cmake -S . -B build-hip -DGHALO_ENABLE_HIP=ON
cmake -S . -B build-mpi -DGHALO_ENABLE_MPI=ON
cmake -S . -B build-gpu-mpi -DGHALO_ENABLE_MPI=ON -DGHALO_ENABLE_HIP=ON -DGHALO_ENABLE_MPI_HIP=ON
```

The exact compiler wrappers, module names, and scheduler commands are expected
to vary by cluster and should be documented under `scripts/` or `examples/` as
the project matures.

For repeatable system-specific builds and result capture, see
[Build and Run Workflow](BUILD_AND_RUN.md).

## Frontier Version 0 Build

For Frontier, use the OLCF programming environment and CMake with the Cray C++
compiler wrapper:

```sh
cmake -S . -B build-frontier -DGHALO_ENABLE_MPI=ON -DCMAKE_CXX_COMPILER=CC
cmake --build build-frontier
srun -n 16 ./build-frontier/ghalo --csv ghalo.csv --json ghalo.json
```

Version 0 does not require ROCm or HIP.

## Frontier GPU-Aware MPI Build

For the HIP device-buffer MPI backend on Frontier, load the accelerator and
ROCm modules and enable GPU-aware Cray MPICH before configuring:

```sh
module load craype-accel-amd-gfx90a
module load rocm
export MPICH_GPU_SUPPORT_ENABLED=1

cmake -S . -B build-frontier-mpi-hip \
  -DGHALO_ENABLE_MPI=ON \
  -DGHALO_ENABLE_HIP=ON \
  -DGHALO_ENABLE_MPI_HIP=ON \
  -DCMAKE_CXX_COMPILER=CC

cmake --build build-frontier-mpi-hip
srun -N 1 -n 4 --ntasks-per-gpu=1 \
  ./build-frontier-mpi-hip/ghalo \
  --backend mpi-hip \
  --target-seconds 0.1
```

Validation is enabled by default for `mpi-hip`. Use `--no-validate` only after
the backend has been validated on the target system.

## CMake Policy

- Keep optional dependencies disabled by default.
- Use `find_package` only inside the relevant feature option.
- Keep HIP language enablement gated behind `GHALO_ENABLE_HIP`.
- Avoid global compile definitions that leak backend assumptions into portable
  code.
- Prefer target-scoped options, includes, definitions, and link libraries.

## Source Organization Policy

Source code should keep portable and platform-specific code separated:

```text
include/ghalo/       public C++ interfaces
src/core/            portable benchmark orchestration
src/backends/mpi/    MPI implementation
src/backends/rccl/   future RCCL implementation
src/backends/ucx/    future UCX implementation
src/gpu/hip/         future HIP kernels and GPU utilities
tests/               portable and backend-specific tests
examples/            runnable configurations and scheduler examples
scripts/             build, launch, and result-processing helpers
```

## Testing Policy

Tests should scale with the feature being added:

- Portable core logic should have local unit tests.
- MPI behavior should have tests that can run under a launcher on a cluster.
- HIP behavior should include device correctness tests.
- Performance tests should be separate from correctness tests.
- Diagnostic tests should validate output schemas and warning logic.

## Style

- Use portable C++20 for host-side code.
- Keep backend interfaces small and explicit.
- Make synchronization behavior visible in code and output.
- Record enough metadata for benchmark results to be reproducible.
- Prefer structured outputs over log scraping.
