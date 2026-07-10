# Build and Run Workflow

This document describes the portable gHALO build-and-run workflow for HPC
systems. The workflow keeps machine-specific build trees, binaries, and result
metadata separate while preserving the benchmark algorithms and timing
semantics implemented by gHALO itself.

## Directory Model

Builds are stored by system and backend:

```text
builds/
  <system-name>/
    mpi/
    mpi-hip/
```

Results are stored by system and timestamp:

```text
results/
  <system-name>/
    <timestamp>_<backend>_<label>/
```

The scripts determine `<system-name>` in this order:

1. `GHALO_SYSTEM_NAME` environment variable;
2. explicit `--system` command-line option;
3. `hostname -s` fallback.

A stable cluster name such as `frontier` is preferred over a login-node or
compute-node hostname. This keeps build and result paths stable across
sessions, allocations, and compute nodes.

## Build Script

Use `scripts/build.sh` to configure and build one backend for one system:

```sh
GHALO_SYSTEM_NAME=frontier scripts/build.sh --backend mpi
GHALO_SYSTEM_NAME=frontier scripts/build.sh --backend mpi-hip --clean
```

Supported options:

```text
--system <name>
--backend mpi|mpi-hip
--build-type Release|Debug
--clean
--jobs <N>
```

The MPI build enables MPI and disables HIP/MPI-HIP:

```text
-DGHALO_ENABLE_MPI=ON
-DGHALO_ENABLE_HIP=OFF
-DGHALO_ENABLE_MPI_HIP=OFF
```

The MPI-HIP build enables MPI, HIP, and MPI-HIP:

```text
-DGHALO_ENABLE_MPI=ON
-DGHALO_ENABLE_HIP=ON
-DGHALO_ENABLE_MPI_HIP=ON
```

Machine-specific CMake additions come from
`scripts/systems/<system-name>.sh`. The portable script does not hard-code
compiler, MPI, ROCm, or scheduler paths.

Each build records metadata under:

```text
builds/<system>/<backend>/build-info/
  backend.txt
  build metadata files
  cmake.txt
  compiler.txt
  configure-command.txt
  environment.txt
  git.txt
  modules.txt
  system.txt
```

The metadata records the Git branch and commit, dirty working-tree state,
compiler and CMake versions, loaded modules when available, relevant MPI/HIP
environment variables, and the exact CMake configure command.

## Run Script

Use `scripts/run.sh` to launch an already-built backend:

```sh
GHALO_SYSTEM_NAME=frontier scripts/run.sh \
  --backend mpi \
  --nodes 2 \
  --ranks 16 \
  --ranks-per-node 8 \
  --target-seconds 3

GHALO_SYSTEM_NAME=frontier scripts/run.sh \
  --backend mpi-hip \
  --nodes 2 \
  --ranks 16 \
  --ranks-per-node 8 \
  --target-seconds 3 \
  --validate
```

Supported options:

```text
--system <name>
--backend mpi|mpi-hip
--nodes <N>
--ranks <N>
--ranks-per-node <N>
--target-seconds <seconds>
--validate
--label <text>
--extra-srun-args "<args>"
```

The run script uses:

```text
builds/<system>/<backend>/ghalo
```

and writes each run to a unique directory:

```text
results/<system>/<timestamp>_<backend>_<label>/
```

The optional label is sanitized for safe filenames. The result directory is
printed before launching the benchmark.

Each run records:

```text
binary-info.txt
command.txt
environment.txt
exit-status.txt
ghalo.csv
ghalo.json
git.txt
hostname.txt
modules.txt
slurm-job.txt
stderr.txt
stdout.txt
```

The current gHALO CLI supports `--csv` and `--json`, so the workflow writes
structured output directly into the result directory. The script launches the
benchmark without pipelines and returns the benchmark exit status.

To prevent accidental cross-use of binaries, `run.sh` checks build metadata
when available and ensures the selected binary comes from the expected
`builds/<system>/<backend>/` directory.

## System Configuration Interface

Each system file is a Bash script at:

```text
scripts/systems/<system-name>.sh
```

It may define these functions:

```sh
ghalo_system_setup_build <backend>
ghalo_system_setup_run <backend>
ghalo_system_cmake_args <backend>
ghalo_system_launch <backend> <nodes> <ranks> <ranks-per-node> <extra-args> <binary> [ghalo args...]
```

The interface is intentionally small:

- setup functions load modules, verify tools, and set system-specific
  environment variables;
- `ghalo_system_cmake_args` prints one CMake argument per line;
- `ghalo_system_launch` prints one launcher argument per line, including the
  binary and gHALO arguments.

Do not put benchmark logic, timing changes, or backend semantics into a system
configuration file. System files should describe how to build and launch gHALO
on a machine, not what gHALO measures.

See `scripts/systems/example.sh` for a template.

## Frontier

`scripts/systems/frontier.sh` is the initial Frontier configuration.

Build examples:

```sh
GHALO_SYSTEM_NAME=frontier scripts/build.sh --backend mpi --jobs 8
GHALO_SYSTEM_NAME=frontier scripts/build.sh --backend mpi-hip --clean --jobs 8
```

Run examples:

```sh
GHALO_SYSTEM_NAME=frontier scripts/run.sh \
  --backend mpi \
  --nodes 2 \
  --ranks 16 \
  --ranks-per-node 8 \
  --target-seconds 3 \
  --label cpu-mpi

GHALO_SYSTEM_NAME=frontier scripts/run.sh \
  --backend mpi-hip \
  --nodes 2 \
  --ranks 16 \
  --ranks-per-node 8 \
  --target-seconds 3 \
  --validate \
  --label gpu-aware
```

The Frontier configuration:

- uses the Cray C++ wrapper `CC`;
- loads or verifies the Cray programming environment where available;
- loads `craype-accel-amd-gfx90a` and `rocm` for `mpi-hip`;
- sets `CMAKE_HIP_ARCHITECTURES=gfx90a` for `mpi-hip`;
- uses `srun` as the launcher;
- sets `MPICH_GPU_SUPPORT_ENABLED=1` only for `mpi-hip` runs and builds;
- unsets `MPICH_GPU_SUPPORT_ENABLED` for CPU MPI runs;
- does not add unvalidated GPU-binding options.

The CMake project contains a target-local MPI header fallback for HIP
compilation that uses `MPICH_DIR/include` only when CMake does not expose an
explicit MPI include directory and `MPICH_DIR/include/mpi.h` exists. The scripts
do not hard-code Cray MPICH versions or `/opt/cray` paths.

The run script can be used inside an existing Slurm allocation or from a batch
script. Batch scripts should request the nodes, time, account, and job
resources; `scripts/run.sh` should be responsible for constructing the gHALO
launch command within that allocation.

## Repository Hygiene

Generated build and result directories are ignored by Git:

```text
builds/
results/*
!results/reference/
```

The exception allows future curated reference results to be tracked under
`results/reference/` without accidentally tracking routine benchmark output.
