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
    rccl/
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
GHALO_SYSTEM_NAME=frontier scripts/build.sh --backend rccl --clean
```

Supported options:

```text
--system <name>
--backend mpi|mpi-hip|rccl
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

The RCCL build enables MPI, HIP, and RCCL but does not enable MPI-HIP unless a
future implementation explicitly shares that path:

```text
-DGHALO_ENABLE_MPI=ON
-DGHALO_ENABLE_HIP=ON
-DGHALO_ENABLE_MPI_HIP=OFF
-DGHALO_ENABLE_RCCL=ON
```

The RCCL backend is experimental. The default path runs the full
two-dimensional halo exchange with correctness-first stream synchronization.
It does not yet implement RCCL phase timing, synchronization optimization, or
performance-comparison claims.

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
--backend mpi|mpi-hip|rccl
--nodes <N>
--ranks <N>
--ranks-per-node <N>
--target-seconds <seconds>
--validate
--phase-timing
--rccl-stage-b
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
submission.txt
system-resolution.txt
```

The current gHALO CLI supports `--csv` and `--json`, so the workflow writes
structured output directly into the result directory. The script launches the
benchmark without pipelines and returns the benchmark exit status.

For RCCL full correctness validation:

```sh
GHALO_SYSTEM_NAME=borg scripts/run.sh \
  --backend rccl \
  --nodes 1 \
  --ranks 8 \
  --ranks-per-node 8 \
  --target-seconds 0.1 \
  --validate
```

`--backend rccl --rccl-stage-b` runs the retained north/south-only debugging
path. Stage B does not emit normal production benchmark timings.

To prevent accidental cross-use of binaries, `run.sh` checks build metadata
when available and ensures the selected binary comes from the expected
`builds/<system>/<backend>/` directory.

### Build Aliases

Most systems use the active system name for both results and binaries:

```text
active system: <system>
build system:  <system>
binary:        builds/<system>/<backend>/ghalo
results:       results/<system>/
```

A system profile may define a build alias when two systems share compatible
hardware, software, and filesystems. In that case, `run.sh` still writes results
under the active system, but resolves the binary from the aliased build system:

```text
active system: borg
build system:  frontier
binary:        builds/frontier/<backend>/ghalo
results:       results/borg/
```

Each run records this resolution in:

```text
system-resolution.txt
```

with fields for `active_system`, `build_system`, `backend`, and `binary`.

Profiles may honor `GHALO_BUILD_SYSTEM_ALIAS=<name>` for explicit aliasing and
`GHALO_USE_NATIVE_BUILD=1` to force the active system's own build tree.

For Borg RCCL experiments, the default build alias also points to Frontier:

```text
active system: borg
build system:  frontier
binary:        builds/frontier/rccl/ghalo
results:       results/borg/<timestamp>_rccl_<label>/
```

Inspect RCCL availability on Frontier or Borg with:

```sh
module avail rccl
module spider rccl
find "${ROCM_PATH:-/opt/rocm}" -name 'librccl.so*' 2>/dev/null
find "${ROCM_PATH:-/opt/rocm}" -path '*include*' \( -name rccl.h -o -name nccl.h \) 2>/dev/null
```

Set `RCCL_ROOT`, `RCCL_PATH`, or `ROCM_PATH` if RCCL is installed outside the
default search paths.

## Batch Submission

Use `scripts/submit.sh` to submit a gHALO run through Slurm `sbatch` instead of
first obtaining an interactive allocation. The submitted Slurm job runs
`scripts/batch-job.sh`, and `batch-job.sh` invokes `scripts/run.sh`. This keeps
interactive and batch runs on the same system profiles, binary resolution,
environment setup, result directory layout, metadata capture, and benchmark
arguments.

Example Borg MPI-HIP phase-timing run:

```sh
GHALO_SYSTEM_NAME=borg scripts/submit.sh \
  --backend mpi-hip \
  --account VEN004 \
  --partition batch \
  --nodes 6 \
  --ranks 48 \
  --ranks-per-node 8 \
  --time 00:20:00 \
  --target-seconds 3 \
  --validate \
  --phase-timing \
  --label phase-6node-48gpu
```

Example Frontier scale run:

```sh
GHALO_SYSTEM_NAME=frontier scripts/submit.sh \
  --backend mpi-hip \
  --account VEN004 \
  --partition batch \
  --nodes 128 \
  --ranks 1024 \
  --ranks-per-node 8 \
  --time 00:30:00 \
  --target-seconds 3 \
  --validate \
  --label scale-128node
```

Supported submission options include:

```text
--system <name>
--backend mpi|mpi-hip
--account <account>
--partition <partition>
--nodes <N>
--ranks <N>
--ranks-per-node <N>
--time <HH:MM:SS>
--target-seconds <seconds>
--validate
--phase-timing
--label <text>
--job-name <name>
--constraint <constraint>
--reservation <reservation>
--qos <qos>
--exclusive
--dependency <dependency>
--extra-sbatch-args "<args>"
--extra-srun-args "<args>"
--confirm-large-run
--dry-run
```

The submit script requires `backend`, `nodes`, `ranks`, `ranks-per-node`, and
wall-clock `time`. It also requires an account unless a system profile provides
a default account. Explicit command-line values override system-profile
defaults. Frontier currently provides a `batch` partition default; accounts are
left explicit because they are allocation-specific.

For normal gHALO runs, `scripts/submit.sh` requires:

```text
ranks == nodes * ranks-per-node
```

Use `scripts/run.sh` directly inside a custom allocation for unusual layouts.

The submit script prints the requested resources and the complete shell-escaped
`sbatch` command before submission. With `--dry-run`, it prints the command but
does not submit it.

Batch stdout and stderr are written under:

```text
batch-logs/<system>/%x-%j.out
batch-logs/<system>/%x-%j.err
```

Routine batch logs are ignored by Git.

Large-run safety:

- runs above 128 nodes print a warning;
- runs above 512 nodes require `--confirm-large-run`;
- these thresholds may be adjusted with `GHALO_LARGE_RUN_WARNING_NODES` and
  `GHALO_LARGE_RUN_CONFIRM_NODES`.

The normal timestamped result directory remains the result hierarchy. When a
run is submitted through `scripts/submit.sh`, `scripts/run.sh` also writes:

```text
submission.txt
```

with the Slurm job ID, job name, submission system, account, partition, node
list, original submit command, and batch stdout/stderr paths.

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
ghalo_system_default_account
ghalo_system_default_partition
ghalo_system_default_batch_time
```

The interface is intentionally small:

- setup functions load modules, verify tools, and set system-specific
  environment variables;
- `ghalo_system_cmake_args` prints one CMake argument per line;
- `ghalo_system_launch` prints one launcher argument per line, including the
  binary and gHALO arguments.
- default functions may print optional Slurm defaults for `scripts/submit.sh`.

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
  --phase-timing \
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

## Borg

Borg is a Frontier hot-spare cabinet. Borg compute blades are
hardware-identical to Frontier compute blades, and Borg shares the same NFS
filesystem as Frontier. Because the hardware, filesystem, Cray programming
environment, Cray MPICH, and `gfx90a` GPU target are compatible, the Borg
profile reuses Frontier build trees by default:

```text
builds/frontier/mpi/ghalo
builds/frontier/mpi-hip/ghalo
results/borg/<timestamp>_mpi_<label>/
results/borg/<timestamp>_mpi-hip_<label>/
```

This keeps Frontier and Borg result histories separate while avoiding duplicate
builds for identical binaries.

Borg CPU MPI smoke run:

```sh
GHALO_SYSTEM_NAME=borg scripts/run.sh \
  --backend mpi \
  --nodes 1 \
  --ranks 4 \
  --ranks-per-node 4 \
  --target-seconds 0.1 \
  --label smoke
```

Borg MPI-HIP smoke run:

```sh
GHALO_SYSTEM_NAME=borg scripts/run.sh \
  --backend mpi-hip \
  --nodes 1 \
  --ranks 4 \
  --ranks-per-node 4 \
  --target-seconds 0.1 \
  --validate \
  --label smoke
```

Borg phase-timing baseline:

```sh
GHALO_SYSTEM_NAME=borg scripts/run.sh \
  --backend mpi-hip \
  --nodes 1 \
  --ranks 4 \
  --ranks-per-node 4 \
  --target-seconds 3 \
  --validate \
  --phase-timing \
  --label phase-baseline
```

The Borg profile:

- uses the Cray C++ wrapper `CC`;
- uses `srun` with `-N`, `-n`, and `--ntasks-per-node` when supplied;
- uses `gfx90a` for HIP builds;
- loads `craype-accel-amd-gfx90a` and `rocm/6.2.4` for `mpi-hip`;
- sets `MPICH_GPU_SUPPORT_ENABLED=1` only for `mpi-hip`;
- unsets `MPICH_GPU_SUPPORT_ENABLED` for CPU MPI;
- avoids unvalidated GPU-binding options;
- does not hard-code Cray MPICH paths.

To create Borg-native builds instead of using Frontier builds:

```sh
GHALO_SYSTEM_NAME=borg scripts/build.sh --backend mpi
GHALO_SYSTEM_NAME=borg scripts/build.sh --backend mpi-hip
GHALO_USE_NATIVE_BUILD=1 GHALO_SYSTEM_NAME=borg scripts/run.sh --backend mpi-hip
```

With `GHALO_USE_NATIVE_BUILD=1`, Borg runs resolve binaries from:

```text
builds/borg/<backend>/ghalo
```

## Repository Hygiene

Generated build and result directories are ignored by Git:

```text
builds/
results/*
!results/reference/
```

The exception allows future curated reference results to be tracked under
`results/reference/` without accidentally tracking routine benchmark output.
`rccl` is accepted by the workflow for experimental full-exchange correctness
runs. Treat timing results as bring-up data until RCCL phase timing and
synchronization policy have been studied.
