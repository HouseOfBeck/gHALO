# gHALO v0.3.0 Release Notes

## Release Overview

gHALO v0.3.0 is the first release-prepared snapshot that combines the
CPU/host-memory MPI reference backend with an AMD HIP device-memory backend
using GPU-aware MPI. The release preserves the HALO benchmark methodology while
adding the operational pieces needed to build, run, validate, capture, and
analyze results on Frontier-class systems.

This release remains a `0.x` research and architecture release. Benchmark
semantics are intentionally conservative, but command-line interfaces and
output schemas may still evolve before `1.0.0`.

## Supported Platforms

Supported development and execution paths:

- macOS or other workstations for editing, documentation, Git, and portable
  Python/shell checks.
- Linux systems with CMake and a C++20 compiler for CPU-only builds.
- Linux MPI systems for the `mpi` backend.
- Frontier with Cray MPICH, ROCm/HIP, AMD GPUs, and
  `MPICH_GPU_SUPPORT_ENABLED=1` for the `mpi-hip` backend.
- Borg, Frontier's hot-spare cabinet, using the Borg system profile and
  Frontier build-tree reuse by default.

Unsupported or not-yet-implemented paths:

- RCCL backend.
- UCX backend.
- GitHub-hosted HIP/GPU performance validation.

## Validated Configurations

The v0.3.0 release preparation includes:

- CPU-only build and portable tests through GitHub CI.
- Linux OpenMPI build and short CPU MPI smoke run through GitHub CI.
- Shell syntax and ShellCheck validation through GitHub CI.
- Frontier MPI-HIP validation at 64 nodes and 512 GPU ranks.
- Frontier and Borg workflow support for system/backend-specific builds,
  interactive runs, Slurm batch submission, metadata capture, and analysis.

## Example Build Command

Frontier MPI-HIP build through the portable workflow:

```sh
GHALO_SYSTEM_NAME=frontier scripts/build.sh \
  --backend mpi-hip \
  --jobs 8
```

The system profile enables MPI, HIP, and MPI-HIP CMake options, configures the
Cray programming environment, and records build metadata under
`builds/frontier/mpi-hip/build-info/`.

## Example Interactive Command

Frontier interactive MPI-HIP run:

```sh
GHALO_SYSTEM_NAME=frontier scripts/run.sh \
  --backend mpi-hip \
  --nodes 1 \
  --ranks 8 \
  --ranks-per-node 8 \
  --target-seconds 3 \
  --validate \
  --label validation
```

Results are written under `results/frontier/<timestamp>_mpi-hip_validation/`
with `ghalo.json`, `ghalo.csv`, command metadata, environment metadata, Git
metadata, Slurm metadata when present, binary metadata, and exit status.

## Example Batch Command

Frontier Slurm batch submission:

```sh
GHALO_SYSTEM_NAME=frontier scripts/submit.sh \
  --backend mpi-hip \
  --account <project> \
  --partition batch \
  --nodes 64 \
  --ranks 512 \
  --ranks-per-node 8 \
  --time 00:30:00 \
  --target-seconds 3 \
  --validate \
  --phase-timing \
  --label frontier-64node
```

Borg runs use `GHALO_SYSTEM_NAME=borg` and write results under `results/borg/`
while reusing compatible Frontier build trees unless native Borg builds are
requested.

## Example Analysis Command

Summarize one run:

```sh
python3 tools/ghalo_analyze.py summarize \
  --include-metadata \
  results/frontier/<run-directory>
```

Compare repeated runs:

```sh
python3 tools/ghalo_analyze.py compare \
  results/frontier/<first-64-node-run> \
  results/frontier/<repeat-64-node-run>
```

Create a latency plot when `matplotlib` is available:

```sh
python3 tools/ghalo_analyze.py plot \
  --kind latency \
  --xscale log2 \
  --labels "Run 1,Run 2" \
  --output frontier-64node-repeatability.png \
  RUN_1 RUN_2
```

## Known Limitations

- MPI-HIP currently uses correctness-first synchronization at GPU-aware
  MPI/HIP handoff points. This is intentionally conservative and may be
  replaced later by lower-overhead stream-aware ordering.
- Phase timing perturbs very small messages because it adds timing calls around
  internal exchange phases.
- Independently `MPI_MAX`-reduced phase maxima may sum to more than the
  independently reduced total exchange maximum.
- RCCL phase timing is diagnostic; synchronization optimization and performance
  comparisons are not yet implemented.
- GitHub-hosted CI does not test HIP, Cray MPICH, Slurm, GPU execution, or
  performance.
- Large-system performance results remain system- and placement-dependent.

## Next Planned Work

- Continue validating MPI-HIP across larger Frontier and Borg placements.
- Investigate lower-overhead synchronization for the MPI-HIP backend while
  preserving correctness and benchmark semantics.
- Continue RCCL bring-up with synchronization analysis and comparison runs.
- Expand diagnostics from summary analysis toward topology and cluster-health
  reporting.
- Continue RCCL backend experiments beyond the initial full-exchange path.
- Continue refining output schema documentation before `1.0.0`.

## Maintainer Note

This document prepares the repository for a v0.3.0 release. It does not create
or push a Git tag; tagging is left to the maintainer.
