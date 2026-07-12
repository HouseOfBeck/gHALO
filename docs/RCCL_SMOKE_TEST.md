# RCCL Smoke And Stage B Validation

This document describes the standalone RCCL smoke test and the integrated RCCL
Stage B validation path.

## Standalone Smoke Test

`ghalo_rccl_smoke` validates RCCL bring-up independently of the benchmark
backend. It is built only when `GHALO_ENABLE_RCCL=ON`.

The smoke test checks:

- MPI initialization;
- node-local rank discovery;
- HIP device selection;
- HIP device allocation;
- RCCL communicator initialization;
- grouped RCCL point-to-point send/receive;
- stream synchronization;
- exact received-data validation;
- cleanup.

Example Borg run:

```sh
srun -N 1 -n 8 --ntasks-per-node=8 \
  builds/borg/rccl/tests/ghalo_rccl_smoke \
  --count 1024 \
  --iterations 1 \
  --validate \
  --verbose
```

The standalone smoke test uses a ring pattern. It is a bring-up diagnostic, not
the HALO benchmark.

## Integrated Stage B

The RCCL backend now includes Stage B integration for north/south halo
correctness. This path uses the normal gHALO application, topology, validation
patterns, HIP stream ownership, and RCCL communicator setup.

Run it explicitly with:

```sh
srun -N 1 -n 8 --ntasks-per-node=8 \
  builds/borg/rccl/ghalo \
  --backend rccl \
  --rccl-stage-b \
  --validate \
  --target-seconds 0.1
```

Stage B validates:

- RCCL communicator initialization;
- local-rank HIP device mapping;
- device-resident north/south `N` exchange;
- device-resident north/south `2N` exchange;
- self-neighbor and duplicate-neighbor cases;
- stream synchronization with `hipStreamSynchronize`;
- exact data values matching the MPI-HIP north/south validation pattern.

Stage B does not implement:

- intermediate `hons` to `hiew` transpose;
- east/west exchange;
- full benchmark timing;
- phase timing;
- performance comparison;
- scaling claims.

Without `--rccl-stage-b`, `--backend rccl` fails clearly because the full halo
exchange is not implemented. This prevents partial north/south correctness
runs from being mistaken for production benchmark results.

## Borg Validation Matrix

Correctness runs should include:

```sh
srun -N 1 -n 1 --ntasks-per-node=1 builds/borg/rccl/ghalo \
  --backend rccl --rccl-stage-b --validate --target-seconds 0.1

srun -N 1 -n 2 --ntasks-per-node=2 builds/borg/rccl/ghalo \
  --backend rccl --rccl-stage-b --validate --target-seconds 0.1

srun -N 1 -n 8 --ntasks-per-node=8 builds/borg/rccl/ghalo \
  --backend rccl --rccl-stage-b --validate --target-seconds 0.1

srun -N 2 -n 16 --ntasks-per-node=8 builds/borg/rccl/ghalo \
  --backend rccl --rccl-stage-b --validate --target-seconds 0.1
```

These are correctness runs only. Do not compare Stage B timings against
`mpi-hip`.
