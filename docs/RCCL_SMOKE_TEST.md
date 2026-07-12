# RCCL Smoke And Validation

This document describes the standalone RCCL smoke test and the integrated RCCL
backend validation paths.

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

## Integrated Stage C

The default `--backend rccl` path now runs the full two-dimensional halo
exchange:

1. north/south RCCL exchange;
2. intermediate `hons` to `hiew` device copy;
3. east/west RCCL exchange;
4. full validation with the same expected values as MPI-HIP.

Run full RCCL validation with:

```sh
srun -N 1 -n 8 --ntasks-per-node=8 \
  builds/borg/rccl/ghalo \
  --backend rccl \
  --validate \
  --target-seconds 0.1
```

The full path participates in the normal gHALO timing loop after validation.
It does not implement RCCL phase timing or make performance-comparison claims.

## Integrated Stage B

Stage B remains available as a north/south-only debugging path:

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

Stage B does not emit normal benchmark timing results. It exists so
north/south RCCL behavior can be debugged independently of transpose and
east/west communication.

## Borg Validation Matrix

Full correctness runs should include:

```sh
srun -N 1 -n 1 --ntasks-per-node=1 builds/borg/rccl/ghalo \
  --backend rccl --validate --target-seconds 0.1

srun -N 1 -n 2 --ntasks-per-node=2 builds/borg/rccl/ghalo \
  --backend rccl --validate --target-seconds 0.1

srun -N 1 -n 4 --ntasks-per-node=4 builds/borg/rccl/ghalo \
  --backend rccl --validate --target-seconds 0.1

srun -N 1 -n 8 --ntasks-per-node=8 builds/borg/rccl/ghalo \
  --backend rccl --validate --target-seconds 0.1

srun -N 2 -n 16 --ntasks-per-node=8 builds/borg/rccl/ghalo \
  --backend rccl --validate --target-seconds 0.1
```

Also rerun Stage B at one and two nodes when debugging RCCL transport issues.
Do not compare RCCL timings against `mpi-hip` until phase timing and
synchronization policy have been studied.
