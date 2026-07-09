# gHALO Version 0 Design

Version 0 implements the original HALO benchmark methodology with modern C++20
structure and an MPI backend. It intentionally does not implement HIP,
GPU-aware MPI, RCCL, UCX, or SHMEM yet.

## Design Decisions

- The benchmark runner depends on the abstract `Backend` interface, not MPI.
- MPI is isolated in `MPIBackend`.
- `MPIBackend` creates a periodic two-dimensional Cartesian communicator with
  `MPI_Cart_create`.
- The Cartesian dimensions follow the original HALO factorization policy so
  version 0 can preserve historical topology behavior.
- Version 0 implements the `sendrecv` exchange algorithm, corresponding to the
  original MPI `HALO2A` variant.
- The timed exchange uses local communication buffers and simulated copy steps.
- The first exchange for each halo size is not timed.
- A five-exchange calibration loop estimates the iteration count needed for
  approximately three seconds of steady-state timing.
- The reported time is the maximum rank elapsed time divided by the iteration
  count.
- Console, CSV, and JSON output are produced by rank zero.

## Compatibility Semantics

For each halo length `N`, version 0 performs:

1. copy previous east-west output into the north-south send buffer;
2. send `N` words south and receive `N` words from north;
3. send `2N` words north and receive `2N` words from south;
4. copy north-south output into the east-west send buffer;
5. send `N` words west and receive `N` words from east;
6. send `2N` words east and receive `2N` words from west.

The payload word is `float`, matching the original benchmark's 4-byte Fortran
`REAL` payload size.

## Frontier Notes

Frontier builds should use the programming environment and compiler wrappers
provided by OLCF. The Frontier user guide documents the programming environment,
compiler wrappers, MPI support, and Slurm `srun` launch model.

Example:

```sh
cmake -S . -B build-frontier -DGHALO_ENABLE_MPI=ON -DCMAKE_CXX_COMPILER=CC
cmake --build build-frontier
srun -n 16 ./build-frontier/ghalo --csv ghalo.csv --json ghalo.json
```

The code does not require ROCm or HIP for version 0.
