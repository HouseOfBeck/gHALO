# Continuous Integration

gHALO uses GitHub Actions for portable checks that can run on hosted Ubuntu
runners. CI is intentionally scoped to development hygiene and CPU/MPI
portability. It does not replace validation on Frontier or Borg.

## GitHub Jobs

The workflow in `.github/workflows/ci.yml` currently runs three jobs.

### CPU-Only CMake

This job configures the project with MPI, HIP, and MPI-HIP disabled:

```sh
cmake -S . -B build/cpu \
  -G Ninja \
  -DGHALO_ENABLE_MPI=OFF \
  -DGHALO_ENABLE_HIP=OFF \
  -DGHALO_ENABLE_MPI_HIP=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu
ctest --test-dir build/cpu --output-on-failure
```

It verifies that the portable C++ core and generic workflow tests remain
buildable without MPI, ROCm, HIP, Slurm, or Cray tooling.

### Linux MPI Build and Tests

This job installs OpenMPI and configures the MPI backend:

```sh
cmake -S . -B build/mpi \
  -G Ninja \
  -DGHALO_ENABLE_MPI=ON \
  -DGHALO_ENABLE_HIP=OFF \
  -DGHALO_ENABLE_MPI_HIP=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/mpi
ctest --test-dir build/mpi --output-on-failure
```

It also runs a short CPU MPI smoke benchmark:

```sh
mpirun --oversubscribe -np 4 \
  ./build/mpi/ghalo \
  --target-seconds 0.01 \
  --csv build/mpi-smoke/ghalo.csv \
  --json build/mpi-smoke/ghalo.json
```

The smoke run verifies that the executable launches under MPI and produces
nonempty CSV and JSON output.

### Shell Validation

This job checks shell syntax and runs ShellCheck:

```sh
bash -n scripts/*.sh scripts/systems/*.sh tests/*.sh
shellcheck -x scripts/*.sh scripts/systems/*.sh tests/*.sh
```

The shell smoke test is also part of CTest and can be run directly:

```sh
bash tests/workflow_smoke.sh
```

The test uses temporary directories and mocked HPC commands so it does not
require Frontier, Borg, Slurm, ROCm, Cray MPICH, or environment modules.

## What GitHub CI Does Not Validate

GitHub-hosted runners do not provide the production HPC environment needed for:

- HIP compilation;
- AMD GPU execution;
- GPU-aware MPI with Cray MPICH;
- RCCL or UCX backends;
- Slurm allocation behavior on Frontier or Borg;
- network topology or performance;
- benchmark timing quality.

Those checks must be performed on Frontier and Borg using the documented
`scripts/build.sh`, `scripts/run.sh`, and `scripts/submit.sh` workflows.

## Failure Diagnostics

Each CMake job runs `ctest --output-on-failure`. On failure, CI uploads
`Testing/Temporary/LastTest.log` as a short-retention artifact when available.

The workflow prints versions for the relevant tools, including CMake, the C++
compiler, OpenMPI, Bash, and ShellCheck.
