# GPU-Aware MPI Backend Design

This document defines the proposed design for the next gHALO milestone:
GPU-aware MPI on AMD GPUs using HIP. It is a design document only. It does not
authorize changes to benchmark semantics and does not include implementation
code.

The CPU/MPI backend remains the reference backend. The GPU-aware MPI backend
must preserve the same HALO exchange definition, topology, timing methodology,
and reporting semantics while moving communication buffers into AMD GPU memory.

## Goals

- Preserve Version 0 HALO benchmark semantics.
- Keep the existing CPU/MPI backend unchanged as the reference backend.
- Add a separate GPU-aware MPI backend for HIP device buffers.
- Allocate send and receive buffers with `hipMalloc`.
- Pass device pointers directly to MPI.
- Exclude initialization, allocation, device setup, validation, and warm-up from
  timed measurements.
- Use one MPI rank per GPU or GPU compute device.
- Keep the design portable beyond Frontier while documenting Frontier-specific
  build and run examples.

## Non-Goals

- Do not add RCCL.
- Do not add UCX.
- Do not add new exchange algorithms.
- Do not introduce host staging in the timed communication path.
- Do not change CPU/MPI reference behavior.
- Do not require ROCm, HIP, or MPI on macOS development systems.

## Benchmark Semantics

The GPU-aware MPI backend must implement the same logical exchange as the
current CPU/MPI backend:

1. copy previous east-west output into the north-south send buffer;
2. exchange `N` words south and receive `N` words from north;
3. exchange `2N` words north and receive `2N` words from south;
4. copy north-south output into the east-west send buffer;
5. exchange `N` words west and receive `N` words from east;
6. exchange `2N` words east and receive `2N` words from west.

The halo sizes remain:

```text
2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
```

The payload remains a 4-byte word for compatibility with the original HALO
methodology and the current CPU/MPI reference backend.

## Timing Semantics

Timing must remain identical to the CPU implementation:

- backend initialization is excluded;
- HIP device discovery and selection are excluded;
- `hipMalloc` allocation is excluded;
- buffer initialization is excluded;
- validation is excluded;
- warm-up exchange is excluded;
- the five-exchange calibration loop remains;
- approximately `target-seconds` of steady-state execution are measured per
  halo size;
- elapsed time is reduced with maximum across ranks;
- the reported metric remains maximum average wall-clock time per exchange.

The compatibility timing metric should use the same wall-clock source and MPI
maximum reduction path as the CPU backend. HIP events may be added later as
supplemental diagnostics, but they must not replace the primary HALO-compatible
wall-clock metric.

## Backend Separation

The existing CPU/MPI backend should remain the reference backend and should not
gain HIP dependencies.

The GPU-aware MPI backend should be separate, for example:

```text
src/backends/mpi/          CPU host-buffer MPI backend
src/backends/mpi_hip/      GPU-aware MPI backend using HIP device buffers
src/gpu/hip/               HIP utilities and kernels, if shared later
```

The existing `Backend` interface is close to sufficient:

```text
setup(halo_words)
exchange()
barrier()
now()
max_time(local_seconds)
topology()
```

The GPU-aware backend should implement this interface rather than changing the
benchmark runner. Interface changes should be avoided unless they are required
for correctness or clear ownership. If validation is added, prefer a separate
optional validation path rather than putting validation behavior into the timed
`exchange()` call.

Recommended names:

- `MPIBackend`: existing host-memory reference backend.
- `MPIHIPBackend`: GPU-aware MPI backend using HIP device buffers.
- `HIPDevice`: small RAII helper for device selection and metadata.
- `HIPBuffer<T>`: RAII helper for `hipMalloc`/`hipFree`.

These helpers should remain implementation details of the HIP-enabled targets
unless a stable public abstraction is needed later.

## Device Memory Model

The GPU-aware MPI backend should allocate all communication buffers in device
memory:

- north-south input/send buffer;
- north-south output/receive buffer;
- east-west input/send buffer;
- east-west output/receive buffer.

Allocation should use `hipMalloc`. Deallocation should use RAII and `hipFree`.

The timed exchange must pass device pointers directly to MPI. Host staging is
not allowed in the timed communication path. If a host-staged comparison mode is
ever added, it must be a separately named backend or mode and must be labeled in
output metadata.

The simulated copy steps from the CPU backend must still exist semantically. For
the GPU-aware backend, these copies should be device-side operations, either:

- HIP kernels, or
- `hipMemcpyAsync` device-to-device copies when that exactly matches the
  intended operation.

The backend must synchronize GPU work as needed before MPI consumes a device
buffer and before GPU work consumes an MPI receive buffer. Synchronization
policy must be explicit and documented because it affects the measured
application-visible communication cost.

## HIP Responsibilities

HIP is responsible for:

- runtime initialization checks;
- device discovery;
- device selection;
- buffer allocation;
- buffer initialization;
- device-side copy or packing work;
- validation kernels or validation preparation;
- synchronization;
- error reporting.

Every HIP API call should be checked. HIP failures should become clear gHALO
errors that abort the MPI job consistently after MPI is initialized.

## Rank-to-GPU Mapping

The intended execution model is one MPI rank per GPU or GPU compute device.

The default portable mapping should be:

```text
device_index = local_rank % visible_device_count
```

This requires a reliable local rank. The implementation should derive local rank
from MPI shared-memory communicators rather than relying only on environment
variables:

1. split `MPI_COMM_WORLD` by shared-memory node with `MPI_Comm_split_type`;
2. get `local_rank` and `local_size`;
3. query `hipGetDeviceCount`;
4. choose the visible device from `local_rank`;
5. call `hipSetDevice`.

If `local_size > visible_device_count` in one-rank-per-GPU mode, the backend
should fail before benchmark timing starts.

### Slurm and `ROCR_VISIBLE_DEVICES`

Schedulers may restrict the set of GPUs visible to each rank. On Frontier,
Slurm GPU options such as `--ntasks-per-gpu=1`, `--gpus-per-node`, or
`--gpus-per-task` are important because they cause `ROCR_VISIBLE_DEVICES` to be
set for each rank. If `ROCR_VISIBLE_DEVICES` exposes one GPU per rank, the
backend will usually see `hipGetDeviceCount() == 1` and select device `0`
inside each rank's restricted view.

The backend should record:

- world rank;
- local rank;
- selected HIP device ordinal;
- visible device count;
- `ROCR_VISIBLE_DEVICES`, if set;
- device PCI bus ID or HIP device UUID, if available;
- device name.

This metadata is required for future diagnostics and for explaining rank/device
placement.

## Device Mapping Modes

Proposed command-line option:

```text
--device-map local-rank
```

Initial supported mode:

- `local-rank`: select device by local rank within the set visible to the
  process.

Future modes may include explicit maps or topology-aware maps, but those should
not be part of the first GPU-aware MPI implementation unless required for
correctness on the target system.

## Error Detection

The GPU-aware backend should fail early, collectively, and clearly.

### No GPU Available

Detection:

- call `hipGetDeviceCount`;
- if the count is zero, report a fatal configuration error.

Expected message:

```text
GPU-aware MPI backend requested, but HIP reports no visible devices.
```

### Invalid Rank-to-Device Mapping

Detection:

- compute local rank and local size;
- query visible device count;
- in one-rank-per-GPU mode, require enough visible devices or a scheduler view
  that exposes one device per rank;
- call `hipSetDevice` and verify success.

Expected message should include world rank, local rank, local size, visible
device count, and `ROCR_VISIBLE_DEVICES`.

### HIP Runtime Errors

Detection:

- wrap every HIP call;
- include the HIP error name and string;
- include source context where practical.

Failure policy:

- abort the MPI job after reporting the error.

### MPI Cannot Communicate With Device Buffers

Detection should happen before benchmark timing.

Recommended checks:

1. verify known runtime configuration when available, such as
   `MPICH_GPU_SUPPORT_ENABLED=1` on Cray MPICH;
2. perform a small untimed device-buffer MPI exchange between neighbors;
3. validate the result by copying only a small validation buffer back to host;
4. fail if MPI returns an error, aborts, or validation fails.

The code should not assume all MPI implementations expose the same capability
query. Runtime probing is more portable than relying only on environment
variables.

On Frontier, Cray MPICH requires GPU-aware support to be enabled and linked.
The OLCF Frontier documentation states that Cray MPICH is GPU-aware, that GPU
buffers can be passed directly to MPI, and that `MPICH_GPU_SUPPORT_ENABLED=1`
is required for GPU-aware use.

## Correctness Validation

Validation must be optional but strongly recommended for GPU-aware MPI runs.

Proposed command-line option:

```text
--validate
```

Validation requirements:

- validation occurs before timed measurement;
- validation does not run inside the timed loop;
- validation uses device-resident benchmark buffers;
- validation copies only validation data back to the host;
- validation verifies that received data came from the expected neighbor.

Recommended validation sequence for each halo size or for a representative set
of halo sizes:

1. initialize each rank's device send buffers with rank- and
   direction-specific patterns;
2. synchronize device work;
3. perform one or more untimed exchanges;
4. copy compact validation regions from receive buffers back to host;
5. verify north, south, east, and west receive regions against expected neighbor
   patterns;
6. collectively reduce validation status across ranks;
7. abort before timing if any rank fails validation.

Pattern design:

```text
value = encode(source_rank, direction, halo_words, element_index)
```

The encoding should be exactly representable in the selected payload type or
use integer validation buffers separate from the timed payload. If separate
validation buffers are used, they must preserve the same communication sizes and
device-pointer MPI path.

Validation should check the asymmetric `N` and `2N` regions separately so that
direction swaps, offset errors, and message-size errors are caught.

## Command-Line Interface

Proposed options:

```text
--backend mpi
--backend mpi-hip
--device-map local-rank
--validate
```

Behavior:

- `--backend mpi`: use the existing CPU/MPI reference backend.
- `--backend mpi-hip`: use the new GPU-aware MPI backend.
- `--device-map local-rank`: select HIP device by local rank within the visible
  device set.
- `--validate`: run untimed correctness validation before benchmark timing.

The default should remain conservative. During early development, defaulting to
`--backend mpi` is preferable so existing CPU/MPI workflows keep working.

Output metadata should record backend name, memory location, device map mode,
selected device, validation status, and whether GPU-aware MPI was requested.

## CMake Design

CPU-only builds must continue to work.

Recommended options:

```cmake
GHALO_ENABLE_MPI              # existing option
GHALO_ENABLE_HIP              # existing option
GHALO_ENABLE_GPU_AWARE_MPI    # new explicit option
```

Rules:

- `GHALO_ENABLE_HIP=OFF` by default.
- `GHALO_ENABLE_GPU_AWARE_MPI=OFF` by default.
- `GHALO_ENABLE_GPU_AWARE_MPI=ON` requires `GHALO_ENABLE_MPI=ON`.
- `GHALO_ENABLE_GPU_AWARE_MPI=ON` requires `GHALO_ENABLE_HIP=ON`.
- HIP language enablement must remain gated by `GHALO_ENABLE_HIP`.
- CMake must not hard-code ROCm paths.
- CMake must not assume ROCm exists on macOS.
- HIP targets should be added only when HIP is explicitly enabled.

Discovery strategy:

- prefer CMake's HIP language and ROCm-provided CMake package configuration;
- use target-scoped include directories, definitions, and link libraries;
- avoid global compile definitions that leak HIP assumptions into CPU-only
  targets;
- keep the CPU/MPI backend target free of HIP headers and libraries.

Frontier-specific flags belong in examples or documentation, not in portable
CMake logic.

## Frontier Build and Run Examples

The following examples are Frontier-oriented, not portable requirements.

Frontier GPU-aware MPI environment:

```sh
module load craype-accel-amd-gfx90a
module load rocm
export MPICH_GPU_SUPPORT_ENABLED=1
```

Configure and build:

```sh
cmake -S . -B build-frontier-mpi-hip \
  -DGHALO_ENABLE_MPI=ON \
  -DGHALO_ENABLE_HIP=ON \
  -DGHALO_ENABLE_GPU_AWARE_MPI=ON \
  -DCMAKE_CXX_COMPILER=CC \
  -DCMAKE_HIP_COMPILER=amdclang++

cmake --build build-frontier-mpi-hip
```

One node, one rank per GPU:

```sh
srun -N 1 -n 8 --ntasks-per-gpu=1 \
  ./build-frontier-mpi-hip/ghalo \
  --backend mpi-hip \
  --device-map local-rank \
  --validate
```

Two nodes, one rank per GPU:

```sh
srun -N 2 -n 16 --ntasks-per-gpu=1 \
  ./build-frontier-mpi-hip/ghalo \
  --backend mpi-hip \
  --device-map local-rank \
  --validate
```

For short smoke testing:

```sh
srun -N 1 -n 4 --ntasks-per-gpu=1 \
  ./build-frontier-mpi-hip/ghalo \
  --backend mpi-hip \
  --device-map local-rank \
  --validate \
  --target-seconds 0.1
```

These examples rely on Slurm setting an appropriate GPU visibility environment
for each rank. If `ROCR_VISIBLE_DEVICES` is unset or exposes an unexpected
device set, the backend should report that in its startup metadata and fail if
the rank-to-device mapping is invalid.

## Portability Notes

The design should not depend on Frontier-specific names except in examples.

Portable assumptions:

- MPI is initialized outside individual backend exchange logic.
- HIP is optional and only used in HIP-enabled targets.
- Device buffers are passed directly to MPI in the GPU-aware backend.
- The backend validates that direct device-buffer MPI works before timing.
- Scheduler-provided GPU visibility is treated as configuration, not as a
  hard-coded rule.

Cluster-specific documentation should describe required modules, compiler
wrappers, GPU visibility behavior, and environment variables for each system.

## Output Metadata

The GPU-aware backend should extend result metadata when possible:

- backend: `MPIHIPBackend`;
- memory location: `device`;
- HIP runtime version;
- selected device ordinal;
- device name;
- visible device count;
- local rank;
- `ROCR_VISIBLE_DEVICES`, if set;
- GPU-aware MPI requested/enabled status;
- validation enabled/disabled;
- validation result.

If the current output structures cannot represent these fields cleanly, add
metadata in a backward-compatible way. Do not change the meaning of existing
Version 0 fields.

## Risks and Mitigations

- **MPI silently stages through host memory.** Mitigate by documenting required
  runtime configuration, probing device-buffer communication, and recording
  metadata. Full detection may be MPI-implementation-specific.
- **Rank-to-GPU mapping is wrong.** Mitigate with shared-memory local rank,
  device metadata, `ROCR_VISIBLE_DEVICES` reporting, and validation.
- **GPU work races MPI.** Mitigate with explicit HIP synchronization before MPI
  consumes send buffers and before device work consumes receive buffers.
- **Validation changes timings.** Keep validation outside the timed loop.
- **CMake breaks macOS development.** Keep HIP disabled by default and isolate
  HIP targets behind feature gates.

## Ordered Implementation Plan

1. Add CMake option `GHALO_ENABLE_GPU_AWARE_MPI`, gated on MPI and HIP.
2. Add HIP RAII helpers for device selection, error checking, and device
   buffers in HIP-only targets.
3. Add command-line parsing for `--backend`, `--device-map`, and `--validate`
   without changing default CPU/MPI behavior.
4. Add `MPIHIPBackend` as a separate backend implementing the existing
   `Backend` interface.
5. Implement rank-to-GPU mapping using MPI shared-memory local rank and HIP
   device discovery.
6. Allocate device-resident HALO buffers with `hipMalloc`.
7. Implement device-side initialization and copy semantics needed by the HALO
   exchange.
8. Pass HIP device pointers directly to MPI and preserve the existing
   send/receive ordering.
9. Add untimed validation that verifies expected neighbor data using compact
   host copies of validation regions.
10. Add startup checks for HIP devices, mapping validity, HIP errors, and
    direct device-buffer MPI capability.
11. Extend output metadata in a backward-compatible way.
12. Validate on Frontier with one-node and two-node one-rank-per-GPU runs before
    considering larger scaling tests.
