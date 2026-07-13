# RCCL Backend Design

This document defines the architecture for the experimental gHALO RCCL backend.
Stage C is implemented as a complete two-dimensional RCCL halo exchange. Stage
B remains available as a north/south-only debugging path.

Frontier and Borg GPU backend workflows default to ROCm 6.4.2. New `mpi-hip`
and `rccl` comparison runs can therefore use the same ROCm stack. The RCCL OFI
network plugin is loaded only for `rccl`; `mpi-hip` uses Cray MPICH GPU-aware
MPI and does not load `rccl-net-plugin/1.0`.

## Purpose And Motivation

The RCCL backend will allow gHALO to compare GPU-resident halo exchange over
RCCL point-to-point operations against the existing GPU-aware MPI path. The goal
is diagnostic: determine whether a stream-oriented RCCL implementation can
reduce some of the fixed synchronization overhead observed in the `mpi-hip`
backend while preserving the same HALO benchmark semantics.

RCCL is not assumed to be faster. It may be worse for small messages, may expose
different transport behavior, and may have different operational constraints on
Frontier and Borg.

## Responsibility Split

MPI remains responsible for process-level coordination:

- `MPI_Init` and `MPI_Finalize` ownership, consistent with the current
  `MPIEnvironment` path.
- Global rank and world size.
- Periodic Cartesian topology construction.
- Node-local rank discovery.
- GPU selection coordination.
- RCCL unique-ID broadcast.
- Benchmark barriers.
- Maximum-rank timing reductions.
- Validation error reduction and reporting.

RCCL is responsible only for GPU data movement:

- Device-to-device point-to-point communication.
- Grouped send and receive operations.
- Stream-ordered completion of communication work.

The benchmark core, output writers, and analysis tools must not depend on RCCL.

## Communicator Initialization

The intended execution model is one MPI rank per GPU or GPU compute device and
one RCCL rank per MPI rank.

Initialization sequence:

1. Initialize MPI through the existing `MPIEnvironment`.
2. Create the same periodic Cartesian communicator used by the MPI backends.
3. Split `MPI_COMM_WORLD` with `MPI_COMM_TYPE_SHARED` to obtain node-local rank.
4. Select a HIP device using the existing local-rank mapping policy.
5. Rank 0 calls `rcclGetUniqueId` or the RCCL-provided NCCL-compatible
   equivalent.
6. Broadcast the unique ID with `MPI_Bcast`.
7. Create one explicit HIP stream owned by the backend.
8. Call `rcclCommInitRank` or the RCCL-provided NCCL-compatible equivalent.

Destruction order should be the reverse of ownership:

1. complete or synchronize outstanding stream work;
2. destroy the RCCL communicator;
3. destroy the HIP stream;
4. free MPI communicators owned by the backend.

Every RCCL, HIP, and MPI error must produce a useful message. Destructors should
use noexcept cleanup paths and must not throw.

## GPU Mapping

The RCCL backend should reuse the `mpi-hip` local-rank-to-HIP-device policy:

```text
device_index = local_rank % visible_device_count
```

Oversubscription detection should be preserved. In normal one-rank-per-GPU
operation, `local_size > visible_device_count` should fail before benchmark
timing starts unless an explicit oversubscription mode is added later.

Schedulers may restrict visibility through `ROCR_VISIBLE_DEVICES`. The backend
should validate the visible device count and record rank, local rank, hostname,
selected HIP device, and device name in output metadata. It should not introduce
new, unvalidated Slurm GPU-binding flags.

## Halo Exchange Mapping

The RCCL exchange must preserve the existing two-stage HALO semantics.

North/south stage:

- send `N` words south and receive `N` words from north;
- send `2N` words north and receive `2N` words from south;
- issue compatible grouped RCCL operations across all ranks.

Intermediate copy:

- preserve the existing `hons_` to `hiew_` logic;
- use HIP stream-aware device operations where practical;
- do not move the data through host memory.

East/west stage:

- send `N` words west and receive `N` words from east;
- send `2N` words east and receive `2N` words from west;
- preserve the existing neighbor mapping and message meaning.

The implementation must not silently fall back to `mpi-hip`.

Stage B implements only the north/south stage. It remains available for
debugging through:

```sh
./ghalo --backend rccl --rccl-stage-b --validate
```

Without `--rccl-stage-b`, `--backend rccl` runs the full Stage C exchange:
north/south, intermediate device copy, and east/west. Stage B does not emit
normal benchmark timing results.

## Stream Model

The implementation uses one explicit HIP stream owned by the RCCL backend.
RCCL operations and required HIP copies are issued on that stream.

The intended final design should avoid `hipDeviceSynchronize` in the timed path
where stream ordering is sufficient. Correctness-first stream synchronization
may be used initially, but each host synchronization point must be documented.
RCCL work must be complete before a timed exchange is considered complete.

Two synchronization modes are available:

- `conservative`: the default correctness reference. It enqueues north/south
  RCCL work, synchronizes the stream, enqueues the intermediate device copy,
  synchronizes the stream, enqueues east/west RCCL work, and synchronizes the
  stream again.
- `stream-ordered`: experimental. It enqueues the input copy, north/south RCCL
  work, intermediate device copy, and east/west RCCL work on the same backend
  stream, then performs one final stream synchronization.

The stream-ordered mode depends on all HIP copies and RCCL send/receive calls
using the same non-default backend stream, and on RCCL honoring stream ordering
for the point-to-point API. It must be validated for every topology edge case
before its timings are interpreted.

RCCL phase timing is implemented as a diagnostic mode. Conservative mode reports
communication, copy, and synchronization phases. Stream-ordered mode reports
host enqueue phases plus the one final stream synchronization; enqueue phases
are not communication-completion times.

## Message Ordering And Deadlock Safety

RCCL point-to-point operations should use `rcclGroupStart` and `rcclGroupEnd`
or the RCCL-provided NCCL-compatible equivalents. All ranks must issue
compatible send and receive operations in the same logical exchange stage.

The implementation must handle:

- 1x1 topologies where every neighbor is self;
- 1x2 and 2x1 topologies where one Cartesian dimension collapses;
- 2x2 periodic topologies where opposite directions may map to the same rank;
- larger topologies where all directions are distinct.

The backend must not assume that north, south, east, and west are four distinct
physical ranks. Self-neighbor behavior must be validated explicitly before it
is trusted in the timed path.

## Validation Strategy

RCCL validation should reuse the current deterministic patterned data strategy:

- initialize each directional send region with rank- and direction-specific
  values;
- run untimed exchanges;
- copy validation data back to host only after communication;
- compare received values against the expected neighbor and direction;
- reduce validation errors across MPI ranks;
- fail collectively with rank, topology, neighbor, expected value, and actual
  value.

Validation must cover 1x1, 1x2, 2x1, 2x2, 2x4, 4x4, multi-node, Frontier, and
Borg cases in both synchronization modes. It must remain outside the timed loop.

## Timing Semantics

The primary metric remains the complete HALO-compatible exchange time:

- initialization, allocation, validation, and warm-up are excluded;
- MPI barriers bracket timed regions;
- elapsed time is reduced with `MPI_MAX`;
- the reported value is the maximum average wall-clock time across ranks.

RCCL communication and stream work required for the exchange must be complete
before timing stops. Optional RCCL phase timing is diagnostic and does not
replace the primary metric. Each phase is locally averaged and independently
reduced with `MPI_MAX`, matching the MPI-HIP phase timing aggregation model.
The phase sum is not forced to match the total exchange maximum.

`--rccl-sync-mode conservative` is the default. Use
`--rccl-sync-mode stream-ordered` only for experimental validation and
comparison runs. Stage B remains a conservative north/south-only debugging path;
stream-ordered mode applies only to the full RCCL backend.

## Error Handling

RCCL calls should be wrapped in a focused helper that reports:

- function name;
- RCCL error code;
- RCCL error string when available;
- MPI world rank;
- node-local rank;
- selected HIP device.

HIP and MPI calls should follow the same philosophy. Fatal failures should be
coordinated with MPI where practical so one rank does not silently exit while
others continue waiting.

## Known Risks

- RCCL point-to-point support and behavior may vary by RCCL version.
- Self-send behavior must not be assumed correct without validation.
- Communicator initialization may expose large-scale sensitivity.
- Stream synchronization mistakes can produce stale or incomplete device data.
- Frontier transport or plugin behavior may differ from GPU-aware MPI.
- Small-message performance may be worse than `mpi-hip`.

## Staged Implementation Plan

Stage A: build detection and backend registration.

- Add optional RCCL detection.
- Add backend scaffolding.
- Register `--backend rccl` only when compiled with RCCL.
- Do not provide a runnable exchange.

Stage B: one-dimensional north/south correctness.

- Initialize communicator and stream.
- Validate north/south `N` and `2N` exchanges.
- Verify self-neighbor and duplicate-neighbor behavior.
- Implemented as validation-only behind `--rccl-stage-b`.
- Does not implement intermediate transpose, east/west exchange, full
  benchmark timing, phase timing, performance comparison, or scaling claims.

Stage C: full two-stage halo correctness.

- Add intermediate HIP copy.
- Add east/west exchange.
- Validate 1x1, 1x2, 2x2, and multi-node cases.
- Implemented as the default `--backend rccl` path.
- Uses correctness-first stream synchronization after north/south RCCL, after
  the intermediate device copy, and after east/west RCCL.
- Does not claim synchronization optimization, persistent/alternate RCCL
  strategies, scaling comparison against MPI-HIP, or production performance.

Stage D: phase timing and comparison.

- Add RCCL diagnostic phase timing. This is implemented for the full Stage C
  path with `--phase-timing`.
- Compare complete-exchange and phase behavior against `mpi-hip`.

Stage E: Borg scaling, then Frontier scaling.

- Validate Borg with native Borg builds, and use `GHALO_BUILD_SYSTEM_ALIAS`
  only for explicit cross-system artifact tests.
- Scale on Frontier after correctness is established.
