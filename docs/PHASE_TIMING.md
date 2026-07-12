# Phase Timing

Phase timing is an optional diagnostic mode for GPU-resident backends. It helps
attribute the time spent inside one halo exchange to communication, device
copies, and synchronization boundaries.

Phase timing does not replace the primary gHALO metric. The authoritative
benchmark result remains the maximum average wall-clock time per complete halo
exchange across all ranks.

## Why It Exists

The GPU-aware MPI backend intentionally preserves the HALO exchange semantics:

1. copy previous east-west output into the north-south input buffer;
2. exchange north-south halo data;
3. copy north-south output into the east-west input buffer;
4. exchange east-west halo data.

On GPU systems, the cost of this sequence may be dominated by different
components on different machines: GPU-aware MPI transfers, device-to-device
copies, or synchronization required at MPI/HIP visibility boundaries. Phase
timing provides a first diagnostic view of those costs without changing the
benchmark algorithm.

## Enabling Phase Timing

Use:

```sh
ghalo --backend mpi-hip --phase-timing
```

Example Frontier run through the portable workflow:

```sh
GHALO_SYSTEM_NAME=frontier scripts/run.sh \
  --backend mpi-hip \
  --nodes 2 \
  --ranks 16 \
  --ranks-per-node 8 \
  --target-seconds 3 \
  --validate \
  --phase-timing \
  --label phase-timing
```

If `--phase-timing` is requested for a backend that does not support it, gHALO
fails clearly instead of emitting fabricated values. The CPU MPI backend does
not currently implement internal phase timing.

## Measured Phases

The initial implementation uses `MPI_Wtime` because the measured boundaries
include MPI calls and host-side HIP synchronization calls. HIP events alone are
not sufficient for phases that include MPI.

For each timed exchange iteration, the `mpi-hip` backend records:

- `input_device_copy`: the existing visibility synchronization before reading
  `hoew`, plus the device-to-device copy from `hoew` to `hins`.
- `north_south_mpi`: the north-south `N` and `2N` GPU-aware MPI exchanges.
- `north_south_sync`: the existing synchronization after north-south MPI
  receives into `hons`, before HIP reads `hons`.
- `transpose_device_copy`: the device-to-device copy from `hons` to `hiew`.
- `transpose_copy_sync`: the existing synchronization after the `hons` to
  `hiew` copy, before east-west MPI reads `hiew`.
- `east_west_mpi`: the east-west `N` and `2N` GPU-aware MPI exchanges.
- `east_west_sync`: reserved for an explicit synchronization after east-west
  MPI. The current timed exchange does not add such a synchronization, so this
  phase is reported as zero.

The implementation does not add synchronization solely for measurement. It
measures synchronization calls that already exist for correctness.

For each timed exchange iteration, the `rccl` backend records:

- `north_south_communication`: grouped RCCL sends and receives for the
  north-south `N` and `2N` segments.
- `north_south_sync`: the stream synchronization after north-south RCCL.
- `transpose_copy`: the device-to-device copy from `hons` to `hiew`.
- `transpose_sync`: the stream synchronization after the transpose copy.
- `east_west_communication`: grouped RCCL sends and receives for the east-west
  `N` and `2N` segments.
- `east_west_sync`: the stream synchronization after east-west RCCL.

The initial RCCL implementation intentionally keeps correctness-first
`hipStreamSynchronize` calls at the three visibility boundaries. These
synchronizations may later be replaced by a lower-overhead stream-aware
mechanism, but they must not be removed merely to improve benchmark numbers.

## Aggregation

For each halo size and phase, gHALO:

1. accumulates local elapsed time over the timed iterations only;
2. divides by the timed iteration count to get a local average per exchange;
3. reduces those local averages with `MPI_MAX` across ranks.

This follows the gHALO benchmark philosophy: the slowest participant determines
application-visible progress.

The output also reports:

- `phase_sum_seconds`: the sum of the reduced phase averages;
- `total_exchange_seconds`: the primary maximum average complete-exchange
  time;
- `total_minus_sum_of_phase_maxima_seconds`:
  `total_exchange_seconds - phase_sum_seconds`.

Small positive or negative total-minus-phase-sum values are expected. The
complete exchange and the individual phases are timed separately, and phase
timings are independently reduced with `MPI_MAX`. gHALO does not adjust phase
values to force the sum to equal the total.

## Output

When phase timing is disabled, console, CSV, and JSON output keep their normal
shape.

When phase timing is enabled:

- console output prints the normal result table first, then a second phase
  table in microseconds;
- CSV output adds phase columns such as
  `phase_north_south_communication_seconds` and
  `phase_total_minus_sum_of_phase_maxima_seconds`;
- JSON output adds a per-result `phase_timing` object and metadata describing
  the timing source and aggregation policy.

The older MPI-HIP-specific field names, such as `north_south_mpi_seconds`, are
retained for compatibility. RCCL communication is serialized with neutral
`*_communication_seconds` names so analysis and reporting do not mislabel RCCL
transfers as MPI.

## Performance Caution

Phase timing is diagnostic. It adds several `MPI_Wtime` calls to each timed
exchange and may perturb very small-message latency. Use normal, non-phase
timed runs for authoritative benchmark numbers. Use phase-timed runs to
understand relative costs and guide deeper investigation.
