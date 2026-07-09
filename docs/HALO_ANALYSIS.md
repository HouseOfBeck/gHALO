# HALO Benchmark Analysis

This document reverse engineers Alan Wallcraft's original HALO benchmark source
tree from `halo.tgz`. The goal is to preserve the benchmark methodology before
extending gHALO.

No gHALO benchmark behavior should be changed from this analysis alone. This
file captures the original benchmark philosophy, mechanics, and compatibility
points that a modern implementation should preserve.

## Source Tree Summary

The archive contains several implementations of the same benchmark idea:

- MPI-1 two-sided message passing
- MPI-2 one-sided `MPI_Get` variants
- SHMEM one-sided variants
- LAPI emulation of SHMEM-style behavior
- OpenMP shared-memory variants
- Co-array Fortran variants
- BSP variants
- platform-specific makefiles, launch scripts, and historical outputs

The most useful compatibility baseline is the MPI-1 implementation because it
is explicit, portable in concept, and includes several exchange algorithms:

- `HALO2A`: `MPI_Sendrecv`
- `HALO2B`: ordered blocking send/receive on alternating ranks
- `HALO2D`: nonblocking sends posted before receives
- `HALO2P`: persistent version of send-before-receive
- `HALO2E`: nonblocking receives posted before sends
- `HALO2Q`: persistent version of receive-before-send

The SHMEM, OpenMP, and MPI-2 variants confirm that the benchmark is meant to
compare communication methods while preserving a common halo pattern, topology,
timing loop, and worst-rank reporting methodology.

## Benchmark Purpose

HALO is a synthetic benchmark for nearest-neighbor halo exchange in a
two-dimensional domain decomposition. It models a common communication step in
finite-difference and stencil-style scientific applications, such as ocean
models.

The benchmark intentionally does not model full application computation. It
isolates the communication pattern that often limits scalability:

1. each process owns a tile,
2. each tile exchanges boundary data with neighboring tiles,
3. all tiles must wait for the exchange before advancing.

The original README explicitly frames HALO as an informal but practical test for
whether a machine and communication library are suitable for domain-decomposed
applications.

## Communication Pattern

Each process is placed on a logical 2-D grid with periodic wrap in both
directions. Every process has north, south, east, and west logical neighbors,
although small grids such as 2x2 can collapse opposite directions onto the same
physical neighbor.

For each halo size `N`, one exchange performs two ordered phases:

1. North-south exchange:
   - send `N` words south,
   - receive `N` words from north,
   - send `2N` words north,
   - receive `2N` words from south.

2. East-west exchange, after north-south data has arrived:
   - send `N` words west,
   - receive `N` words from east,
   - send `2N` words east,
   - receive `2N` words from west.

The order matters. The benchmark is not four independent neighbor messages
issued as a single fully overlapped operation. It models a staged halo update in
which north-south results are available before the east-west phase begins.

For the original MPI and SHMEM implementations, a word is a 4-byte Fortran
`REAL`. Each exchange therefore moves `3N` words in the north-south phase and
`3N` words in the east-west phase per process, for `6N` words total per process
per full halo exchange.

## Message Sizes

The benchmark tests powers of two:

```text
N = 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
```

For each `N`, the benchmark uses buffers of length `3N` words:

- first segment: `N` words,
- second segment: `2N` words.

With 4-byte words, per-message payloads are:

```text
N-word message:   4N bytes
2N-word message:  8N bytes
full exchange:   24N bytes per process
```

The historical `README.times` treats `N=2` as the latency point and `N=128` as a
representative tile edge length for comparing wall time.

## Process Topology

The original `GRID2D` routine maps ranks into a row-major 2-D periodic grid.

The grid factorization attempts to make the two dimensions approximately equal
while allowing first dimensions of the form:

```text
1 * 2^k
3 * 2^k
5 * 2^k
```

In the MPI-1 implementation:

1. if `NPES` is divisible by 25, start with `N1 = 5`;
2. else if `NPES` is divisible by 9, start with `N1 = 3`;
3. otherwise start with `N1 = 1`;
4. repeatedly double `N1` while the remaining factor is even and the grid is not
   already square or past square;
5. set `N2 = NPES / N1`.

For rank `MYPE`:

```text
i = MYPE mod N1
j = MYPE / N1
north = i + N1 * ((j + 1) mod N2)
south = i + N1 * ((j + N2 - 1) mod N2)
east  = ((i + 1) mod N1) + N1 * j
west  = ((i + N1 - 1) mod N1) + N1 * j
```

Periodic wrap is part of the benchmark. It makes small runs less physically
realistic but keeps every rank active and makes results more consistent with
large practical decompositions.

## Exchange Algorithm

The core algorithm is:

1. Copy simulated existing data into a north-south send buffer.
2. Exchange the first `N` words with one vertical neighbor.
3. Exchange the next `2N` words with the opposite vertical neighbor.
4. Copy received vertical halo data into an east-west send buffer.
5. Exchange the first `N` words with one horizontal neighbor.
6. Exchange the next `2N` words with the opposite horizontal neighbor.
7. Leave the final result in the east-west receive buffer for use by the next
   exchange.

The source does not allocate real 2-D arrays. It simulates array-to-buffer and
buffer-to-array movement using linear buffers:

- `HINS`: north-south input/send buffer
- `HONS`: north-south output/receive buffer
- `HIEW`: east-west input/send buffer
- `HOEW`: east-west output/receive buffer

The next exchange starts by copying from `HOEW` into `HINS`, so each iteration
has a data dependency that discourages the compiler from removing the work.

## Why Communication Buffers Are Used

The original README gives three reasons:

1. The benchmark simulates copying from an application array into a local halo
   buffer without needing to allocate full 2-D arrays.
2. User-level buffers allow MPI persistent communication requests to be used.
3. Buffers avoid compiler optimization problems around nonblocking calls.

There is also a benchmarking reason: explicit buffers make the communication
payload and layout simple, repeatable, and comparable across communication
libraries.

The original author notes that real applications may sometimes avoid local
user-level buffers, but that doing so is only worthwhile on very fast networks.
For benchmark compatibility, the buffer-based mode is essential.

## Timing Methodology

Each algorithm and halo size uses the same timing structure:

1. Run one unmeasured exchange for the current algorithm and `N`.
2. Synchronize all processes.
3. Time exactly five exchanges.
4. Reduce the five-exchange time using a global maximum.
5. Estimate the number of repetitions needed for about three seconds:

   ```text
   LREP = max(5, round(3.0 / (TALL / 5.0)))
   ```

6. Synchronize all processes again.
7. Time `LREP` exchanges.
8. Reduce the final elapsed time using a global maximum.
9. Report:

   ```text
   TALL / LREP
   ```

This makes the reported number the maximum, across all ranks, of the average
wall-clock time per full halo exchange.

## Synchronization

The benchmark synchronizes at two levels.

At the outer measurement level, all processes synchronize before timed regions.
MPI versions use `MPI_Barrier`; SHMEM versions use `SHMEM_BARRIER_ALL`; OpenMP
versions use OpenMP barriers.

Inside each halo exchange, synchronization depends on backend and algorithm:

- MPI `Sendrecv` synchronizes each pairwise send/receive operation naturally.
- MPI nonblocking variants use `MPI_Waitall` after posting four operations for
  each phase.
- MPI persistent variants use `MPI_Startall` and `MPI_Waitall`.
- MPI-2 one-sided variants use fence or post/start/complete/wait epochs.
- SHMEM variants use global barriers or pairwise ring-style synchronization.
- OpenMP variants use global barriers or custom pairwise barrier structures.

The internal synchronization is part of the operation being measured. The outer
barriers align ranks before the measurement begins; they are not included in
the timed halo loop.

## Why Startup Costs Are Excluded

The benchmark explicitly excludes one-time startup overhead because production
codes perform many halo exchanges after setup. Startup costs such as first-use
communication setup, persistent request creation, window creation, or cache and
runtime initialization should not dominate the reported per-exchange steady
state.

The unmeasured first exchange for each algorithm and `N` is especially important
for persistent and one-sided variants:

- MPI persistent requests are created on the first call for a new `N`.
- MPI-2 windows are created or recreated when `N` changes.
- communication libraries may perform first-use setup.
- caches and internal protocol state are warmed before measurement.

This is a compatibility point. A modern implementation should distinguish setup
time from steady-state exchange time and should report both only if clearly
labeled.

## Why Each Halo Size Runs About Three Seconds

The benchmark uses a short calibration run of five exchanges to estimate the
current per-exchange cost. It then chooses `LREP` so the final measured loop
lasts approximately three seconds.

This design reduces measurement noise:

- very fast small messages need many repetitions for stable timing;
- larger messages need fewer repetitions;
- every halo size receives roughly comparable wall-clock sampling time;
- timer granularity matters less than it would for a fixed small iteration
  count.

The `max(5, ...)` guard ensures that even slow cases run at least five exchanges.
The final duration is approximate because `LREP` is computed from a calibration
measurement and rounded to an integer.

## Why the Maximum Average Wall-Clock Time Is Reported

Halo exchange is a collective progress constraint even when implemented through
point-to-point communication. In a domain-decomposed application, the timestep
cannot continue until all ranks have received the halo data they need.

Reporting the fastest or average rank would hide stragglers, topology
imbalances, OS noise, and degraded links. The original benchmark therefore
reduces local elapsed times with a maximum and reports the slowest rank's
average exchange time.

This is central to HALO's diagnostic value. It measures the time the application
would actually pay.

## Essential Compatibility Points

A modern gHALO compatibility mode should preserve these behaviors:

- Logical 2-D process grid with periodic north, south, east, and west neighbors.
- Original grid factorization policy, or at least a mode that reproduces it.
- Halo sizes `N = 2..1024` by powers of two.
- 4-byte word payload compatibility mode.
- Two-phase exchange: north-south first, east-west second.
- Asymmetric `N` and `2N` messages in each phase.
- Full exchange definition of `6N` words per process.
- Explicit communication buffers and simulated copy steps.
- Unmeasured warmup/setup exchange before timing each algorithm and size.
- Five-exchange calibration loop.
- Approximately three-second final timed loop per halo size.
- Global maximum reduction of elapsed time before dividing by repetition count.
- Output that records algorithm, process count, halo size, and time per exchange.
- Backend algorithms treated as comparable implementations of the same exchange
  pattern, not as different benchmarks.

## Parts That Should Be Modernized

The following should not be copied literally:

- Fixed-form Fortran and common blocks.
- Hard-coded static buffer limits as the only supported configuration.
- Lack of correctness validation.
- Text-only output as the only result format.
- Backend-specific source duplication.
- Implicit use of `REAL*4` without explicit payload type metadata.
- Limited process topology reporting.
- No rank-pair timing, topology annotation, or diagnostic context.
- No separation between benchmark configuration, execution, and reporting.
- No machine-readable schema.
- No GPU-resident data model.
- No clear distinction between host-staged and device-resident transfers.
- Historical platform scripts and machine-specific launch assumptions.

Some historical behaviors should remain available as compatibility modes, even
if they are not the default modern mode.

## Recommended Modern Architecture

gHALO should separate benchmark semantics from communication backends.

Recommended layers:

```text
CLI / configuration
        |
Run plan and parameter sweep
        |
Topology model
        |
Halo pattern model
        |
Buffer model
        |
Timing and statistics
        |
Backend interface
        |
MPI / GPU-aware MPI / RCCL / UCX / future backends
        |
Host memory / HIP device memory
```

## Core Model

The portable core should define:

- rank topology and neighbor mapping;
- halo sizes and payload word sizes;
- exchange phase descriptions;
- repetition calibration policy;
- warmup/setup policy;
- timing reduction policy;
- result records;
- diagnostic metadata.

This core should build on macOS without MPI, HIP, ROCm, RCCL, or UCX.

## Backend Interface

Each backend should implement the same conceptual operation:

```text
setup(size, topology, buffer_model)
warmup()
exchange()
teardown()
metadata()
```

The timed loop should call only the steady-state `exchange()` operation.
Backend setup must occur outside timed compatibility measurements unless a run
mode explicitly asks to measure setup cost.

Backends should report:

- library and version where available;
- communication mode;
- memory location;
- synchronization policy;
- message sizes;
- rank topology;
- whether transfers are host-staged or device-resident.

## GPU-Native Modernization

For GPU systems, gHALO should preserve the original benchmark shape while
modernizing memory and communication paths:

- HIP device buffers as first-class buffers.
- Optional device-side packing and unpacking kernels.
- GPU-aware MPI mode using device pointers.
- Host-staged MPI mode for comparison.
- RCCL and UCX backends behind the same exchange semantics.
- Explicit stream, event, and synchronization policy metadata.
- Result fields that distinguish host timer, GPU event timing, and end-to-end
  application-visible time.

The compatibility timing metric should remain end-to-end wall-clock time across
ranks, because that is what the original benchmark reports and what applications
pay.

## Diagnostic Extensions

gHALO should extend HALO's philosophy into diagnostics:

- preserve the maximum-rank timing metric;
- additionally collect min, mean, median, and distribution summaries;
- optionally collect per-rank and per-neighbor timings;
- emit JSON and CSV;
- generate communication heat maps;
- detect asymmetric neighbor performance;
- annotate rank placement and hardware topology when available;
- compare host-staged, GPU-aware MPI, RCCL, and UCX modes under the same halo
  pattern.

These features should add diagnostic depth without changing the compatibility
definition of a HALO exchange.

## Implementation Cautions

- Do not optimize away the simulated copy steps in compatibility mode.
- Do not collapse north-south and east-west into one fully concurrent exchange
  when claiming HALO compatibility.
- Do not include backend setup in steady-state timing unless explicitly labeled.
- Do not report rank-average time as the primary compatibility metric.
- Do not assume local development machines have MPI, HIP, ROCm, RCCL, UCX, or
  GPUs.
- Do not make GPU-aware MPI the only path; host-staged and CPU-buffer baselines
  remain useful controls.

## Summary

The original HALO benchmark is small but carefully shaped. Its value comes from
measuring a communication pattern that real domain-decomposed applications pay:
periodic nearest-neighbor halo exchange, staged by dimension, with short and
moderate message sizes, repeated long enough for stable timing, and reported as
the slowest rank's average wall-clock time.

gHALO should preserve those semantics in a documented compatibility mode while
modernizing the implementation around portable C++20, isolated backends,
GPU-resident buffers, structured output, and richer diagnostics.
