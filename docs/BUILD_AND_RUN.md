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

Results are stored by system, ROCm version, category, and timestamp:

```text
results/
  <system-name>/
    rocm-<version>/
      <category>/
        <timestamp>_<backend>_<label>/
```

GPU runs normalize the ROCm version as values such as `rocm-6.4.2`. CPU-only
runs use `rocm-none`. Supported categories are `validation`, `scaling`,
`repeatability`, and `phase-timing`.

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
It implements diagnostic RCCL phase timing, but does not yet claim
synchronization optimization or production performance comparisons.

Frontier and Borg GPU backends default to ROCm 6.4.2:

- `mpi-hip`: loads `rocm/6.4.2`, sets `MPICH_GPU_SUPPORT_ENABLED=1`, and does
  not load `rccl-net-plugin/1.0`.
- `rccl`: loads `rocm/6.4.2`, loads `rccl-net-plugin/1.0`, and sets
  `RCCL_ROOT="$ROCM_PATH"`.

Older MPI-HIP result sets collected with ROCm 6.2.4 remain valid historical
measurements, but they are a different software configuration from new ROCm
6.4.2 runs.

The default synchronization mode is:

```sh
--rccl-sync-mode conservative
```

An experimental stream-ordered mode is available for full RCCL runs:

```sh
--rccl-sync-mode stream-ordered
```

Stream-ordered mode enqueues the input copy, north/south RCCL work,
intermediate device copy, and east/west RCCL work on the same backend HIP
stream, then performs one final stream synchronization. It must be validated
for each topology before timings are interpreted. Stage B remains a
conservative north/south-only debugging path.

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
--min-halo <N>
--max-halo <N>
--halo-multiplier <N>
--samples-per-halo <N>
--record-iteration-times
--iteration-stall-threshold-us <microseconds>
--validate
--phase-timing
--rccl-stage-b
--rccl-sync-mode conservative|stream-ordered
--category validation|scaling|repeatability|phase-timing
--label <text>
--extra-srun-args "<args>"
```

The run script uses:

```text
builds/<system>/<backend>/ghalo
```

and writes each run to a unique directory:

```text
results/<system>/rocm-<version>/<category>/<timestamp>_<backend>_<label>/
```

The optional label is sanitized for safe filenames. If the label already begins
with the backend name, that duplicate prefix is removed so names such as
`rccl_rccl_conservative` are not produced. If a generated directory already
exists, the workflow appends a numeric suffix rather than overwriting it. The
result directory is printed before launching the benchmark.

`results/` is the canonical location for benchmark artifacts intended for
analysis. Regression harnesses such as `run_loop.sh` may keep combined logs,
build logs, `ldd` output, shell-test output, and standalone smoke-test output
under `test-logs/`, but MPI-HIP and RCCL benchmark runs should still be
launched through `scripts/run.sh` so `ghalo.csv`, `ghalo.json`, and metadata are
written under `results/`.

Each run records:

```text
binary-info.txt
command.txt
environment.txt
exit-status.txt
ghalo.csv
ghalo.json
iteration-times.csv
git.txt
hostname.txt
modules.txt
result-metadata.txt
slurm-job.txt
stderr.txt
stdout.txt
submission.txt
system-resolution.txt
```

The current gHALO CLI supports `--csv` and `--json`, so the workflow writes
structured output directly into the result directory. The script launches the
benchmark without pipelines and returns the benchmark exit status.
`iteration-times.csv` is written only when per-iteration timing diagnostics are
enabled.

The default halo sweep remains `2, 4, ..., 1024`. Short validation runs can keep
the default sweep while reducing timing duration:

```sh
GHALO_SYSTEM_NAME=frontier scripts/run.sh \
  --backend mpi-hip \
  --nodes 1 \
  --ranks 8 \
  --ranks-per-node 8 \
  --target-seconds 0.1 \
  --validate \
  --category validation \
  --label validation-1node
```

Extended sweeps do not require source edits:

```sh
GHALO_SYSTEM_NAME=frontier scripts/run.sh \
  --backend rccl \
  --nodes 64 \
  --ranks 512 \
  --ranks-per-node 8 \
  --min-halo 2 \
  --max-halo 262144 \
  --halo-multiplier 2 \
  --target-seconds 3 \
  --category scaling \
  --label 64node-extended
```

## Persistent-Process Repeatability

Use `--samples-per-halo <N>` to collect multiple independent timed samples for
each halo size within a single gHALO executable. For each halo, gHALO configures
the backend once, performs the usual warmup and calibration once, fixes the
iteration count once, and then records `N` timed samples without reinitializing
MPI, HIP, RCCL, streams, communicators, or the backend. This mode is intended to
distinguish process-lifetime state from invocation-to-invocation state.

The default is `--samples-per-halo 1`, which preserves normal single-sample
behavior. CSV and JSON output include `sample_index` and `sample_count`.

Example Borg RCCL conservative run for halo 128 with 50 samples:

```sh
GHALO_SYSTEM_NAME=borg scripts/run.sh \
  --backend rccl \
  --rccl-sync-mode conservative \
  --nodes 1 \
  --ranks 8 \
  --ranks-per-node 8 \
  --min-halo 128 \
  --max-halo 128 \
  --samples-per-halo 50 \
  --target-seconds 0.1 \
  --validate \
  --category repeatability \
  --label conservative-halo128-50samples
```

Example Borg RCCL stream-ordered run for halo 64 with 50 samples:

```sh
GHALO_SYSTEM_NAME=borg scripts/run.sh \
  --backend rccl \
  --rccl-sync-mode stream-ordered \
  --nodes 1 \
  --ranks 8 \
  --ranks-per-node 8 \
  --min-halo 64 \
  --max-halo 64 \
  --samples-per-halo 50 \
  --target-seconds 0.1 \
  --validate \
  --category repeatability \
  --label stream-ordered-halo64-50samples
```

## Iteration Timing Diagnostics

Use `--record-iteration-times` to capture per-iteration wall-clock timing for
measured benchmark iterations. This mode is intended to diagnose rare slow
samples while keeping MPI, HIP, RCCL, streams, communicators, and the selected
backend alive for the duration of the executable.

When enabled, gHALO records one diagnostic observation per measured iteration
after warmup and calibration. For each observation, it records the maximum
iteration duration across ranks and the rank that observed that maximum when
the backend can provide it. The diagnostic records are written to
`iteration-times.csv` in the result directory, while the normal `ghalo.csv` and
`ghalo.json` summary fields retain their usual meaning.

`--iteration-stall-threshold-us <VALUE>` filters the records written to
`iteration-times.csv`. A missing value or `0` emits every measured iteration.
Positive values emit only iterations whose global maximum duration is at least
the threshold. Summary counts in JSON and `result-metadata.txt` record total
observed iterations, emitted records, stall count, and threshold.

This mode intentionally adds diagnostic overhead after each timed sample to
identify the global maximum and rank for the sample's measured iterations. It
does not add synchronization or timed collectives inside the exchange loop, and
it is opt-in so ordinary benchmark runs are unaffected.

Borg RCCL conservative example for halo 128, 100 persistent samples, and a
1000 us stall threshold:

```sh
GHALO_SYSTEM_NAME=borg scripts/run.sh \
  --backend rccl \
  --rccl-sync-mode conservative \
  --nodes 1 \
  --ranks 8 \
  --ranks-per-node 8 \
  --min-halo 128 \
  --max-halo 128 \
  --samples-per-halo 100 \
  --target-seconds 0.1 \
  --validate \
  --record-iteration-times \
  --iteration-stall-threshold-us 1000 \
  --category repeatability \
  --label conservative-halo128-100samples-iter
```

MPI-HIP control run using the same placement and halo size:

```sh
GHALO_SYSTEM_NAME=borg scripts/run.sh \
  --backend mpi-hip \
  --nodes 1 \
  --ranks 8 \
  --ranks-per-node 8 \
  --min-halo 128 \
  --max-halo 128 \
  --samples-per-halo 100 \
  --target-seconds 0.1 \
  --validate \
  --record-iteration-times \
  --iteration-stall-threshold-us 1000 \
  --category repeatability \
  --label mpi-hip-halo128-100samples-iter
```

Borg RCCL stream-ordered example for halo 64 with 50 persistent samples:

```sh
GHALO_SYSTEM_NAME=borg scripts/run.sh \
  --backend rccl \
  --rccl-sync-mode stream-ordered \
  --nodes 1 \
  --ranks 8 \
  --ranks-per-node 8 \
  --min-halo 64 \
  --max-halo 64 \
  --samples-per-halo 50 \
  --target-seconds 0.1 \
  --validate \
  --record-iteration-times \
  --iteration-stall-threshold-us 1000 \
  --category repeatability \
  --label stream-ordered-halo64-50samples-iter
```

Use `--record-iteration-phase-times` with `--record-iteration-times` to record
phase timing for only those measured iterations that pass the existing
`--iteration-stall-threshold-us` filter. gHALO writes these records to
`iteration-phase-times.csv` using a normalized schema: one row per emitted
iteration and phase, with `phase_name`, `phase_seconds`, and
`phase_max_rank`.

The phase diagnostic records local phase durations during the measured
iteration, then performs max-rank reductions after the complete timed sample.
This avoids adding global reductions inside the timed exchange sequence. The
diagnostic still adds overhead after each sample and should be used for
diagnosis rather than publication timing. Phase maxima may come from different
ranks, so the sum of phase maxima is not expected to equal the total iteration
maximum.

Example Borg RCCL conservative phase diagnostic:

```sh
scripts/run.sh \
  --system borg \
  --backend rccl \
  --nodes 8 \
  --ranks 64 \
  --ranks-per-node 8 \
  --target-seconds 0.1 \
  --min-halo 128 \
  --max-halo 128 \
  --samples-per-halo 100 \
  --record-iteration-times \
  --record-iteration-phase-times \
  --iteration-stall-threshold-us 1000 \
  --validate \
  --phase-timing \
  --rccl-sync-mode conservative \
  --category repeatability \
  --label borg-iteration-phase-rccl-conservative-halo128
```

Example Frontier RCCL conservative phase diagnostic:

```sh
scripts/run.sh \
  --system frontier \
  --backend rccl \
  --nodes 8 \
  --ranks 64 \
  --ranks-per-node 8 \
  --target-seconds 0.1 \
  --min-halo 128 \
  --max-halo 128 \
  --samples-per-halo 100 \
  --record-iteration-times \
  --record-iteration-phase-times \
  --iteration-stall-threshold-us 1000 \
  --validate \
  --phase-timing \
  --rccl-sync-mode conservative \
  --category repeatability \
  --label frontier-iteration-phase-rccl-conservative-halo128
```

Submit the packaged batch experiments with:

```sh
sbatch slurm/borg_iteration_phase_diagnostic.sbatch
sbatch slurm/frontier_iteration_phase_diagnostic.sbatch
```

Analyze phase diagnostics with:

```sh
python3 tools/analyze_iteration_stalls.py \
  --system borg \
  --rocm-version 6.4.2 \
  --category repeatability \
  --label-filter iteration-phase \
  --threshold-us 1000 \
  --output-dir analysis/borg-iteration-phase
```

The analyzer reports the dominant phase by iteration count, cumulative phase
time, median phase time, maximum phase time, and phase max-rank frequency. For
RCCL conservative runs, compare north/south communication, north/south sync,
transpose copy/sync, east/west communication, and east/west sync. For
stream-ordered runs, compare north/south enqueue, transpose enqueue, east/west
enqueue, and final stream sync. For MPI-HIP, compare the available GPU-aware MPI
and synchronization phases.

### Borg Iteration Stall Diagnostic

The focused Borg stall diagnostic batch script compares RCCL conservative,
RCCL stream-ordered, and MPI-HIP inside the same 8-node allocation. Each case
uses one gHALO executable, 100 persistent samples for one halo size, validation,
phase timing, and per-iteration timing records filtered at 1000 us.

Submit the job from the repository root:

```sh
JOBID=$(sbatch slurm/borg_iteration_stall_diagnostic.sbatch | awk '{print $4}')
echo "${JOBID}"
```

The script represents these Slurm resources:

```text
#SBATCH -A VEN004
#SBATCH -p testing
#SBATCH --reservation=jlbeck.testing
#SBATCH -N 8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH -t 00:20:00
```

Monitor the job with:

```sh
squeue -j "${JOBID}"
scontrol show job "${JOBID}"
tail -f "batch-logs/borg-iteration-stall-${JOBID}.out"
```

The three result labels are:

```text
borg-iteration-stall-rccl-conservative-halo128
borg-iteration-stall-rccl-stream-halo64
borg-iteration-stall-mpi-hip-halo128
```

After completion, analyze the repeatability result bundles:

```sh
python3 tools/analyze_iteration_stalls.py \
  --system borg \
  --rocm-version 6.4.2 \
  --category repeatability \
  --label-filter borg-iteration-stall \
  --threshold-us 1000 \
  --output-dir analysis/borg-iteration-stall-${JOBID}
```

The analyzer writes:

```text
iteration-stall-report.md
iteration-stall-summary.csv
iteration-stall-summary.json
```

Interpretation:

- `F` in the sample sequence means a sample had no iteration above the
  threshold; `S` means at least one iteration exceeded it.
- One isolated bad iteration suggests a brief transient or rank-local outlier.
- Multiple bad iterations in one sample suggest a longer disruption inside the
  same persistent process.
- Nearly-all-slow samples suggest a sustained slow mode for that sample rather
  than a single catastrophic iteration.
- Compare RCCL conservative, RCCL stream-ordered, and MPI-HIP before assigning
  the cause to RCCL synchronization policy rather than placement or system
  noise.

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

To validate the stream-ordered experiment on the same placement:

```sh
GHALO_SYSTEM_NAME=borg scripts/run.sh \
  --backend rccl \
  --rccl-sync-mode stream-ordered \
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

Set `GHALO_BUILD_SYSTEM_ALIAS=<name>` only when intentionally testing
cross-system artifacts on compatible hardware, software, and filesystems. In
that case, `run.sh` still writes results under the active system, but resolves
the binary from the aliased build system:

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

For example, this explicitly runs on Borg while using Frontier build artifacts:

```text
active system: borg
build system:  frontier
binary:        builds/frontier/rccl/ghalo
results:       results/borg/rocm-<version>/<category>/<timestamp>_rccl_<label>/
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
--backend mpi|mpi-hip|rccl
--account <account>
--partition <partition>
--nodes <N>
--ranks <N>
--ranks-per-node <N>
--time <HH:MM:SS>
--target-seconds <seconds>
--min-halo <N>
--max-halo <N>
--halo-multiplier <N>
--samples-per-halo <N>
--record-iteration-times
--iteration-stall-threshold-us <microseconds>
--validate
--phase-timing
--rccl-stage-b
--rccl-sync-mode conservative|stream-ordered
--category validation|scaling|repeatability|phase-timing
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

### Frontier ROCm 6.4.2 Validation Workflow

This is the concise workflow for reproducing the Frontier validation campaign
from a login node or workflow checkout. It assumes the site batch script exists
at `slurm/validation.sbatch` and delegates execution to the repository scripts
described above.

1. Build the GPU backends:

   ```sh
   GHALO_SYSTEM_NAME=frontier scripts/build.sh --backend mpi-hip --clean --jobs 8
   GHALO_SYSTEM_NAME=frontier scripts/build.sh --backend rccl --clean --jobs 8
   ```

2. Submit validation:

   ```sh
   sbatch slurm/validation.sbatch
   ```

   The validation job should run MPI-HIP, RCCL conservative, and RCCL
   stream-ordered cases through `scripts/run.sh` or
   `scripts/run_validation_suite.sh` so result metadata and directory layout
   remain consistent.

3. Inspect results:

   ```text
   results/frontier/<rocm-version>/<suite>/<run>/
   ```

   For the ROCm 6.4.2 validation baseline, this is typically:

   ```text
   results/frontier/rocm-6.4.2/validation/<timestamp>_<backend>_<label>/
   ```

4. Generate reports:

   ```sh
   python3 tools/ghalo_analyze.py summarize \
     --include-metadata \
     --phase-timing \
     --format markdown \
     results/frontier/rocm-6.4.2/validation

   python3 tools/ghalo_analyze.py report \
     --style publication \
     --output-dir analysis/frontier-rocm-6.4.2-validation \
     results/frontier/rocm-6.4.2/validation

   python3 tools/ghalo_analyze.py plot \
     --kind latency \
     --output analysis/frontier-rocm-6.4.2-validation/plots/latency.png \
     results/frontier/rocm-6.4.2/validation
   ```

   The convenience wrapper is equivalent for the common report bundle:

   ```sh
   scripts/generate_analysis_report.sh \
     --input results/frontier/rocm-6.4.2/validation \
     --output analysis/frontier-rocm-6.4.2-validation
   ```

5. Reproduce the tagged baseline source:

   ```sh
   git fetch --tags
   git checkout frontier-rocm-6.4.2-validation-baseline
   ```

   After checking out the tag, rebuild before comparing new results against
   the baseline. Result directories are intentionally separate from Git tags;
   keep the result bundle paths and `provenance.json` with any report.

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
filesystem as Frontier. Build resolution defaults to the active system, so the
Borg profile uses Borg-native build trees by default:

```text
builds/borg/mpi/ghalo
builds/borg/mpi-hip/ghalo
results/borg/rocm-<version>/<category>/<timestamp>_mpi_<label>/
results/borg/rocm-<version>/<category>/<timestamp>_mpi-hip_<label>/
```

This keeps build provenance and result histories aligned with the active
system. To prepare a clean Borg environment for GPU builds and runs:

```sh
source scripts/setenv.borg
```

The helper purges existing modules, loads the Borg Cray programming
environment, loads `craype-accel-amd-gfx90a`, selects `rocm/6.4.2`, and exports
the standard gHALO Borg/ROCm metadata variables. It does not set user-specific
paths.

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

Concise RCCL mode comparison sequence for one node:

```sh
GHALO_SYSTEM_NAME=borg scripts/run.sh \
  --backend rccl \
  --rccl-sync-mode conservative \
  --nodes 1 \
  --ranks 8 \
  --ranks-per-node 8 \
  --target-seconds 3 \
  --validate \
  --phase-timing \
  --label rccl-conservative

GHALO_SYSTEM_NAME=borg scripts/run.sh \
  --backend rccl \
  --rccl-sync-mode stream-ordered \
  --nodes 1 \
  --ranks 8 \
  --ranks-per-node 8 \
  --target-seconds 3 \
  --validate \
  --phase-timing \
  --label rccl-stream-ordered

GHALO_SYSTEM_NAME=borg scripts/run.sh \
  --backend mpi-hip \
  --nodes 1 \
  --ranks 8 \
  --ranks-per-node 8 \
  --target-seconds 3 \
  --validate \
  --phase-timing \
  --label mpi-hip-reference
```

Treat these as bring-up comparisons only. Current RCCL and MPI-HIP workflows
both use ROCm 6.4.2 by default, but placement effects can dominate small
messages and multiple repeats are required before drawing performance
conclusions. Historical MPI-HIP baselines collected with ROCm 6.2.4 should be
kept separate in analysis.

The Borg profile:

- uses the Cray C++ wrapper `CC`;
- uses `srun` with `-N`, `-n`, and `--ntasks-per-node` when supplied;
- uses `gfx90a` for HIP builds;
- loads `craype-accel-amd-gfx90a` and `rocm/6.4.2` for `mpi-hip`;
- loads `rccl-net-plugin/1.0` only for `rccl`;
- sets `MPICH_GPU_SUPPORT_ENABLED=1` only for `mpi-hip`;
- unsets `MPICH_GPU_SUPPORT_ENABLED` for CPU MPI;
- avoids unvalidated GPU-binding options;
- does not hard-code Cray MPICH paths.

Intentional cross-system artifact testing remains available through
`GHALO_BUILD_SYSTEM_ALIAS`. For example, to run on Borg with explicitly selected
Frontier build artifacts:

```sh
export GHALO_BUILD_SYSTEM_ALIAS=frontier
GHALO_SYSTEM_NAME=borg scripts/run.sh --backend mpi-hip
```

Without that explicit override, Borg runs resolve binaries from:

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
runs. Treat timing results as bring-up data until RCCL synchronization policy
and performance behavior have been studied.

## Migrating Flat Results

Older gHALO runs used the flat layout
`results/<system>/<timestamp>_<backend>_<label>/`. They remain analyzable, but
they can be copied into the versioned hierarchy with:

```sh
python3 tools/migrate_results.py --dry-run results/frontier
python3 tools/migrate_results.py --apply results/frontier
```

The helper infers system, ROCm version, category, backend, and label from
available metadata and falls back conservatively when metadata is missing. It
copies bundles into the new tree, avoids destination collisions, removes
duplicated backend prefixes in destination names, and never deletes the source
directories automatically.
