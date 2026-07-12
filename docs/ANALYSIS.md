# Result Analysis

gHALO includes portable result-analysis tooling in
`tools/ghalo_analyze.py`. The tool is intended for post-processing result
directories produced by `scripts/run.sh` and `scripts/submit.sh`, while also
accepting explicit `ghalo.json` or `ghalo.csv` files.

Core textual, CSV, JSON, and Markdown analysis uses only the Python 3 standard
library. Plotting is optional and requires `matplotlib`; no pandas, NumPy,
SciPy, seaborn, Jupyter, ROCm, MPI, or GPU runtime is required for analysis.

## Inputs

The analyzer accepts:

- a result directory containing `ghalo.json` or `ghalo.csv`;
- an explicit `ghalo.json` file;
- an explicit `ghalo.csv` file;
- multiple shell-expanded result paths such as `results/frontier/*mpi-hip*`.

When both `ghalo.json` and `ghalo.csv` exist in a directory, `ghalo.json` is
preferred because it preserves nested topology, rank mapping, metadata, and
phase-timing structure.

Optional metadata files are read when present:

- `system-resolution.txt`
- `submission.txt`
- `environment.txt`
- `git.txt`
- `slurm-job.txt`
- `command.txt`
- `exit-status.txt`

Missing metadata files do not cause analysis failure. A nonzero
`exit-status.txt` is reported as a warning because the result directory may
contain useful partial output, but the run should not be treated as a clean
benchmark result without investigation.

## Summarize One Run

```sh
python3 tools/ghalo_analyze.py summarize \
  results/frontier/<64-node-run>
```

Use other output formats with:

```sh
python3 tools/ghalo_analyze.py summarize \
  --format csv \
  --output summary.csv \
  --include-metadata \
  results/frontier/<64-node-run>
```

The summary table reports the authoritative measured fields:

- halo length `N`;
- iteration count;
- maximum average wall-clock seconds;
- transferred bytes per rank;
- backend, algorithm, memory type, rank count, and Cartesian dimensions.

It also reports derived fields:

- microseconds, calculated from `max_average_seconds`;
- `per-rank effective transferred-byte rate`, calculated as
  `total_exchange_bytes_per_rank / max_average_seconds`;
- the same derived rate in GiB/s.

This derived rate is not physical link bandwidth, injection bandwidth,
aggregate network bandwidth, or achieved hardware bandwidth. It is a
per-rank effective transferred-byte rate based on the benchmark's measured
complete halo-exchange time.

## Compare Two Runs

```sh
python3 tools/ghalo_analyze.py compare \
  results/frontier/<first-64-node-run> \
  results/frontier/<repeat-64-node-run>
```

By default, both runs must contain the same halo sizes. For each size, the
comparison reports:

- time A and time B;
- absolute difference;
- observed percent difference;
- ratio `B/A`;
- bytes per rank for both runs;
- iteration counts;
- whether selected configuration metadata differs.

Use a regression threshold for automation:

```sh
python3 tools/ghalo_analyze.py compare \
  --threshold-percent 10 \
  --fail-on-regression \
  RUN_A RUN_B
```

`--fail-on-regression` returns nonzero only when run B is slower than run A by
more than the threshold. A two-run comparison is reported as an observed
difference, not a statistically significant regression.

## Aggregate Repeated Runs

```sh
python3 tools/ghalo_analyze.py aggregate \
  results/frontier/*frontier-64node*
```

For each halo size, aggregation reports:

- count;
- minimum and maximum;
- arithmetic mean;
- median;
- population standard deviation;
- sample standard deviation when count is greater than one;
- coefficient of variation;
- first and third quartiles.

Quartiles use an inclusive linear percentile rule over sorted data:
`position = (count - 1) * percentile`. If the position is between two samples,
the reported value is linearly interpolated.

Aggregates require compatible configurations by default. Compatibility includes
backend, rank count, node count when known, Cartesian dimensions, bytes per
rank for every halo size, RCCL synchronization mode, and phase-timing state. Use `--allow-mixed` with
`--group-by system`, `--group-by backend`, `--group-by nodes`, or
`--group-by ranks` when unlike runs should be grouped rather than rejected.
Aggregate output uses readable group labels while retaining complete grouping
metadata in CSV and JSON columns such as backend, ranks, nodes, Cartesian
dimensions, phase-timing state, RCCL synchronization mode, and bytes per rank
by halo size.

## Fixed-Message-Size Scaling

```sh
python3 tools/ghalo_analyze.py scaling \
  results/borg/*mpi-hip* \
  results/frontier/*mpi-hip*
```

Scaling analysis compares latency at fixed local halo message sizes across
node or rank counts. It reports system, nodes, ranks, ranks per node,
Cartesian dimensions, time in microseconds, ratio relative to a selected
baseline, percent increase relative to the baseline, and the per-rank
effective transferred-byte rate.

Baseline options:

```sh
--baseline smallest-nodes
--baseline PATH
--baseline-path PATH
--baseline-nodes N
```

The tool intentionally avoids the term strong-scaling efficiency here because
the local halo message size is held constant while the participating rank count
changes. Prefer terms such as fixed-message-size scaling, latency scaling, or
scale-dependent slowdown.

For scaling campaigns with a constant number of GPU ranks per node, node count
is usually the clearest x-axis because it exposes system-scale effects while
keeping the rank layout readable. Rank count remains available with
`--x-axis ranks` when node metadata is missing or when comparing layouts that do
not keep ranks per node constant. The analyzer never guesses node count from
rank count alone.

## Plotting

Plotting is optional:

```sh
python3 tools/ghalo_analyze.py plot \
  --kind latency \
  --output frontier-64node-latency.png \
  results/frontier/*frontier-64node*
```

Frontier scaling by node count:

```sh
python3 tools/ghalo_analyze.py plot \
  --kind scaling \
  --x-axis nodes \
  --style publication \
  --output analysis/plots/frontier-scaling-run2.pdf \
  results/frontier/*scaling-*run2*
```

Effective transferred-byte rate:

```sh
python3 tools/ghalo_analyze.py plot \
  --kind effective-rate \
  --style publication \
  --output analysis/plots/frontier-effective-rate-run2.pdf \
  results/frontier/*scaling-*run2*
```

64-node repeatability with explicit labels:

```sh
python3 tools/ghalo_analyze.py plot \
  --kind latency \
  --label "Run 1" \
  --label "Run 2" \
  --style publication \
  --output analysis/plots/frontier-64node-repeatability.pdf \
  RUN_1 RUN_2
```

Observed difference between two repeat runs:

```sh
python3 tools/ghalo_analyze.py plot \
  --kind percent-difference \
  --label "Run 1" \
  --label "Run 2" \
  --style publication \
  --output analysis/plots/frontier-64node-difference.pdf \
  RUN_1 RUN_2
```

Comma-separated labels are also accepted:

```sh
python3 tools/ghalo_analyze.py plot \
  --kind latency \
  --labels "Run 1,Run 2" \
  --output frontier-64node-repeatability.png \
  RUN_1 RUN_2
```

Supported plot kinds:

- `latency`: halo length `N` versus maximum average time in microseconds;
- `effective-rate`: halo length `N` versus per-rank effective transferred-byte
  rate in GiB/s;
- `scaling`: nodes or ranks versus time in microseconds, one series per halo
  size;
- `phase`: phase timing values for one selected run;
- `percent-difference`: observed
  `100 * (time_B - time_A) / time_A` versus halo length for two compatible
  runs, with a zero reference line.

Scaling plots support:

```text
--x-axis auto|nodes|ranks
```

`auto` uses nodes when all inputs have known node counts and falls back to
ranks otherwise. The axis label is explicit: `Node Count` or
`GPU Rank Count`. It is never labeled "nodes or ranks."

Latency plots default to a log2 halo-length axis, measured halo sizes as tick
labels, markers at every measured point, and a light grid. Supported scales are
`linear`, `log2`, and `log10` where applicable. Plots do not smooth or
interpolate data. This is intentional: non-monotonic behavior can indicate real
topology, routing, synchronization, or congestion effects and should remain
visible.

Effective-rate plots use the label
`Per-rank Effective Transferred-Byte Rate (GiB/s)`. This value remains the
derived rate:

```text
bytes_per_rank / maximum_average_exchange_seconds
```

It is not NIC bandwidth, injection bandwidth, aggregate network bandwidth, or
physical-link bandwidth.

Automatic legend labels use active-system metadata, node count, and rank count
instead of timestamp-heavy directory names. For example, Borg runs remain
labeled as Borg even when they use Frontier build artifacts through the build
alias mechanism. Full source paths and timestamps remain available in result
directories and analysis provenance.

When `--title` is not supplied, plots derive a concise title from backend,
active system, node count, rank count, and ranks per node. A repeated Frontier
MPI-HIP run may produce:

```text
gHALO MPI-HIP Repeatability
Frontier — 64 Nodes — 512 GPU Ranks
```

A Frontier scaling campaign with constant rank layout may produce:

```text
gHALO MPI-HIP Scaling
Frontier — 8 GPU Ranks per Node
```

`--style publication` increases font size, line width, marker size, DPI, and
uses a high-contrast color cycle with restrained grid lines. `--format` may be
`png`, `pdf`, or `svg`; when omitted, the format is inferred from the output
extension. Publication PNG output defaults to 600 DPI unless `--dpi` is
provided. PDF and SVG outputs are vector formats. `--legend-position` accepts
`auto`, `inside`, or `outside`.

If `matplotlib` is unavailable, the plot command fails clearly and the textual,
CSV, JSON, and Markdown commands continue to work.

## Report Generation

The optional `report` subcommand creates a modest one-command analysis bundle:

```sh
python3 tools/ghalo_analyze.py report \
  --style publication \
  --output-dir analysis/frontier-scaling-run2 \
  results/frontier/*scaling-*run2*
```

The report command writes, as applicable:

```text
analysis/frontier-scaling-run2/
  report.md
  summary.csv
  summary.json
  scaling.csv
  comparison.csv
  plots/
    latency.png
    effective-rate.png
    scaling.png
    percent-difference.png
    phase.png
  provenance.json
```

Skipped outputs and reasons are listed in `report.md` and `provenance.json`.
The command records exact input paths, input checksums, the analysis command,
UTC generation time, Python version, Git commit, and matplotlib version when
available. Source result directories are never modified.

Single-run comparisons and percent-difference plots are observed differences,
not statistical conclusions. Positive percent differences mean run B was slower
than run A; negative values mean run B was faster.

## Phase Timing

When phase timing is present, use:

```sh
python3 tools/ghalo_analyze.py summarize \
  --phase-timing \
  results/frontier/<phase-run>
```

The analyzer reports:

- input device copy when present;
- north/south communication;
- north/south synchronization;
- transpose device copy;
- transpose synchronization;
- east/west communication;
- east/west synchronization;
- phase sum;
- total;
- total minus phase sum.

It also derives broad categories:

- device-copy total;
- communication total;
- MPI total for older MPI-HIP fields;
- synchronization total;
- `total_minus_sum_of_phase_maxima_seconds`.

These category totals are sums of independently `MPI_MAX`-reduced phase
maxima. They may exceed the independently reduced complete exchange time. The
analyzer preserves negative total-minus-phase-sum values and does not rename
them as missing work.

For RCCL, `rccl_sync_mode` is compatibility metadata. Conservative and
stream-ordered runs are not aggregated together unless explicitly mixed, and
mixed backend groups still retain sync mode as a grouping discriminator.
Stream-ordered phase plots show enqueue phases plus `final_stream_sync`; they
do not reinterpret enqueue time as communication completion time.

## Output Directory

For a compact analysis bundle:

```sh
python3 tools/ghalo_analyze.py summarize \
  --output-dir frontier-analysis \
  results/frontier/*frontier-64node*
```

This creates:

```text
frontier-analysis/
  analysis/
    summary.csv
    summary.json
    summary.md
    comparison.csv        # when the first two inputs can be compared
    aggregate.csv         # when inputs are compatible
    plots/
    provenance.json
```

`provenance.json` records input paths, input checksums, the analysis command,
the analysis tool Git commit when available from CI, generation time in UTC,
Python version, and matplotlib version when installed.

The analyzer never modifies source result directories.

## Wrapper

The convenience wrapper delegates directly to the Python tool:

```sh
scripts/analyze.sh compare RUN_A RUN_B
scripts/analyze.sh aggregate results/frontier/*frontier-64node*
```

Set `PYTHON=/path/to/python3` to choose a specific interpreter.

## Why Maximum-Rank Timing Matters

gHALO preserves the HALO benchmark philosophy that the slowest participant
determines application-visible progress. A rank that waits on a slow neighbor
cannot advance the halo exchange even if most ranks were faster. For that
reason, the primary timing field is the maximum average wall-clock time across
all ranks.

Analysis tools may compute summaries and rates from this field, but they do
not change its meaning.
