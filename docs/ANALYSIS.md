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
rank for every halo size, and phase-timing state. Use `--allow-mixed` with
`--group-by system`, `--group-by backend`, `--group-by nodes`, or
`--group-by ranks` when unlike runs should be grouped rather than rejected.

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

## Plotting

Plotting is optional:

```sh
python3 tools/ghalo_analyze.py plot \
  --kind latency \
  --output frontier-64node-latency.png \
  results/frontier/*frontier-64node*
```

Supported plot kinds:

- `latency`: halo length `N` versus maximum average time in microseconds;
- `effective-rate`: halo length `N` versus per-rank effective transferred-byte
  rate in GiB/s;
- `scaling`: nodes or ranks versus time in microseconds, one series per halo
  size;
- `phase`: phase timing values for one selected run.

Supported scales are `linear`, `log2`, and `log10` where applicable. Plots use
markers at measured halo sizes and do not smooth or interpolate data. This is
intentional: non-monotonic behavior can indicate real topology, routing,
synchronization, or congestion effects and should remain visible.

If `matplotlib` is unavailable, the plot command fails clearly and the textual,
CSV, JSON, and Markdown commands continue to work.

## Phase Timing

When phase timing is present, use:

```sh
python3 tools/ghalo_analyze.py summarize \
  --phase-timing \
  results/frontier/<phase-run>
```

The analyzer reports:

- input device copy;
- north/south MPI;
- north/south synchronization;
- transpose device copy;
- transpose synchronization;
- east/west MPI;
- east/west synchronization;
- phase sum;
- total;
- total minus phase sum.

It also derives broad categories:

- device-copy total;
- MPI total;
- synchronization total;
- `total_minus_sum_of_phase_maxima_seconds`.

These category totals are sums of independently `MPI_MAX`-reduced phase
maxima. They may exceed the independently reduced complete exchange time. The
analyzer preserves negative total-minus-phase-sum values and does not rename
them as missing work.

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
