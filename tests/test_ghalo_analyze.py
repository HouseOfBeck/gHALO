#!/usr/bin/env python3
"""Unit tests for portable gHALO result analysis tooling."""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import ghalo_analyze as analyze  # noqa: E402

DATA = ROOT / "tests" / "data" / "analysis"


class FakeAxis:
    def __init__(self) -> None:
        self.xscale = None
        self.yscale = None
        self.plots = []
        self.hlines = []
        self.xlabel = ""
        self.ylabel = ""
        self.xticks = []
        self.xticklabels = []

    def set_xscale(self, *args, **kwargs):
        self.xscale = (args, kwargs)

    def set_yscale(self, *args, **kwargs):
        self.yscale = (args, kwargs)

    def plot(self, *args, **kwargs):
        self.plots.append((args, kwargs))

    def axhline(self, *args, **kwargs):
        self.hlines.append((args, kwargs))

    def set_xlabel(self, label):
        self.xlabel = label

    def set_ylabel(self, label):
        self.ylabel = label

    def set_xticks(self, ticks):
        self.xticks = ticks

    def set_xticklabels(self, labels):
        self.xticklabels = labels


class GhaloAnalyzeTests(unittest.TestCase):
    def test_json_parsing_and_metadata(self) -> None:
        run = analyze.load_run(str(DATA / "cpu_mpi_run"))
        self.assertEqual(run.version, "0.3.0")
        self.assertEqual(run.system, "frontier")
        self.assertEqual(run.build_system, "frontier")
        self.assertEqual(run.git_commit, "abcdef0")
        self.assertEqual(run.slurm_job_id, "12345")
        self.assertEqual(run.ranks, 4)
        self.assertEqual(run.cartesian, "2x2")
        self.assertEqual(run.results[0].metadata["memory_location"], "host")

    def test_csv_fallback(self) -> None:
        run = analyze.load_run(str(DATA / "csv_only"))
        self.assertEqual(run.source_format, "csv")
        self.assertEqual(len(run.results), 2)
        self.assertEqual(run.results[0].backend, "MPIBackend")
        self.assertEqual(run.results[0].topology["world_size"], 4)

    def test_summary_effective_rate(self) -> None:
        run = analyze.load_run(str(DATA / "cpu_mpi_run"))
        rows = analyze.summary_rows(run, include_metadata=True)
        self.assertEqual(rows[0]["bytes_per_rank"], 48)
        self.assertAlmostEqual(
            rows[0]["per_rank_effective_transferred_byte_rate_Bps"],
            4_800_000.0,
        )
        self.assertEqual(rows[0]["git_commit"], "abcdef0")

    def test_compare_ratios_and_percent_difference(self) -> None:
        a = analyze.load_run(str(DATA / "repeat_a"))
        b = analyze.load_run(str(DATA / "repeat_b"))
        rows = analyze.compare_rows(a, b)
        self.assertEqual(rows[0]["halo_words"], 2)
        self.assertAlmostEqual(rows[0]["ratio_B_over_A"], 1.1)
        self.assertAlmostEqual(rows[0]["observed_percent_difference"], 10.0)

    def test_fail_on_regression_exit_status(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            status = analyze.main(
                [
                    "compare",
                    "--threshold-percent",
                    "5",
                    "--fail-on-regression",
                    "--format",
                    "json",
                    str(DATA / "repeat_a"),
                    str(DATA / "repeat_b"),
                ]
            )
        self.assertEqual(status, 1)

    def test_aggregate_statistics(self) -> None:
        runs = [
            analyze.load_run(str(DATA / "repeat_a")),
            analyze.load_run(str(DATA / "repeat_b")),
        ]
        rows = analyze.aggregate_rows(runs, allow_mixed=False, group_by=[])
        by_halo = {row["halo_words"]: row for row in rows}
        self.assertEqual(by_halo[2]["count"], 2)
        self.assertAlmostEqual(by_halo[2]["minimum_seconds"], 0.000010)
        self.assertAlmostEqual(by_halo[2]["maximum_seconds"], 0.000011)
        self.assertAlmostEqual(by_halo[2]["mean_seconds"], 0.0000105)
        self.assertAlmostEqual(by_halo[2]["median_seconds"], 0.0000105)
        self.assertGreater(by_halo[2]["population_stddev_seconds"], 0.0)
        self.assertGreater(by_halo[2]["sample_stddev_seconds"], 0.0)

    def test_aggregate_compatibility_check(self) -> None:
        runs = [
            analyze.load_run(str(DATA / "repeat_a")),
            analyze.load_run(str(DATA / "mixed_run")),
        ]
        with self.assertRaises(analyze.AnalysisError):
            analyze.aggregate_rows(runs, allow_mixed=False, group_by=[])
        rows = analyze.aggregate_rows(runs, allow_mixed=True, group_by=["backend"])
        groups = {row["group"] for row in rows}
        self.assertIn("MPIHIPBackend", groups)
        self.assertIn("MPIBackend", groups)

    def test_readable_aggregate_group_labels(self) -> None:
        rows = analyze.aggregate_rows(
            [
                analyze.load_run(str(DATA / "repeat_a")),
                analyze.load_run(str(DATA / "repeat_b")),
            ],
            allow_mixed=False,
            group_by=[],
        )
        self.assertEqual(rows[0]["group"], "Frontier | 1 node | 8 GPU ranks | repeat")
        self.assertNotIn("((", rows[0]["group"])
        self.assertEqual(rows[0]["group_backend"], "MPIHIPBackend")
        self.assertIn("N=2:48B/rank", rows[0]["group_bytes_per_rank_by_halo"])

    def test_scaling_baseline_logic(self) -> None:
        runs = [
            analyze.load_run(str(DATA / "cpu_mpi_run")),
            analyze.load_run(str(DATA / "mpi_hip_run")),
        ]
        baseline = analyze.choose_baseline(
            Namespace(baseline="smallest-nodes", baseline_path=None, baseline_nodes=1),
            runs,
        )
        self.assertEqual(baseline.system, "borg")
        path_baseline = analyze.choose_baseline(
            Namespace(
                baseline=str(DATA / "cpu_mpi_run"),
                baseline_path=None,
                baseline_nodes=None,
            ),
            runs,
        )
        self.assertEqual(path_baseline.system, "frontier")
        rows = analyze.scaling_rows(runs, baseline)
        self.assertTrue(any(row["ratio_relative_to_baseline"] > 1.0 for row in rows))

    def test_phase_category_calculation_and_negative_difference(self) -> None:
        run = analyze.load_run(str(DATA / "phase_run"))
        result = run.results[0]
        categories = analyze.phase_categories(result)
        self.assertAlmostEqual(categories["device_copy_total_seconds"], 0.000004)
        self.assertAlmostEqual(categories["mpi_total_seconds"], 0.000007)
        self.assertAlmostEqual(categories["synchronization_total_seconds"], 0.000003)
        self.assertLess(categories["total_minus_sum_of_phase_maxima_seconds"], 0.0)
        rows = analyze.summary_rows(run, phase_timing=True)
        self.assertIn("phase_phase_sum_seconds", rows[0])
        labeled = analyze.summary_rows(run, label_override="explicit")
        self.assertEqual(labeled[0]["label"], "explicit")

    def test_concise_labels_and_borg_active_system(self) -> None:
        frontier = analyze.load_run(str(DATA / "repeat_a"))
        borg = analyze.load_run(str(DATA / "mpi_hip_run"))
        self.assertEqual(frontier.label, "Frontier | 1 node | 8 GPU ranks | repeat")
        self.assertEqual(borg.label, "Borg | 1 node | 8 GPU ranks")
        labels = analyze.run_labels([frontier, borg], ["Run 1", "Run 2"])
        self.assertEqual(labels, ["Run 1", "Run 2"])

    def test_log2_x_axis_selection_and_halo_ticks(self) -> None:
        axis = FakeAxis()
        run = analyze.load_run(str(DATA / "cpu_mpi_run"))
        analyze.apply_axis_scale(axis, "x", "log2")
        analyze.set_halo_ticks(axis, [run])
        self.assertEqual(axis.xscale, (("log",), {"base": 2}))
        self.assertEqual(axis.xticks, [2, 4])
        self.assertEqual(axis.xticklabels, ["2", "4"])
        args = analyze.build_parser().parse_args(
            ["plot", "--kind", "latency", str(DATA / "cpu_mpi_run")]
        )
        self.assertEqual(args.xscale, "log2")

    def test_default_plot_title_uses_metadata(self) -> None:
        title = analyze.default_plot_title(
            "latency",
            [
                analyze.load_run(str(DATA / "mpi_hip_run")),
                analyze.load_run(str(DATA / "mpi_hip_run")),
            ],
        )
        self.assertIn("gHALO MPI-HIP Repeatability", title)
        self.assertIn("Borg", title)
        self.assertIn("1 Node", title)
        self.assertIn("8 GPU Ranks", title)

    def test_percent_difference_plot_rows_and_zero_reference(self) -> None:
        runs = [
            analyze.load_run(str(DATA / "repeat_a")),
            analyze.load_run(str(DATA / "repeat_b")),
        ]
        rows = analyze.percent_difference_rows_for_plot(runs)
        self.assertAlmostEqual(rows[0]["observed_percent_difference"], 10.0)

        axis = FakeAxis()
        analyze.draw_percent_difference_plot(axis, rows)
        self.assertEqual(axis.plots[0][0][0], [2, 4])
        self.assertAlmostEqual(axis.plots[0][0][1][0], 10.0)
        self.assertEqual(axis.hlines[0][0][0], 0.0)
        self.assertEqual(axis.ylabel, "Observed Percent Difference")

    def test_percent_difference_requires_exactly_two_inputs(self) -> None:
        run = analyze.load_run(str(DATA / "repeat_a"))
        with self.assertRaisesRegex(analyze.AnalysisError, "exactly two"):
            analyze.percent_difference_rows_for_plot([run])
        with self.assertRaisesRegex(analyze.AnalysisError, "exactly two"):
            analyze.percent_difference_rows_for_plot([run, run, run])

    def test_malformed_and_incomplete_input_errors(self) -> None:
        with self.assertRaises(analyze.AnalysisError):
            analyze.load_run(str(DATA / "malformed_json"))
        with self.assertRaises(analyze.AnalysisError):
            analyze.compare_rows(
                analyze.load_run(str(DATA / "repeat_a")),
                analyze.load_run(str(DATA / "missing_halo")),
            )

    def test_nonzero_exit_status_warns_but_loads(self) -> None:
        run = analyze.load_run(str(DATA / "nonzero_exit"))
        self.assertEqual(len(run.results), 1)
        self.assertTrue(any("nonzero" in warning for warning in run.warnings))

    def test_markdown_csv_json_output(self) -> None:
        run = analyze.load_run(str(DATA / "cpu_mpi_run"))
        rows = analyze.summary_rows(run)
        csv_text = analyze.rows_to_csv(rows)
        md_text = analyze.rows_to_markdown(rows)
        self.assertIn("halo_words", csv_text)
        self.assertIn("| halo_words |", md_text)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            analyze.write_output(rows, "json", None)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload[0]["halo_words"], 2)

    def test_plot_without_matplotlib_fails_clearly(self) -> None:
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        args = Namespace(
            kind="latency",
            xscale="linear",
            yscale="linear",
            title=None,
            output=None,
            dpi=120,
            baseline="smallest-nodes",
            baseline_path=None,
            baseline_nodes=None,
            inputs=[str(DATA / "cpu_mpi_run")],
        )
        with mock.patch("builtins.__import__", side_effect=blocked_import):
            with self.assertRaisesRegex(analyze.AnalysisError, "requires matplotlib"):
                analyze.command_plot(args)

    def test_compare_allows_missing_sizes(self) -> None:
        rows = analyze.compare_rows(
            analyze.load_run(str(DATA / "repeat_a")),
            analyze.load_run(str(DATA / "missing_halo")),
            allow_missing=True,
        )
        missing = [row for row in rows if row.get("missing")]
        self.assertEqual(missing[0]["halo_words"], 4)
        self.assertEqual(missing[0]["missing"], "B")


if __name__ == "__main__":
    unittest.main()
