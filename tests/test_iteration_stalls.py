#!/usr/bin/env python3
"""Unit tests for gHALO iteration stall diagnostics."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import analyze_iteration_stalls as stalls  # noqa: E402


def write_case(
    root: Path,
    name: str,
    *,
    iterations: int = 10,
    samples: int = 4,
    halo: int = 128,
    records=None,
    metadata_threshold: float = 1000.0,
    backend: str = "RCCLBackend",
    sync_mode: str = "conservative",
    include_iteration_file: bool = True,
    empty_iteration_file: bool = False,
) -> Path:
    path = root / name
    path.mkdir(parents=True)
    results = []
    for sample in range(1, samples + 1):
        results.append(
            {
                "backend": backend,
                "algorithm": "rccl" if backend == "RCCLBackend" else "mpi-hip-sendrecv",
                "halo_words": halo,
                "iterations": iterations,
                "sample_index": sample,
                "sample_count": samples,
                "max_average_seconds": 1.0e-5,
                "max_total_seconds": iterations * 1.0e-5,
                "total_exchange_bytes_per_rank": halo * 24,
                "metadata": {
                    "rccl_sync_mode": sync_mode,
                    "rocm_version": "6.4.2",
                    "ranks": [{"hostname": "node0"}],
                },
                "topology": {"world_size": 64},
            }
        )
    (path / "ghalo.json").write_text(json.dumps({"results": results}), encoding="utf-8")
    (path / "result-metadata.txt").write_text(
        "result_category=repeatability\n"
        "rocm_version=6.4.2\n"
        f"iteration_stall_threshold_us={metadata_threshold}\n",
        encoding="utf-8",
    )
    (path / "system-resolution.txt").write_text(
        "active_system=borg\nbackend=rccl\n", encoding="utf-8"
    )
    if include_iteration_file:
        iteration_path = path / "iteration-times.csv"
        if empty_iteration_file:
            iteration_path.write_text("", encoding="utf-8")
        else:
            with iteration_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "halo_words",
                        "sample_index",
                        "iteration_index",
                        "global_max_iteration_seconds",
                        "max_rank",
                        "backend",
                        "rccl_sync_mode",
                        "world_size",
                    ],
                )
                writer.writeheader()
                for sample, iteration, duration_us, rank in records or []:
                    writer.writerow(
                        {
                            "halo_words": halo,
                            "sample_index": sample,
                            "iteration_index": iteration,
                            "global_max_iteration_seconds": duration_us * 1.0e-6,
                            "max_rank": rank,
                            "backend": backend,
                            "rccl_sync_mode": sync_mode,
                            "world_size": 64,
                        }
                    )
    return path


class IterationStallAnalysisTests(unittest.TestCase):
    def test_no_stalls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_case(Path(tmp), "no-stalls", records=[])
            summary = stalls.analyze_case(case, 1000.0)
            self.assertEqual(summary.samples_found, 4)
            self.assertEqual(summary.total_measured_iterations, 40)
            self.assertEqual(summary.stall_count, 0)
            self.assertEqual(summary.samples_with_stalls, 0)
            self.assertEqual(summary.sample_sequence, "FFFF")
            self.assertEqual(summary.longest_stall_free_sequence, 4)

    def test_one_isolated_stall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_case(Path(tmp), "isolated", records=[(2, 7, 1500.0, 3)])
            summary = stalls.analyze_case(case, 1000.0)
            self.assertEqual(summary.stall_count, 1)
            self.assertEqual(summary.samples_with_stalls, 1)
            self.assertEqual(summary.sample_sequence, "FSFF")
            self.assertEqual(summary.isolated_bad_samples, 1)
            self.assertEqual(summary.multiple_bad_samples, 0)
            self.assertEqual(summary.affected_indices[0][:3], (128, 2, 7))

    def test_multiple_and_nearly_all_stalls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = [(1, 1, 1200.0, 4), (1, 2, 1300.0, 4)]
            records.extend((3, iteration, 2000.0, 5) for iteration in range(1, 10))
            case = write_case(Path(tmp), "multiple", records=records)
            summary = stalls.analyze_case(case, 1000.0)
            self.assertEqual(summary.stall_count, 11)
            self.assertEqual(summary.samples_with_stalls, 2)
            self.assertEqual(summary.sample_sequence, "SFSF")
            self.assertEqual(summary.multiple_bad_samples, 1)
            self.assertEqual(summary.nearly_all_slow_samples, 1)
            self.assertEqual(summary.max_stalls_in_one_sample, 9)

    def test_repeated_stalls_and_rank_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_case(
                Path(tmp),
                "repeated",
                records=[(1, 1, 1100.0, 1), (2, 1, 1200.0, 2), (4, 1, 1300.0, 2)],
            )
            summary = stalls.analyze_case(case, 1000.0)
            self.assertEqual(summary.sample_sequence, "SSFS")
            self.assertEqual(summary.longest_stall_free_sequence, 1)
            self.assertEqual(summary.rank_frequency, {1: 1, 2: 2})

    def test_threshold_boundary_is_inclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_case(Path(tmp), "boundary", records=[(1, 1, 1000.0, 9)])
            summary = stalls.analyze_case(case, 1000.0)
            self.assertEqual(summary.stall_count, 1)
            self.assertEqual(summary.rank_frequency, {9: 1})

    def test_missing_and_empty_iteration_files_warn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = write_case(Path(tmp), "missing", include_iteration_file=False)
            missing_summary = stalls.analyze_case(missing, 1000.0)
            self.assertEqual(missing_summary.stall_count, 0)
            self.assertTrue(any("missing iteration" in w for w in missing_summary.warnings))

            empty = write_case(Path(tmp), "empty", empty_iteration_file=True)
            empty_summary = stalls.analyze_case(empty, 1000.0)
            self.assertEqual(empty_summary.stall_count, 0)
            self.assertTrue(any("empty iteration" in w for w in empty_summary.warnings))

    def test_inconsistent_sample_metadata_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_case(Path(tmp), "bad", samples=2)
            payload = json.loads((case / "ghalo.json").read_text(encoding="utf-8"))
            payload["results"][1]["sample_index"] = 1
            (case / "ghalo.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(stalls.StallAnalysisError):
                stalls.analyze_case(case, 1000.0)

    def test_outputs_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = write_case(root, "case", records=[(1, 1, 1500.0, 3)])
            summary = stalls.analyze_case(case, 1000.0)
            output = root / "analysis"
            stalls.write_outputs(output, [summary])
            self.assertTrue((output / "iteration-stall-report.md").exists())
            self.assertTrue((output / "iteration-stall-summary.csv").exists())
            self.assertTrue((output / "iteration-stall-summary.json").exists())


if __name__ == "__main__":
    unittest.main()
