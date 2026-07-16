#!/usr/bin/env python3
"""Unit tests for RCCL version comparison tooling."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import compare_rccl_versions as compare  # noqa: E402


def write_version_case(
    root: Path,
    version: str,
    code: str,
    *,
    stalls=None,
    validation_passed: bool = True,
    result_rocm_version: str | None = None,
    backend: str = "RCCLBackend",
) -> Path:
    path = (
        root
        / "results"
        / "borg"
        / f"rocm-{version}"
        / "repeatability"
        / f"20260716T000000Z_rccl_borg-rccl-version-rocm-{version}"
    )
    path.mkdir(parents=True)
    stalls = stalls or []
    results = []
    for sample in range(1, 4):
      results.append(
          {
              "backend": backend,
              "algorithm": "rccl",
              "halo_words": 128,
              "iterations": 10,
              "sample_index": sample,
              "sample_count": 3,
              "max_average_seconds": (20.0 + sample) * 1.0e-6,
              "max_total_seconds": (20.0 + sample) * 10.0e-6,
              "metadata": {
                  "rocm_version": result_rocm_version or version,
                  "hip_runtime_version": f"hip-{version}",
                  "rccl_version": code,
                  "rccl_sync_mode": "conservative",
                  "validation_passed": validation_passed,
                  "rccl_plugin_root": f"/opt/plugins/{version}",
              },
              "topology": {"world_size": 64},
          }
      )
    (path / "ghalo.json").write_text(json.dumps({"results": results}), encoding="utf-8")
    (path / "result-metadata.txt").write_text(
        "\n".join(
            [
                "result_category=repeatability",
                f"requested_rocm_version={version}",
                f"rocm_version={version}",
                f"observed_rccl_version_code={code}",
                f"expected_rccl_version_code={compare.EXPECTED_CODES.get(version, code)}",
                f"libnccl_net_realpath=/opt/plugins/{version}/lib/libnccl-net.so",
                "iteration_stall_threshold_us=1000",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (path / "system-resolution.txt").write_text("active_system=borg\n", encoding="utf-8")

    with (path / "iteration-times.csv").open("w", newline="", encoding="utf-8") as handle:
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
        for sample, iteration, duration_us, rank in stalls:
            writer.writerow(
                {
                    "halo_words": 128,
                    "sample_index": sample,
                    "iteration_index": iteration,
                    "global_max_iteration_seconds": duration_us * 1.0e-6,
                    "max_rank": rank,
                    "backend": backend,
                    "rccl_sync_mode": "conservative",
                    "world_size": 64,
                }
            )

    with (path / "iteration-phase-times.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "version",
                "backend",
                "rccl_sync_mode",
                "world_size",
                "halo_words",
                "sample_index",
                "sample_count",
                "iteration_index",
                "iterations_in_sample",
                "stall_threshold_us",
                "total_iteration_seconds",
                "total_iteration_max_rank",
                "backend_schema",
                "phase_name",
                "phase_seconds",
                "phase_max_rank",
            ],
        )
        writer.writeheader()
        for sample, iteration, duration_us, rank in stalls:
            writer.writerow(
                {
                    "version": "0.3.0",
                    "backend": backend,
                    "rccl_sync_mode": "conservative",
                    "world_size": 64,
                    "halo_words": 128,
                    "sample_index": sample,
                    "sample_count": 3,
                    "iteration_index": iteration,
                    "iterations_in_sample": 10,
                    "stall_threshold_us": 1000,
                    "total_iteration_seconds": duration_us * 1.0e-6,
                    "total_iteration_max_rank": rank,
                    "backend_schema": "rccl-conservative",
                    "phase_name": "north_south_sync",
                    "phase_seconds": duration_us * 0.8e-6,
                    "phase_max_rank": rank,
                }
            )

    with (path / "stalled-rank-times.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "version",
                "backend",
                "rccl_sync_mode",
                "world_size",
                "halo_words",
                "sample_index",
                "sample_count",
                "iteration_index",
                "iterations_in_sample",
                "stall_threshold_us",
                "world_rank",
                "local_rank",
                "hostname",
                "selected_hip_device",
                "cart_rank",
                "cart_row",
                "cart_col",
                "local_total_iteration_seconds",
                "global_max_iteration_seconds",
                "global_max_iteration_rank",
                "backend_schema",
            ],
        )
        writer.writeheader()
        for sample, iteration, duration_us, rank in stalls:
            for world_rank in range(4):
                writer.writerow(
                    {
                        "version": "0.3.0",
                        "backend": backend,
                        "rccl_sync_mode": "conservative",
                        "world_size": 64,
                        "halo_words": 128,
                        "sample_index": sample,
                        "sample_count": 3,
                        "iteration_index": iteration,
                        "iterations_in_sample": 10,
                        "stall_threshold_us": 1000,
                        "world_rank": world_rank,
                        "local_rank": world_rank,
                        "hostname": f"node{world_rank // 2}",
                        "selected_hip_device": world_rank,
                        "cart_rank": world_rank,
                        "cart_row": world_rank // 2,
                        "cart_col": world_rank % 2,
                        "local_total_iteration_seconds": (duration_us if world_rank == rank else 100.0) * 1.0e-6,
                        "global_max_iteration_seconds": duration_us * 1.0e-6,
                        "global_max_iteration_rank": rank,
                        "backend_schema": "rccl-conservative",
                    }
                )
    return path


class RCCLVersionComparisonTests(unittest.TestCase):
    def test_discovers_and_orders_three_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_version_case(root, "7.2.0", "22707", stalls=[(1, 1, 1500.0, 2)])
            write_version_case(root, "6.4.2", "22203", stalls=[(1, 1, 1200.0, 1)])
            write_version_case(root, "7.0.2", "22606", stalls=[])
            dirs = compare.discover_result_dirs(root, "borg", "repeatability")
            summaries = compare.validate_complete_matrix(
                [compare.summarize_result(path, 1000.0) for path in dirs]
            )
            self.assertEqual([summary.rocm_version for summary in summaries], compare.VERSION_ORDER)
            self.assertEqual(summaries[1].stalled_iterations, 0)
            self.assertEqual(summaries[1].stall_percentage, 0.0)

    def test_validation_failure_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_version_case(
                Path(tmp), "6.4.2", "22203", validation_passed=False
            )
            summary = compare.summarize_result(case, 1000.0)
            self.assertFalse(summary.validation_passed)

    def test_wrong_expected_rccl_version_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_version_case(Path(tmp), "7.0.2", "99999")
            with self.assertRaises(compare.RCCLVersionCompareError):
                compare.summarize_result(case, 1000.0)

    def test_missing_result_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_version_case(root, "6.4.2", "22203")
            write_version_case(root, "7.0.2", "22606")
            dirs = compare.discover_result_dirs(root, "borg", "repeatability")
            with self.assertRaises(compare.RCCLVersionCompareError):
                compare.validate_complete_matrix(
                    [compare.summarize_result(path, 1000.0) for path in dirs]
                )

    def test_mixed_metadata_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_version_case(
                Path(tmp), "6.4.2", "22203", result_rocm_version="7.0.2"
            )
            with self.assertRaises(compare.RCCLVersionCompareError):
                compare.summarize_result(case, 1000.0)

    def test_outputs_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                write_version_case(root, "6.4.2", "22203", stalls=[(1, 1, 1200.0, 1)]),
                write_version_case(root, "7.0.2", "22606"),
                write_version_case(root, "7.2.0", "22707", stalls=[(2, 3, 2000.0, 3)]),
            ]
            summaries = compare.validate_complete_matrix(
                [compare.summarize_result(path, 1000.0) for path in cases]
            )
            output = root / "analysis"
            compare.write_outputs(output, summaries)
            self.assertTrue((output / "rccl-version-comparison.md").exists())
            self.assertTrue((output / "rccl-version-comparison.csv").exists())
            self.assertTrue((output / "rccl-version-comparison.json").exists())
            payload = json.loads(
                (output / "rccl-version-comparison.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(payload["cases"]), 3)


if __name__ == "__main__":
    unittest.main()
