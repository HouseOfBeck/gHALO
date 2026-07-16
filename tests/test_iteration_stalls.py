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
    phase_records=None,
    rank_records=None,
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
    if phase_records is not None:
        with (path / "iteration-phase-times.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
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
            for sample, iteration, phase_name, phase_us, rank in phase_records:
                writer.writerow(
                    {
                        "version": "0.3.0",
                        "backend": backend,
                        "rccl_sync_mode": sync_mode,
                        "world_size": 64,
                        "halo_words": halo,
                        "sample_index": sample,
                        "sample_count": samples,
                        "iteration_index": iteration,
                        "iterations_in_sample": iterations,
                        "stall_threshold_us": metadata_threshold,
                        "total_iteration_seconds": 0.002,
                        "total_iteration_max_rank": rank,
                        "backend_schema": "rccl-stream-ordered"
                        if sync_mode == "stream-ordered"
                        else "rccl-conservative",
                        "phase_name": phase_name,
                        "phase_seconds": phase_us * 1.0e-6,
                        "phase_max_rank": rank,
                    }
                )
    if rank_records is not None:
        max_rank_by_iteration = {}
        for row in rank_records:
            sample, iteration, rank, *_rest, local_us = row
            key = (sample, iteration)
            if key not in max_rank_by_iteration or local_us > max_rank_by_iteration[key][1]:
                max_rank_by_iteration[key] = (rank, local_us)
        with (path / "stalled-rank-times.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
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
                    "north_south_communication_seconds",
                    "north_south_sync_seconds",
                    "transpose_copy_seconds",
                    "transpose_sync_seconds",
                    "east_west_communication_seconds",
                    "east_west_sync_seconds",
                    "north_south_enqueue_seconds",
                    "transpose_enqueue_seconds",
                    "east_west_enqueue_seconds",
                    "final_stream_sync_seconds",
                    "input_device_copy_seconds",
                    "north_south_mpi_seconds",
                    "transpose_device_copy_seconds",
                    "transpose_copy_sync_seconds",
                    "east_west_mpi_seconds",
                ],
            )
            writer.writeheader()
            for row in rank_records:
                sample, iteration, rank, host, gpu, cart_row, cart_col, local_us = row
                max_rank = max_rank_by_iteration[(sample, iteration)][0]
                writer.writerow(
                    {
                        "version": "0.3.0",
                        "backend": backend,
                        "rccl_sync_mode": sync_mode,
                        "world_size": 64,
                        "halo_words": halo,
                        "sample_index": sample,
                        "sample_count": samples,
                        "iteration_index": iteration,
                        "iterations_in_sample": iterations,
                        "stall_threshold_us": metadata_threshold,
                        "world_rank": rank,
                        "local_rank": rank % 8,
                        "hostname": host,
                        "selected_hip_device": gpu,
                        "cart_rank": rank,
                        "cart_row": cart_row,
                        "cart_col": cart_col,
                        "local_total_iteration_seconds": local_us * 1.0e-6,
                        "global_max_iteration_seconds": 2000.0 * 1.0e-6,
                        "global_max_iteration_rank": max_rank,
                        "backend_schema": "rccl-stream-ordered"
                        if sync_mode == "stream-ordered"
                        else "rccl-conservative",
                        "north_south_communication_seconds": local_us * 0.5e-6,
                        "north_south_sync_seconds": local_us * 0.1e-6,
                        "transpose_copy_seconds": 0.0,
                        "transpose_sync_seconds": 0.0,
                        "east_west_communication_seconds": local_us * 0.2e-6,
                        "east_west_sync_seconds": 0.0,
                        "north_south_enqueue_seconds": 0.0,
                        "transpose_enqueue_seconds": 0.0,
                        "east_west_enqueue_seconds": 0.0,
                        "final_stream_sync_seconds": 0.0,
                        "input_device_copy_seconds": 0.0,
                        "north_south_mpi_seconds": 0.0,
                        "transpose_device_copy_seconds": 0.0,
                        "transpose_copy_sync_seconds": 0.0,
                        "east_west_mpi_seconds": 0.0,
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
            case = write_case(
                Path(tmp),
                "isolated",
                records=[(2, 7, 1500.0, 3)],
                phase_records=[
                    (2, 7, "north_south_communication", 1200.0, 3),
                    (2, 7, "east_west_sync", 100.0, 4),
                ],
            )
            summary = stalls.analyze_case(case, 1000.0)
            self.assertEqual(summary.stall_count, 1)
            self.assertEqual(summary.samples_with_stalls, 1)
            self.assertEqual(summary.sample_sequence, "FSFF")
            self.assertEqual(summary.isolated_bad_samples, 1)
            self.assertEqual(summary.multiple_bad_samples, 0)
            self.assertEqual(summary.affected_indices[0][:3], (128, 2, 7))
            self.assertEqual(summary.stalled_iterations_with_phase_data, 1)
            self.assertEqual(summary.stall_cause_classification, "communication/enqueue")
            self.assertEqual(summary.phase_summaries[0].phase_name, "east_west_sync")
            dominant = {
                phase.phase_name: phase.dominant_iterations
                for phase in summary.phase_summaries
            }
            self.assertEqual(dominant["north_south_communication"], 1)
            north_south = next(
                phase for phase in summary.phase_summaries
                if phase.phase_name == "north_south_communication"
            )
            self.assertEqual(north_south.max_rank_frequency, {3: 1})

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

    def test_mixed_phase_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_case(
                Path(tmp),
                "mixed-phase",
                records=[(1, 1, 1500.0, 1), (2, 1, 1600.0, 2)],
                phase_records=[
                    (1, 1, "north_south_communication", 1200.0, 1),
                    (1, 1, "east_west_sync", 100.0, 2),
                    (2, 1, "north_south_communication", 100.0, 1),
                    (2, 1, "east_west_sync", 1300.0, 2),
                ],
            )
            summary = stalls.analyze_case(case, 1000.0)
            self.assertEqual(summary.stalled_iterations_with_phase_data, 2)
            self.assertEqual(summary.stall_cause_classification, "mixed/unattributed")
            self.assertEqual(len(summary.phase_iteration_breakdown), 2)

    def test_stream_ordered_phase_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_case(
                Path(tmp),
                "stream",
                sync_mode="stream-ordered",
                records=[(1, 1, 1800.0, 6)],
                phase_records=[
                    (1, 1, "north_south_enqueue", 300.0, 1),
                    (1, 1, "transpose_enqueue", 200.0, 2),
                    (1, 1, "east_west_enqueue", 400.0, 3),
                    (1, 1, "final_stream_sync", 1700.0, 6),
                ],
            )
            summary = stalls.analyze_case(case, 1000.0)
            self.assertEqual(summary.stall_cause_classification, "synchronization/final sync")
            self.assertIn("final_stream_sync", {p.phase_name for p in summary.phase_summaries})

    def test_mpi_hip_phase_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_case(
                Path(tmp),
                "mpi-hip",
                backend="MPIHIPBackend",
                sync_mode="",
                records=[(1, 1, 1800.0, 6)],
                phase_records=[
                    (1, 1, "input_device_copy", 100.0, 1),
                    (1, 1, "north_south_mpi", 1600.0, 6),
                    (1, 1, "east_west_sync", 200.0, 2),
                ],
            )
            summary = stalls.analyze_case(case, 1000.0)
            self.assertEqual(summary.stall_cause_classification, "communication/enqueue")
            self.assertIn("north_south_mpi", {p.phase_name for p in summary.phase_summaries})

    def test_stalled_rank_collective_wide_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                (1, 1, 0, "node0", 0, 0, 0, 1900.0),
                (1, 1, 1, "node0", 1, 0, 1, 1800.0),
                (1, 1, 2, "node1", 0, 1, 0, 1850.0),
                (1, 1, 3, "node1", 1, 1, 1, 2000.0),
            ]
            case = write_case(
                Path(tmp),
                "rank-collective",
                records=[(1, 1, 2000.0, 3)],
                phase_records=[
                    (1, 1, "north_south_communication", 1500.0, 3),
                ],
                rank_records=rows,
            )
            summary = stalls.analyze_case(case, 1000.0)
            self.assertEqual(
                summary.rank_event_classification_frequency,
                {"collective-wide": 1},
            )
            self.assertEqual(summary.slow_rank_frequency, {0: 1, 1: 1, 2: 1, 3: 1})
            self.assertEqual(summary.dominant_phase_rank_matches, 1)

    def test_stalled_rank_locality_classifications(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            node_case = write_case(
                Path(tmp),
                "rank-node",
                records=[(1, 1, 2000.0, 2)],
                rank_records=[
                    (1, 1, 0, "node0", 0, 0, 0, 1700.0),
                    (1, 1, 1, "node0", 1, 0, 1, 1800.0),
                    (1, 1, 2, "node0", 2, 1, 0, 2000.0),
                    (1, 1, 3, "node1", 0, 1, 1, 300.0),
                ],
            )
            self.assertEqual(
                stalls.analyze_case(node_case, 1000.0)
                .rank_event_classification_frequency,
                {"node-localized": 1},
            )

            row_case = write_case(
                Path(tmp),
                "rank-row",
                records=[(1, 1, 2000.0, 2)],
                rank_records=[
                    (1, 1, 0, "node0", 0, 0, 0, 1700.0),
                    (1, 1, 1, "node1", 0, 0, 1, 1800.0),
                    (1, 1, 2, "node2", 0, 0, 2, 2000.0),
                    (1, 1, 3, "node3", 0, 1, 0, 200.0),
                ],
            )
            self.assertEqual(
                stalls.analyze_case(row_case, 1000.0)
                .rank_event_classification_frequency,
                {"row-localized": 1},
            )

            column_case = write_case(
                Path(tmp),
                "rank-column",
                records=[(1, 1, 2000.0, 2)],
                rank_records=[
                    (1, 1, 0, "node0", 0, 0, 0, 1700.0),
                    (1, 1, 1, "node1", 0, 1, 0, 1800.0),
                    (1, 1, 2, "node2", 0, 2, 0, 2000.0),
                    (1, 1, 3, "node3", 0, 0, 1, 200.0),
                ],
            )
            self.assertEqual(
                stalls.analyze_case(column_case, 1000.0)
                .rank_event_classification_frequency,
                {"column-localized": 1},
            )

    def test_stalled_rank_rank_localized_and_duplicate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_case(
                Path(tmp),
                "rank-local",
                records=[(1, 1, 2000.0, 1)],
                rank_records=[
                    (1, 1, 0, "node0", 0, 0, 0, 500.0),
                    (1, 1, 1, "node1", 0, 0, 1, 2000.0),
                    (1, 1, 2, "node2", 0, 1, 0, 400.0),
                    (1, 1, 3, "node3", 0, 1, 1, 300.0),
                ],
            )
            summary = stalls.analyze_case(case, 1000.0)
            self.assertEqual(
                summary.rank_event_classification_frequency,
                {"rank-localized": 1},
            )
            self.assertEqual(summary.slow_node_frequency, {"node1": 1})

            duplicate = write_case(
                Path(tmp),
                "duplicate-rank",
                records=[(1, 1, 2000.0, 1)],
                rank_records=[
                    (1, 1, 1, "node1", 0, 0, 1, 2000.0),
                    (1, 1, 1, "node1", 0, 0, 1, 1900.0),
                ],
            )
            with self.assertRaises(stalls.StallAnalysisError):
                stalls.analyze_case(duplicate, 1000.0)

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

    def test_missing_and_empty_phase_files_warn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = write_case(Path(tmp), "missing-phase", records=[(1, 1, 1500.0, 1)])
            missing_summary = stalls.analyze_case(missing, 1000.0)
            self.assertTrue(any("missing iteration phase" in w for w in missing_summary.warnings))

            empty = write_case(Path(tmp), "empty-phase", records=[(1, 1, 1500.0, 1)])
            (empty / "iteration-phase-times.csv").write_text("", encoding="utf-8")
            empty_summary = stalls.analyze_case(empty, 1000.0)
            self.assertTrue(any("empty iteration phase" in w for w in empty_summary.warnings))

    def test_inconsistent_sample_metadata_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_case(Path(tmp), "bad", samples=2)
            payload = json.loads((case / "ghalo.json").read_text(encoding="utf-8"))
            payload["results"][1]["sample_index"] = 1
            (case / "ghalo.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(stalls.StallAnalysisError):
                stalls.analyze_case(case, 1000.0)

    def test_inconsistent_phase_metadata_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = write_case(
                Path(tmp),
                "bad-phase",
                records=[(1, 1, 1500.0, 1)],
                phase_records=[(99, 1, "east_west_sync", 1500.0, 1)],
            )
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
