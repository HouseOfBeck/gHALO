#!/usr/bin/env python3
"""Portable analysis tooling for gHALO result directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

GIB = 1024.0**3

REQUIRED_FIELDS = (
    "backend",
    "algorithm",
    "halo_words",
    "total_exchange_bytes_per_rank",
    "iterations",
    "max_average_seconds",
)

PHASE_FIELDS = (
    "input_device_copy_seconds",
    "north_south_mpi_seconds",
    "north_south_sync_seconds",
    "transpose_device_copy_seconds",
    "transpose_copy_sync_seconds",
    "east_west_mpi_seconds",
    "east_west_sync_seconds",
    "phase_sum_seconds",
    "unattributed_seconds",
)


class AnalysisError(RuntimeError):
    """Raised for invalid or incomplete result inputs."""


@dataclass
class RunResult:
    halo_words: int
    backend: str
    algorithm: str
    iterations: int
    max_average_seconds: float
    max_total_seconds: float
    total_exchange_bytes_per_rank: int
    word_bytes: Optional[int] = None
    n_message_bytes: Optional[int] = None
    two_n_message_bytes: Optional[int] = None
    topology: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    phase_timing: Optional[Dict[str, float]] = None


@dataclass
class Run:
    input_path: Path
    result_path: Path
    version: str
    results: List[RunResult]
    metadata_files: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    source_format: str = "json"

    @property
    def label(self) -> str:
        system = display_system(self.system)
        nodes = self.nodes
        ranks = self.ranks
        if system or nodes or ranks:
            parts = [system] if system else []
            if nodes is not None:
                parts.append(plural(nodes, "node"))
            if ranks is not None:
                rank_kind = "GPU rank" if self.memory_location == "device" else "rank"
                parts.append(plural(ranks, rank_kind))
            if "repeat" in self.result_path.name.lower():
                parts.append("repeat")
            return " | ".join(parts)
        return self.input_path.name

    @property
    def backend(self) -> str:
        return self.results[0].backend if self.results else ""

    @property
    def algorithm(self) -> str:
        return self.results[0].algorithm if self.results else ""

    @property
    def memory_location(self) -> str:
        if not self.results:
            return ""
        return str(self.results[0].metadata.get("memory_location", ""))

    @property
    def ranks(self) -> Optional[int]:
        if not self.results:
            return None
        value = self.results[0].topology.get("world_size")
        return int(value) if value not in (None, "") else None

    @property
    def rows(self) -> Optional[int]:
        if not self.results:
            return None
        value = self.results[0].topology.get("rows")
        return int(value) if value not in (None, "") else None

    @property
    def cols(self) -> Optional[int]:
        if not self.results:
            return None
        value = self.results[0].topology.get("cols")
        return int(value) if value not in (None, "") else None

    @property
    def cartesian(self) -> str:
        if self.rows is None or self.cols is None:
            return ""
        return f"{self.rows}x{self.cols}"

    @property
    def system(self) -> str:
        return self.meta_value("active_system") or infer_system_from_path(self.result_path)

    @property
    def build_system(self) -> str:
        return self.meta_value("build_system") or ""

    @property
    def nodes(self) -> Optional[int]:
        value = self.meta_value("nodes")
        if value is None:
            value = self.meta_value("NodeCnt", from_file="slurm-job.txt")
        return parse_int(value)

    @property
    def ranks_per_node(self) -> Optional[int]:
        nodes = self.nodes
        ranks = self.ranks
        if nodes and ranks and ranks % nodes == 0:
            return ranks // nodes
        return parse_int(self.meta_value("ranks_per_node"))

    @property
    def git_commit(self) -> str:
        return (
            self.meta_value("commit", from_file="git.txt")
            or self.meta_value("git_commit")
            or ""
        )

    @property
    def slurm_job_id(self) -> str:
        return self.meta_value("slurm_job_id", from_file="submission.txt") or ""

    @property
    def phase_timing_enabled(self) -> bool:
        return any(result.phase_timing for result in self.results)

    def meta_value(self, key: str, from_file: Optional[str] = None) -> Optional[str]:
        files = [from_file] if from_file else list(self.metadata_files)
        for filename in files:
            if not filename:
                continue
            values = parse_metadata_text(self.metadata_files.get(filename, ""))
            if key in values:
                return values[key]
            if filename == "slurm-job.txt":
                slurm_values = parse_slurm_text(self.metadata_files.get(filename, ""))
                if key in slurm_values:
                    return slurm_values[key]
        return None


def parse_int(value: Optional[str]) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def infer_system_from_path(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        if index + 1 < len(parts):
            return parts[index + 1]
    return ""


def display_system(system: str) -> str:
    if not system:
        return ""
    known = {"frontier": "Frontier", "borg": "Borg"}
    return known.get(system.lower(), system)


def display_backend(backend: str) -> str:
    if "hip" in backend.lower():
        return "MPI-HIP"
    if "mpi" in backend.lower():
        return "MPI"
    return backend


def plural(count: int, unit: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {unit}{suffix}"


def parse_metadata_text(text: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_slurm_text(text: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for token in text.replace("\n", " ").split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    return values


def load_run(path_text: str) -> Run:
    path = Path(path_text)
    result_path = path
    source = path
    source_format = "json"
    warnings: List[str] = []

    if path.is_dir():
        json_path = path / "ghalo.json"
        csv_path = path / "ghalo.csv"
        if json_path.exists():
            source = json_path
            source_format = "json"
        elif csv_path.exists():
            source = csv_path
            source_format = "csv"
        else:
            raise AnalysisError(f"{path}: result directory lacks ghalo.json or ghalo.csv")
    elif path.suffix.lower() == ".json":
        result_path = path.parent
        source_format = "json"
    elif path.suffix.lower() == ".csv":
        result_path = path.parent
        source_format = "csv"
    else:
        raise AnalysisError(f"{path}: expected result directory, ghalo.json, or ghalo.csv")

    metadata_files = read_metadata_files(result_path)
    exit_status = metadata_files.get("exit-status.txt")
    if exit_status is not None:
        stripped = exit_status.strip()
        if stripped and stripped != "0":
            warnings.append(f"{result_path}: exit-status.txt is nonzero ({stripped})")

    if source_format == "json":
        run = load_json_run(source, result_path, metadata_files, warnings)
    else:
        run = load_csv_run(source, result_path, metadata_files, warnings)
    validate_run(run)
    return run


def read_metadata_files(result_path: Path) -> Dict[str, str]:
    names = (
        "system-resolution.txt",
        "submission.txt",
        "environment.txt",
        "git.txt",
        "slurm-job.txt",
        "command.txt",
        "exit-status.txt",
    )
    files: Dict[str, str] = {}
    if result_path.is_dir():
        for name in names:
            candidate = result_path / name
            if candidate.exists():
                files[name] = candidate.read_text(encoding="utf-8", errors="replace")
    return files


def load_json_run(
    source: Path,
    result_path: Path,
    metadata_files: Dict[str, str],
    warnings: List[str],
) -> Run:
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise AnalysisError(f"{source}: malformed JSON: {error}") from error

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise AnalysisError(f"{source}: missing results array")

    results = [result_from_json(item, source) for item in raw_results]
    return Run(
        input_path=source,
        result_path=result_path,
        version=str(payload.get("version", "")),
        results=results,
        metadata_files=metadata_files,
        warnings=warnings,
        source_format="json",
    )


def result_from_json(item: Dict[str, Any], source: Path) -> RunResult:
    if not isinstance(item, dict):
        raise AnalysisError(f"{source}: result entry is not an object")
    for field_name in REQUIRED_FIELDS:
        if field_name not in item:
            raise AnalysisError(f"{source}: missing required result field {field_name}")
    phase = item.get("phase_timing")
    return RunResult(
        halo_words=int(item["halo_words"]),
        backend=str(item["backend"]),
        algorithm=str(item["algorithm"]),
        iterations=int(item["iterations"]),
        max_average_seconds=float(item["max_average_seconds"]),
        max_total_seconds=float(item.get("max_total_seconds", 0.0)),
        total_exchange_bytes_per_rank=int(item["total_exchange_bytes_per_rank"]),
        word_bytes=optional_int(item.get("word_bytes")),
        n_message_bytes=optional_int(item.get("n_message_bytes")),
        two_n_message_bytes=optional_int(item.get("two_n_message_bytes")),
        topology=dict(item.get("topology", {})),
        metadata=dict(item.get("metadata", {})),
        phase_timing={name: float(phase.get(name, 0.0)) for name in PHASE_FIELDS}
        if isinstance(phase, dict)
        else None,
    )


def load_csv_run(
    source: Path,
    result_path: Path,
    metadata_files: Dict[str, str],
    warnings: List[str],
) -> Run:
    with source.open(newline="", encoding="utf-8") as handle:
      reader = csv.DictReader(handle)
      rows = list(reader)
    if not rows:
        raise AnalysisError(f"{source}: empty CSV result set")
    results = [result_from_csv(row, source) for row in rows]
    return Run(
        input_path=source,
        result_path=result_path,
        version=str(rows[0].get("version", "")),
        results=results,
        metadata_files=metadata_files,
        warnings=warnings,
        source_format="csv",
    )


def result_from_csv(row: Dict[str, str], source: Path) -> RunResult:
    for field_name in REQUIRED_FIELDS:
        if row.get(field_name) in (None, ""):
            raise AnalysisError(f"{source}: missing required result field {field_name}")
    topology = {
        "world_size": optional_int(row.get("world_size")),
        "world_rank": optional_int(row.get("root_world_rank")),
        "cart_rank": optional_int(row.get("root_cart_rank")),
        "rows": optional_int(row.get("rows")),
        "cols": optional_int(row.get("cols")),
        "row": optional_int(row.get("root_row")),
        "col": optional_int(row.get("root_col")),
        "north": optional_int(row.get("root_north")),
        "south": optional_int(row.get("root_south")),
        "east": optional_int(row.get("root_east")),
        "west": optional_int(row.get("root_west")),
    }
    metadata = {
        "memory_location": row.get("memory_location", ""),
        "mpi_library_version": unquote_csv_string(row.get("mpi_library_version", "")),
        "hip_runtime_version": unquote_csv_string(row.get("hip_runtime_version", "")),
        "validation_enabled": row.get("validation_enabled", ""),
        "validation_passed": row.get("validation_passed", ""),
    }
    phase = None
    if any(row.get("phase_" + name) for name in PHASE_FIELDS):
        phase = {
            name: float(row.get("phase_" + name) or 0.0) for name in PHASE_FIELDS
        }
    return RunResult(
        halo_words=int(row["halo_words"]),
        backend=row["backend"],
        algorithm=row["algorithm"],
        iterations=int(row["iterations"]),
        max_average_seconds=float(row["max_average_seconds"]),
        max_total_seconds=float(row.get("max_total_seconds") or 0.0),
        total_exchange_bytes_per_rank=int(row["total_exchange_bytes_per_rank"]),
        word_bytes=optional_int(row.get("word_bytes")),
        n_message_bytes=optional_int(row.get("n_message_bytes")),
        two_n_message_bytes=optional_int(row.get("two_n_message_bytes")),
        topology=topology,
        metadata=metadata,
        phase_timing=phase,
    )


def optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


def unquote_csv_string(value: str) -> str:
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return value


def validate_run(run: Run) -> None:
    if not run.results:
        raise AnalysisError(f"{run.input_path}: empty result set")
    seen = set()
    rank_count = run.results[0].topology.get("world_size")
    for result in run.results:
        if result.halo_words in seen:
            raise AnalysisError(f"{run.input_path}: duplicate halo size {result.halo_words}")
        seen.add(result.halo_words)
        if result.max_average_seconds <= 0.0:
            raise AnalysisError(f"{run.input_path}: nonpositive max_average_seconds")
        if rank_count != result.topology.get("world_size"):
            raise AnalysisError(f"{run.input_path}: inconsistent rank counts")


def effective_bytes_per_second(result: RunResult) -> float:
    return result.total_exchange_bytes_per_rank / result.max_average_seconds


def effective_gib_per_second(result: RunResult) -> float:
    return effective_bytes_per_second(result) / GIB


def phase_categories(result: RunResult) -> Dict[str, float]:
    phase = result.phase_timing or {}
    device_copy = (
        phase.get("input_device_copy_seconds", 0.0)
        + phase.get("transpose_device_copy_seconds", 0.0)
    )
    mpi = (
        phase.get("north_south_mpi_seconds", 0.0)
        + phase.get("east_west_mpi_seconds", 0.0)
    )
    sync = (
        phase.get("north_south_sync_seconds", 0.0)
        + phase.get("transpose_copy_sync_seconds", 0.0)
        + phase.get("east_west_sync_seconds", 0.0)
    )
    return {
        "device_copy_total_seconds": device_copy,
        "mpi_total_seconds": mpi,
        "synchronization_total_seconds": sync,
        "total_minus_sum_of_phase_maxima_seconds": phase.get(
            "unattributed_seconds",
            result.max_average_seconds - phase.get("phase_sum_seconds", 0.0),
        ),
    }


def summary_rows(
    run: Run,
    include_metadata: bool = False,
    phase_timing: bool = False,
    label_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows = []
    for result in sorted(run.results, key=lambda item: item.halo_words):
        row: Dict[str, Any] = {
            "label": label_override or run.label,
            "system": run.system,
            "backend": result.backend,
            "algorithm": result.algorithm,
            "memory_type": result.metadata.get("memory_location", ""),
            "ranks": result.topology.get("world_size", ""),
            "cartesian_dimensions": f"{result.topology.get('rows', '')}x{result.topology.get('cols', '')}",
            "nodes": run.nodes if run.nodes is not None else "",
            "ranks_per_node": run.ranks_per_node if run.ranks_per_node is not None else "",
            "halo_words": result.halo_words,
            "iterations": result.iterations,
            "max_average_seconds": result.max_average_seconds,
            "microseconds": result.max_average_seconds * 1.0e6,
            "bytes_per_rank": result.total_exchange_bytes_per_rank,
            "per_rank_effective_transferred_byte_rate_Bps": effective_bytes_per_second(result),
            "per_rank_effective_transferred_byte_rate_GiBps": effective_gib_per_second(result),
        }
        if include_metadata:
            row.update(
                {
                    "build_system": run.build_system,
                    "git_commit": run.git_commit,
                    "slurm_job_id": run.slurm_job_id,
                    "source_format": run.source_format,
                    "input_path": str(run.input_path),
                }
            )
        if phase_timing and result.phase_timing:
            for name in PHASE_FIELDS:
                row["phase_" + name] = result.phase_timing.get(name, 0.0)
            row.update(phase_categories(result))
        rows.append(row)
    return rows


def load_runs(paths: Sequence[str]) -> List[Run]:
    if not paths:
        raise AnalysisError("at least one input path is required")
    return [load_run(path) for path in paths]


def write_output(rows_or_payload: Any, fmt: str, output: Optional[str]) -> None:
    if fmt == "json":
        text = json.dumps(rows_or_payload, indent=2, sort_keys=True) + "\n"
    elif fmt == "csv":
        rows = ensure_rows(rows_or_payload)
        text = rows_to_csv(rows)
    elif fmt == "markdown":
        text = rows_to_markdown(ensure_rows(rows_or_payload))
    elif fmt == "table":
        text = rows_to_table(ensure_rows(rows_or_payload))
    else:
        raise AnalysisError(f"unsupported output format {fmt}")
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def ensure_rows(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and "rows" in value:
        return list(value["rows"])
    raise AnalysisError("selected format requires tabular rows")


def rows_to_csv(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    import io

    output = io.StringIO()
    fields = ordered_fields(rows)
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def rows_to_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._\n"
    fields = ordered_fields(rows)
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def rows_to_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No rows.\n"
    fields = ordered_fields(rows)
    widths = {
        field: max(len(field), *(len(format_cell(row.get(field, ""))) for row in rows))
        for field in fields
    }
    lines = ["  ".join(field.ljust(widths[field]) for field in fields)]
    lines.append("  ".join("-" * widths[field] for field in fields))
    for row in rows:
        lines.append("  ".join(format_cell(row.get(field, "")).ljust(widths[field]) for field in fields))
    return "\n".join(lines) + "\n"


def ordered_fields(rows: List[Dict[str, Any]]) -> List[str]:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def command_summarize(args: argparse.Namespace) -> int:
    runs = load_runs(args.inputs)
    if args.output_dir:
        write_analysis_directory(args.output_dir, runs)
    rows: List[Dict[str, Any]] = []
    for run in runs:
        rows.extend(
            summary_rows(
                run,
                args.include_metadata,
                args.phase_timing,
                args.label,
            )
        )
    write_output(rows, args.format, args.output)
    write_warnings(runs)
    return 0


def results_by_halo(run: Run) -> Dict[int, RunResult]:
    return {result.halo_words: result for result in run.results}


def configuration_signature(run: Run) -> Dict[str, Any]:
    return {
        "active_system": run.system,
        "build_system": run.build_system,
        "backend": run.backend,
        "ranks": run.ranks,
        "nodes": run.nodes,
        "ranks_per_node": run.ranks_per_node,
        "cartesian_dimensions": run.cartesian,
        "git_commit": run.git_commit,
        "phase_timing_enabled": run.phase_timing_enabled,
        "target_seconds": extract_target_seconds(run),
    }


def extract_target_seconds(run: Run) -> str:
    command = run.metadata_files.get("command.txt", "")
    parts = command.split()
    for index, part in enumerate(parts[:-1]):
        if part == "--target-seconds":
            return parts[index + 1]
    return ""


def configuration_differences(a: Run, b: Run) -> Dict[str, Tuple[Any, Any]]:
    left = configuration_signature(a)
    right = configuration_signature(b)
    return {
        key: (left[key], right[key])
        for key in left
        if left.get(key) != right.get(key)
    }


def label_overrides(args: argparse.Namespace, count: int) -> List[Optional[str]]:
    labels: List[str] = []
    for value in getattr(args, "label", None) or []:
        labels.append(value)
    comma_labels = getattr(args, "labels", None)
    if comma_labels:
        labels.extend(label.strip() for label in comma_labels.split(","))
    if labels and len(labels) != count:
        raise AnalysisError(
            f"received {len(labels)} label override(s) for {count} input run(s)"
        )
    return labels or [None for _ in range(count)]


def run_labels(runs: Sequence[Run], overrides: Sequence[Optional[str]]) -> List[str]:
    labels = [override or run.label for run, override in zip(runs, overrides)]
    seen: Dict[str, int] = {}
    unique = []
    for label in labels:
        count = seen.get(label, 0)
        seen[label] = count + 1
        unique.append(label if count == 0 else f"{label} | repeat {count + 1}")
    return unique


def compare_rows(a: Run, b: Run, allow_missing: bool = False) -> List[Dict[str, Any]]:
    a_results = results_by_halo(a)
    b_results = results_by_halo(b)
    halos = sorted(set(a_results) | set(b_results)) if allow_missing else sorted(set(a_results) & set(b_results))
    if not allow_missing and set(a_results) != set(b_results):
        raise AnalysisError("compare requires matching halo sizes; use --allow-missing-sizes to relax")
    differences = configuration_differences(a, b)
    rows = []
    for halo in halos:
        ra = a_results.get(halo)
        rb = b_results.get(halo)
        if ra is None or rb is None:
            rows.append({"halo_words": halo, "missing": "A" if ra is None else "B"})
            continue
        diff = rb.max_average_seconds - ra.max_average_seconds
        percent = (diff / ra.max_average_seconds) * 100.0
        rows.append(
            {
                "halo_words": halo,
                "time_A_seconds": ra.max_average_seconds,
                "time_B_seconds": rb.max_average_seconds,
                "absolute_difference_seconds": diff,
                "observed_percent_difference": percent,
                "ratio_B_over_A": rb.max_average_seconds / ra.max_average_seconds,
                "bytes_per_rank_A": ra.total_exchange_bytes_per_rank,
                "bytes_per_rank_B": rb.total_exchange_bytes_per_rank,
                "iterations_A": ra.iterations,
                "iterations_B": rb.iterations,
                "configuration_metadata_differs": bool(differences),
            }
        )
    return rows


def command_compare(args: argparse.Namespace) -> int:
    a, b = load_runs([args.run_a, args.run_b])
    rows = compare_rows(a, b, args.allow_missing_sizes)
    write_output(rows, args.format, args.output)
    write_warnings([a, b])
    if args.fail_on_regression:
        for row in rows:
            if row.get("observed_percent_difference", 0.0) > args.threshold_percent:
                return 1
    return 0


def percentile(values: Sequence[float], fraction: float) -> float:
    """Inclusive linear percentile over sorted data using standard-library math."""
    if not values:
        raise AnalysisError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def compatibility_key(run: Run) -> Tuple[Any, ...]:
    bytes_by_halo = tuple(
        (result.halo_words, result.total_exchange_bytes_per_rank)
        for result in sorted(run.results, key=lambda item: item.halo_words)
    )
    return (
        run.backend,
        run.ranks,
        run.nodes,
        run.cartesian,
        bytes_by_halo,
        run.phase_timing_enabled,
    )


def grouping_key(run: Run, group_by: Sequence[str]) -> Tuple[Any, ...]:
    values = []
    for name in group_by:
        if name == "system":
            values.append(run.system)
        elif name == "backend":
            values.append(run.backend)
        elif name == "nodes":
            values.append(run.nodes)
        elif name == "ranks":
            values.append(run.ranks)
        else:
            raise AnalysisError(f"unsupported group-by field {name}")
    return tuple(values)


def aggregate_group_metadata(run: Run, key: Tuple[Any, ...], group_by: Sequence[str]) -> Dict[str, Any]:
    if group_by:
        metadata = {
            f"group_{name}": value for name, value in zip(group_by, key)
        }
        label_parts = []
        for name, value in zip(group_by, key):
            if value in (None, ""):
                continue
            if name == "system":
                label_parts.append(display_system(str(value)))
            elif name == "nodes":
                label_parts.append(plural(int(value), "node"))
            elif name == "ranks":
                label_parts.append(plural(int(value), "rank"))
            else:
                label_parts.append(str(value))
        metadata["group"] = " | ".join(label_parts) if label_parts else "all runs"
        return metadata

    bytes_by_halo = ";".join(
        f"N={result.halo_words}:{result.total_exchange_bytes_per_rank}B/rank"
        for result in sorted(run.results, key=lambda item: item.halo_words)
    )
    return {
        "group": run.label,
        "group_backend": run.backend,
        "group_ranks": run.ranks if run.ranks is not None else "",
        "group_nodes": run.nodes if run.nodes is not None else "",
        "group_cartesian_dimensions": run.cartesian,
        "group_phase_timing_enabled": run.phase_timing_enabled,
        "group_bytes_per_rank_by_halo": bytes_by_halo,
    }


def aggregate_rows(runs: List[Run], allow_mixed: bool, group_by: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, ...], List[Run]] = {}
    for run in runs:
        key = grouping_key(run, group_by) if allow_mixed and group_by else compatibility_key(run)
        groups.setdefault(key, []).append(run)
    if not allow_mixed and len(groups) > 1:
        raise AnalysisError("aggregate inputs have incompatible configurations; use --allow-mixed to group them")

    rows: List[Dict[str, Any]] = []
    for key, group_runs in groups.items():
        group_metadata = aggregate_group_metadata(group_runs[0], key, group_by)
        halos = sorted({result.halo_words for run in group_runs for result in run.results})
        for halo in halos:
            values = [results_by_halo(run)[halo].max_average_seconds for run in group_runs if halo in results_by_halo(run)]
            if not values:
                continue
            mean = statistics.fmean(values)
            row = {
                **group_metadata,
                "halo_words": halo,
                "count": len(values),
                "minimum_seconds": min(values),
                "maximum_seconds": max(values),
                "mean_seconds": mean,
                "median_seconds": statistics.median(values),
                "population_stddev_seconds": statistics.pstdev(values),
                "sample_stddev_seconds": statistics.stdev(values) if len(values) > 1 else "",
                "coefficient_of_variation": statistics.pstdev(values) / mean if mean else "",
                "first_quartile_seconds": percentile(values, 0.25),
                "third_quartile_seconds": percentile(values, 0.75),
            }
            rows.append(row)
    return rows


def command_aggregate(args: argparse.Namespace) -> int:
    runs = load_runs(args.inputs)
    rows = aggregate_rows(runs, args.allow_mixed, args.group_by)
    write_output(rows, args.format, args.output)
    write_warnings(runs)
    return 0


def choose_baseline(args: argparse.Namespace, runs: List[Run]) -> Run:
    if args.baseline_path:
        baseline = load_run(args.baseline_path)
        return baseline
    if args.baseline and args.baseline != "smallest-nodes":
        return load_run(args.baseline)
    if args.baseline_nodes is not None:
        matches = [run for run in runs if run.nodes == args.baseline_nodes]
        if not matches:
            raise AnalysisError(f"no run matched --baseline-nodes {args.baseline_nodes}")
        return matches[0]
    if args.baseline == "smallest-nodes":
        known = [run for run in runs if run.nodes is not None]
        if known:
            return min(known, key=lambda run: (run.nodes or 0, run.ranks or 0))
        return min(runs, key=lambda run: run.ranks or 0)
    raise AnalysisError(f"unsupported baseline {args.baseline}")


def scaling_rows(runs: List[Run], baseline: Run) -> List[Dict[str, Any]]:
    baseline_results = results_by_halo(baseline)
    rows = []
    for run in runs:
        for result in sorted(run.results, key=lambda item: item.halo_words):
            base = baseline_results.get(result.halo_words)
            if base is None:
                continue
            ratio = result.max_average_seconds / base.max_average_seconds
            rows.append(
                {
                    "system": run.system,
                    "nodes": run.nodes if run.nodes is not None else "",
                    "ranks": run.ranks if run.ranks is not None else "",
                    "ranks_per_node": run.ranks_per_node if run.ranks_per_node is not None else "",
                    "cartesian_dimensions": run.cartesian,
                    "halo_words": result.halo_words,
                    "time_microseconds": result.max_average_seconds * 1.0e6,
                    "ratio_relative_to_baseline": ratio,
                    "percent_increase_relative_to_baseline": (ratio - 1.0) * 100.0,
                    "per_rank_effective_transferred_byte_rate_Bps": effective_bytes_per_second(result),
                    "per_rank_effective_transferred_byte_rate_GiBps": effective_gib_per_second(result),
                    "baseline_label": baseline.label,
                }
            )
    return rows


def command_scaling(args: argparse.Namespace) -> int:
    runs = load_runs(args.inputs)
    baseline = choose_baseline(args, runs)
    rows = scaling_rows(runs, baseline)
    write_output(rows, args.format, args.output)
    write_warnings(runs)
    return 0


def percent_difference_rows_for_plot(runs: Sequence[Run]) -> List[Dict[str, Any]]:
    if len(runs) != 2:
        raise AnalysisError("percent-difference plot requires exactly two input runs")
    rows = compare_rows(runs[0], runs[1], allow_missing=False)
    for row in rows:
        if row.get("bytes_per_rank_A") != row.get("bytes_per_rank_B"):
            raise AnalysisError(
                "percent-difference plot requires matching bytes per rank for each halo size"
            )
    return rows


def default_plot_title(kind: str, runs: Sequence[Run]) -> str:
    if kind == "percent-difference":
        headline = "gHALO Observed Percent Difference"
    elif len(runs) > 1:
        headline = f"gHALO {display_backend(runs[0].backend)} Repeatability"
    else:
        headline = f"gHALO {display_backend(runs[0].backend)}"
    system = display_system(runs[0].system)
    nodes = runs[0].nodes
    ranks = runs[0].ranks
    details = []
    if system:
        details.append(system)
    if nodes is not None:
        details.append(plural(nodes, "Node"))
    if ranks is not None:
        rank_kind = "GPU Rank" if runs[0].memory_location == "device" else "Rank"
        details.append(plural(ranks, rank_kind))
    return headline if not details else f"{headline}\n{', '.join(details)}"


def set_halo_ticks(ax: Any, runs: Sequence[Run]) -> None:
    halos = sorted({result.halo_words for run in runs for result in run.results})
    if halos:
        ax.set_xticks(halos)
        ax.set_xticklabels([str(halo) for halo in halos])


def draw_percent_difference_plot(ax: Any, rows: Sequence[Dict[str, Any]]) -> None:
    x = [row["halo_words"] for row in rows]
    y = [row["observed_percent_difference"] for row in rows]
    ax.plot(x, y, marker="o", label="Observed percent difference")
    ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Halo Length (elements)")
    ax.set_ylabel("Observed Percent Difference")


def command_plot(args: argparse.Namespace) -> int:
    runs = load_runs(args.inputs)
    overrides = label_overrides(args, len(runs))
    labels = run_labels(runs, overrides)
    percent_difference_rows = (
        percent_difference_rows_for_plot(runs)
        if args.kind == "percent-difference"
        else None
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise AnalysisError(
            "plot output requires matplotlib; install matplotlib or use summarize/compare/aggregate/scaling formats"
        )

    fig, ax = plt.subplots()
    if args.kind in ("latency", "effective-rate"):
        for run, label in zip(runs, labels):
            ordered = sorted(run.results, key=lambda item: item.halo_words)
            x = [result.halo_words for result in ordered]
            if args.kind == "latency":
                y = [result.max_average_seconds * 1.0e6 for result in ordered]
                ax.set_ylabel("Maximum Average Exchange Time (µs)")
            else:
                y = [effective_gib_per_second(result) for result in ordered]
                ax.set_ylabel("per-rank effective transferred-byte rate (GiB/s)")
            ax.plot(x, y, marker="o", label=label)
        ax.set_xlabel("Halo Length (elements)")
        set_halo_ticks(ax, runs)
    elif args.kind == "scaling":
        baseline = choose_baseline(args, runs)
        rows = scaling_rows(runs, baseline)
        for halo in sorted({row["halo_words"] for row in rows}):
            series = [row for row in rows if row["halo_words"] == halo]
            x = [row["nodes"] or row["ranks"] for row in series]
            y = [row["time_microseconds"] for row in series]
            ax.plot(x, y, marker="o", label=f"N={halo}")
        ax.set_xlabel("nodes or ranks")
        ax.set_ylabel("max average time (microseconds)")
    elif args.kind == "percent-difference":
        draw_percent_difference_plot(ax, percent_difference_rows or [])
        set_halo_ticks(ax, runs)
    elif args.kind == "phase":
        run = runs[0]
        phase_results = [result for result in run.results if result.phase_timing]
        if not phase_results:
            raise AnalysisError("phase plot requires a run with phase_timing results")
        labels = [str(result.halo_words) for result in phase_results]
        bottoms = [0.0 for _ in phase_results]
        for name in PHASE_FIELDS[:7]:
            values = [result.phase_timing.get(name, 0.0) * 1.0e6 for result in phase_results]  # type: ignore[union-attr]
            ax.bar(labels, values, bottom=bottoms, label=name)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        ax.set_xlabel("halo length N")
        ax.set_ylabel("independently reduced phase maxima (microseconds)")
    else:
        raise AnalysisError(f"unsupported plot kind {args.kind}")

    xscale = "linear" if args.kind == "phase" and args.xscale == "log2" else args.xscale
    apply_axis_scale(ax, "x", xscale)
    apply_axis_scale(ax, "y", args.yscale)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(args.title or default_plot_title(args.kind, runs))
    ax.legend()
    fig.tight_layout()
    output = args.output or f"{args.kind}.png"
    fig.savefig(output, dpi=args.dpi)
    return 0


def write_warnings(runs: Iterable[Run]) -> None:
    for run in runs:
        for warning in run.warnings:
            print(f"warning: {warning}", file=sys.stderr)


def checksum(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def write_analysis_directory(output_root: str, runs: List[Run]) -> None:
    output_dir = Path(output_root) / "analysis"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary = [row for run in runs for row in summary_rows(run, True, True)]
    (output_dir / "summary.csv").write_text(rows_to_csv(summary), encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(rows_to_markdown(summary), encoding="utf-8")
    if len(runs) >= 2:
        try:
            comparison = compare_rows(runs[0], runs[1], allow_missing=False)
            (output_dir / "comparison.csv").write_text(
                rows_to_csv(comparison),
                encoding="utf-8",
            )
        except AnalysisError:
            pass
        try:
            aggregate = aggregate_rows(runs, allow_mixed=False, group_by=[])
            (output_dir / "aggregate.csv").write_text(
                rows_to_csv(aggregate),
                encoding="utf-8",
            )
        except AnalysisError:
            pass
    provenance = {
        "input_paths": [str(run.input_path) for run in runs],
        "input_checksums": {str(run.input_path): checksum(run.input_path) for run in runs},
        "analysis_command": sys.argv,
        "analysis_tool_git_commit": os.environ.get("GITHUB_SHA", ""),
        "generation_time_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "matplotlib_version": matplotlib_version(),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def matplotlib_version() -> str:
    try:
        import matplotlib

        return str(matplotlib.__version__)
    except ImportError:
        return ""


def add_common_format_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("table", "csv", "json", "markdown"), default="table")
    parser.add_argument("--output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze gHALO result directories and output files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize", help="summarize one or more runs")
    add_common_format_options(summarize)
    summarize.add_argument("--include-metadata", action="store_true")
    summarize.add_argument("--phase-timing", action="store_true")
    summarize.add_argument("--output-dir")
    summarize.add_argument("--label")
    summarize.add_argument("inputs", nargs="+")
    summarize.set_defaults(func=command_summarize)

    compare = subparsers.add_parser("compare", help="compare two runs")
    add_common_format_options(compare)
    compare.add_argument("--allow-missing-sizes", action="store_true")
    compare.add_argument("--threshold-percent", type=float, default=0.0)
    compare.add_argument("--fail-on-regression", action="store_true")
    compare.add_argument("run_a")
    compare.add_argument("run_b")
    compare.set_defaults(func=command_compare)

    aggregate = subparsers.add_parser("aggregate", help="aggregate repeated runs")
    add_common_format_options(aggregate)
    aggregate.add_argument("--group-by", action="append", choices=("system", "backend", "nodes", "ranks"), default=[])
    aggregate.add_argument("--allow-mixed", action="store_true")
    aggregate.add_argument("inputs", nargs="+")
    aggregate.set_defaults(func=command_aggregate)

    scaling = subparsers.add_parser("scaling", help="compare fixed-message-size scaling")
    add_common_format_options(scaling)
    scaling.add_argument("--baseline", default="smallest-nodes")
    scaling.add_argument("--baseline-path")
    scaling.add_argument("--baseline-nodes", type=int)
    scaling.add_argument("inputs", nargs="+")
    scaling.set_defaults(func=command_scaling)

    plot = subparsers.add_parser("plot", help="create optional matplotlib plots")
    plot.add_argument("--kind", choices=("latency", "effective-rate", "scaling", "phase", "percent-difference"), default="latency")
    plot.add_argument("--xscale", choices=("linear", "log2", "log10"), default="log2")
    plot.add_argument("--yscale", choices=("linear", "log10"), default="linear")
    plot.add_argument("--title")
    plot.add_argument("--label", action="append", default=[])
    plot.add_argument("--labels")
    plot.add_argument("--output")
    plot.add_argument("--dpi", type=int, default=120)
    plot.add_argument("--baseline", default="smallest-nodes")
    plot.add_argument("--baseline-path")
    plot.add_argument("--baseline-nodes", type=int)
    plot.add_argument("inputs", nargs="+")
    plot.set_defaults(func=command_plot)

    return parser


def apply_axis_scale(ax: Any, axis: str, scale: str) -> None:
    setter = ax.set_xscale if axis == "x" else ax.set_yscale
    if scale == "linear":
        setter("linear")
    elif scale == "log2":
        setter("log", base=2)
    elif scale == "log10":
        setter("log", base=10)
    else:
        raise AnalysisError(f"unsupported {axis}-axis scale {scale}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AnalysisError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
