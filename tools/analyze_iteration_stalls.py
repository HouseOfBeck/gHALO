#!/usr/bin/env python3
"""Analyze opt-in gHALO per-iteration timing diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class StallAnalysisError(RuntimeError):
    """Raised for malformed iteration timing inputs."""


@dataclass(frozen=True, order=True)
class SampleKey:
    halo_words: int
    sample_index: int


@dataclass
class SampleInfo:
    key: SampleKey
    iterations: int
    sample_count: int


@dataclass
class IterationRecord:
    halo_words: int
    sample_index: int
    iteration_index: int
    duration_seconds: float
    max_rank: int
    backend: str
    rccl_sync_mode: str
    world_size: int

    @property
    def duration_us(self) -> float:
        return self.duration_seconds * 1.0e6


@dataclass
class PhaseRecord:
    halo_words: int
    sample_index: int
    iteration_index: int
    phase_name: str
    phase_seconds: float
    phase_max_rank: int
    total_iteration_seconds: float
    total_iteration_max_rank: int
    backend_schema: str

    @property
    def phase_us(self) -> float:
        return self.phase_seconds * 1.0e6


@dataclass
class PhaseSummary:
    phase_name: str
    dominant_iterations: int
    cumulative_ms: float
    median_us: float
    max_us: float
    max_rank_frequency: Dict[int, int]


@dataclass
class CaseSummary:
    result_dir: Path
    label: str
    backend: str
    rccl_sync_mode: str
    system: str
    rocm_version: str
    category: str
    threshold_us: float
    samples_found: int
    total_measured_iterations: int
    emitted_iteration_records: int
    stall_count: int
    stalled_iteration_percent: float
    samples_with_stalls: int
    samples_with_stalls_percent: float
    max_stalls_in_one_sample: int
    longest_stall_free_sequence: int
    min_stall_duration_us: Optional[float]
    median_stall_duration_us: Optional[float]
    mean_stall_duration_us: Optional[float]
    max_stall_duration_us: Optional[float]
    rank_frequency: Dict[int, int]
    sample_sequence: str
    affected_indices: List[Tuple[int, int, int, float, int]]
    isolated_bad_samples: int
    multiple_bad_samples: int
    nearly_all_slow_samples: int
    stalled_iterations_with_phase_data: int = 0
    phase_summaries: List[PhaseSummary] = field(default_factory=list)
    phase_iteration_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    stall_cause_classification: str = "mixed/unattributed"
    warnings: List[str] = field(default_factory=list)


def parse_key_value_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def normalize_rocm(value: str) -> str:
    value = value.strip()
    if value.startswith("rocm-"):
        return value
    return f"rocm-{value}" if value else ""


def result_dirs_from_inputs(inputs: Sequence[str]) -> List[Path]:
    discovered: List[Path] = []
    for text in inputs:
        path = Path(text)
        if (path / "ghalo.json").exists():
            discovered.append(path)
        elif path.is_dir():
            discovered.extend(sorted(p.parent for p in path.rglob("ghalo.json")))
        else:
            raise StallAnalysisError(f"input path does not exist: {path}")
    return unique_paths(discovered)


def discover_result_dirs(
    root: Path, system: str, rocm_version: str, category: str, label_filter: str
) -> List[Path]:
    base = root / "results" / system / normalize_rocm(rocm_version) / category
    if not base.exists():
        return []
    result_dirs = sorted(p.parent for p in base.rglob("ghalo.json"))
    if label_filter:
        result_dirs = [p for p in result_dirs if label_filter in p.name]
    return unique_paths(result_dirs)


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    unique: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def load_samples(result_dir: Path) -> Tuple[List[SampleInfo], Dict[str, Any]]:
    json_path = result_dir / "ghalo.json"
    if not json_path.exists() or json_path.stat().st_size == 0:
        raise StallAnalysisError(f"missing or empty ghalo.json: {json_path}")
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise StallAnalysisError(f"ghalo.json lacks a non-empty results array: {json_path}")

    samples: List[SampleInfo] = []
    seen = set()
    for result in results:
        key = SampleKey(
            int(result.get("halo_words", 0)),
            int(result.get("sample_index", 1)),
        )
        if key in seen:
            raise StallAnalysisError(
                f"inconsistent sample metadata in {result_dir}: duplicate halo "
                f"{key.halo_words} sample {key.sample_index}"
            )
        seen.add(key)
        samples.append(
            SampleInfo(
                key=key,
                iterations=int(result.get("iterations", 0)),
                sample_count=int(result.get("sample_count", 1)),
            )
        )
    return sorted(samples, key=lambda sample: sample.key), payload


def load_iteration_records(path: Path) -> Tuple[List[IterationRecord], List[str]]:
    warnings: List[str] = []
    if not path.exists():
        warnings.append(f"missing iteration timing file: {path}")
        return [], warnings
    if path.stat().st_size == 0:
        warnings.append(f"empty iteration timing file: {path}")
        return [], warnings

    records: List[IterationRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "halo_words",
            "sample_index",
            "iteration_index",
            "global_max_iteration_seconds",
            "max_rank",
            "backend",
            "rccl_sync_mode",
            "world_size",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise StallAnalysisError(f"iteration-times.csv missing required columns: {path}")
        for row in reader:
            records.append(
                IterationRecord(
                    halo_words=int(row["halo_words"]),
                    sample_index=int(row["sample_index"]),
                    iteration_index=int(row["iteration_index"]),
                    duration_seconds=float(row["global_max_iteration_seconds"]),
                    max_rank=int(row["max_rank"]),
                    backend=row.get("backend", ""),
                    rccl_sync_mode=row.get("rccl_sync_mode", ""),
                    world_size=int(row.get("world_size", "0") or 0),
                )
            )
    return records, warnings


def load_phase_records(path: Path) -> Tuple[List[PhaseRecord], List[str]]:
    warnings: List[str] = []
    if not path.exists():
        warnings.append(f"missing iteration phase timing file: {path}")
        return [], warnings
    if path.stat().st_size == 0:
        warnings.append(f"empty iteration phase timing file: {path}")
        return [], warnings

    records: List[PhaseRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "halo_words",
            "sample_index",
            "iteration_index",
            "total_iteration_seconds",
            "total_iteration_max_rank",
            "backend_schema",
            "phase_name",
            "phase_seconds",
            "phase_max_rank",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise StallAnalysisError(
                f"iteration-phase-times.csv missing required columns: {path}"
            )
        for row in reader:
            records.append(
                PhaseRecord(
                    halo_words=int(row["halo_words"]),
                    sample_index=int(row["sample_index"]),
                    iteration_index=int(row["iteration_index"]),
                    phase_name=row["phase_name"],
                    phase_seconds=float(row["phase_seconds"]),
                    phase_max_rank=int(row["phase_max_rank"]),
                    total_iteration_seconds=float(row["total_iteration_seconds"]),
                    total_iteration_max_rank=int(row["total_iteration_max_rank"]),
                    backend_schema=row["backend_schema"],
                )
            )
    return records, warnings


def longest_stall_free_sequence(sequence: str) -> int:
    longest = 0
    current = 0
    for marker in sequence:
        if marker == "F":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def classify_sample(stalls: int, iterations: int) -> str:
    if stalls == 0:
        return "none"
    if stalls == 1:
        return "isolated"
    if iterations > 0 and stalls >= math.ceil(0.9 * iterations):
        return "nearly_all"
    return "multiple"


def phase_category(phase_name: str) -> str:
    if "sync" in phase_name:
        return "synchronization/final sync"
    if "copy" in phase_name or "transpose" in phase_name:
        return "copy/transpose"
    if "communication" in phase_name or "enqueue" in phase_name or "mpi" in phase_name:
        return "communication/enqueue"
    return "mixed/unattributed"


def analyze_phase_records(
    phase_records: Sequence[PhaseRecord],
    sample_by_key: Dict[SampleKey, SampleInfo],
    affected_iterations: Optional[set[Tuple[int, int, int]]] = None,
) -> Tuple[int, List[PhaseSummary], List[Dict[str, Any]], str, List[str]]:
    warnings: List[str] = []
    by_iteration: Dict[Tuple[int, int, int], List[PhaseRecord]] = defaultdict(list)
    for record in phase_records:
        key = SampleKey(record.halo_words, record.sample_index)
        if key not in sample_by_key:
            raise StallAnalysisError(
                "iteration phase record references missing sample metadata: "
                f"halo={record.halo_words} sample={record.sample_index}"
            )
        iteration_key = (record.halo_words, record.sample_index, record.iteration_index)
        if affected_iterations is not None and iteration_key not in affected_iterations:
            continue
        by_iteration[(record.halo_words, record.sample_index, record.iteration_index)].append(record)

    phase_values: Dict[str, List[PhaseRecord]] = defaultdict(list)
    dominant_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    breakdown: List[Dict[str, Any]] = []
    for key in sorted(by_iteration):
        records = by_iteration[key]
        for record in records:
            phase_values[record.phase_name].append(record)
        dominant = max(records, key=lambda item: (item.phase_seconds, -item.phase_max_rank, item.phase_name))
        dominant_counter[dominant.phase_name] += 1
        category_counter[phase_category(dominant.phase_name)] += 1
        breakdown.append(
            {
                "halo_words": key[0],
                "sample_index": key[1],
                "iteration_index": key[2],
                "dominant_phase": dominant.phase_name,
                "dominant_phase_us": dominant.phase_us,
                "dominant_phase_max_rank": dominant.phase_max_rank,
                "total_iteration_us": dominant.total_iteration_seconds * 1.0e6,
                "total_iteration_max_rank": dominant.total_iteration_max_rank,
            }
        )

    summaries: List[PhaseSummary] = []
    for phase_name in sorted(phase_values):
        records = phase_values[phase_name]
        phase_us = [record.phase_us for record in records]
        summaries.append(
            PhaseSummary(
                phase_name=phase_name,
                dominant_iterations=dominant_counter.get(phase_name, 0),
                cumulative_ms=sum(phase_us) / 1000.0,
                median_us=statistics.median(phase_us) if phase_us else 0.0,
                max_us=max(phase_us) if phase_us else 0.0,
                max_rank_frequency=dict(sorted(Counter(record.phase_max_rank for record in records).items())),
            )
        )

    if not by_iteration:
        return 0, [], [], "mixed/unattributed", warnings
    if category_counter:
        category, count = category_counter.most_common(1)[0]
        classification = category if count > len(by_iteration) / 2.0 else "mixed/unattributed"
    else:
        classification = "mixed/unattributed"
    return len(by_iteration), summaries, breakdown, classification, warnings


def analyze_case(result_dir: Path, threshold_override_us: Optional[float] = None) -> CaseSummary:
    samples, payload = load_samples(result_dir)
    result_metadata = parse_key_value_file(result_dir / "result-metadata.txt")
    system_metadata = parse_key_value_file(result_dir / "system-resolution.txt")
    records, warnings = load_iteration_records(result_dir / "iteration-times.csv")
    phase_records, phase_warnings = load_phase_records(
        result_dir / "iteration-phase-times.csv"
    )
    warnings.extend(phase_warnings)

    sample_by_key = {sample.key: sample for sample in samples}
    total_iterations = sum(sample.iterations for sample in samples)
    metadata_threshold = result_metadata.get("iteration_stall_threshold_us", "0")
    threshold_us = threshold_override_us
    if threshold_us is None:
        try:
            threshold_us = float(metadata_threshold)
        except ValueError:
            threshold_us = 0.0
    if threshold_us < 0.0:
        raise StallAnalysisError("--threshold-us must be nonnegative")

    emitted_count = len(records)
    stalls_by_sample: Dict[SampleKey, List[IterationRecord]] = defaultdict(list)
    rank_counter: Counter[int] = Counter()
    affected: List[Tuple[int, int, int, float, int]] = []
    affected_iteration_keys = set()
    stall_durations = []
    for record in records:
        key = SampleKey(record.halo_words, record.sample_index)
        if key not in sample_by_key:
            warnings.append(
                f"iteration record references missing sample metadata: "
                f"halo={record.halo_words} sample={record.sample_index}"
            )
        if record.duration_us >= threshold_us:
            stalls_by_sample[key].append(record)
            rank_counter[record.max_rank] += 1
            stall_durations.append(record.duration_us)
            affected.append(
                (
                    record.halo_words,
                    record.sample_index,
                    record.iteration_index,
                    record.duration_us,
                    record.max_rank,
                )
            )
            affected_iteration_keys.add(
                (record.halo_words, record.sample_index, record.iteration_index)
            )

    (
        stalled_iterations_with_phase_data,
        phase_summaries,
        phase_iteration_breakdown,
        stall_cause_classification,
        phase_analysis_warnings,
    ) = analyze_phase_records(phase_records, sample_by_key, affected_iteration_keys)
    warnings.extend(phase_analysis_warnings)

    sequence = "".join(
        "S" if sample.key in stalls_by_sample else "F" for sample in samples
    )
    stalled_samples = sum(1 for sample in samples if sample.key in stalls_by_sample)
    max_stalls = max((len(values) for values in stalls_by_sample.values()), default=0)

    isolated = 0
    multiple = 0
    nearly_all = 0
    for sample in samples:
        classification = classify_sample(
            len(stalls_by_sample.get(sample.key, [])), sample.iterations
        )
        if classification == "isolated":
            isolated += 1
        elif classification == "multiple":
            multiple += 1
        elif classification == "nearly_all":
            nearly_all += 1

    first_result = payload["results"][0]
    first_metadata = first_result.get("metadata", {})
    backend = first_result.get("backend", "")
    rccl_sync_mode = first_metadata.get("rccl_sync_mode", "")
    rocm_version = (
        first_metadata.get("rocm_version")
        or result_metadata.get("rocm_version", "")
        or infer_rocm_from_path(result_dir)
    )
    system = (
        system_metadata.get("active_system")
        or result_metadata.get("system", "")
        or infer_system_from_path(result_dir)
    )
    category = result_metadata.get("result_category") or infer_category_from_path(result_dir)
    label = result_dir.name

    return CaseSummary(
        result_dir=result_dir,
        label=label,
        backend=backend,
        rccl_sync_mode=rccl_sync_mode,
        system=system,
        rocm_version=normalize_rocm(rocm_version),
        category=category,
        threshold_us=threshold_us,
        samples_found=len(samples),
        total_measured_iterations=total_iterations,
        emitted_iteration_records=emitted_count,
        stall_count=len(stall_durations),
        stalled_iteration_percent=percent(len(stall_durations), total_iterations),
        samples_with_stalls=stalled_samples,
        samples_with_stalls_percent=percent(stalled_samples, len(samples)),
        max_stalls_in_one_sample=max_stalls,
        longest_stall_free_sequence=longest_stall_free_sequence(sequence),
        min_stall_duration_us=min(stall_durations) if stall_durations else None,
        median_stall_duration_us=statistics.median(stall_durations)
        if stall_durations
        else None,
        mean_stall_duration_us=statistics.fmean(stall_durations)
        if stall_durations
        else None,
        max_stall_duration_us=max(stall_durations) if stall_durations else None,
        rank_frequency=dict(sorted(rank_counter.items())),
        sample_sequence=sequence,
        affected_indices=affected,
        isolated_bad_samples=isolated,
        multiple_bad_samples=multiple,
        nearly_all_slow_samples=nearly_all,
        stalled_iterations_with_phase_data=stalled_iterations_with_phase_data,
        phase_summaries=phase_summaries,
        phase_iteration_breakdown=phase_iteration_breakdown,
        stall_cause_classification=stall_cause_classification,
        warnings=warnings,
    )


def percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * float(numerator) / float(denominator)


def infer_system_from_path(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        if index + 1 < len(parts):
            return parts[index + 1]
    return ""


def infer_rocm_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("rocm-"):
            return part
    return ""


def infer_category_from_path(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        if index + 3 < len(parts):
            return parts[index + 3]
    return ""


def format_optional(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def markdown_report(summaries: Sequence[CaseSummary]) -> str:
    lines = [
        "# gHALO Iteration Stall Analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "| Case | Samples | Iterations | Stalls | Stall % | Samples with stalls | Max/sample | Longest F run | Min us | Median us | Mean us | Max us |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.label,
                    str(summary.samples_found),
                    str(summary.total_measured_iterations),
                    str(summary.stall_count),
                    f"{summary.stalled_iteration_percent:.3f}",
                    f"{summary.samples_with_stalls} ({summary.samples_with_stalls_percent:.3f}%)",
                    str(summary.max_stalls_in_one_sample),
                    str(summary.longest_stall_free_sequence),
                    format_optional(summary.min_stall_duration_us),
                    format_optional(summary.median_stall_duration_us),
                    format_optional(summary.mean_stall_duration_us),
                    format_optional(summary.max_stall_duration_us),
                ]
            )
            + " |"
        )
    lines.append("")

    for summary in summaries:
        lines.extend(
            [
                f"## {summary.label}",
                "",
                f"- Result directory: `{summary.result_dir}`",
                f"- System: {summary.system or 'unknown'}",
                f"- ROCm version: {summary.rocm_version or 'unknown'}",
                f"- Backend: {summary.backend}",
                f"- RCCL sync mode: {summary.rccl_sync_mode or 'n/a'}",
                f"- Threshold: {summary.threshold_us:g} us",
                f"- Sample sequence: `{summary.sample_sequence}`",
                f"- Isolated bad samples: {summary.isolated_bad_samples}",
                f"- Multiple-bad-iteration samples: {summary.multiple_bad_samples}",
                f"- Nearly-all-slow samples: {summary.nearly_all_slow_samples}",
                f"- Stalled iterations with phase data: {summary.stalled_iterations_with_phase_data}",
                f"- Primary stall classification: {summary.stall_cause_classification}",
                "",
                "Rank frequency:",
                "",
            ]
        )
        if summary.rank_frequency:
            lines.append("| Rank | Stalls |")
            lines.append("| ---: | ---: |")
            for rank, count in summary.rank_frequency.items():
                lines.append(f"| {rank} | {count} |")
        else:
            lines.append("No stalls above threshold.")
        lines.extend(["", "Affected sample/iteration indices:", ""])
        if summary.affected_indices:
            lines.append("| Halo | Sample | Iteration | Duration us | Max rank |")
            lines.append("| ---: | ---: | ---: | ---: | ---: |")
            for halo, sample, iteration, duration, rank in summary.affected_indices:
                lines.append(
                    f"| {halo} | {sample} | {iteration} | {duration:.3f} | {rank} |"
                )
        else:
            lines.append("None.")
        lines.extend(["", "Phase dominance:", ""])
        if summary.phase_summaries:
            lines.append(
                "| Phase | Dominant iterations | Cumulative ms | Median us | Max us |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for phase in summary.phase_summaries:
                lines.append(
                    f"| {phase.phase_name} | {phase.dominant_iterations} | "
                    f"{phase.cumulative_ms:.3f} | {phase.median_us:.3f} | "
                    f"{phase.max_us:.3f} |"
                )
        else:
            lines.append("No phase data available.")
        lines.extend(["", "Per-stalled-iteration phase breakdown:", ""])
        if summary.phase_iteration_breakdown:
            lines.append(
                "| Halo | Sample | Iteration | Dominant phase | Phase us | Phase rank | Total us | Total rank |"
            )
            lines.append("| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
            for row in summary.phase_iteration_breakdown:
                lines.append(
                    f"| {row['halo_words']} | {row['sample_index']} | "
                    f"{row['iteration_index']} | {row['dominant_phase']} | "
                    f"{row['dominant_phase_us']:.3f} | "
                    f"{row['dominant_phase_max_rank']} | "
                    f"{row['total_iteration_us']:.3f} | "
                    f"{row['total_iteration_max_rank']} |"
                )
        else:
            lines.append("None.")
        if summary.warnings:
            lines.extend(["", "Warnings:"])
            for warning in summary.warnings:
                lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines)


def write_csv_summary(path: Path, summaries: Sequence[CaseSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "result_dir",
                "system",
                "rocm_version",
                "category",
                "backend",
                "rccl_sync_mode",
                "threshold_us",
                "samples_found",
                "total_measured_iterations",
                "emitted_iteration_records",
                "stall_count",
                "stalled_iteration_percent",
                "samples_with_stalls",
                "samples_with_stalls_percent",
                "max_stalls_in_one_sample",
                "longest_stall_free_sequence",
                "min_stall_duration_us",
                "median_stall_duration_us",
                "mean_stall_duration_us",
                "max_stall_duration_us",
                "isolated_bad_samples",
                "multiple_bad_samples",
                "nearly_all_slow_samples",
                "stalled_iterations_with_phase_data",
                "stall_cause_classification",
                "sample_sequence",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "label": summary.label,
                    "result_dir": str(summary.result_dir),
                    "system": summary.system,
                    "rocm_version": summary.rocm_version,
                    "category": summary.category,
                    "backend": summary.backend,
                    "rccl_sync_mode": summary.rccl_sync_mode,
                    "threshold_us": summary.threshold_us,
                    "samples_found": summary.samples_found,
                    "total_measured_iterations": summary.total_measured_iterations,
                    "emitted_iteration_records": summary.emitted_iteration_records,
                    "stall_count": summary.stall_count,
                    "stalled_iteration_percent": summary.stalled_iteration_percent,
                    "samples_with_stalls": summary.samples_with_stalls,
                    "samples_with_stalls_percent": summary.samples_with_stalls_percent,
                    "max_stalls_in_one_sample": summary.max_stalls_in_one_sample,
                    "longest_stall_free_sequence": summary.longest_stall_free_sequence,
                    "min_stall_duration_us": summary.min_stall_duration_us,
                    "median_stall_duration_us": summary.median_stall_duration_us,
                    "mean_stall_duration_us": summary.mean_stall_duration_us,
                    "max_stall_duration_us": summary.max_stall_duration_us,
                    "isolated_bad_samples": summary.isolated_bad_samples,
                    "multiple_bad_samples": summary.multiple_bad_samples,
                    "nearly_all_slow_samples": summary.nearly_all_slow_samples,
                    "stalled_iterations_with_phase_data": summary.stalled_iterations_with_phase_data,
                    "stall_cause_classification": summary.stall_cause_classification,
                    "sample_sequence": summary.sample_sequence,
                }
            )


def summary_to_json(summary: CaseSummary) -> Dict[str, Any]:
    return {
        "label": summary.label,
        "result_dir": str(summary.result_dir),
        "system": summary.system,
        "rocm_version": summary.rocm_version,
        "category": summary.category,
        "backend": summary.backend,
        "rccl_sync_mode": summary.rccl_sync_mode,
        "threshold_us": summary.threshold_us,
        "samples_found": summary.samples_found,
        "total_measured_iterations": summary.total_measured_iterations,
        "emitted_iteration_records": summary.emitted_iteration_records,
        "stall_count": summary.stall_count,
        "stalled_iteration_percent": summary.stalled_iteration_percent,
        "samples_with_stalls": summary.samples_with_stalls,
        "samples_with_stalls_percent": summary.samples_with_stalls_percent,
        "max_stalls_in_one_sample": summary.max_stalls_in_one_sample,
        "longest_stall_free_sequence": summary.longest_stall_free_sequence,
        "min_stall_duration_us": summary.min_stall_duration_us,
        "median_stall_duration_us": summary.median_stall_duration_us,
        "mean_stall_duration_us": summary.mean_stall_duration_us,
        "max_stall_duration_us": summary.max_stall_duration_us,
        "rank_frequency": summary.rank_frequency,
        "sample_sequence": summary.sample_sequence,
        "affected_indices": [
            {
                "halo_words": halo,
                "sample_index": sample,
                "iteration_index": iteration,
                "duration_us": duration,
                "max_rank": rank,
            }
            for halo, sample, iteration, duration, rank in summary.affected_indices
        ],
        "isolated_bad_samples": summary.isolated_bad_samples,
        "multiple_bad_samples": summary.multiple_bad_samples,
        "nearly_all_slow_samples": summary.nearly_all_slow_samples,
        "stalled_iterations_with_phase_data": summary.stalled_iterations_with_phase_data,
        "stall_cause_classification": summary.stall_cause_classification,
        "phase_summaries": [
            {
                "phase_name": phase.phase_name,
                "dominant_iterations": phase.dominant_iterations,
                "cumulative_ms": phase.cumulative_ms,
                "median_us": phase.median_us,
                "max_us": phase.max_us,
                "max_rank_frequency": phase.max_rank_frequency,
            }
            for phase in summary.phase_summaries
        ],
        "phase_iteration_breakdown": summary.phase_iteration_breakdown,
        "warnings": summary.warnings,
    }


def write_json_summary(path: Path, summaries: Sequence[CaseSummary]) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cases": [summary_to_json(summary) for summary in summaries],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_outputs(output_dir: Path, summaries: Sequence[CaseSummary]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "iteration-stall-report.md").write_text(
        markdown_report(summaries) + "\n", encoding="utf-8"
    )
    write_csv_summary(output_dir / "iteration-stall-summary.csv", summaries)
    write_json_summary(output_dir / "iteration-stall-summary.json", summaries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze gHALO iteration-times.csv stall diagnostics."
    )
    parser.add_argument("inputs", nargs="*", help="Result directories or parent directories")
    parser.add_argument("--threshold-us", type=float, default=None)
    parser.add_argument("--system", default="borg")
    parser.add_argument("--rocm-version", default="6.4.2")
    parser.add_argument("--category", default="repeatability")
    parser.add_argument("--label-filter", default="")
    parser.add_argument("--output-dir", default="analysis/iteration-stalls")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threshold_us is not None and args.threshold_us < 0.0:
        parser.error("--threshold-us must be nonnegative")

    try:
        if args.inputs:
            result_dirs = result_dirs_from_inputs(args.inputs)
        else:
            result_dirs = discover_result_dirs(
                Path.cwd(), args.system, args.rocm_version, args.category, args.label_filter
            )
        if args.label_filter and args.inputs:
            result_dirs = [p for p in result_dirs if args.label_filter in p.name]
        if not result_dirs:
            raise StallAnalysisError("no matching result directories found")

        summaries = [analyze_case(path, args.threshold_us) for path in result_dirs]
        write_outputs(Path(args.output_dir), summaries)
        print(markdown_report(summaries))
        print(f"\nWrote iteration stall analysis to {args.output_dir}")
    except (OSError, json.JSONDecodeError, ValueError, StallAnalysisError) as error:
        print(f"analyze_iteration_stalls.py: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
