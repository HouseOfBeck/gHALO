#!/usr/bin/env python3
"""Compare Borg RCCL stall diagnostics across ROCm/RCCL stacks."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import analyze_iteration_stalls as stalls  # noqa: E402


EXPECTED_CODES = {
    "6.4.2": "22203",
    "7.0.2": "22606",
    "7.2.0": "22707",
}
VERSION_ORDER = ["6.4.2", "7.0.2", "7.2.0"]


class RCCLVersionCompareError(RuntimeError):
    """Raised for malformed or incompatible RCCL version comparison inputs."""


@dataclass
class VersionSummary:
    rocm_version: str
    expected_rccl_code: str
    observed_rccl_code: str
    result_dir: Path
    label: str
    total_samples: int
    total_iterations: int
    stalled_iterations: int
    stall_percentage: float
    samples_containing_stalls: int
    stall_events: int
    median_stall_us: Optional[float]
    max_stall_us: Optional[float]
    median_event_length: Optional[float]
    max_event_length: int
    dominant_phase_counts: Dict[str, int]
    rank_event_classifications: Dict[str, int]
    fast_path_median_us: Optional[float]
    validation_passed: bool
    hip_runtime_version: str
    rccl_version: str
    plugin_path: str
    warnings: List[str] = field(default_factory=list)


def normalize_rocm(value: str) -> str:
    value = value.strip()
    value = value.removeprefix("rocm-").removeprefix("rocm/")
    return value


def parse_metadata(path: Path) -> Dict[str, str]:
    return stalls.parse_key_value_file(path)


def result_dirs_from_inputs(inputs: Sequence[str]) -> List[Path]:
    result_dirs: List[Path] = []
    for text in inputs:
        path = Path(text)
        if (path / "ghalo.json").exists():
            result_dirs.append(path)
        elif path.is_dir():
            result_dirs.extend(sorted(p.parent for p in path.rglob("ghalo.json")))
        else:
            raise RCCLVersionCompareError(f"missing result path: {path}")
    return unique_paths(result_dirs)


def discover_result_dirs(root: Path, system: str, category: str) -> List[Path]:
    base = root / "results" / system
    if not base.exists():
        return []
    result_dirs = []
    for version in VERSION_ORDER:
        pattern_base = base / f"rocm-{version}" / category
        if pattern_base.exists():
            result_dirs.extend(
                sorted(
                    p.parent
                    for p in pattern_base.rglob("ghalo.json")
                    if f"borg-rccl-version-rocm-{version}" in p.parent.name
                )
            )
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


def load_json(path: Path) -> Dict[str, Any]:
    json_path = path / "ghalo.json"
    if not json_path.exists() or json_path.stat().st_size == 0:
        raise RCCLVersionCompareError(f"missing or empty ghalo.json: {json_path}")
    with json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("results"), list) or not payload["results"]:
        raise RCCLVersionCompareError(f"ghalo.json lacks results: {json_path}")
    return payload


def sample_fast_median_us(payload: Dict[str, Any], stalled_samples: set[Tuple[int, int]]) -> Optional[float]:
    values = []
    fallback = []
    for result in payload["results"]:
        halo = int(result.get("halo_words", 0))
        sample = int(result.get("sample_index", 1))
        value = float(result.get("max_average_seconds", 0.0)) * 1.0e6
        fallback.append(value)
        if (halo, sample) not in stalled_samples:
            values.append(value)
    if values:
        return statistics.median(values)
    return statistics.median(fallback) if fallback else None


def event_lengths(affected_indices: Sequence[Tuple[int, int, int, float, int]]) -> List[int]:
    by_sample: Dict[Tuple[int, int], List[int]] = {}
    for halo, sample, iteration, _duration, _rank in affected_indices:
        by_sample.setdefault((halo, sample), []).append(iteration)
    lengths: List[int] = []
    for iterations in by_sample.values():
        current = 0
        previous = None
        for iteration in sorted(iterations):
            if previous is None or iteration == previous + 1:
                current += 1
            else:
                lengths.append(current)
                current = 1
            previous = iteration
        if current:
            lengths.append(current)
    return lengths


def infer_version(result_dir: Path, payload: Dict[str, Any], metadata: Dict[str, str]) -> str:
    first_metadata = payload["results"][0].get("metadata", {})
    value = (
        metadata.get("requested_rocm_version")
        or first_metadata.get("rocm_version", "")
        or stalls.infer_rocm_from_path(result_dir)
    )
    return normalize_rocm(value)


def validate_payload(result_dir: Path, payload: Dict[str, Any], metadata: Dict[str, str], version: str) -> None:
    expected = EXPECTED_CODES.get(version)
    if expected is None:
        raise RCCLVersionCompareError(f"unexpected ROCm version for comparison: {version}")
    observed = metadata.get("observed_rccl_version_code", "")
    if observed and observed != expected:
        raise RCCLVersionCompareError(
            f"{result_dir} expected RCCL version code {expected} for ROCm {version}, observed {observed}"
        )
    for result in payload["results"]:
        result_metadata = result.get("metadata", {})
        backend = result.get("backend", "")
        if "RCCL" not in backend:
            raise RCCLVersionCompareError(f"{result_dir} is not an RCCL result")
        result_version = normalize_rocm(result_metadata.get("rocm_version", version))
        if result_version and result_version != version:
            raise RCCLVersionCompareError(
                f"{result_dir} mixes ROCm metadata: path/requested {version}, result {result_version}"
            )
        if result_metadata.get("rccl_sync_mode", "") != "conservative":
            raise RCCLVersionCompareError(
                f"{result_dir} is not an RCCL conservative comparison run"
            )


def summarize_result(result_dir: Path, threshold_us: Optional[float]) -> VersionSummary:
    payload = load_json(result_dir)
    metadata = parse_metadata(result_dir / "result-metadata.txt")
    version = infer_version(result_dir, payload, metadata)
    validate_payload(result_dir, payload, metadata, version)
    summary = stalls.analyze_case(result_dir, threshold_us)
    affected_samples = {(halo, sample) for halo, sample, _iteration, _duration, _rank in summary.affected_indices}
    lengths = event_lengths(summary.affected_indices)
    phase_counts = {}
    for row in summary.phase_iteration_breakdown:
        phase = str(row.get("dominant_phase", ""))
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    first_metadata = payload["results"][0].get("metadata", {})
    validation_passed = all(
        bool(result.get("metadata", {}).get("validation_passed", False))
        for result in payload["results"]
    )
    observed_code = metadata.get("observed_rccl_version_code", "")
    expected_code = EXPECTED_CODES[version]
    if not observed_code:
        observed_code = expected_code if first_metadata.get("rccl_version") else ""
    return VersionSummary(
        rocm_version=version,
        expected_rccl_code=expected_code,
        observed_rccl_code=observed_code,
        result_dir=result_dir,
        label=result_dir.name,
        total_samples=summary.samples_found,
        total_iterations=summary.total_measured_iterations,
        stalled_iterations=summary.stall_count,
        stall_percentage=summary.stalled_iteration_percent,
        samples_containing_stalls=summary.samples_with_stalls,
        stall_events=len(lengths),
        median_stall_us=summary.median_stall_duration_us,
        max_stall_us=summary.max_stall_duration_us,
        median_event_length=statistics.median(lengths) if lengths else None,
        max_event_length=max(lengths) if lengths else 0,
        dominant_phase_counts=dict(sorted(phase_counts.items())),
        rank_event_classifications=summary.rank_event_classification_frequency,
        fast_path_median_us=sample_fast_median_us(payload, affected_samples),
        validation_passed=validation_passed,
        hip_runtime_version=first_metadata.get("hip_runtime_version", ""),
        rccl_version=first_metadata.get("rccl_version", ""),
        plugin_path=(
            metadata.get("libnccl_net_realpath")
            or metadata.get("olcf_ofi_nccl_root")
            or first_metadata.get("rccl_plugin_root", "")
        ),
        warnings=summary.warnings,
    )


def validate_complete_matrix(summaries: Sequence[VersionSummary]) -> List[VersionSummary]:
    by_version: Dict[str, VersionSummary] = {}
    for summary in summaries:
        if summary.rocm_version in by_version:
            raise RCCLVersionCompareError(f"duplicate ROCm version: {summary.rocm_version}")
        by_version[summary.rocm_version] = summary
    missing = [version for version in VERSION_ORDER if version not in by_version]
    if missing:
        raise RCCLVersionCompareError(f"missing required ROCm result(s): {', '.join(missing)}")
    return [by_version[version] for version in VERSION_ORDER]


def fmt_optional(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def markdown_report(summaries: Sequence[VersionSummary]) -> str:
    lines = [
        "# Borg RCCL ROCm Stack Comparison",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "This is a ROCm/RCCL stack comparison, not a pure RCCL-only test. HIP runtime, RCCL, and other ROCm components change together.",
        "Zero observed stalls in this sample is not proof that a stack is immune to stalls; compare the sample and iteration counts.",
        "",
        "| ROCm | RCCL | Iterations | Stalls | Stall % | Samples affected | Events | Median stall us | Max stall us | Fast median us |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.rocm_version} | {summary.observed_rccl_code or summary.rccl_version} | "
            f"{summary.total_iterations} | {summary.stalled_iterations} | "
            f"{summary.stall_percentage:.3f} | {summary.samples_containing_stalls}/{summary.total_samples} | "
            f"{summary.stall_events} | {fmt_optional(summary.median_stall_us)} | "
            f"{fmt_optional(summary.max_stall_us)} | {fmt_optional(summary.fast_path_median_us)} |"
        )
    lines.append("")
    for summary in summaries:
        lines.extend(
            [
                f"## ROCm {summary.rocm_version}",
                "",
                f"- Result: `{summary.result_dir}`",
                f"- Expected RCCL version code: {summary.expected_rccl_code}",
                f"- Observed RCCL version code: {summary.observed_rccl_code or 'unknown'}",
                f"- HIP runtime: {summary.hip_runtime_version or 'unknown'}",
                f"- RCCL version: {summary.rccl_version or 'unknown'}",
                f"- Plugin path: `{summary.plugin_path or 'unknown'}`",
                f"- Validation passed: {summary.validation_passed}",
                f"- Median event length: {fmt_optional(summary.median_event_length)} iterations",
                f"- Max event length: {summary.max_event_length} iterations",
                f"- Dominant phase counts: `{json.dumps(summary.dominant_phase_counts, sort_keys=True)}`",
                f"- Rank-event classifications: `{json.dumps(summary.rank_event_classifications, sort_keys=True)}`",
                "",
            ]
        )
        if summary.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in summary.warnings)
            lines.append("")
    return "\n".join(lines)


def write_outputs(output_dir: Path, summaries: Sequence[VersionSummary]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rccl-version-comparison.md").write_text(
        markdown_report(summaries) + "\n", encoding="utf-8"
    )
    with (output_dir / "rccl-version-comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rocm_version",
                "expected_rccl_code",
                "observed_rccl_code",
                "result_dir",
                "total_samples",
                "total_iterations",
                "stalled_iterations",
                "stall_percentage",
                "samples_containing_stalls",
                "stall_events",
                "median_stall_us",
                "max_stall_us",
                "median_event_length",
                "max_event_length",
                "fast_path_median_us",
                "validation_passed",
                "hip_runtime_version",
                "rccl_version",
                "plugin_path",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "rocm_version": summary.rocm_version,
                    "expected_rccl_code": summary.expected_rccl_code,
                    "observed_rccl_code": summary.observed_rccl_code,
                    "result_dir": str(summary.result_dir),
                    "total_samples": summary.total_samples,
                    "total_iterations": summary.total_iterations,
                    "stalled_iterations": summary.stalled_iterations,
                    "stall_percentage": summary.stall_percentage,
                    "samples_containing_stalls": summary.samples_containing_stalls,
                    "stall_events": summary.stall_events,
                    "median_stall_us": summary.median_stall_us,
                    "max_stall_us": summary.max_stall_us,
                    "median_event_length": summary.median_event_length,
                    "max_event_length": summary.max_event_length,
                    "fast_path_median_us": summary.fast_path_median_us,
                    "validation_passed": summary.validation_passed,
                    "hip_runtime_version": summary.hip_runtime_version,
                    "rccl_version": summary.rccl_version,
                    "plugin_path": summary.plugin_path,
                }
            )
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cases": [summary.__dict__ | {"result_dir": str(summary.result_dir)} for summary in summaries],
    }
    (output_dir / "rccl-version-comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="Result directories or parent directories")
    parser.add_argument("--system", default="borg")
    parser.add_argument("--category", default="repeatability")
    parser.add_argument("--threshold-us", type=float, default=None)
    parser.add_argument("--output-dir", default="analysis/borg-rccl-version-comparison")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.inputs:
            result_dirs = result_dirs_from_inputs(args.inputs)
        else:
            result_dirs = discover_result_dirs(Path.cwd(), args.system, args.category)
        summaries = validate_complete_matrix(
            [summarize_result(path, args.threshold_us) for path in result_dirs]
        )
        write_outputs(Path(args.output_dir), summaries)
        print(markdown_report(summaries))
        print(f"\nWrote RCCL version comparison to {args.output_dir}")
    except (OSError, json.JSONDecodeError, ValueError, stalls.StallAnalysisError, RCCLVersionCompareError) as error:
        print(f"compare_rccl_versions.py: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
