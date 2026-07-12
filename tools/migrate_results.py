#!/usr/bin/env python3
"""Migrate flat gHALO result bundles into the versioned result hierarchy."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


BACKENDS = ("mpi-hip", "rccl", "mpi")
CATEGORIES = ("validation", "scaling", "repeatability", "phase-timing")


def sanitize(value: str, default: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip().replace(" ", "_"))
    value = value.strip("_")
    return value or default


def normalize_backend(value: str) -> str:
    lowered = value.lower()
    if "mpi-hip" in lowered or "mpihip" in lowered:
        return "mpi-hip"
    if "rccl" in lowered:
        return "rccl"
    if "mpi" in lowered:
        return "mpi"
    return sanitize(lowered, "unknown")


def normalize_rocm(value: str) -> str:
    value = value.strip()
    if not value:
        return "none"
    if value.startswith("rocm/"):
        value = value.split("/", 1)[1]
    if "rocm-" in value:
        value = value.rsplit("rocm-", 1)[1]
    value = value.split("/", 1)[0]
    return sanitize(value.removeprefix("rocm-"), "none")


def normalize_label(label: str, backend: str) -> str:
    label = sanitize(label, "run")
    while label.startswith(f"{backend}_") or label.startswith(f"{backend}-"):
        label = label.removeprefix(f"{backend}_").removeprefix(f"{backend}-")
    return label or "run"


def parse_key_value_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def split_flat_name(name: str) -> Tuple[str, str, str]:
    match = re.match(r"^(\d{8}T\d{6}Z)_([^_]+)_(.*)$", name)
    if not match:
        return ("unknown", "unknown", name)
    timestamp, backend, label = match.groups()
    return (timestamp, normalize_backend(backend), label)


def infer_system(bundle: Path, metadata: Dict[str, str]) -> str:
    if metadata.get("active_system"):
        return sanitize(metadata["active_system"], "unknown")
    parts = bundle.parts
    if "results" in parts:
        index = parts.index("results")
        if index + 1 < len(parts):
            return sanitize(parts[index + 1], "unknown")
    return sanitize(bundle.parent.name, "unknown")


def infer_category(label: str, result: Dict[str, Any]) -> str:
    if result.get("phase_timing"):
        return "phase-timing"
    metadata = result.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("validation_enabled") is True:
        return "validation"
    lowered = label.lower()
    if "phase" in lowered:
        return "phase-timing"
    if "valid" in lowered or "smoke" in lowered:
        return "validation"
    if "scaling" in lowered or lowered.startswith(("scale-", "scale_")):
        return "scaling"
    return "repeatability"


def infer_bundle(bundle: Path) -> Tuple[str, str, str, str, str]:
    timestamp, backend, label = split_flat_name(bundle.name)
    system_metadata = parse_key_value_file(bundle / "system-resolution.txt")
    result_metadata = parse_key_value_file(bundle / "result-metadata.txt")
    data = load_json(bundle / "ghalo.json")
    results = data.get("results", []) if isinstance(data.get("results"), list) else []
    first = results[0] if results and isinstance(results[0], dict) else {}
    first_metadata = first.get("metadata", {}) if isinstance(first.get("metadata"), dict) else {}

    system = infer_system(bundle, system_metadata)
    backend = normalize_backend(str(first.get("backend") or system_metadata.get("backend") or backend))
    rocm = normalize_rocm(
        str(
            first_metadata.get("rocm_version")
            or result_metadata.get("rocm_version")
            or result_metadata.get("loaded_rocm_module")
            or "none"
        )
    )
    category = result_metadata.get("result_category") or infer_category(label, first)
    if category not in CATEGORIES:
        category = "repeatability"
    label = normalize_label(label, backend)
    return system, rocm, category, backend, label


def unique_destination(
    root: Path,
    system: str,
    rocm: str,
    category: str,
    source: Path,
    backend: str,
    label: str,
) -> Path:
    parent = root / "results" / system / f"rocm-{rocm}" / category
    timestamp, _, _ = split_flat_name(source.name)
    stem = f"{timestamp}_{backend}_{label}" if timestamp != "unknown" else f"{backend}_{label}"
    candidate = parent / stem
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{stem}_{suffix}"
        suffix += 1
    return candidate


def candidate_bundles(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir() and ((path / "ghalo.json").exists() or (path / "ghalo.csv").exists()):
            yield path
            continue
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.is_dir() and re.match(r"^\d{8}T\d{6}Z_", child.name):
                    if (child / "ghalo.json").exists() or (child / "ghalo.csv").exists():
                        yield child


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy flat gHALO result bundles into results/<system>/rocm-<version>/<category>/."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true", help="perform the copy")
    parser.add_argument("--dry-run", action="store_true", help="print actions without copying")
    args = parser.parse_args()

    apply = args.apply and not args.dry_run
    for bundle in candidate_bundles(args.paths):
        system, rocm, category, backend, label = infer_bundle(bundle)
        destination = unique_destination(args.root, system, rocm, category, bundle, backend, label)
        print(
            f"{'COPY' if apply else 'DRY-RUN'} {bundle} -> {destination} "
            f"(system={system} rocm={rocm} category={category} backend={backend} label={label})"
        )
        if apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(bundle, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
