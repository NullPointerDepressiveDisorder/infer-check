"""JSON export for infer-check results.

Scans a results directory for cached JSON files, classifies each file by its
Pydantic model type, merges them into a single structured document, and writes
the result to a specified output path.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infer_check.types import (
    ComparisonResult,
    DeterminismResult,
    StressResult,
    SweepResult,
)

__all__ = ["export"]

_UNKNOWN_VERSION = "unknown"


def _try_load_sweep(data: Any) -> SweepResult | None:
    """Attempt to parse *data* as a ``SweepResult``."""
    try:
        if isinstance(data, dict):
            return SweepResult.model_validate(data)
    except Exception:
        pass
    return None


def _try_load_comparison_list(data: Any) -> list[ComparisonResult] | None:
    """Attempt to parse *data* as a list of ``ComparisonResult`` objects."""
    try:
        if isinstance(data, list) and data:
            return [ComparisonResult.model_validate(item) for item in data]
    except Exception:
        pass
    return None


def _try_load_stress(data: Any) -> StressResult | None:
    """Attempt to parse *data* as a ``StressResult``."""
    try:
        if isinstance(data, dict):
            return StressResult.model_validate(data)
    except Exception:
        pass
    return None


def _try_load_stress_list(data: Any) -> list[StressResult] | None:
    """Attempt to parse *data* as a list of ``StressResult`` objects."""
    try:
        if isinstance(data, list) and data:
            return [StressResult.model_validate(item) for item in data]
    except Exception:
        pass
    return None


def _try_load_determinism(data: Any) -> DeterminismResult | None:
    """Attempt to parse *data* as a ``DeterminismResult``."""
    try:
        if isinstance(data, dict):
            return DeterminismResult.model_validate(data)
    except Exception:
        pass
    return None


def _try_load_determinism_list(data: Any) -> list[DeterminismResult] | None:
    """Attempt to parse *data* as a list of ``DeterminismResult`` objects."""
    try:
        if isinstance(data, list) and data:
            return [DeterminismResult.model_validate(item) for item in data]
    except Exception:
        pass
    return None


def _classify_file(path: Path) -> tuple[str, Any] | None:
    """Load and classify a single JSON result file.

    Args:
        path: Path to a ``.json`` file in the results directory.

    Returns:
        A ``(section_name, parsed_object)`` tuple, or ``None`` if the file
        cannot be parsed as any known result type.  The *section_name* is one
        of ``"sweep"``, ``"diff"``, ``"stress"``, or ``"determinism"``.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    # Try SweepResult first (contains 'comparisons' key as a list).
    sweep = _try_load_sweep(raw)
    if sweep is not None:
        return ("sweep", sweep)

    # ComparisonResult list (diff output).
    comp_list = _try_load_comparison_list(raw)
    if comp_list is not None:
        return ("diff", comp_list)

    # StressResult (single or list).
    stress = _try_load_stress(raw)
    if stress is not None:
        return ("stress", [stress])
    stress_list = _try_load_stress_list(raw)
    if stress_list is not None:
        return ("stress", stress_list)

    # DeterminismResult (single or list).
    det = _try_load_determinism(raw)
    if det is not None:
        return ("determinism", [det])
    det_list = _try_load_determinism_list(raw)
    if det_list is not None:
        return ("determinism", det_list)

    return None


def export(results_dir: Path, output_path: Path) -> Path:
    """Merge all cached result JSON files into a single structured export.

    Scans *results_dir* recursively for ``.json`` files, classifies each by
    its Pydantic model type, and writes a merged document to *output_path*.
    Files that cannot be parsed as a known result type are silently skipped.

    Args:
        results_dir: Directory containing cached ``.json`` result files
            produced by the ``TestRunner``.
        output_path: Destination path for the merged JSON export.  Parent
            directories are created automatically.

    Returns:
        The resolved *output_path* after writing.

    Raises:
        OSError: If *output_path* cannot be written.

    Examples:
        >>> import tempfile, pathlib
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     d = pathlib.Path(tmp)
        ...     out = d / "report.json"
        ...     result = export(d, out)
        ...     result == out
        True
    """
    sections: dict[str, list[Any]] = {
        "sweep": [],
        "diff": [],
        "stress": [],
        "determinism": [],
    }

    json_files = sorted(results_dir.rglob("*.json"))
    for path in json_files:
        # Skip the output file itself if it happens to be in the same dir.
        if path.resolve() == output_path.resolve():
            continue
        classified = _classify_file(path)
        if classified is None:
            continue
        section, obj = classified
        if isinstance(obj, list) and section in ("stress", "determinism"):
            sections[section].extend(obj)
        else:
            sections[section].append(obj)

    # Determine tool version.
    try:
        tool_version = importlib.metadata.version("infer-check")
    except importlib.metadata.PackageNotFoundError:
        tool_version = _UNKNOWN_VERSION

    document: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "tool_version": tool_version,
            "machine": platform.machine(),
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "hostname": os.uname().nodename if hasattr(os, "uname") else platform.node(),
        },
        "summary": {
            "sweep_count": len(sections["sweep"]),
            "diff_batches": len(sections["diff"]),
            "stress_count": len(sections["stress"]),
            "determinism_count": len(sections["determinism"]),
            "total_files_scanned": len(json_files),
        },
        "sweep": [s.model_dump(mode="json") if hasattr(s, "model_dump") else s for s in sections["sweep"]],
        "diff": [[c.model_dump(mode="json") for c in batch] for batch in sections["diff"]],
        "stress": [s.model_dump(mode="json") if hasattr(s, "model_dump") else s for s in sections["stress"]],
        "determinism": [d.model_dump(mode="json") if hasattr(d, "model_dump") else d for d in sections["determinism"]],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    return output_path
