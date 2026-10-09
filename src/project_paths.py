"""Shared discovery of the organiser-provided Data directory."""

from __future__ import annotations

from pathlib import Path


def find_project_root(source_file: str | Path) -> Path:
    """Return the nearest ancestor containing the supplied Data directory."""
    source = Path(source_file).resolve()
    for candidate in source.parents:
        if (candidate / "Data").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the organiser-provided Data directory. "
        "Extract the submission folder beside Data."
    )
