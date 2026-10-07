"""Discovery of bundled and user-supplied dataset files.

This module covers two independent, unrelated dataset families that happen
to both live under ``mexpsimu/data/``:

- the legacy AFM/BEPS ``.npz`` presets (``default_data_dir`` /
  ``discover_datasets`` / ``list_presets`` / ``resolve_dataset_path`` /
  ``parse_scan_size_um``) -- not required by stm_simu, kept for future
  integration with the STM path.
- the STM pyNSID ``.h5`` presets (``default_stm_data_dir`` /
  ``discover_stm_datasets`` / ``list_stm_presets`` /
  ``resolve_stm_dataset_path``), used by :mod:`mexpsimu.stm_dataset`.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

from .exceptions import DatasetNotFoundError

__all__ = [
    "default_data_dir",
    "discover_datasets",
    "list_presets",
    "resolve_dataset_path",
    "parse_scan_size_um",
    "default_stm_data_dir",
    "discover_stm_datasets",
    "list_stm_presets",
    "resolve_stm_dataset_path",
]

# --------------------------------------------------------------------------
# Legacy AFM/BEPS .npz presets -- not required by stm_simu.
# --------------------------------------------------------------------------

# Matches a scan size embedded in a filename, e.g. "3um" -> 3.0, "1d3um" -> 1.3
# ("d" stands in for a decimal point, as is common in filenames that avoid dots).
# Not required by stm_simu: STM scan size comes from real sidpy X/Y
# dimensions (see mexpsimu.stm_dataset), not from filename parsing.
_SIZE_RE = re.compile(r"(\d+)(?:d(\d+))?um", re.IGNORECASE)


def parse_scan_size_um(name: str) -> float | None:
    """Extract a physical (square) scan size in microns from a filename, if present.

    Not required by stm_simu.
    """
    match = _SIZE_RE.search(name)
    if not match:
        return None
    whole, frac = match.groups()
    return float(f"{whole}.{frac}") if frac is not None else float(whole)


def default_data_dir() -> Path:
    """The directory containing the legacy AFM/BEPS sample datasets shipped
    with mexpsimu. Not required by stm_simu."""
    return Path(resources.files("mexpsimu").joinpath("data", "afm_data"))


def discover_datasets(data_dir: str | Path | None = None) -> dict[str, Path]:
    """Return ``{preset_name: path}`` for every ``.npz`` file in ``data_dir``.

    ``preset_name`` is the filename without its extension, so a preset can be
    referenced by that name from :class:`mexpsimu.microscope.Microscope`.

    Not required by stm_simu.
    """
    directory = Path(data_dir) if data_dir is not None else default_data_dir()
    if not directory.is_dir():
        return {}
    return {p.stem: p for p in sorted(directory.glob("*.npz"))}


def list_presets(data_dir: str | Path | None = None) -> list[str]:
    """Names of all discoverable legacy AFM/BEPS dataset presets, sorted
    alphabetically. Not required by stm_simu."""
    return sorted(discover_datasets(data_dir))


def resolve_dataset_path(name_or_path: str | Path, data_dir: str | Path | None = None) -> Path:
    """Resolve a preset name or an explicit file path to an existing ``.npz`` path.

    Not required by stm_simu.
    """
    candidate = Path(name_or_path)
    if candidate.suffix == ".npz" and candidate.is_file():
        return candidate

    datasets = discover_datasets(data_dir)
    key = candidate.stem if candidate.suffix else str(name_or_path)
    if key in datasets:
        return datasets[key]

    raise DatasetNotFoundError(
        f"unknown dataset {name_or_path!r}; available presets: {sorted(datasets)}"
    )


# --------------------------------------------------------------------------
# STM pyNSID .h5 presets.
# --------------------------------------------------------------------------


def default_stm_data_dir() -> Path:
    """The directory containing the bundled STM sample datasets."""
    return Path(resources.files("mexpsimu").joinpath("data", "stm_data"))


STM_SIDPY_SUBDIR = "sidpy"


def _folder_h5_files(folder: Path) -> list[Path]:
    """The converted ``.h5`` files of a raw-data folder (``<folder>/sidpy/*.h5``)."""
    return sorted((folder / STM_SIDPY_SUBDIR).glob("*.h5"))


def discover_stm_datasets(data_dir: str | Path | None = None) -> dict[str, Path]:
    """Return ``{preset_name: path}`` for every STM ``.h5`` dataset in ``data_dir``.

    Two layouts are discovered:

    - ``<data_dir>/<name>.h5`` -- preset ``<name>`` (the file stem);
    - ``<data_dir>/<folder>/sidpy/<file>.h5`` -- a raw Nanonis data folder
      converted by ``scripts/combine_stm_datasets.py``. Preset ``<folder>``
      if the folder has exactly one such file, else ``<folder>/<file stem>``.

    A preset can be referenced by name from :class:`mexpsimu.stm_microscope.STMMicroscope`.
    """
    directory = Path(data_dir) if data_dir is not None else default_stm_data_dir()
    if not directory.is_dir():
        return {}
    found = {p.stem: p for p in sorted(directory.glob("*.h5"))}
    for folder in sorted(p for p in directory.iterdir() if p.is_dir()):
        files = _folder_h5_files(folder)
        if len(files) == 1:
            found[folder.name] = files[0]
        else:
            found.update({f"{folder.name}/{f.stem}": f for f in files})
    return found


def list_stm_presets(data_dir: str | Path | None = None) -> list[str]:
    """Names of all discoverable STM dataset presets, sorted alphabetically."""
    return sorted(discover_stm_datasets(data_dir))


def resolve_stm_dataset_path(name_or_path: str | Path, data_dir: str | Path | None = None) -> Path:
    """Resolve a preset name, an ``.h5`` path, or a converted raw-data folder
    (one holding exactly one ``sidpy/*.h5``) to an existing ``.h5`` path."""
    candidate = Path(name_or_path)
    if candidate.suffix == ".h5" and candidate.is_file():
        return candidate
    if candidate.is_dir() and len(_folder_h5_files(candidate)) == 1:
        return _folder_h5_files(candidate)[0]

    datasets = discover_stm_datasets(data_dir)
    key = str(name_or_path) if str(name_or_path) in datasets else candidate.stem
    if key in datasets:
        return datasets[key]

    raise DatasetNotFoundError(
        f"unknown STM dataset {name_or_path!r}; available presets: {sorted(datasets)}"
    )
