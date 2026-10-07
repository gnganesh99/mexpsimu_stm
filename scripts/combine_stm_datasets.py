"""Generic helpers for consolidating pyNSID/sidpy scanning-probe datasets
into a single combined pyNSID-format HDF5 file.

These functions are sample-agnostic: any pyNSID-format .h5 file works as a
source, regardless of instrument or sample, as long as it has the standard
NSID channel-group layout that SciFiReaders.NSIDReader / pyNSID's
write_nsid_dataset both produce and consume -- a top-level group whose
children are channel groups, each carrying a "channel_key" attr and a
nested Main/Main dataset.

Because write_nsid_dataset re-serializes every dict-valued property of a
sidpy.Dataset (original_metadata, provenance, ...), per-channel instrument
metadata survives the round trip unchanged; only the container layout
changes.

The channel-group read/write primitives (read_channels/write_channels) live
in mexpsimu.nsid_io -- the same module mexpsimu.stm_dataset uses to load
files for the simulator -- so this script and the simulator's loader share
one implementation instead of two copies.

convert_nanonis_folder() builds a simulator-ready file straight from raw
Nanonis data: a folder with one grid spectroscopy .3ds file and, optionally,
one .sxm topography scan. It writes a single file to <folder>/sidpy/ --
combined .sxm + .3ds when both exist, else .3ds only. The .3ds reader also
yields a 2-D "Topography" channel (the Z-controller height at each grid
pixel), which goes into the "topography" section, so a .3ds-only folder
still gets a real topography image. Requires the gnganesh99/SciFiReaders
fork (fixed .3ds channel parsing + Topography channel).

Running this file converts every raw data folder under
mexpsimu/data/stm_data/:

    python scripts/combine_stm_datasets.py [--overwrite]

To combine existing pyNSID .h5 files instead, call combine_sidpy_datasets().
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from mexpsimu.nanonis_io import read_nanonis_file, read_nanonis_sections
from mexpsimu.nsid_io import open_readonly, read_channels, write_channels
from mexpsimu.registry import default_stm_data_dir

__all__ = [
    "read_channels",
    "write_channels",
    "combine_sidpy_datasets",
    "read_nanonis_file",
    "convert_nanonis_folder",
    "convert_all_nanonis_folders",
]

SIDPY_SUBDIR = "sidpy"


def combine_sidpy_datasets(
    sources: dict[str, str | Path],
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Combine any number of pyNSID-format sidpy dataset files into one.

    Parameters
    ----------
    sources:
        Mapping of ``{section_name: source_file_path}``. Every channel in
        each source file is re-written under a top-level group named
        ``section_name`` in the output file. Works with two files (e.g.
        topography + spectroscopy) or more.
    output_path:
        Where to write the combined file.
    overwrite:
        If ``False`` (default) and ``output_path`` already exists, raises
        ``FileExistsError`` rather than silently overwriting it.

    Returns
    -------
    Path to the written file.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass overwrite=True to replace it"
        )

    channel_sets: dict[str, dict[str, "sidpy.Dataset"]] = {}
    for section, path in sources.items():
        with open_readonly(path) as h5f:
            root_group = h5f[list(h5f.keys())[0]]
            channels = read_channels(root_group)
            channel_sets[section] = channels
            print(f"read {len(channels)} channel(s) from {path} -> section {section!r}")

            # Each sidpy.Dataset is backed by a dask array pointing at this
            # file, so write it out to the combined file before this file
            # handle goes out of scope.
            with h5py.File(output_path, "a") as out:
                write_channels(channels, out.create_group(section))

    print(f"wrote combined file: {output_path}")
    return output_path


def _single_file(folder: Path, suffix: str, required: bool) -> Path | None:
    files = sorted(folder.glob(f"*{suffix}"))
    if len(files) > 1:
        raise ValueError(f"{folder} has more than one {suffix} file: {[f.name for f in files]}")
    if not files:
        if required:
            raise FileNotFoundError(f"{folder} has no {suffix} file")
        return None
    return files[0]


def convert_nanonis_folder(folder: str | Path, overwrite: bool = False) -> Path:
    """Convert one raw Nanonis data folder into a single simulator-ready file.

    ``folder`` must hold exactly one ``.3ds`` file and at most one ``.sxm``
    file. The output goes to ``<folder>/sidpy/``:

    - ``<3ds stem>_<sxm stem>_combined_sd.h5`` when both exist: the
      ``"topography"`` section holds every ``.sxm`` channel plus the ``.3ds``
      ``"Topography"`` channel;
    - ``<3ds stem>_3ds_sd.h5`` when only the ``.3ds`` exists: the
      ``"topography"`` section holds the ``.3ds`` ``"Topography"`` channel.

    In both cases the ``"spectroscopy"`` section holds the ``.3ds`` 3-D
    channels. Any other ``.h5`` already in ``<folder>/sidpy/`` is removed
    (with ``overwrite=True``) so the folder keeps exactly one file.
    """
    folder = Path(folder)
    grid_path = _single_file(folder, ".3ds", required=True)
    scan_path = _single_file(folder, ".sxm", required=False)

    spectroscopy, topography = read_nanonis_sections(grid_path, scan_path)
    if scan_path is not None:
        output_name = f"{grid_path.stem}_{scan_path.stem}_combined_sd.h5"
    else:
        output_name = f"{grid_path.stem}_3ds_sd.h5"

    out_dir = folder / SIDPY_SUBDIR
    out_dir.mkdir(exist_ok=True)
    output_path = out_dir / output_name
    existing = sorted(out_dir.glob("*.h5"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{out_dir} already has {[p.name for p in existing]}; pass overwrite=True to replace"
        )
    for old in existing:
        old.unlink()

    with h5py.File(output_path, "w") as out:
        write_channels(spectroscopy, out.create_group("spectroscopy"))
        if topography:
            write_channels(topography, out.create_group("topography"))

    sources = grid_path.name if scan_path is None else f"{scan_path.name} + {grid_path.name}"
    print(f"{sources} -> {output_path} ({len(topography)} topography, {len(spectroscopy)} spectroscopy channels)")
    return output_path


def convert_all_nanonis_folders(data_dir: str | Path | None = None, overwrite: bool = False) -> list[Path]:
    """Run :func:`convert_nanonis_folder` on every subfolder of ``data_dir``
    (default: the bundled ``stm_data``) that contains a ``.3ds`` file."""
    data_dir = Path(data_dir) if data_dir is not None else default_stm_data_dir()
    folders = [p for p in sorted(data_dir.iterdir()) if p.is_dir() and any(p.glob("*.3ds"))]
    return [convert_nanonis_folder(folder, overwrite=overwrite) for folder in folders]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", nargs="?", default=None, help="defaults to mexpsimu/data/stm_data")
    parser.add_argument("--overwrite", action="store_true", help="replace existing files in <folder>/sidpy/")
    args = parser.parse_args()
    convert_all_nanonis_folders(args.data_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
