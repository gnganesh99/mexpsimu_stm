"""Reading raw Nanonis STM files (``.3ds`` grid spectroscopy, ``.sxm`` scans).

Shared by :meth:`mexpsimu.stm_dataset.STMScanDataset.from_nanonis` (load raw
files straight into the simulator) and ``scripts/combine_stm_datasets.py``
(convert raw files to a pyNSID ``.h5``), so both split the files into the
same ``spectroscopy`` / ``topography`` sections.

Requires the gnganesh99/SciFiReaders fork: its ``.3ds`` reader parses
channel names correctly (e.g. ``LIX 1 omega (A)`` is a forward channel in
``A``), writes physical X/Y axes, and adds a 2-D ``Topography`` channel (the
Z-controller height at each grid pixel). Loading already-converted ``.h5``
files does not need it.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["read_nanonis_file", "read_nanonis_sections"]


def read_nanonis_file(path: str | Path) -> dict[str, "sidpy.Dataset"]:
    """Read a raw Nanonis ``.3ds`` or ``.sxm`` file into ``{key: sidpy.Dataset}``."""
    from SciFiReaders import Nanonis3dsReader, NanonisSXMReader

    if not hasattr(Nanonis3dsReader, "_split_channel_name"):
        raise RuntimeError(
            "the installed SciFiReaders has the old .3ds parser (mislabels channels "
            "such as 'LIX 1 omega (A)' as backward and has no Topography channel); "
            "install the gnganesh99/SciFiReaders fork to read raw Nanonis data"
        )
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".3ds":
        return Nanonis3dsReader(str(path)).read()
    if suffix == ".sxm":
        return NanonisSXMReader(str(path)).read()
    raise ValueError(f"{path} is not a Nanonis .3ds or .sxm file")


def read_nanonis_sections(
    file_3ds: str | Path, file_sxm: str | Path | None = None
) -> tuple[dict[str, "sidpy.Dataset"], dict[str, "sidpy.Dataset"]]:
    """Read a ``.3ds`` grid (and optionally an ``.sxm`` scan of the same frame)
    into ``(spectroscopy, topography)`` channel dicts.

    ``spectroscopy`` holds the ``.3ds`` 3-D channels (``Current``,
    ``LIX 1 omega``, ...). ``topography`` holds the ``.sxm`` channels
    (``Z forward``, ...), if given, followed by the ``.3ds`` 2-D
    ``Topography`` channel.
    """
    if Path(file_3ds).suffix.lower() != ".3ds":
        raise ValueError(f"file_3ds must be a .3ds file, got {file_3ds}")
    if file_sxm is not None and Path(file_sxm).suffix.lower() != ".sxm":
        raise ValueError(f"file_sxm must be an .sxm file, got {file_sxm}")
    grid = read_nanonis_file(file_3ds)
    spectroscopy = {key: ds for key, ds in grid.items() if ds.ndim == 3}
    topography = {key: ds for key, ds in grid.items() if ds.ndim == 2}
    if file_sxm is not None:
        scan = read_nanonis_file(file_sxm)
        clashes = set(scan) & set(topography)
        if clashes:
            raise ValueError(
                f"channel names in both {Path(file_sxm).name} and {Path(file_3ds).name}: {sorted(clashes)}"
            )
        topography = {**scan, **topography}
    return spectroscopy, topography
