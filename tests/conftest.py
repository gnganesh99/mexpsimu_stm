"""Shared test helpers for building tiny synthetic pyNSID/sidpy files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import sidpy

from mexpsimu.nsid_io import open_readonly, read_channels, write_channels
from mexpsimu.registry import resolve_stm_dataset_path


def make_sidpy_channel(
    data: np.ndarray,
    quantity: str,
    units: str = "A",
    x_units: str = "nm",
    bias: np.ndarray | None = None,
    size_xy_m: tuple[float, float] | None = None,
) -> "sidpy.Dataset":
    """Build a minimal sidpy.Dataset for a 2-D image or 3-D spectral-image channel."""
    ds = sidpy.Dataset.from_array(data, title=quantity, units=units, quantity=quantity)
    rows, cols = data.shape[0], data.shape[1]
    ds.data_type = "SPECTRAL_IMAGE" if data.ndim == 3 else "IMAGE"
    # (Y, X[, Bias]) axis order, as written by the Nanonis .3ds/.sxm readers
    ds.set_dimension(
        0, sidpy.Dimension(np.arange(rows, dtype=float), name="Y", units=x_units, quantity="Length", dimension_type="spatial")
    )
    ds.set_dimension(
        1, sidpy.Dimension(np.arange(cols, dtype=float), name="X", units=x_units, quantity="Length", dimension_type="spatial")
    )
    if data.ndim == 3:
        if bias is None:
            bias = np.linspace(-1.0, 1.0, data.shape[2])
        ds.set_dimension(2, sidpy.Dimension(bias, name="Bias", units="V", quantity="Bias", dimension_type="spectral"))
    if size_xy_m is not None:
        ds.original_metadata = {"size_xy": list(size_xy_m)}
    return ds


def write_combined_h5(
    path: Path,
    spectroscopy: dict[str, "sidpy.Dataset"],
    topography: dict[str, "sidpy.Dataset"] | None = None,
) -> Path:
    """Write a combined pyNSID .h5 file with 'spectroscopy' (required) and
    optional 'topography' sections, matching scripts/combine_stm_datasets.py's
    output layout."""
    with h5py.File(path, "w") as f:
        write_channels(spectroscopy, f.create_group("spectroscopy"))
        if topography is not None:
            write_channels(topography, f.create_group("topography"))
    return Path(path)


def write_large_area_3ds_only(path: Path) -> Path:
    """Write the bundled ``large_area`` dataset as a ``.3ds``-only file: its
    spectroscopy section plus only the grid ``Topography`` channel, the
    layout ``convert_nanonis_folder`` writes for a folder with no ``.sxm``."""
    with open_readonly(resolve_stm_dataset_path("large_area")) as src, h5py.File(path, "w") as out:
        write_channels(read_channels(src["spectroscopy"]), out.create_group("spectroscopy"))
        topography = read_channels(src["topography"])
        write_channels({"Topography": topography["Topography"]}, out.create_group("topography"))
    return Path(path)
