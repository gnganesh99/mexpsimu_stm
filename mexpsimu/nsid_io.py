"""Low-level read/write helpers for pyNSID-format ``.h5`` files.

Shared by :mod:`mexpsimu.stm_dataset` (the STM simulator's loader) and
``scripts/combine_stm_datasets.py`` (the file-combining tool), so the two
don't carry independent copies of the same NSID channel-group traversal
logic.

These helpers are sample-agnostic: they only assume the standard NSID
channel-group layout that ``SciFiReaders.NSIDReader`` / pyNSID's
``write_nsid_dataset`` both produce and consume -- a group whose children
are channel groups, each carrying a ``channel_key`` attr and a nested
``Main/Main`` dataset.

Reading is done via ``pyNSID.io.hdf_utils.read_h5py_dataset`` directly
on a strictly read-only ``h5py.File``, deliberately *not* through
``SciFiReaders.NSIDReader`` -- that reader always opens its file in ``'r+'``
mode, which we found rewrites the file's internal HDF5 bookkeeping bytes on
close even when nothing logically changes (same array/attr content,
different bytes on disk). For a git-tracked data file, that means every
*read* looks like a write. ``read_h5py_dataset`` is the same function
``NSIDReader.read()`` calls internally for a single dataset, so behavior is
identical; only the file mode differs.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import h5py
import pyNSID.io as pio
from pyNSID.io.hdf_utils import read_h5py_dataset

__all__ = ["open_readonly", "read_channels", "write_channels"]


@contextmanager
def open_readonly(path: str | Path):
    """Open a pyNSID-format ``.h5`` file strictly read-only."""
    f = h5py.File(str(path), "r")
    try:
        yield f
    finally:
        f.close()


def read_channels(group: h5py.Group) -> dict[str, "sidpy.Dataset"]:
    """Read every named channel under ``group`` into a
    ``{channel_name: sidpy.Dataset}`` dict.

    ``group`` is a section group (e.g. an open file's ``"topography"`` or
    ``"spectroscopy"`` group) or, for a single-source file with no such
    section, the file's root group. Each channel is identified by a
    ``channel_key`` attr on its subgroup and a nested ``Main/Main`` dataset
    -- the same layout ``pyNSID.io.write_nsid_dataset`` produces.
    """
    channels = {}
    for name, grp in group.items():
        if isinstance(grp, h5py.Group) and "channel_key" in grp.attrs:
            main_h5_dset = grp["Main"]["Main"]
            channels[grp.attrs["channel_key"]] = read_h5py_dataset(main_h5_dset)
    return channels


def write_channels(channels: dict[str, "sidpy.Dataset"], parent_group: h5py.Group) -> None:
    """Write ``{channel_name: sidpy.Dataset}`` into ``parent_group``, one NSID
    channel group each."""
    for index, (name, dataset) in enumerate(channels.items()):
        channel_group = parent_group.create_group(name)
        channel_group.attrs["channel_index"] = index
        channel_group.attrs["channel_key"] = name
        pio.write_nsid_dataset(dataset, channel_group, main_data_name="Main")
