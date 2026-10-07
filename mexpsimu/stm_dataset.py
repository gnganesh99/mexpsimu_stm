"""Loading of combined STM pyNSID/sidpy datasets (``.h5`` files) into memory.

A combined file (see ``scripts/combine_stm_datasets.py``) holds up to two
top-level sections of NSID channels:

- ``"topography"`` (optional) -- 2-D image channels, each ``(rows, cols)``:
  the ``.sxm`` scan channels (e.g. ``Z forward``, ``Current backward``) and/or
  the ``.3ds`` grid's ``Topography`` (Z-controller height at each grid
  pixel). Channels may differ in resolution (e.g. a 512x512 ``.sxm`` image
  next to a 180x180 grid ``Topography``) but cover the same frame.
- ``"spectroscopy"`` (required) -- 3-D channels from a grid spectroscopy
  scan (e.g. ``Current``, ``LIX 1 omega``, ``Current [bwd]``), each
  ``(rows, cols, n_bias)``, sharing one pixel grid and one ``Bias`` axis.

Axes are ``(Y, X[, Bias])``: rows run along Y, columns along X, both in nm.

Raw Nanonis files can also be loaded directly, without a converted ``.h5``:
:meth:`STMScanDataset.from_nanonis` (``.3ds``, optionally with an ``.sxm`` of
the same frame) builds the same two sections in memory.

If a file has no ``"topography"`` section, :attr:`STMScanDataset.image_channels`
falls back to a single synthesized ``"current_avg"`` channel: the
spectroscopy section's ``Current`` cube averaged over bias. Check
:attr:`STMScanDataset.has_topography` to tell which case a loaded dataset is in.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from .exceptions import NoSpectroscopyDataError
from .nanonis_io import read_nanonis_sections
from .nsid_io import open_readonly, read_channels

__all__ = ["STMScanDataset", "AxisInfo", "Channel"]


@dataclass
class AxisInfo:
    """A single dimension of a channel (e.g. its X, Y, or Bias axis)."""

    name: str
    quantity: str
    units: str
    values: np.ndarray


@dataclass
class Channel:
    """One materialized NSID channel: its array plus axis metadata.

    Unlike the lazy, file-backed ``sidpy.Dataset`` it's read from, ``data``
    is a plain in-memory numpy array -- the source file doesn't need to stay
    open once a :class:`STMScanDataset` is loaded.
    """

    name: str
    data: np.ndarray
    quantity: str
    units: str
    axes: dict[int, AxisInfo]


def _materialize_channel(name: str, dataset: "sidpy.Dataset") -> Channel:
    axes = {
        i: AxisInfo(name=dim.name, quantity=dim.quantity, units=dim.units, values=np.array(dim.values))
        for i, dim in dataset._axes.items()
    }
    return Channel(
        name=name,
        data=np.array(dataset),
        quantity=str(dataset.quantity),
        units=str(dataset.units),
        axes=axes,
    )


def _is_backward(name: str) -> bool:
    """Backward-direction channel: ``.3ds`` style ``"Current [bwd]"`` or
    ``.sxm`` style ``"Z backward"``."""
    return "[bwd]" in name or name.lower().endswith("backward")


def _pick_channel(channels: dict[str, Channel], quantity_prefix: str) -> tuple[str, Channel]:
    """Pick a channel whose quantity starts with ``quantity_prefix``
    (case-insensitive), preferring an explicit ``"... forward"`` channel (an
    ``.sxm`` image), then any non-backward one (e.g. a ``.3ds`` ``Topography``
    or ``Current``), then a backward one; else just the first channel."""
    prefix = quantity_prefix.lower()
    matches = [(name, ch) for name, ch in channels.items() if ch.quantity.lower().startswith(prefix)]
    if matches:
        forward = [(name, ch) for name, ch in matches if name.lower().endswith("forward")]
        not_backward = [(name, ch) for name, ch in matches if not _is_backward(name)]
        return (forward or not_backward or matches)[0]
    return next(iter(channels.items()))


def _spatial_axis(channel: Channel, name: str, default_index: int) -> np.ndarray:
    """Values of the spatial axis called ``name`` (``"X"``/``"Y"``), falling
    back to its conventional position (Y = axis 0 = rows, X = axis 1 = cols)."""
    for index in (0, 1):
        axis = channel.axes.get(index)
        if axis is not None and axis.name.upper() == name:
            return axis.values
    return channel.axes[default_index].values


@dataclass
class STMScanDataset:
    """An in-memory STM dataset: optional topography channels plus grid
    spectroscopy channels, loaded from a combined pyNSID ``.h5`` file.

    Parameters
    ----------
    name:
        A human-readable identifier (defaults to the source file's stem).
    path:
        The ``.h5`` file this dataset was loaded from (the ``.3ds`` file for
        :meth:`from_nanonis`).
    topography:
        Mapping of channel name -> :class:`Channel`, from the file's
        ``"topography"`` section, or ``None`` if the file has none. Channels
        may differ in resolution.
    spectroscopy:
        Mapping of channel name -> :class:`Channel`, from the file's
        ``"spectroscopy"`` section (required). All channels share one pixel
        grid and one Bias axis.
    """

    name: str
    path: Path
    topography: dict[str, Channel] | None
    spectroscopy: dict[str, Channel]

    def __post_init__(self) -> None:
        if not self.spectroscopy:
            raise NoSpectroscopyDataError(f"dataset {self.name!r} has no spectroscopy channels")

        shapes = {name: ch.data.shape for name, ch in self.spectroscopy.items()}
        if len(set(shapes.values())) > 1:
            raise ValueError(f"dataset {self.name!r}: mismatched spectroscopy channel shapes: {shapes}")

        biases = {name: ch.axes[2].values for name, ch in self.spectroscopy.items()}
        first_name, first_bias = next(iter(biases.items()))
        for name, bias in biases.items():
            if not np.allclose(bias, first_bias):
                raise ValueError(
                    f"dataset {self.name!r}: spectroscopy channel {name!r} has a different "
                    f"bias axis than {first_name!r}"
                )

        if self.topography is not None:
            if not self.topography:
                raise ValueError(f"dataset {self.name!r}: 'topography' section is present but empty")
            not_2d = {name: ch.data.shape for name, ch in self.topography.items() if ch.data.ndim != 2}
            if not_2d:
                raise ValueError(f"dataset {self.name!r}: topography channels must be 2-D, got {not_2d}")

    @classmethod
    def load(cls, path: str | Path, name: str | None = None) -> "STMScanDataset":
        """Load a combined dataset from a pyNSID ``.h5`` file."""
        path = Path(path)
        with open_readonly(path) as h5f:
            if "spectroscopy" not in h5f:
                raise NoSpectroscopyDataError(f"{path} has no 'spectroscopy' section")
            spectroscopy = {
                ch_name: _materialize_channel(ch_name, ds)
                for ch_name, ds in read_channels(h5f["spectroscopy"]).items()
            }

            topography = None
            if "topography" in h5f:
                topo_raw = read_channels(h5f["topography"])
                if topo_raw:
                    topography = {
                        ch_name: _materialize_channel(ch_name, ds) for ch_name, ds in topo_raw.items()
                    }

        return cls(name=name or path.stem, path=path, topography=topography, spectroscopy=spectroscopy)

    @classmethod
    def from_nanonis(
        cls, file_3ds: str | Path, file_sxm: str | Path | None = None, name: str | None = None
    ) -> "STMScanDataset":
        """Load raw Nanonis files directly (no converted ``.h5`` needed).

        ``file_3ds`` is the grid spectroscopy file; its 3-D channels become the
        spectroscopy section and its ``Topography`` (Z at each grid pixel) a
        topography channel. ``file_sxm`` optionally adds the image channels
        of an ``.sxm`` scan of the same frame. Requires the
        gnganesh99/SciFiReaders fork (see :mod:`mexpsimu.nanonis_io`).
        """
        file_3ds = Path(file_3ds)
        spectroscopy, topography = read_nanonis_sections(file_3ds, file_sxm)
        return cls(
            name=name or file_3ds.stem,
            path=file_3ds,
            topography={ch_name: _materialize_channel(ch_name, ds) for ch_name, ds in topography.items()} or None,
            spectroscopy={ch_name: _materialize_channel(ch_name, ds) for ch_name, ds in spectroscopy.items()},
        )

    @property
    def has_topography(self) -> bool:
        """Whether the source file had a real ``"topography"`` section."""
        return self.topography is not None

    @property
    def spectroscopy_shape(self) -> tuple[int, int]:
        """The pixel grid shape ``(rows, cols)`` the tip is addressed on."""
        return next(iter(self.spectroscopy.values())).data.shape[:2]

    @property
    def x_values(self) -> np.ndarray:
        """X position (nm) of each spectroscopy-grid column."""
        return _spatial_axis(next(iter(self.spectroscopy.values())), "X", default_index=1)

    @property
    def y_values(self) -> np.ndarray:
        """Y position (nm) of each spectroscopy-grid row."""
        return _spatial_axis(next(iter(self.spectroscopy.values())), "Y", default_index=0)

    @property
    def scan_extent(self) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` in nm: the area covered by the
        spectroscopy grid and every topography image together (the order
        matplotlib's ``imshow(extent=...)`` uses). Images are often a little
        wider than the grid -- e.g. a 512-point ``.sxm`` axis ends at 249.5 nm,
        a 180-point grid at 248.6 nm, both over a 250 nm frame."""
        channels = [next(iter(self.spectroscopy.values()))] + list((self.topography or {}).values())
        xs = np.concatenate([_spatial_axis(ch, "X", default_index=1) for ch in channels])
        ys = np.concatenate([_spatial_axis(ch, "Y", default_index=0) for ch in channels])
        return float(np.nanmin(xs)), float(np.nanmax(xs)), float(np.nanmin(ys)), float(np.nanmax(ys))

    @property
    def bias(self) -> np.ndarray:
        """The excitation bias (V) paired with each step of every spectroscopy channel."""
        return next(iter(self.spectroscopy.values())).axes[2].values

    @cached_property
    def image_channels(self) -> dict[str, np.ndarray]:
        """Full-frame image channels for :meth:`STMMicroscope.scan_image`.

        These are the real ``"topography"`` section channels, each at its own
        native resolution, when present;
        otherwise a single synthesized ``"current_avg"`` channel (the
        spectroscopy ``Current`` cube averaged over bias) -- see
        :attr:`has_topography` to tell which case applies.
        """
        if self.topography is not None:
            return {name: ch.data for name, ch in self.topography.items()}
        _, current_ch = _pick_channel(self.spectroscopy, quantity_prefix="current")
        return {"current_avg": current_ch.data.mean(axis=2)}

    @property
    def default_image_channel(self) -> str:
        if self.topography is not None:
            name, _ = _pick_channel(self.topography, quantity_prefix="z")
            return name
        return "current_avg"

    @property
    def default_spectroscopy_channel(self) -> str:
        name, _ = _pick_channel(self.spectroscopy, quantity_prefix="current")
        return name
