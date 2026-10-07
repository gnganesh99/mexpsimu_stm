"""A digital-twin scanning-probe microscope backed by recorded scan data.

Note: this is the legacy AFM/BEPS npz-based microscope -- not required by
stm_simu. Kept as-is for possible future integration; see
:mod:`mexpsimu.stm_microscope` for the STM-specific microscope.

:class:`Microscope` is the main entry point of mexpsimu. It wraps a
:class:`~mexpsimu.dataset.ScanDataset` and exposes the three operations a
real experiment control loop needs:

* :meth:`Microscope.scan_image` -- perform an image scan (return a recorded
  image channel for the whole frame).
* :meth:`Microscope.move_tip` -- move the tip to a coordinate on the sample.
* :meth:`Microscope.spectroscopy` -- perform a local spectroscopy
  measurement at the tip's current (or a given) coordinate.

Coordinates in physical units are microns, with the origin ``(0, 0)`` at the
top-left pixel of the scan and ``(scan_size_um, scan_size_um)`` at the
bottom-right pixel; ``x`` runs along columns and ``y`` along rows. Pixel
coordinates use the same ``(x, y)`` = ``(col, row)`` convention.

The tip is addressed on the dataset's spectroscopy grid
(:attr:`ScanDataset.measurement_shape`), which may be coarser than an image
channel's own pixel grid -- :meth:`Microscope.scan_image` returns a channel
at its native resolution regardless of where the tip currently is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import ScanDataset
from .exceptions import ChannelNotFoundError, NoSpectroscopyDataError, OutOfScanRangeError
from .registry import resolve_dataset_path

__all__ = ["Microscope", "TipPosition"]


@dataclass(frozen=True)
class TipPosition:
    """Where the tip currently is, in both pixel and physical coordinates."""

    row: int
    col: int
    x_um: float | None
    y_um: float | None


class Microscope:
    """Simulates image scan, tip motion, and local spectroscopy against a dataset."""

    def __init__(
        self,
        dataset: str | Path | ScanDataset,
        data_dir: str | Path | None = None,
        default_channel: str | None = None,
        **waveform_kwargs,
    ) -> None:
        """Create a microscope backed by ``dataset``.

        ``dataset`` may be a :class:`ScanDataset`, a preset name (e.g.
        ``"PTO_BEPS_1d3um"``), or an explicit path to a ``.npz`` file.
        ``data_dir`` overrides where preset names are looked up (defaults to
        mexpsimu's bundled sample data). ``**waveform_kwargs`` are forwarded
        to :class:`ScanDataset` to control the synthesized excitation
        waveform (``waveform_kind``, ``waveform_v_max``, ``waveform_cycles``).
        """
        if isinstance(dataset, ScanDataset):
            self.dataset = dataset
        else:
            path = resolve_dataset_path(dataset, data_dir=data_dir)
            self.dataset = ScanDataset.load(path, **waveform_kwargs)

        if default_channel is not None and default_channel not in self.dataset.channels:
            raise ChannelNotFoundError(
                f"channel {default_channel!r} not in dataset {self.dataset.name!r}; "
                f"available channels: {self.dataset.channel_names}"
            )
        self.default_channel = default_channel or self.dataset.channel_names[0]

        rows, cols = self.dataset.measurement_shape
        self._position = self._pixel_to_position(rows // 2, cols // 2)

    def __repr__(self) -> str:
        return (
            f"Microscope(dataset={self.dataset.name!r}, "
            f"image_shape={self.dataset.image_shape}, "
            f"measurement_shape={self.dataset.measurement_shape}, "
            f"position={self._position})"
        )

    # -- image scan -----------------------------------------------------

    def list_channels(self) -> list[str]:
        """Names of the image channels available for :meth:`scan_image`."""
        return self.dataset.channel_names

    def scan_image(self, channel: str | None = None) -> np.ndarray:
        """Perform a full-frame image scan and return the recorded 2-D image.

        ``channel`` selects which recorded channel to return (defaults to
        the microscope's ``default_channel``, e.g. topography).
        """
        channel = channel or self.default_channel
        if channel not in self.dataset.channels:
            raise ChannelNotFoundError(
                f"channel {channel!r} not in dataset {self.dataset.name!r}; "
                f"available channels: {self.dataset.channel_names}"
            )
        return self.dataset.channels[channel].copy()

    # -- tip motion -------------------------------------------------------

    @property
    def position(self) -> TipPosition:
        """The tip's current position."""
        return self._position

    def move_tip(self, x: float, y: float, units: str = "um") -> TipPosition:
        """Move the tip to coordinate ``(x, y)`` and return the new position.

        ``units`` is ``"um"`` (physical microns, the default) or ``"px"``
        (pixel indices). Raises :class:`OutOfScanRangeError` if the
        coordinate falls outside the scanned area.
        """
        col, row = self._to_pixel(x, y, units)
        return self.move_tip_pixel(col, row)

    def move_tip_pixel(self, col: int, row: int) -> TipPosition:
        """Move the tip directly to pixel ``(col, row)`` and return the new position."""
        rows, cols = self.dataset.measurement_shape
        row_i, col_i = int(round(row)), int(round(col))
        if not (0 <= row_i < rows and 0 <= col_i < cols):
            raise OutOfScanRangeError(
                f"pixel (col={col_i}, row={row_i}) is outside the {cols}x{rows} scan area"
            )
        self._position = self._pixel_to_position(row_i, col_i)
        return self._position

    def _to_pixel(self, x: float, y: float, units: str) -> tuple[float, float]:
        """Convert an (x, y) coordinate to a (col, row) pixel coordinate."""
        if units == "px":
            return x, y
        if units == "um":
            size = self.dataset.scan_size_um
            if size is None:
                raise ValueError(
                    f"dataset {self.dataset.name!r} has no known physical scan size; "
                    "use units='px' or set scan_size_um explicitly"
                )
            rows, cols = self.dataset.measurement_shape
            col = x / size * (cols - 1) if cols > 1 else 0.0
            row = y / size * (rows - 1) if rows > 1 else 0.0
            return col, row
        raise ValueError(f"unknown units {units!r}; expected 'um' or 'px'")

    def _pixel_to_position(self, row: int, col: int) -> TipPosition:
        size = self.dataset.scan_size_um
        rows, cols = self.dataset.measurement_shape
        x_um = (col / (cols - 1) * size) if (size is not None and cols > 1) else (0.0 if size is not None else None)
        y_um = (row / (rows - 1) * size) if (size is not None and rows > 1) else (0.0 if size is not None else None)
        return TipPosition(row=row, col=col, x_um=x_um, y_um=y_um)

    # -- spectroscopy -----------------------------------------------------

    @property
    def excitation_waveform(self) -> np.ndarray | None:
        """The excitation voltage waveform paired with each spectroscopy reading.

        See :mod:`mexpsimu.waveform` -- this is synthesized, not recorded,
        since the real drive signal was not saved with these datasets.
        """
        return self.dataset.excitation_waveform

    @property
    def voltage_source(self) -> str | None:
        """``"recorded"`` if the dataset stored a real voltage array (``vstep``),
        ``"synthesized"`` if it was generated, or ``None`` if there's no
        spectroscopy data at all."""
        if not self.dataset.has_spectroscopy:
            return None
        return "recorded" if self.dataset.has_recorded_voltage else "synthesized"

    def spectroscopy(
        self, x: float | None = None, y: float | None = None, units: str = "um"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform a local spectroscopy measurement and return ``(voltage, response)``.

        If ``x``/``y`` are given, the tip is moved there first (see
        :meth:`move_tip`); otherwise the measurement is taken at the tip's
        current position.
        """
        if x is not None or y is not None:
            if x is None or y is None:
                raise ValueError("both x and y must be given, or neither")
            self.move_tip(x, y, units=units)

        if not self.dataset.has_spectroscopy:
            raise NoSpectroscopyDataError(f"dataset {self.dataset.name!r} has no spectroscopy data")

        response = self.dataset.spectra[self._position.row, self._position.col].copy()
        voltage = self.dataset.excitation_waveform
        return voltage, response
