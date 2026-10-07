"""A digital-twin STM: replays recorded Nanonis data as if it were a live instrument.

:class:`STMMicroscope` is the STM counterpart of :class:`mexpsimu.microscope.Microscope`.
It wraps a :class:`~mexpsimu.stm_dataset.STMScanDataset` -- recorded topography
images plus a grid-spectroscopy (``.3ds``) cube -- and exposes the three
operations an acquisition loop needs:

* :meth:`STMMicroscope.scan_image` -- image scan: return a full-frame 2-D image.
* :meth:`STMMicroscope.move_tip` -- tip motion: move the tip to an ``(x, y)``
  position in nm.
* :meth:`STMMicroscope.spectroscopy` -- local spectroscopy: return a bias
  sweep (e.g. I(V) or the lock-in dI/dV) at the tip position.

Where the data comes from
-------------------------
* A preset name or ``.h5`` path -- ``STMMicroscope("large_area")`` -- loads a
  converted pyNSID file (see ``scripts/combine_stm_datasets.py``).
* Raw Nanonis files -- :meth:`STMMicroscope.from_nanonis` -- reads a ``.3ds``
  grid, optionally with an ``.sxm`` scan of the same frame.

Coordinates and the spectroscopy grid
-------------------------------------
Positions are in nanometers in the scan frame: ``x`` runs along image
columns, ``y`` along image rows, both starting at 0 at the frame corner,
taken from the dataset's own recorded X/Y axes. The tip may be placed
anywhere in the scan area (:attr:`STMScanDataset.scan_extent` -- the grid and
the images together). Spectra were recorded only on the ``.3ds`` grid (e.g.
180x180 points over 250 nm, ~1.4 nm apart): between grid points
:meth:`STMMicroscope.spectroscopy` bilinearly interpolates the four
surrounding grid spectra (see :meth:`STMMicroscope.neighbour_weights`);
in the thin strip between the last grid point and the image edge it uses
the nearest edge grid spectra. Image channels are independent of
the grid and are always returned at their own native resolution (e.g. a
512x512 ``.sxm`` image).

Measurement noise
-----------------
By default the simulator is deterministic: the same call always returns the
same spectrum. ``STMMicroscope(..., measurement_noise=0.02)`` adds Gaussian
noise to every :meth:`STMMicroscope.spectroscopy` result, with a standard
deviation of 2 % of the channel's RMS signal over the whole grid; pass
``seed`` for reproducible noise. Images from :meth:`STMMicroscope.scan_image`
are not affected. Keep it below 1 % (e.g. 0.001 -- 0.01): the RMS is
dominated by the large high-bias signal, so larger values swamp the small
low-bias signal (e.g. 2 % of a ``Current`` RMS of ~2.8 nA is ~57 pA).

Example
-------
>>> scope = STMMicroscope("large_area")
>>> image = scope.scan_image()                 # .sxm "Z forward", 512x512, in m
>>> scope.move_tip(110.0, 120.0)               # nm
>>> bias, current = scope.spectroscopy()       # "Current" vs bias (V), in A
>>> bias, didv = scope.spectroscopy(channel="LIX 1 omega")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .exceptions import ChannelNotFoundError, OutOfScanRangeError
from .registry import resolve_stm_dataset_path
from .stm_dataset import STMScanDataset

__all__ = ["STMMicroscope", "STMTipPosition"]

INTERPOLATION_MODES = ("bilinear", "nearest")


@dataclass(frozen=True)
class STMTipPosition:
    """Where the tip currently is.

    Attributes
    ----------
    row, col:
        Indices of the spectroscopy-grid pixel nearest to the tip
        (``row`` along Y, ``col`` along X). Useful for indexing the raw
        ``dataset.spectroscopy[...].data[row, col]`` cube.
    x_nm, y_nm:
        The exact tip position in nm. With :meth:`STMMicroscope.move_tip`
        this can lie between grid points; with
        :meth:`STMMicroscope.move_tip_pixel` it is exactly on grid point
        ``(row, col)``.
    """

    row: int
    col: int
    x_nm: float
    y_nm: float


def _fractional_index(values: np.ndarray, v: float) -> float:
    """Position of ``v`` along a monotonic axis, in (fractional) index units."""
    n = len(values)
    if n == 1:
        return 0.0
    index = np.arange(n, dtype=float)
    if values[-1] < values[0]:
        return float(np.interp(v, values[::-1], index[::-1]))
    return float(np.interp(v, values, index))


def _bilinear_neighbours(frac_row: float, frac_col: float, shape: tuple[int, int]) -> list[tuple[int, int, float]]:
    """The four grid pixels around fractional position ``(frac_row, frac_col)``
    as ``(row, col, weight)``; weights are bilinear and sum to 1."""
    rows, cols = shape
    r0 = min(int(np.floor(frac_row)), max(rows - 2, 0))
    c0 = min(int(np.floor(frac_col)), max(cols - 2, 0))
    r1, c1 = min(r0 + 1, rows - 1), min(c0 + 1, cols - 1)
    ty, tx = frac_row - r0, frac_col - c0
    return [
        (r0, c0, (1 - ty) * (1 - tx)),
        (r0, c1, (1 - ty) * tx),
        (r1, c0, ty * (1 - tx)),
        (r1, c1, ty * tx),
    ]


class STMMicroscope:
    """Simulated STM: image scan, tip motion and local spectroscopy on recorded data.

    The microscope holds one :class:`STMScanDataset` (``self.dataset``) and a
    tip position (``self.position``), which starts at the centre of the
    spectroscopy grid. Nothing is ever measured: :meth:`scan_image` returns a
    copy of a recorded image and :meth:`spectroscopy` builds a spectrum from
    the recorded grid, so repeated calls give identical results.

    Attributes
    ----------
    dataset:
        The underlying :class:`STMScanDataset` (recorded channels and axes).
    default_image_channel:
        Channel returned by :meth:`scan_image` when none is given.
    default_spectroscopy_channel:
        Channel returned by :meth:`spectroscopy` when none is given.
    interpolation:
        ``"bilinear"`` or ``"nearest"`` -- how :meth:`spectroscopy` turns grid
        spectra into the spectrum at the tip position.
    measurement_noise:
        Relative noise level added to :meth:`spectroscopy` results (0 = none);
        can be changed after creation.
    """

    def __init__(
        self,
        dataset: str | Path | STMScanDataset,
        data_dir: str | Path | None = None,
        default_image_channel: str | None = None,
        default_spectroscopy_channel: str | None = None,
        interpolation: str = "bilinear",
        measurement_noise: float = 0.0,
        seed: int | None = None,
    ) -> None:
        """Create a microscope backed by a converted dataset.

        Parameters
        ----------
        dataset:
            One of: a preset name (e.g. ``"large_area"``, see
            :func:`mexpsimu.list_stm_presets`); a path to a combined ``.h5``
            file or to a data folder holding one ``sidpy/*.h5``; or an
            already-loaded :class:`STMScanDataset`. To start from raw Nanonis
            ``.3ds``/``.sxm`` files instead, use :meth:`from_nanonis`.
        data_dir:
            Folder to look preset names up in (defaults to mexpsimu's bundled
            ``data/stm_data``).
        default_image_channel:
            Channel :meth:`scan_image` returns by default. Defaults to
            ``"Z forward"`` (``.sxm`` topography) if present, else the
            ``.3ds`` grid ``"Topography"``, else the synthesized
            ``"current_avg"``.
        default_spectroscopy_channel:
            Channel :meth:`spectroscopy` returns by default. Defaults to the
            forward ``"Current"`` channel.
        interpolation:
            How :meth:`spectroscopy` builds the spectrum at the tip position:

            - ``"bilinear"`` (default): distance-weighted average of the four
              grid spectra around the tip (see :meth:`neighbour_weights`);
              equals the recorded spectrum when the tip is on a grid point.
            - ``"nearest"``: the recorded spectrum of the nearest grid point,
              unmodified.
        measurement_noise:
            Simulated measurement noise for :meth:`spectroscopy`, as a
            fraction of each channel's RMS signal over the whole grid: every
            spectrum gets independent Gaussian noise with standard deviation
            ``measurement_noise * rms(channel)`` added at each bias point.
            ``0`` (default) adds no noise; e.g. ``0.005`` is 0.5 % noise. The
            same fraction applies to every channel. Recommended: below 1 %
            (``< 0.01``) -- the RMS is dominated by the large high-bias
            signal, so larger values swamp the small low-bias part of the
            spectra. Can be changed later via the ``measurement_noise``
            attribute.
        seed:
            Seed for the noise generator, for reproducible noisy spectra;
            ``None`` (default) gives different noise on every run.

        Raises
        ------
        DatasetNotFoundError
            If ``dataset`` is not a known preset or an existing file/folder.
        ChannelNotFoundError
            If a given default channel does not exist in the dataset.
        ValueError
            If ``interpolation`` is not ``"bilinear"`` or ``"nearest"``, or
            ``measurement_noise`` is negative.
        """
        if interpolation not in INTERPOLATION_MODES:
            raise ValueError(f"unknown interpolation {interpolation!r}; expected one of {INTERPOLATION_MODES}")
        self.interpolation = interpolation
        self.measurement_noise = measurement_noise
        self._rng = np.random.default_rng(seed)
        self._channel_rms: dict[str, float] = {}

        if isinstance(dataset, STMScanDataset):
            self.dataset = dataset
        else:
            path = resolve_stm_dataset_path(dataset, data_dir=data_dir)
            self.dataset = STMScanDataset.load(path)

        if default_image_channel is not None and default_image_channel not in self.dataset.image_channels:
            raise ChannelNotFoundError(
                f"image channel {default_image_channel!r} not in dataset {self.dataset.name!r}; "
                f"available channels: {list(self.dataset.image_channels)}"
            )
        self.default_image_channel = default_image_channel or self.dataset.default_image_channel

        if (
            default_spectroscopy_channel is not None
            and default_spectroscopy_channel not in self.dataset.spectroscopy
        ):
            raise ChannelNotFoundError(
                f"spectroscopy channel {default_spectroscopy_channel!r} not in dataset "
                f"{self.dataset.name!r}; available channels: {list(self.dataset.spectroscopy)}"
            )
        self.default_spectroscopy_channel = (
            default_spectroscopy_channel or self.dataset.default_spectroscopy_channel
        )

        rows, cols = self.dataset.spectroscopy_shape
        self._position = self._pixel_to_position(rows // 2, cols // 2)

    @classmethod
    def from_nanonis(
        cls,
        file_3ds: str | Path,
        file_sxm: str | Path | None = None,
        default_image_channel: str | None = None,
        default_spectroscopy_channel: str | None = None,
        interpolation: str = "bilinear",
        measurement_noise: float = 0.0,
        seed: int | None = None,
    ) -> "STMMicroscope":
        """Create a microscope straight from raw Nanonis files (no ``.h5`` needed).

        Two variants:

        - ``STMMicroscope.from_nanonis(file_3ds=..., file_sxm=...)`` -- the
          ``.sxm`` scan provides the image channels (default ``"Z forward"``),
          the ``.3ds`` grid the spectroscopy channels. The ``.3ds`` grid's own
          ``"Topography"`` is available as an extra image channel.
        - ``STMMicroscope.from_nanonis(file_3ds=...)`` -- the ``.3ds`` grid
          alone; its ``"Topography"`` (Z-controller height at each grid
          point) is the only image channel.

        The ``.sxm`` must cover the same frame as the ``.3ds`` grid (same
        centre, size and angle), so that tip positions mean the same thing in
        both. Reading raw files requires the gnganesh99/SciFiReaders fork.

        Parameters
        ----------
        file_3ds:
            Path to the Nanonis ``.3ds`` grid-spectroscopy file (required).
        file_sxm:
            Path to a Nanonis ``.sxm`` scan of the same frame (optional).
        default_image_channel, default_spectroscopy_channel, interpolation, measurement_noise, seed:
            As for :class:`STMMicroscope`.

        Raises
        ------
        ValueError
            If ``file_3ds`` is not a ``.3ds`` or ``file_sxm`` not an ``.sxm`` file.
        FileNotFoundError
            If a file does not exist.
        RuntimeError
            If the installed SciFiReaders lacks the fork's ``.3ds`` parser.
        """
        return cls(
            STMScanDataset.from_nanonis(file_3ds, file_sxm),
            default_image_channel=default_image_channel,
            default_spectroscopy_channel=default_spectroscopy_channel,
            interpolation=interpolation,
            measurement_noise=measurement_noise,
            seed=seed,
        )

    def __repr__(self) -> str:
        return (
            f"STMMicroscope(dataset={self.dataset.name!r}, "
            f"has_topography={self.has_topography}, "
            f"spectroscopy_shape={self.dataset.spectroscopy_shape}, "
            f"measurement_noise={self.measurement_noise}, "
            f"position={self._position})"
        )

    @property
    def measurement_noise(self) -> float:
        """Relative noise level added to :meth:`spectroscopy` results: the
        Gaussian standard deviation as a fraction of the channel's RMS signal
        (same fraction for every channel). ``0`` means no noise; keep it
        below 1 % (``< 0.01``), see :class:`STMMicroscope`. Settable, e.g.
        ``scope.measurement_noise = 0`` to switch noise off."""
        return self._measurement_noise

    @measurement_noise.setter
    def measurement_noise(self, value: float) -> None:
        value = float(value)
        if not value >= 0.0:
            raise ValueError(f"measurement_noise must be >= 0, got {value}")
        self._measurement_noise = value

    # -- image scan -----------------------------------------------------

    @property
    def has_topography(self) -> bool:
        """``True`` if the dataset has a recorded topography image (``.sxm``
        channels and/or the ``.3ds`` grid ``Topography``); ``False`` if the only
        image is the synthesized ``"current_avg"`` (the ``Current`` cube
        averaged over bias)."""
        return self.dataset.has_topography

    def list_channels(self) -> list[str]:
        """Names of the image channels :meth:`scan_image` can return.

        e.g. ``["Z forward", "Z backward", "Current forward", ...,
        "Topography"]`` for ``.sxm`` + ``.3ds`` data, ``["Topography"]`` for
        ``.3ds``-only data.
        """
        return list(self.dataset.image_channels)

    def scan_image(self, channel: str | None = None) -> np.ndarray:
        """Perform a full-frame image scan and return the recorded 2-D image.

        Parameters
        ----------
        channel:
            Which image channel to return (see :meth:`list_channels`);
            defaults to ``default_image_channel``.

        Returns
        -------
        numpy.ndarray
            A ``(rows, cols)`` array -- row index along Y, column index along
            X, row 0 at the bottom of the frame (plot with
            ``origin="lower"``) -- at the channel's native resolution, in the
            channel's units (``Z``: m, ``Current``/``LIX``: A). It is a copy:
            modifying it does not change the dataset. The tip position does
            not affect the result.

        Raises
        ------
        ChannelNotFoundError
            If ``channel`` is not an image channel of this dataset.
        """
        channel = channel or self.default_image_channel
        channels = self.dataset.image_channels
        if channel not in channels:
            raise ChannelNotFoundError(
                f"image channel {channel!r} not in dataset {self.dataset.name!r}; "
                f"available channels: {list(channels)}"
            )
        return channels[channel].copy()

    # -- tip motion -------------------------------------------------------

    @property
    def position(self) -> STMTipPosition:
        """The tip's current position: exact ``x_nm``/``y_nm`` plus the nearest
        grid pixel ``row``/``col`` (see :class:`STMTipPosition`)."""
        return self._position

    def move_tip(self, x: float, y: float, units: str = "nm") -> STMTipPosition:
        """Move the tip to ``(x, y)`` and return the new position.

        Parameters
        ----------
        x, y:
            Target position. ``x`` runs along image columns, ``y`` along image
            rows, both from 0 at the frame corner.
        units:
            ``"nm"`` (default): physical position in nanometers; the tip goes
            to exactly ``(x, y)``, anywhere in the scan area
            (:attr:`STMScanDataset.scan_extent`), including between grid
            points and beyond the last grid point up to the image edge.
            ``"px"``: grid indices (``x`` = column, ``y`` = row), rounded to
            the nearest grid point -- same as :meth:`move_tip_pixel`.

        Returns
        -------
        STMTipPosition
            The new position (exact ``x_nm``/``y_nm`` and nearest grid
            ``row``/``col``).

        Raises
        ------
        OutOfScanRangeError
            If ``(x, y)`` is outside the scan area
            (:attr:`STMScanDataset.scan_extent`, e.g. 0 -- 249.5 nm).
        ValueError
            If ``units`` is not ``"nm"`` or ``"px"``.
        """
        if units == "px":
            return self.move_tip_pixel(x, y)
        if units != "nm":
            raise ValueError(f"unknown units {units!r}; expected 'nm' or 'px'")
        x_vals, y_vals = self._x_values(), self._y_values()
        x_lo, x_hi, y_lo, y_hi = self.dataset.scan_extent
        if not (x_lo <= x <= x_hi) or not (y_lo <= y <= y_hi):
            raise OutOfScanRangeError(
                f"({x}, {y}) nm is outside the scanned area "
                f"x=[{x_lo:.2f}, {x_hi:.2f}] nm, y=[{y_lo:.2f}, {y_hi:.2f}] nm"
            )
        col = int(np.argmin(np.abs(x_vals - x)))
        row = int(np.argmin(np.abs(y_vals - y)))
        self._position = STMTipPosition(row=row, col=col, x_nm=float(x), y_nm=float(y))
        return self._position

    def move_tip_pixel(self, col: int, row: int) -> STMTipPosition:
        """Move the tip exactly onto spectroscopy grid point ``(col, row)``.

        A spectrum taken there is the recorded spectrum of that grid point,
        whatever the ``interpolation`` mode.

        Parameters
        ----------
        col, row:
            Grid indices (``col`` along X, ``row`` along Y); non-integers are
            rounded.

        Returns
        -------
        STMTipPosition
            The new position, with ``x_nm``/``y_nm`` set to that grid point.

        Raises
        ------
        OutOfScanRangeError
            If ``(col, row)`` is outside the grid
            (:attr:`STMScanDataset.spectroscopy_shape`).
        """
        rows, cols = self.dataset.spectroscopy_shape
        row_i, col_i = int(round(row)), int(round(col))
        if not (0 <= row_i < rows and 0 <= col_i < cols):
            raise OutOfScanRangeError(
                f"pixel (col={col_i}, row={row_i}) is outside the {cols}x{rows} scan area"
            )
        self._position = self._pixel_to_position(row_i, col_i)
        return self._position

    def _x_values(self) -> np.ndarray:
        return self.dataset.x_values

    def _y_values(self) -> np.ndarray:
        return self.dataset.y_values

    def _pixel_to_position(self, row: int, col: int) -> STMTipPosition:
        x_vals, y_vals = self._x_values(), self._y_values()
        return STMTipPosition(row=row, col=col, x_nm=float(x_vals[col]), y_nm=float(y_vals[row]))

    # -- spectroscopy -----------------------------------------------------

    @property
    def bias(self) -> np.ndarray:
        """The bias sweep (V): the x-axis shared by every spectrum
        :meth:`spectroscopy` returns (e.g. 121 points from -1.5 to 2.5 V)."""
        return self.dataset.bias

    def list_spectroscopy_channels(self) -> list[str]:
        """Names of the channels :meth:`spectroscopy` can return.

        e.g. ``["Current", "LIX 1 omega", "LIY 1 omega"]``: the tunnelling
        current and the lock-in X/Y outputs (``LIX`` is proportional to
        dI/dV). A backward-sweep channel, if recorded, is named
        ``"<name> [bwd]"``.
        """
        return list(self.dataset.spectroscopy)

    def neighbour_weights(self) -> list[tuple[int, int, float]]:
        """The grid points (and weights) the spectrum at the current tip
        position is built from.

        Returns
        -------
        list of (row, col, weight)
            ``"bilinear"`` mode: the four grid points surrounding the tip.
            With ``dx``, ``dy`` the tip's offset from the lower-left one in
            grid-spacing units (0 -- 1), the weights are
            ``(1-dx)(1-dy)``, ``dx(1-dy)``, ``(1-dx)dy`` and ``dx*dy`` -- the
            closer a grid point, the larger its weight. They sum to 1; on a
            grid point that point gets weight 1 and the others 0. Beyond the
            last grid point (between the grid edge and the image edge) the
            position is clamped to the grid edge, so the nearest edge grid
            spectra are used. ``"nearest"`` mode: just the nearest grid
            point, weight 1.

        Example
        -------
        >>> scope.move_tip(110.0, 120.0)
        >>> scope.neighbour_weights()  # weights rounded here
        [(86, 79, 0.48), (86, 80, 0.12), (87, 79, 0.32), (87, 80, 0.08)]
        """
        pos = self._position
        if self.interpolation == "nearest":
            return [(pos.row, pos.col, 1.0)]
        frac_row = _fractional_index(self._y_values(), pos.y_nm)
        frac_col = _fractional_index(self._x_values(), pos.x_nm)
        return _bilinear_neighbours(frac_row, frac_col, self.dataset.spectroscopy_shape)

    def spectroscopy(
        self,
        x: float | None = None,
        y: float | None = None,
        channel: str | None = None,
        units: str = "nm",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Take a local spectrum (a bias sweep) at the tip position.

        Parameters
        ----------
        x, y:
            Optional position to move the tip to first (see
            :meth:`move_tip`); give both or neither. Without them the spectrum
            is taken at the current :attr:`position`.
        channel:
            Which spectroscopy channel to return (see
            :meth:`list_spectroscopy_channels`); defaults to
            ``default_spectroscopy_channel`` (``"Current"``).
        units:
            Units of ``x``/``y``: ``"nm"`` (default) or ``"px"``.

        Returns
        -------
        (bias, response) : tuple of numpy.ndarray
            ``bias`` -- the sweep in V (same as :attr:`bias`); ``response``
            -- the channel's value at each bias point, in the channel's units
            (A for ``Current`` and ``LIX``/``LIY``), as a new float64 array.

            With ``interpolation="bilinear"`` (default) ``response`` is the
            weighted average of the grid spectra listed by
            :meth:`neighbour_weights`, so it is an interpolated estimate,
            not a recorded spectrum, unless the tip is exactly on a grid
            point (e.g. after :meth:`move_tip_pixel`). With
            ``interpolation="nearest"`` it is the recorded spectrum of the
            nearest grid point. If a neighbour has missing (NaN) values --
            e.g. an unfinished grid -- it is left out at those bias points
            and the remaining weights renormalized; a bias point with no
            valid neighbour is NaN.

            If ``measurement_noise`` > 0, Gaussian noise (standard deviation
            ``measurement_noise`` x the channel's RMS signal) is added to
            ``response``, freshly drawn on every call -- so repeating a
            measurement at the same spot gives a different spectrum, as on a
            real instrument.

        Raises
        ------
        ChannelNotFoundError
            If ``channel`` is not a spectroscopy channel of this dataset.
        OutOfScanRangeError
            If ``(x, y)`` is outside the scan area
            (:attr:`STMScanDataset.scan_extent`).
        ValueError
            If only one of ``x``/``y`` is given.
        """
        if x is not None or y is not None:
            if x is None or y is None:
                raise ValueError("both x and y must be given, or neither")
            self.move_tip(x, y, units=units)

        channel = channel or self.default_spectroscopy_channel
        if channel not in self.dataset.spectroscopy:
            raise ChannelNotFoundError(
                f"spectroscopy channel {channel!r} not in dataset {self.dataset.name!r}; "
                f"available channels: {list(self.dataset.spectroscopy)}"
            )
        cube = self.dataset.spectroscopy[channel].data
        weighted_sum = np.zeros(cube.shape[2])
        weight_total = np.zeros(cube.shape[2])
        for row, col, weight in self.neighbour_weights():
            if weight == 0.0:
                continue
            spectrum = cube[row, col]
            finite = np.isfinite(spectrum)
            weighted_sum += weight * np.where(finite, spectrum, 0.0)
            weight_total += weight * finite
        response = np.full(cube.shape[2], np.nan)
        np.divide(weighted_sum, weight_total, out=response, where=weight_total > 0)
        if self.measurement_noise > 0.0:
            sigma = self.measurement_noise * self._rms(channel)
            response += self._rng.normal(0.0, sigma, size=response.shape)
        return self.dataset.bias, response

    def _rms(self, channel: str) -> float:
        """RMS of a spectroscopy channel over the whole grid (cached): the
        noise scale for :attr:`measurement_noise`."""
        if channel not in self._channel_rms:
            cube = self.dataset.spectroscopy[channel].data
            self._channel_rms[channel] = float(np.sqrt(np.nanmean(np.square(cube, dtype=float))))
        return self._channel_rms[channel]
