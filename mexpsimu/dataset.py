"""Loading of recorded scan datasets (``.npz`` files) into memory.

Note: this is the legacy AFM/BEPS npz-based loader -- not required by
stm_simu. Kept as-is for possible future integration; see
:mod:`mexpsimu.stm_dataset` for the STM-specific (pyNSID ``.h5``) loader.

Each dataset file holds one or more 2-D image channels (e.g. topography, an
averaged piezoresponse map) and, optionally, a 3-D ``spectra`` cube of shape
``(rows, cols, n_steps)`` holding a local spectroscopy measurement recorded
at every pixel of a (possibly coarser) spectroscopy grid, plus the recorded
excitation voltage for each of those ``n_steps`` steps under a ``vstep`` (or
``v_step``) key. Datasets that predate ``vstep`` being recorded have no such
key; :attr:`ScanDataset.excitation_waveform` falls back to a synthesized
waveform for those (see :mod:`mexpsimu.waveform`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from . import waveform as _waveform
from .registry import parse_scan_size_um

__all__ = ["ScanDataset"]

_SPECTRA_KEY = "spectra"
_VOLTAGE_KEYS = ("vstep", "v_step")


@dataclass
class ScanDataset:
    """An in-memory scanning-probe dataset: image channel(s) plus spectroscopy.

    Parameters
    ----------
    name:
        A human-readable identifier (defaults to the source file's stem).
    path:
        The ``.npz`` file this dataset was loaded from.
    channels:
        Mapping of channel name -> 2-D array, e.g. ``{"image": ..., "image2": ...}``.
        All channels must share the same pixel grid.
    spectra:
        ``(rows, cols, n_steps)`` array of local spectroscopy responses, or
        ``None`` if the dataset has no spectroscopy data. Its pixel grid
        (``rows``, ``cols``) may be coarser than the image channels' grid --
        spectroscopy is often recorded on fewer points than a full image scan.
    recorded_voltage:
        The actual excitation voltage applied at each of ``spectra``'s
        ``n_steps`` steps, if the source file recorded one (under a
        ``vstep``/``v_step`` key); otherwise ``None``.
    scan_size_um:
        Physical side length of the (assumed square) scanned area, in
        microns, if it could be determined; otherwise ``None``.
    waveform_kind, waveform_v_max, waveform_cycles:
        Parameters used to synthesize :attr:`excitation_waveform` when
        ``recorded_voltage`` is unavailable (see :mod:`mexpsimu.waveform`).
    """

    name: str
    path: Path
    channels: dict[str, np.ndarray]
    spectra: np.ndarray | None = None
    recorded_voltage: np.ndarray | None = None
    scan_size_um: float | None = None
    waveform_kind: str = "bipolar_triangle"
    waveform_v_max: float = 10.0
    waveform_cycles: int = 1

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError(f"dataset {self.name!r} has no image channels")
        shapes = {name: arr.shape[:2] for name, arr in self.channels.items()}
        if len(set(shapes.values())) > 1:
            raise ValueError(f"dataset {self.name!r} has mismatched channel shapes: {shapes}")
        if (
            self.spectra is not None
            and self.recorded_voltage is not None
            and self.recorded_voltage.shape[-1] != self.spectra.shape[-1]
        ):
            raise ValueError(
                f"dataset {self.name!r}: recorded_voltage length "
                f"{self.recorded_voltage.shape[-1]} does not match spectra's last "
                f"axis (length {self.spectra.shape[-1]})"
            )

    @classmethod
    def load(cls, path: str | Path, name: str | None = None, **waveform_kwargs) -> "ScanDataset":
        """Load a dataset from a ``.npz`` file.

        The ``"spectra"`` key (if present) becomes :attr:`spectra`; a
        ``"vstep"``/``"v_step"`` key (if present) becomes
        :attr:`recorded_voltage`. Every other array is treated as an image
        channel.
        """
        path = Path(path)
        with np.load(path) as npz:
            arrays = {key: np.asarray(npz[key]) for key in npz.files}
        spectra = arrays.pop(_SPECTRA_KEY, None)
        recorded_voltage = None
        for key in _VOLTAGE_KEYS:
            if key in arrays:
                recorded_voltage = arrays.pop(key)
                break
        return cls(
            name=name or path.stem,
            path=path,
            channels=arrays,
            spectra=spectra,
            recorded_voltage=recorded_voltage,
            scan_size_um=parse_scan_size_um(path.name),
            **waveform_kwargs,
        )

    @property
    def image_shape(self) -> tuple[int, int]:
        """The pixel grid shape ``(rows, cols)`` shared by all image channels."""
        return next(iter(self.channels.values())).shape[:2]

    @property
    def spectra_shape(self) -> tuple[int, int] | None:
        """The pixel grid shape ``(rows, cols)`` of the spectroscopy measurement."""
        return self.spectra.shape[:2] if self.spectra is not None else None

    @property
    def measurement_shape(self) -> tuple[int, int]:
        """The grid the tip is addressed on: the spectroscopy grid if present
        (it may be coarser than the image channels), else the image grid."""
        return self.spectra_shape or self.image_shape

    @property
    def channel_names(self) -> list[str]:
        return list(self.channels)

    @property
    def has_spectroscopy(self) -> bool:
        return self.spectra is not None

    @property
    def has_recorded_voltage(self) -> bool:
        return self.recorded_voltage is not None

    @cached_property
    def excitation_waveform(self) -> np.ndarray | None:
        """The excitation voltage paired with each step of :attr:`spectra`.

        This is the actual recorded waveform when the source file included a
        ``vstep``/``v_step`` array; otherwise it is synthesized (see
        :mod:`mexpsimu.waveform`) and its length matches ``spectra.shape[-1]``.
        """
        if self.recorded_voltage is not None:
            return self.recorded_voltage
        if self.spectra is None:
            return None
        return _waveform.generate(
            self.spectra.shape[-1],
            kind=self.waveform_kind,
            v_max=self.waveform_v_max,
            n_cycles=self.waveform_cycles,
        )
