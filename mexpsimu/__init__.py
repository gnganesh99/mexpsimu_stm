"""mexpsimu: a microscopy experiment simulator.

Two independent simulator paths live in this package:

- **STM** (:class:`STMMicroscope`, :class:`STMScanDataset`) -- drives
  combined pyNSID/sidpy ``.h5`` datasets (grid spectroscopy + optional
  topography). This is the actively developed path.
- **Legacy AFM/BEPS** (:class:`Microscope`, :class:`ScanDataset`) -- drives
  ``.npz`` datasets. Not required by stm_simu; kept as-is for possible
  future integration.

Both expose the same three operations a real acquisition control loop
needs: an image scan, tip motion, and local spectroscopy.

Quick start (STM)
------------------
>>> import mexpsimu
>>> mexpsimu.list_stm_presets()
['large_area', ...]
>>> scope = mexpsimu.STMMicroscope("large_area")
>>> image = scope.scan_image()
>>> scope.move_tip(50.0, 50.0)  # nanometers
>>> bias, response = scope.spectroscopy()

Quick start (legacy AFM/BEPS)
------------------------------
>>> mexpsimu.list_presets()
['PTO_BEPS_1d3um', 'PTO_BEPS_1d7um', 'PTO_BEPS_3um']
>>> scope = mexpsimu.Microscope("PTO_BEPS_1d3um")
>>> image = scope.scan_image()
>>> scope.move_tip(0.5, 0.5)  # microns
>>> voltage, response = scope.spectroscopy()
"""

from __future__ import annotations

from . import waveform  # legacy AFM/BEPS only -- not required by stm_simu
from .dataset import ScanDataset  # legacy AFM/BEPS only -- not required by stm_simu
from .exceptions import (
    ChannelNotFoundError,
    DatasetNotFoundError,
    MExpSimuError,
    NoSpectroscopyDataError,
    OutOfScanRangeError,
)
from .microscope import Microscope, TipPosition  # legacy AFM/BEPS only -- not required by stm_simu
from .registry import (
    default_data_dir,
    default_stm_data_dir,
    discover_datasets,
    discover_stm_datasets,
    list_presets,
    list_stm_presets,
    parse_scan_size_um,
    resolve_dataset_path,
    resolve_stm_dataset_path,
)
from .stm_dataset import STMScanDataset
from .stm_microscope import STMMicroscope, STMTipPosition

__version__ = "0.1.0"

__all__ = [
    # STM
    "STMMicroscope",
    "STMTipPosition",
    "STMScanDataset",
    "list_stm_presets",
    "discover_stm_datasets",
    "default_stm_data_dir",
    "resolve_stm_dataset_path",
    # legacy AFM/BEPS -- not required by stm_simu
    "Microscope",
    "TipPosition",
    "ScanDataset",
    "waveform",
    "list_presets",
    "discover_datasets",
    "default_data_dir",
    "resolve_dataset_path",
    "parse_scan_size_um",
    # shared
    "MExpSimuError",
    "DatasetNotFoundError",
    "ChannelNotFoundError",
    "OutOfScanRangeError",
    "NoSpectroscopyDataError",
]
