"""Exception types raised by mexpsimu."""

from __future__ import annotations


class MExpSimuError(Exception):
    """Base class for all mexpsimu errors."""


class DatasetNotFoundError(MExpSimuError, KeyError):
    """Raised when a requested dataset preset or file cannot be located."""


class ChannelNotFoundError(MExpSimuError, KeyError):
    """Raised when a requested image channel is not present in a dataset."""


class OutOfScanRangeError(MExpSimuError, ValueError):
    """Raised when a requested tip coordinate falls outside the scanned area."""


class NoSpectroscopyDataError(MExpSimuError):
    """Raised when spectroscopy is requested on a dataset that has none."""
