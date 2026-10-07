"""Synthetic excitation (bias/drive) waveforms.

Note: not required by stm_simu -- the STM sidpy datasets always carry a
real recorded Bias array (see :mod:`mexpsimu.stm_dataset`), so this
synthesis path is only exercised by the legacy AFM/BEPS npz loader
(:mod:`mexpsimu.dataset`). Kept as-is for possible future integration.

The bundled sample datasets record only the *response* measured at each step
of a local spectroscopy sweep (see :mod:`mexpsimu.dataset`); the excitation
signal actually applied by the instrument during acquisition was not saved
alongside it. To let :class:`mexpsimu.microscope.Microscope` report a
(voltage, response) pair for a spectroscopy measurement, this module
generates a representative excitation waveform shaped like the DC-bias
sweeps used in switching-spectroscopy PFM / BEPS experiments, sized to match
the number of steps present in the recorded spectra.

All waveforms are pure functions of ``n_points`` and start and end at 0 V.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bipolar_triangle",
    "unipolar_triangle",
    "sine",
    "generate",
    "WAVEFORM_KINDS",
]


def bipolar_triangle(n_points: int, v_max: float = 10.0, n_cycles: int = 1) -> np.ndarray:
    """A bipolar triangular sweep: 0 -> +v_max -> 0 -> -v_max -> 0, repeated.

    This is the classic switching-spectroscopy PFM / BEPS DC-bias shape.
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    phase = np.linspace(0.0, float(n_cycles), n_points, endpoint=True)
    frac = phase - np.floor(phase)
    triangle = np.where(
        frac < 0.25,
        frac * 4.0,
        np.where(frac < 0.75, 2.0 - frac * 4.0, frac * 4.0 - 4.0),
    )
    return v_max * triangle


def unipolar_triangle(n_points: int, v_max: float = 10.0, n_cycles: int = 1) -> np.ndarray:
    """A unipolar triangular sweep: 0 -> +v_max -> 0, repeated."""
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    phase = np.linspace(0.0, float(n_cycles), n_points, endpoint=True)
    frac = phase - np.floor(phase)
    triangle = np.where(frac < 0.5, frac * 2.0, 2.0 - frac * 2.0)
    return v_max * triangle


def sine(n_points: int, v_max: float = 10.0, n_cycles: int = 1) -> np.ndarray:
    """A sinusoidal drive waveform, ``n_cycles`` full periods."""
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    phase = np.linspace(0.0, 2.0 * np.pi * n_cycles, n_points, endpoint=True)
    return v_max * np.sin(phase)


WAVEFORM_KINDS = {
    "bipolar_triangle": bipolar_triangle,
    "unipolar_triangle": unipolar_triangle,
    "sine": sine,
}


def generate(n_points: int, kind: str = "bipolar_triangle", v_max: float = 10.0, n_cycles: int = 1) -> np.ndarray:
    """Dispatch to one of :data:`WAVEFORM_KINDS` by name."""
    try:
        fn = WAVEFORM_KINDS[kind]
    except KeyError as exc:
        raise ValueError(
            f"unknown waveform kind {kind!r}; choose one of {sorted(WAVEFORM_KINDS)}"
        ) from exc
    return fn(n_points, v_max=v_max, n_cycles=n_cycles)
