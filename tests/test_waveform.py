import numpy as np
import pytest

from mexpsimu import waveform


@pytest.mark.parametrize("kind", ["bipolar_triangle", "unipolar_triangle", "sine"])
def test_generate_shape_and_endpoints(kind):
    w = waveform.generate(50, kind=kind, v_max=5.0)
    assert w.shape == (50,)
    assert w[0] == pytest.approx(0.0)
    assert w[-1] == pytest.approx(0.0, abs=1e-9)


def test_bipolar_triangle_peaks_at_plus_and_minus_v_max():
    w = waveform.bipolar_triangle(101, v_max=8.0, n_cycles=1)
    assert w.max() == pytest.approx(8.0)
    assert w.min() == pytest.approx(-8.0)


def test_unipolar_triangle_never_goes_negative():
    w = waveform.unipolar_triangle(101, v_max=8.0)
    assert w.min() >= -1e-9
    assert w.max() == pytest.approx(8.0)


def test_sine_amplitude():
    w = waveform.sine(1000, v_max=3.0, n_cycles=2)
    assert np.max(np.abs(w)) <= 3.0 + 1e-9
    assert np.max(w) == pytest.approx(3.0, abs=1e-2)


def test_generate_unknown_kind_raises():
    with pytest.raises(ValueError):
        waveform.generate(10, kind="not_a_real_waveform")


def test_generate_too_few_points_raises():
    with pytest.raises(ValueError):
        waveform.generate(1)
