import numpy as np
import pytest

from mexpsimu.dataset import ScanDataset
from mexpsimu.registry import resolve_dataset_path


@pytest.mark.parametrize(
    ("preset", "expected_image_shape", "expected_spectra_shape", "expected_size_um", "has_vstep"),
    [
        ("PTO_BEPS_1d3um", (65, 65), (65, 65), 1.3, True),
        ("PTO_BEPS_1d7um", (100, 100), (100, 100), 1.7, True),
        ("PTO_BEPS_3um", (100, 100), (100, 100), 3.0, True),
        ("artificial_defect_cAFM", (50, 50), (50, 50), None, True),
    ],
)
def test_load_bundled_datasets(
    preset, expected_image_shape, expected_spectra_shape, expected_size_um, has_vstep
):
    ds = ScanDataset.load(resolve_dataset_path(preset))
    assert ds.image_shape == expected_image_shape
    assert ds.spectra_shape == expected_spectra_shape
    assert ds.measurement_shape == expected_spectra_shape
    assert ds.scan_size_um == expected_size_um
    assert ds.has_spectroscopy is True
    assert ds.has_recorded_voltage is has_vstep
    assert "image" in ds.channel_names
    assert ds.spectra.shape[:2] == expected_spectra_shape


def test_1d3um_has_two_image_channels():
    ds = ScanDataset.load(resolve_dataset_path("PTO_BEPS_1d3um"))
    assert set(ds.channel_names) == {"image", "image2"}
    assert ds.channels["image"].shape == ds.channels["image2"].shape == (65, 65)


def test_artificial_defect_cafm_has_single_channel_and_no_scan_size():
    ds = ScanDataset.load(resolve_dataset_path("artificial_defect_cAFM"))
    assert ds.channel_names == ["image"]
    assert ds.scan_size_um is None
    assert ds.excitation_waveform.min() == pytest.approx(0.0)
    assert ds.excitation_waveform.max() == pytest.approx(1.25)


def test_recorded_voltage_used_verbatim_when_present():
    ds = ScanDataset.load(resolve_dataset_path("PTO_BEPS_1d3um"))
    assert ds.excitation_waveform is ds.recorded_voltage
    assert ds.excitation_waveform.shape == (ds.spectra.shape[-1],)
    assert ds.excitation_waveform.min() == pytest.approx(-12.0)
    assert ds.excitation_waveform.max() == pytest.approx(12.0)


def test_all_bundled_datasets_have_recorded_voltage():
    # As of the current sample data, every bundled dataset records a real
    # vstep array; the synthesized fallback below covers datasets that don't.
    for preset in ("PTO_BEPS_1d3um", "PTO_BEPS_1d7um", "PTO_BEPS_3um"):
        ds = ScanDataset.load(resolve_dataset_path(preset))
        assert ds.has_recorded_voltage is True
        assert ds.excitation_waveform is ds.recorded_voltage


def test_excitation_waveform_synthesized_when_no_recorded_voltage():
    ds = ScanDataset(
        name="no_vstep",
        path=None,
        channels={"image": np.zeros((10, 10))},
        spectra=np.zeros((10, 10, 50)),
    )
    assert ds.has_recorded_voltage is False
    assert ds.excitation_waveform.shape == (50,)


def test_excitation_waveform_is_cached():
    ds = ScanDataset.load(resolve_dataset_path("PTO_BEPS_3um"))
    assert ds.excitation_waveform is ds.excitation_waveform


def test_dataset_without_spectra_has_no_excitation_waveform():
    ds = ScanDataset(name="synthetic", path=None, channels={"image": np.zeros((4, 4))})
    assert ds.has_spectroscopy is False
    assert ds.excitation_waveform is None
    assert ds.measurement_shape == ds.image_shape


def test_mismatched_channel_shapes_rejected():
    with pytest.raises(ValueError):
        ScanDataset(
            name="bad",
            path=None,
            channels={"image": np.zeros((4, 4)), "image2": np.zeros((5, 5))},
        )


def test_image_and_spectra_may_have_different_shapes():
    # Spectroscopy is often recorded on a coarser grid than a full image scan.
    ds = ScanDataset(
        name="mixed_grid",
        path=None,
        channels={"image": np.zeros((10, 10))},
        spectra=np.zeros((4, 4, 8)),
    )
    assert ds.image_shape == (10, 10)
    assert ds.spectra_shape == (4, 4)
    assert ds.measurement_shape == (4, 4)


def test_mismatched_recorded_voltage_length_rejected():
    with pytest.raises(ValueError):
        ScanDataset(
            name="bad_voltage",
            path=None,
            channels={"image": np.zeros((4, 4))},
            spectra=np.zeros((4, 4, 8)),
            recorded_voltage=np.zeros(5),
        )


def test_no_channels_rejected():
    with pytest.raises(ValueError):
        ScanDataset(name="empty", path=None, channels={})
