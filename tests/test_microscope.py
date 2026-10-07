import numpy as np
import pytest

from mexpsimu.exceptions import ChannelNotFoundError, DatasetNotFoundError, OutOfScanRangeError
from mexpsimu.microscope import Microscope


@pytest.fixture()
def scope():
    return Microscope("PTO_BEPS_1d3um")


def test_starts_parked_at_center():
    scope = Microscope("PTO_BEPS_1d3um")
    rows, cols = 65, 65
    assert scope.position.row == rows // 2
    assert scope.position.col == cols // 2


def test_scan_image_returns_full_frame_at_its_native_resolution(scope):
    image = scope.scan_image()
    assert image.shape == (65, 65)
    assert np.array_equal(image, scope.dataset.channels["image"])
    # scan_image must not let callers mutate the stored dataset
    image[0, 0] = 12345.0
    assert scope.dataset.channels["image"][0, 0] != 12345.0


def test_scan_image_can_select_either_channel(scope):
    assert scope.list_channels() == ["image", "image2"]
    img1 = scope.scan_image("image")
    img2 = scope.scan_image("image2")
    assert img1.shape == img2.shape == (65, 65)
    assert not np.array_equal(img1, img2)


def test_scan_image_unknown_channel_raises(scope):
    with pytest.raises(ChannelNotFoundError):
        scope.scan_image("not_a_channel")


def test_move_tip_in_microns_lands_on_expected_pixel(scope):
    pos = scope.move_tip(0.0, 0.0, units="um")
    assert (pos.row, pos.col) == (0, 0)

    pos = scope.move_tip(1.3, 1.3, units="um")
    assert (pos.row, pos.col) == (64, 64)


def test_move_tip_pixel_direct(scope):
    pos = scope.move_tip_pixel(col=10, row=20)
    assert (pos.row, pos.col) == (20, 10)


def test_move_tip_out_of_range_raises(scope):
    with pytest.raises(OutOfScanRangeError):
        scope.move_tip(100.0, 100.0, units="um")
    with pytest.raises(OutOfScanRangeError):
        scope.move_tip_pixel(col=-1, row=0)


def test_move_tip_um_requires_known_scan_size():
    ds_channels = {"image": np.zeros((10, 10))}
    from mexpsimu.dataset import ScanDataset

    ds = ScanDataset(name="no_size", path=None, channels=ds_channels, scan_size_um=None)
    scope = Microscope(ds)
    with pytest.raises(ValueError):
        scope.move_tip(1.0, 1.0, units="um")
    # pixel units still work
    pos = scope.move_tip(3, 4, units="px")
    assert (pos.row, pos.col) == (4, 3)


def test_spectroscopy_at_current_position_matches_raw_array(scope):
    scope.move_tip_pixel(col=5, row=7)
    voltage, response = scope.spectroscopy()
    assert np.array_equal(response, scope.dataset.spectra[7, 5])
    assert voltage.shape == response.shape


def test_spectroscopy_uses_recorded_voltage_when_available(scope):
    voltage, _response = scope.spectroscopy()
    assert scope.voltage_source == "recorded"
    assert voltage is scope.dataset.recorded_voltage


def test_spectroscopy_falls_back_to_synthesized_voltage():
    # All bundled datasets currently carry a recorded vstep; exercise the
    # synthesized-fallback path with a dataset that doesn't.
    from mexpsimu.dataset import ScanDataset

    ds = ScanDataset(
        name="no_vstep",
        path=None,
        channels={"image": np.zeros((10, 10))},
        spectra=np.zeros((10, 10, 50)),
    )
    scope = Microscope(ds)
    voltage, _response = scope.spectroscopy()
    assert scope.voltage_source == "synthesized"
    assert voltage.shape == (scope.dataset.spectra.shape[-1],)


def test_spectroscopy_moves_tip_when_coordinate_given(scope):
    voltage, response = scope.spectroscopy(0.6, 0.6, units="um")
    assert scope.position.x_um == pytest.approx(0.6, abs=0.05)
    assert np.array_equal(response, scope.dataset.spectra[scope.position.row, scope.position.col])


def test_spectroscopy_requires_both_coordinates(scope):
    with pytest.raises(ValueError):
        scope.spectroscopy(x=0.5)


def test_spectroscopy_without_data_raises():
    from mexpsimu.dataset import ScanDataset
    from mexpsimu.exceptions import NoSpectroscopyDataError

    ds = ScanDataset(name="no_spectra", path=None, channels={"image": np.zeros((5, 5))})
    scope = Microscope(ds)
    assert scope.voltage_source is None
    with pytest.raises(NoSpectroscopyDataError):
        scope.spectroscopy()


def test_default_channel_can_be_overridden():
    scope = Microscope("PTO_BEPS_1d7um", default_channel="image2")
    assert np.array_equal(scope.scan_image(), scope.dataset.channels["image2"])


def test_waveform_kwargs_are_forwarded_when_no_recorded_voltage():
    # All three bundled presets now have a recorded vstep, which always wins
    # over synthesis parameters -- use a dataset without one instead.
    from mexpsimu.dataset import ScanDataset

    ds = ScanDataset(
        name="no_vstep",
        path=None,
        channels={"image": np.zeros((10, 10))},
        spectra=np.zeros((10, 10, 50)),
        waveform_kind="sine",
        waveform_v_max=2.0,
    )
    scope = Microscope(ds)
    assert np.max(np.abs(scope.excitation_waveform)) <= 2.0 + 1e-9


def test_waveform_kwargs_forwarded_through_preset_loading():
    scope = Microscope("PTO_BEPS_1d7um", waveform_kind="sine", waveform_v_max=2.0)
    assert scope.dataset.waveform_kind == "sine"
    assert scope.dataset.waveform_v_max == 2.0
    # the dataset's own recorded vstep still wins for the actual waveform used
    assert scope.voltage_source == "recorded"


def test_microscope_loads_preset_by_name_from_custom_data_dir(tmp_path):
    np.savez(tmp_path / "my_scan.npz", image=np.arange(16.0).reshape(4, 4))

    scope = Microscope("my_scan", data_dir=tmp_path)
    assert scope.dataset.name == "my_scan"
    assert np.array_equal(scope.scan_image(), np.arange(16.0).reshape(4, 4))


def test_microscope_loads_from_explicit_path_outside_data_dir(tmp_path):
    file_path = tmp_path / "my_scan.npz"
    np.savez(file_path, image=np.arange(16.0).reshape(4, 4))

    scope = Microscope(file_path)
    assert scope.dataset.name == "my_scan"
    assert np.array_equal(scope.scan_image(), np.arange(16.0).reshape(4, 4))


def test_microscope_custom_data_dir_does_not_see_bundled_presets(tmp_path):
    np.savez(tmp_path / "my_scan.npz", image=np.zeros((4, 4)))

    with pytest.raises(DatasetNotFoundError):
        Microscope("PTO_BEPS_1d3um", data_dir=tmp_path)
