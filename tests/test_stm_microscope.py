import numpy as np
import pytest

from mexpsimu.exceptions import ChannelNotFoundError, DatasetNotFoundError, OutOfScanRangeError
from mexpsimu.stm_dataset import STMScanDataset
from mexpsimu.stm_microscope import STMMicroscope

from conftest import make_sidpy_channel, write_combined_h5, write_large_area_3ds_only


@pytest.fixture()
def scope():
    return STMMicroscope("large_area")


def test_starts_parked_at_center(scope):
    rows, cols = 180, 180
    assert scope.position.row == rows // 2
    assert scope.position.col == cols // 2


def test_has_topography(scope):
    assert scope.has_topography is True


def test_scan_image_returns_default_channel_at_native_resolution(scope):
    image = scope.scan_image()
    assert image.shape == (512, 512)
    assert np.array_equal(image, scope.dataset.image_channels["Z forward"])
    # scan_image must not let callers mutate the stored dataset
    image[0, 0] = 12345.0
    assert scope.dataset.image_channels["Z forward"][0, 0] != 12345.0


def test_scan_image_can_select_other_channel(scope):
    assert "Current forward" in scope.list_channels()
    img = scope.scan_image("Current forward")
    assert img.shape == (512, 512)


def test_scan_image_unknown_channel_raises(scope):
    with pytest.raises(ChannelNotFoundError):
        scope.scan_image("not_a_channel")


def test_move_tip_in_nm_lands_on_expected_pixel(scope):
    pos = scope.move_tip(0.0, 0.0, units="nm")
    assert (pos.row, pos.col) == (0, 0)

    pos = scope.move_tip_pixel(col=179, row=179)
    assert (pos.row, pos.col) == (179, 179)


def test_move_tip_maps_x_to_columns_and_y_to_rows(scope):
    pos = scope.move_tip(200.0, 20.0, units="nm")
    assert pos.col > pos.row
    assert pos.x_nm == pytest.approx(200.0, abs=1.5)
    assert pos.y_nm == pytest.approx(20.0, abs=1.5)


def test_move_tip_out_of_range_raises(scope):
    with pytest.raises(OutOfScanRangeError):
        scope.move_tip(10_000.0, 10_000.0, units="nm")
    with pytest.raises(OutOfScanRangeError):
        scope.move_tip_pixel(col=-1, row=0)


def test_list_spectroscopy_channels(scope):
    assert set(scope.list_spectroscopy_channels()) == {"Current", "LIX 1 omega", "LIY 1 omega"}


def test_spectroscopy_default_channel_matches_raw_array(scope):
    scope.move_tip_pixel(col=5, row=7)
    bias, response = scope.spectroscopy()
    assert np.array_equal(response, scope.dataset.spectroscopy["Current"].data[7, 5])
    assert bias.shape == response.shape


def test_spectroscopy_can_select_channel(scope):
    bias, response = scope.spectroscopy(channel="LIX 1 omega")
    assert np.array_equal(response, scope.dataset.spectroscopy["LIX 1 omega"].data[scope.position.row, scope.position.col])


def test_spectroscopy_unknown_channel_raises(scope):
    with pytest.raises(ChannelNotFoundError):
        scope.spectroscopy(channel="not_a_channel")


def test_spectroscopy_moves_tip_when_coordinate_given(scope):
    scope.spectroscopy(50.0, 50.0, units="nm")
    assert scope.position.x_nm == pytest.approx(50.0, abs=2.0)
    assert scope.position.y_nm == pytest.approx(50.0, abs=2.0)


def test_spectroscopy_requires_both_coordinates(scope):
    with pytest.raises(ValueError):
        scope.spectroscopy(x=50.0)


def test_default_channel_override_validated_at_construction():
    with pytest.raises(ChannelNotFoundError):
        STMMicroscope("large_area", default_image_channel="nope")
    with pytest.raises(ChannelNotFoundError):
        STMMicroscope("large_area", default_spectroscopy_channel="nope")


def test_default_channel_override_accepted():
    scope = STMMicroscope("large_area", default_image_channel="Current forward")
    assert scope.default_image_channel == "Current forward"
    assert np.array_equal(scope.scan_image(), scope.dataset.image_channels["Current forward"])


def test_3ds_only_dataset_scans_grid_topography(tmp_path):
    scope = STMMicroscope(write_large_area_3ds_only(tmp_path / "3ds_only.h5"))
    assert scope.has_topography is True
    assert scope.list_channels() == ["Topography"]
    assert scope.scan_image().shape == scope.dataset.spectroscopy_shape


def test_no_topography_dataset_uses_current_avg_as_only_channel(tmp_path):
    current = np.arange(2 * 2 * 4, dtype=float).reshape(2, 2, 4)
    path = write_combined_h5(
        tmp_path / "spec_only.h5",
        spectroscopy={"Current forward": make_sidpy_channel(current, quantity="Current")},
    )
    scope = STMMicroscope(STMScanDataset.load(path))

    assert scope.has_topography is False
    assert scope.list_channels() == ["current_avg"]
    assert np.array_equal(scope.scan_image(), current.mean(axis=2))
    assert np.array_equal(scope.scan_image("current_avg"), current.mean(axis=2))


def test_microscope_loads_preset_by_name_from_custom_data_dir(tmp_path):
    current = np.zeros((2, 2, 3))
    write_combined_h5(
        tmp_path / "my_scan.h5",
        spectroscopy={"Current forward": make_sidpy_channel(current, quantity="Current")},
    )

    scope = STMMicroscope("my_scan", data_dir=tmp_path)
    assert scope.dataset.name == "my_scan"


def test_microscope_loads_from_explicit_path_outside_data_dir(tmp_path):
    current = np.zeros((2, 2, 3))
    path = write_combined_h5(
        tmp_path / "my_scan.h5",
        spectroscopy={"Current forward": make_sidpy_channel(current, quantity="Current")},
    )

    scope = STMMicroscope(path)
    assert scope.dataset.name == "my_scan"


def test_microscope_custom_data_dir_does_not_see_bundled_presets(tmp_path):
    write_combined_h5(
        tmp_path / "my_scan.h5",
        spectroscopy={"Current forward": make_sidpy_channel(np.zeros((2, 2, 3)), quantity="Current")},
    )

    with pytest.raises(DatasetNotFoundError):
        STMMicroscope("large_area", data_dir=tmp_path)


def _has_scifireaders_fork() -> bool:
    try:
        from SciFiReaders import Nanonis3dsReader
    except ImportError:
        return False
    return hasattr(Nanonis3dsReader, "_split_channel_name")


requires_fork = pytest.mark.skipif(
    not _has_scifireaders_fork(), reason="reading raw Nanonis files needs the gnganesh99/SciFiReaders fork"
)


@pytest.fixture(scope="module")
def raw_large_area():
    from mexpsimu.registry import default_stm_data_dir

    folder = default_stm_data_dir() / "large_area"
    return next(folder.glob("*.3ds")), next(folder.glob("*.sxm"))


@requires_fork
def test_from_nanonis_sxm_and_3ds_matches_converted_preset(raw_large_area):
    file_3ds, file_sxm = raw_large_area
    raw = STMMicroscope.from_nanonis(file_3ds=file_3ds, file_sxm=file_sxm)
    preset = STMMicroscope("large_area")

    assert raw.default_image_channel == "Z forward"
    assert sorted(raw.list_channels()) == sorted(preset.list_channels())
    assert raw.list_spectroscopy_channels() == preset.list_spectroscopy_channels()
    assert np.array_equal(raw.scan_image(), preset.scan_image())
    # axis values round-trip through the .h5 at slightly different precision,
    # so interpolation weights (and spectra) agree to float tolerance
    assert np.allclose(raw.spectroscopy(60.0, 180.0)[1], preset.spectroscopy(60.0, 180.0)[1], rtol=1e-6)
    assert raw.position == preset.position


@requires_fork
def test_from_nanonis_3ds_only_uses_grid_topography(raw_large_area):
    file_3ds, _ = raw_large_area
    scope = STMMicroscope.from_nanonis(file_3ds=file_3ds)

    assert scope.has_topography is True
    assert scope.list_channels() == ["Topography"]
    assert scope.default_image_channel == "Topography"
    assert scope.scan_image().shape == scope.dataset.spectroscopy_shape == (180, 180)
    assert scope.dataset.name == file_3ds.stem


@requires_fork
def test_from_nanonis_rejects_wrong_file_type(raw_large_area):
    _, file_sxm = raw_large_area
    with pytest.raises(ValueError):
        STMMicroscope.from_nanonis(file_3ds=file_sxm)


# -- 4-nearest-neighbour (bilinear) spectroscopy interpolation ---------------


def _grid_scope(tmp_path, cube, **kwargs):
    """Microscope on a synthetic grid whose X/Y axes are 0, 1, 2, ... nm."""
    path = write_combined_h5(
        tmp_path / "grid.h5", spectroscopy={"Current": make_sidpy_channel(cube, quantity="Current")}
    )
    return STMMicroscope(STMScanDataset.load(path), **kwargs)


def test_move_tip_keeps_exact_position_between_grid_points(tmp_path):
    scope = _grid_scope(tmp_path, np.zeros((3, 3, 2)))
    pos = scope.move_tip(0.4, 1.7)
    assert (pos.x_nm, pos.y_nm) == (0.4, 1.7)
    assert (pos.row, pos.col) == (2, 0)  # nearest grid pixel


def test_spectroscopy_on_grid_point_is_the_recorded_spectrum(tmp_path):
    cube = np.random.default_rng(0).normal(size=(3, 4, 5))
    scope = _grid_scope(tmp_path, cube)
    _, response = scope.spectroscopy(2.0, 1.0)  # x=2 -> col 2, y=1 -> row 1
    assert np.allclose(response, cube[1, 2])
    assert [(r, c) for r, c, w in scope.neighbour_weights() if w > 0] == [(1, 2)]


def test_spectroscopy_is_bilinear_weighted_average_of_four_neighbours(tmp_path):
    cube = np.random.default_rng(1).normal(size=(3, 3, 4))
    scope = _grid_scope(tmp_path, cube)
    _, response = scope.spectroscopy(0.25, 1.5)  # between cols 0-1, rows 1-2

    weights = {(r, c): w for r, c, w in scope.neighbour_weights()}
    assert weights == pytest.approx({(1, 0): 0.375, (1, 1): 0.125, (2, 0): 0.375, (2, 1): 0.125})
    expected = 0.375 * cube[1, 0] + 0.125 * cube[1, 1] + 0.375 * cube[2, 0] + 0.125 * cube[2, 1]
    assert np.allclose(response, expected)


def test_interpolation_is_linear_along_one_axis(tmp_path):
    # spectra that grow linearly with x must interpolate exactly
    cube = np.broadcast_to(np.arange(4.0)[None, :, None] * np.ones(3), (2, 4, 3)).copy()
    scope = _grid_scope(tmp_path, cube)
    for x in (0.0, 0.3, 1.5, 2.9, 3.0):
        assert np.allclose(scope.spectroscopy(x, 0.6)[1], x)


def test_nan_neighbour_is_skipped_and_weights_renormalized(tmp_path):
    cube = np.ones((2, 2, 3))
    cube[1, 1] = np.nan  # e.g. an incomplete grid's padding
    cube[0, 0] = 3.0
    scope = _grid_scope(tmp_path, cube)
    _, response = scope.spectroscopy(0.5, 0.5)  # equal 0.25 weights
    assert np.allclose(response, (3.0 + 1.0 + 1.0) / 3)


def test_nearest_interpolation_returns_nearest_recorded_spectrum(tmp_path):
    cube = np.random.default_rng(2).normal(size=(3, 3, 4))
    scope = _grid_scope(tmp_path, cube, interpolation="nearest")
    _, response = scope.spectroscopy(0.4, 1.7)
    assert np.array_equal(response, cube[2, 0])
    assert scope.neighbour_weights() == [(2, 0, 1.0)]


def test_unknown_interpolation_rejected(tmp_path):
    with pytest.raises(ValueError):
        _grid_scope(tmp_path, np.zeros((2, 2, 2)), interpolation="cubic")


# -- measurement noise --------------------------------------------------------


def test_no_noise_by_default_spectra_are_repeatable(scope):
    assert scope.measurement_noise == 0.0
    assert np.array_equal(scope.spectroscopy(110.0, 120.0)[1], scope.spectroscopy(110.0, 120.0)[1])


def test_noise_has_expected_size_and_changes_every_call(tmp_path):
    cube = np.random.default_rng(3).normal(size=(4, 4, 2000))
    noisy = _grid_scope(tmp_path, cube, measurement_noise=0.05, seed=0)
    clean = _grid_scope(tmp_path, cube)

    a = noisy.spectroscopy(1.5, 2.5)[1]
    b = noisy.spectroscopy(1.5, 2.5)[1]
    reference = clean.spectroscopy(1.5, 2.5)[1]
    assert not np.array_equal(a, b)
    rms = np.sqrt(np.mean(cube**2))
    assert np.std(a - reference) == pytest.approx(0.05 * rms, rel=0.1)


def test_noise_is_reproducible_with_seed(tmp_path):
    cube = np.random.default_rng(4).normal(size=(2, 2, 10))
    one = _grid_scope(tmp_path, cube, measurement_noise=0.1, seed=42)
    two = _grid_scope(tmp_path, cube, measurement_noise=0.1, seed=42)
    assert np.array_equal(one.spectroscopy(0.5, 0.5)[1], two.spectroscopy(0.5, 0.5)[1])


def test_noise_can_be_switched_off_after_creation(tmp_path):
    cube = np.random.default_rng(5).normal(size=(2, 2, 10))
    scope = _grid_scope(tmp_path, cube, measurement_noise=0.1)
    scope.measurement_noise = 0
    assert np.allclose(scope.spectroscopy(0.0, 0.0)[1], cube[0, 0])


def test_noise_does_not_affect_scan_image(tmp_path):
    cube = np.random.default_rng(6).normal(size=(2, 2, 10))
    scope = _grid_scope(tmp_path, cube, measurement_noise=0.5)
    assert np.array_equal(scope.scan_image(), scope.scan_image())


def test_negative_noise_rejected(tmp_path):
    with pytest.raises(ValueError):
        _grid_scope(tmp_path, np.zeros((2, 2, 2)), measurement_noise=-0.1)


# -- scan extent and edge clamping -------------------------------------------


def test_scan_extent_covers_grid_and_images(scope):
    x_min, x_max, y_min, y_max = scope.dataset.scan_extent
    assert (x_min, y_min) == (0.0, 0.0)
    assert x_max == pytest.approx(250.0 * 511 / 512)  # .sxm axis, wider than the grid's 250 * 179 / 180
    assert y_max == pytest.approx(250.0 * 511 / 512)


def test_tip_beyond_last_grid_point_uses_edge_spectrum(scope):
    pos = scope.move_tip(249.3, 249.3)  # past the grid's 248.6 nm, inside the .sxm image
    assert (pos.row, pos.col) == (179, 179)
    assert scope.neighbour_weights()[0] == (178, 178, pytest.approx(0.0))
    _, response = scope.spectroscopy()
    assert np.allclose(response, scope.dataset.spectroscopy["Current"].data[179, 179])


def test_tip_between_grid_edge_and_image_edge_clamps_along_one_axis(tmp_path):
    cube = np.random.default_rng(7).normal(size=(3, 3, 4))  # grid at 0, 1, 2 nm
    path = write_combined_h5(
        tmp_path / "wide_image.h5",
        spectroscopy={"Current": make_sidpy_channel(cube, quantity="Current")},
        topography={"Z forward": make_sidpy_channel(np.zeros((4, 4)), quantity="Z", units="m")},  # 0..3 nm
    )
    scope = STMMicroscope(STMScanDataset.load(path))
    _, response = scope.spectroscopy(2.6, 0.5)  # x beyond the grid, y between rows 0 and 1
    assert np.allclose(response, 0.5 * cube[0, 2] + 0.5 * cube[1, 2])
    with pytest.raises(OutOfScanRangeError):
        scope.move_tip(3.1, 0.5)  # outside the image too
