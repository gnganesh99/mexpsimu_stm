import numpy as np
import pytest

from mexpsimu.exceptions import NoSpectroscopyDataError
from mexpsimu.registry import resolve_stm_dataset_path
from mexpsimu.stm_dataset import STMScanDataset

from conftest import make_sidpy_channel, write_combined_h5, write_large_area_3ds_only


SXM_CHANNELS = {
    "Z forward", "Z backward",
    "Current forward", "Current backward",
    "LIX_1_omega forward", "LIX_1_omega backward",
    "LIY_1_omega forward", "LIY_1_omega backward",
}
GRID_CHANNELS = {"Current", "LIX 1 omega", "LIY 1 omega"}


def test_bundled_presets():
    from mexpsimu.registry import list_stm_presets

    assert "large_area" in list_stm_presets()


def test_load_bundled_combined_dataset():
    ds = STMScanDataset.load(resolve_stm_dataset_path("large_area"))
    assert ds.has_topography is True
    assert ds.spectroscopy_shape == (180, 180)
    assert set(ds.spectroscopy) == GRID_CHANNELS
    assert all(ch.units == "A" for ch in ds.spectroscopy.values())
    # .sxm channels plus the .3ds grid's own Topography, at their native resolutions
    assert set(ds.topography) == SXM_CHANNELS | {"Topography"}
    assert ds.image_channels["Z forward"].shape == (512, 512)
    assert ds.image_channels["Topography"].shape == (180, 180)
    assert ds.bias.min() == pytest.approx(-1.5)
    assert ds.bias.max() == pytest.approx(2.5)
    assert ds.default_image_channel == "Z forward"
    assert ds.default_spectroscopy_channel == "Current"
    # real physical axes (nm), not pixel indices
    assert ds.x_values[-1] == pytest.approx(250.0 * 179 / 180)
    assert ds.y_values[-1] == pytest.approx(250.0 * 179 / 180)


def test_3ds_only_dataset_uses_grid_topography(tmp_path):
    ds = STMScanDataset.load(write_large_area_3ds_only(tmp_path / "3ds_only.h5"))
    assert ds.has_topography is True
    assert ds.spectroscopy_shape == (180, 180)
    assert set(ds.spectroscopy) == GRID_CHANNELS
    assert set(ds.topography) == {"Topography"}
    assert ds.default_image_channel == "Topography"
    topo = ds.topography["Topography"]
    assert topo.data.shape == (180, 180)
    assert (topo.quantity, topo.units) == ("Z", "m")
    assert np.all(np.isfinite(topo.data)) and np.ptp(topo.data) > 0


def test_no_topography_falls_back_to_current_avg(tmp_path):
    current = np.arange(2 * 2 * 4, dtype=float).reshape(2, 2, 4)
    lix = np.zeros((2, 2, 4))
    path = write_combined_h5(
        tmp_path / "spec_only.h5",
        spectroscopy={
            "Current forward": make_sidpy_channel(current, quantity="Current"),
            "LIX backward": make_sidpy_channel(lix, quantity="LIX"),
        },
    )
    ds = STMScanDataset.load(path)

    assert ds.has_topography is False
    assert ds.topography is None
    assert set(ds.image_channels) == {"current_avg"}
    assert ds.default_image_channel == "current_avg"
    assert np.array_equal(ds.image_channels["current_avg"], current.mean(axis=2))


def test_no_spectroscopy_section_raises(tmp_path):
    path = tmp_path / "empty.h5"
    import h5py

    with h5py.File(path, "w") as f:
        f.create_group("topography")

    with pytest.raises(NoSpectroscopyDataError):
        STMScanDataset.load(path)


def test_mismatched_spectroscopy_shapes_rejected(tmp_path):
    path = write_combined_h5(
        tmp_path / "bad.h5",
        spectroscopy={
            "Current forward": make_sidpy_channel(np.zeros((2, 2, 4)), quantity="Current"),
            "LIX backward": make_sidpy_channel(np.zeros((3, 3, 4)), quantity="LIX"),
        },
    )
    with pytest.raises(ValueError):
        STMScanDataset.load(path)


def test_mismatched_bias_axis_rejected(tmp_path):
    path = write_combined_h5(
        tmp_path / "bad_bias.h5",
        spectroscopy={
            "Current forward": make_sidpy_channel(
                np.zeros((2, 2, 4)), quantity="Current", bias=np.linspace(-1, 1, 4)
            ),
            "LIX backward": make_sidpy_channel(
                np.zeros((2, 2, 4)), quantity="LIX", bias=np.linspace(-2, 2, 4)
            ),
        },
    )
    with pytest.raises(ValueError):
        STMScanDataset.load(path)


def test_topography_channels_may_differ_in_resolution(tmp_path):
    path = write_combined_h5(
        tmp_path / "mixed_topo.h5",
        spectroscopy={"Current": make_sidpy_channel(np.zeros((2, 2, 4)), quantity="Current")},
        topography={
            "Z forward": make_sidpy_channel(np.zeros((8, 8)), quantity="Z", units="m"),
            "Topography": make_sidpy_channel(np.zeros((2, 2)), quantity="Z", units="m"),
        },
    )
    ds = STMScanDataset.load(path)
    assert ds.image_channels["Z forward"].shape == (8, 8)
    assert ds.image_channels["Topography"].shape == (2, 2)
    assert ds.default_image_channel == "Z forward"


def test_non_2d_topography_channel_rejected(tmp_path):
    path = write_combined_h5(
        tmp_path / "bad_topo.h5",
        spectroscopy={"Current": make_sidpy_channel(np.zeros((2, 2, 4)), quantity="Current")},
        topography={"Topography": make_sidpy_channel(np.zeros((2, 2, 4)), quantity="Z", units="m")},
    )
    with pytest.raises(ValueError):
        STMScanDataset.load(path)


def test_default_channel_prefers_forward_direction(tmp_path):
    path = write_combined_h5(
        tmp_path / "directions.h5",
        spectroscopy={
            "Current backward": make_sidpy_channel(np.zeros((2, 2, 3)), quantity="Current"),
            "Current forward": make_sidpy_channel(np.ones((2, 2, 3)), quantity="Current"),
        },
        topography={
            "Z backward": make_sidpy_channel(np.zeros((4, 4)), quantity="Z", units="m"),
            "Z forward": make_sidpy_channel(np.ones((4, 4)), quantity="Z", units="m"),
        },
    )
    ds = STMScanDataset.load(path)
    assert ds.default_spectroscopy_channel == "Current forward"
    assert ds.default_image_channel == "Z forward"


def test_default_channel_skips_3ds_bwd_channels(tmp_path):
    path = write_combined_h5(
        tmp_path / "bwd.h5",
        spectroscopy={
            "Current [bwd]": make_sidpy_channel(np.zeros((2, 2, 3)), quantity="Current"),
            "Current": make_sidpy_channel(np.ones((2, 2, 3)), quantity="Current"),
        },
    )
    assert STMScanDataset.load(path).default_spectroscopy_channel == "Current"


def test_x_y_values_follow_axis_names_not_positions(tmp_path):
    current = make_sidpy_channel(np.zeros((2, 3, 4)), quantity="Current")  # 2 rows (Y), 3 cols (X)
    ds = STMScanDataset.load(write_combined_h5(tmp_path / "xy.h5", spectroscopy={"Current": current}))
    assert len(ds.y_values) == 2
    assert len(ds.x_values) == 3
