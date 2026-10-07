import numpy as np
import pytest

from mexpsimu.exceptions import DatasetNotFoundError
from mexpsimu.registry import (
    default_data_dir,
    discover_datasets,
    list_presets,
    parse_scan_size_um,
    resolve_dataset_path,
)


def _write_fake_dataset(path):
    np.savez(path, image=np.zeros((4, 4)), spectra=np.zeros((4, 4, 3)))


def test_list_presets_finds_bundled_datasets():
    presets = list_presets()
    assert presets == [
        "PTO_BEPS_1d3um",
        "PTO_BEPS_1d7um",
        "PTO_BEPS_3um",
        "artificial_defect_cAFM",
    ]


def test_discover_datasets_maps_names_to_paths():
    datasets = discover_datasets()
    assert set(datasets) == {
        "PTO_BEPS_1d3um",
        "PTO_BEPS_1d7um",
        "PTO_BEPS_3um",
        "artificial_defect_cAFM",
    }
    for path in datasets.values():
        assert path.is_file()
        assert path.parent == default_data_dir()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("PTO_BEPS_1d3um.npz", 1.3),
        ("PTO_BEPS_1d7um.npz", 1.7),
        ("PTO_BEPS_3um.npz", 3.0),
        ("no_size_here.npz", None),
        ("artificial_defect_cAFM.npz", None),
    ],
)
def test_parse_scan_size_um(name, expected):
    assert parse_scan_size_um(name) == expected


def test_resolve_dataset_path_by_preset_name():
    path = resolve_dataset_path("PTO_BEPS_1d3um")
    assert path.name == "PTO_BEPS_1d3um.npz"


def test_resolve_dataset_path_unknown_raises():
    with pytest.raises(DatasetNotFoundError):
        resolve_dataset_path("does_not_exist")


def test_discover_datasets_with_custom_data_dir(tmp_path):
    _write_fake_dataset(tmp_path / "my_scan.npz")
    _write_fake_dataset(tmp_path / "another_scan.npz")

    datasets = discover_datasets(tmp_path)
    assert set(datasets) == {"my_scan", "another_scan"}
    for path in datasets.values():
        assert path.parent == tmp_path
    # the bundled presets are not mixed in
    assert "PTO_BEPS_1d3um" not in datasets


def test_discover_datasets_missing_dir_returns_empty():
    assert discover_datasets("/no/such/directory") == {}


def test_list_presets_with_custom_data_dir(tmp_path):
    _write_fake_dataset(tmp_path / "zzz_scan.npz")
    _write_fake_dataset(tmp_path / "aaa_scan.npz")

    assert list_presets(tmp_path) == ["aaa_scan", "zzz_scan"]


def test_resolve_dataset_path_by_name_in_custom_data_dir(tmp_path):
    _write_fake_dataset(tmp_path / "my_scan.npz")

    path = resolve_dataset_path("my_scan", data_dir=tmp_path)
    assert path == tmp_path / "my_scan.npz"


def test_resolve_dataset_path_unknown_in_custom_data_dir_raises(tmp_path):
    _write_fake_dataset(tmp_path / "my_scan.npz")

    with pytest.raises(DatasetNotFoundError):
        resolve_dataset_path("PTO_BEPS_1d3um", data_dir=tmp_path)


def test_resolve_dataset_path_explicit_file_ignores_data_dir(tmp_path):
    # An explicit path to an existing .npz file is used directly, regardless
    # of data_dir (or the bundled presets).
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    file_path = other_dir / "my_scan.npz"
    _write_fake_dataset(file_path)

    resolved = resolve_dataset_path(file_path, data_dir=tmp_path)
    assert resolved == file_path
