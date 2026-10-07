# MExpSimu (STM)

A scanning tunneling microscopy (STM) experiment simulator. This is the STM
path of `mexpsimu`, playing back recorded STM/STS datasets through the same
three operations a real acquisition control loop uses:

- **image scan** -- perform a full-frame scan and get back a recorded image channel
- **move tip** -- move the probe tip to a coordinate on the sample
- **spectroscopy** -- take a local bias-sweep measurement (current, conductance) at the tip's position

It's meant as a stand-in for real instrument hardware when developing or
testing acquisition logic, agents, or analysis pipelines that need something
to talk to.

> The original AFM/BEPS path (`.npz`-based, piezoresponse spectroscopy) still
> exists in this repo -- see [README_afm.md](README_afm.md).

## Installation

```bash
pip install -e ".[dev]"   # editable install, with pytest for the test suite
```

Requires Python >= 3.10, numpy, h5py, `pyNSID`, and `SciFiReaders`.
Loading the bundled `.h5` presets works with any SciFiReaders; *converting*
raw Nanonis `.3ds`/`.sxm` files needs the
[gnganesh99/SciFiReaders](https://github.com/gnganesh99/SciFiReaders) fork
(fixed `.3ds` channel parsing, physical X/Y axes, grid `Topography` channel).

## Quick start

```python
import mexpsimu

mexpsimu.list_stm_presets()
# ['large_area', ...]

scope = mexpsimu.STMMicroscope("large_area")

# 1. Image scan -- pick a channel by name (defaults to .sxm 'Z forward')
scope.list_channels()                    # ['Current backward', ..., 'Topography', 'Z backward', 'Z forward']
image = scope.scan_image()               # -> (512, 512) array
grid_topo = scope.scan_image("Topography")  # -> (180, 180) Z map from the .3ds grid

# 2. Move the tip to a physical coordinate, in nanometers
scope.move_tip(125.0, 125.0)
scope.position                            # STMTipPosition(row=..., col=..., x_nm=..., y_nm=...)

# 3. Local spectroscopy at the current tip position
scope.list_spectroscopy_channels()        # ['Current', 'LIX 1 omega', 'LIY 1 omega']
bias, current = scope.spectroscopy()
scope.has_topography                      # True -- this dataset has a real topography scan
```

`spectroscopy()` works anywhere on the grid, not just on grid points: the
spectrum is the bilinear (distance-weighted) average of the four nearest grid
spectra -- `scope.neighbour_weights()` lists them as `(row, col, weight)`. On
a grid point it is exactly the recorded spectrum. The tip can go anywhere in
the scan area (`scope.dataset.scan_extent`); in the thin strip between the
last grid point and the image edge, the nearest edge grid spectra are used. Pass
`STMMicroscope(..., interpolation="nearest")` to always get the nearest grid
point's recorded spectrum instead.

The simulator is deterministic by default. For noisy spectra, set
`measurement_noise` when creating it -- Gaussian noise with a standard
deviation of that fraction of the channel's RMS signal, redrawn on every
`spectroscopy()` call (`scan_image()` is unaffected). Keep it below 1 %: the
RMS is dominated by the large high-bias signal, so larger values swamp the
small low-bias signal:

```python
scope = mexpsimu.STMMicroscope("large_area", measurement_noise=0.005, seed=0)  # 0.5 % noise, reproducible
scope.measurement_noise = 0                                                    # switch it off again
```

Raw Nanonis files can be loaded directly too, without a converted `.h5`
(needs the SciFiReaders fork, see Installation):

```python
scope = mexpsimu.STMMicroscope.from_nanonis(file_3ds="scan.3ds", file_sxm="scan.sxm")  # .sxm image + .3ds grid
scope = mexpsimu.STMMicroscope.from_nanonis(file_3ds="scan.3ds")                       # .3ds grid only
```

Every `.3ds` grid records the Z-controller height at each pixel, so a
`.3ds`-only dataset still has real topography: its
`Topography` channel, at the grid resolution, is the default image. Only a
dataset with no topography section at all falls back to a synthesized
`current_avg` channel -- the recorded `Current` cube averaged over bias --
with `has_topography` `False`.

Channel names are those of the SciFiReaders Nanonis readers: `.3ds` channels
are `Current`, `LIX 1 omega`, ... (`Current [bwd]` for a backward sweep);
`.sxm` channels are `Z forward`, `Z backward`, .... Axes are `(Y, X[, Bias])`
in nm/V: rows run along Y, columns along X.

Open `notebooks/stm_simulator_demo.ipynb` for an executed tour with plots of
both `from_nanonis` variants (`.sxm` + `.3ds`, and `.3ds` only), and
`notebooks/stm_sts_dataload_test.ipynb` for the raw-vs-converted data checks.

## Sample data

Raw Nanonis data lives in one folder per measurement under
`mexpsimu/data/stm_data/`; each folder's converted, simulator-ready file is in
its `sidpy/` subfolder. The preset name is the folder name.

Only `large_area` is tracked in git (it's the example used by the tests and
notebooks, for both the combined and the `.3ds`-only case); every other
folder under `stm_data/` is git-ignored and local-only, like the two below.

| preset                          | raw files              | `sidpy/` file                                   | spectroscopy        | topography |
|----------------------------------|------------------------|--------------------------------------------------|---------------------|------------|
| `large_area`                     | `.sxm` + `.3ds`        | `EuZn2As2008_EnZn2As2122_combined_sd.h5`         | 180x180x121, -1.5 -> 2.5 V, 250 nm | 8 `.sxm` channels (512x512) + grid `Topography` (180x180) |
| `4ao defect_with other defects`  | `.sxm` + `.3ds`        | `EuZn2As2023_EnZn2As2135_combined_sd.h5`         | 50x50x121, -1.5 -> 1.5 V, 12 nm    | 8 `.sxm` channels (512x512) + grid `Topography` (50x50) |
| `region4_3defects`               | `.3ds` only            | `EuZn2As2006_3ds_sd.h5`                          | 160x160x121, -1.5 -> 2.3 V, 12 nm  | grid `Topography` (160x160) |

Each `.h5` has a `spectroscopy` section (the `.3ds` 3-D channels) and a
`topography` section (the `.sxm` channels, if any, plus the `.3ds` grid
`Topography`). In each pair, the `.sxm` and `.3ds` share the same frame
(centre, size, angle).

## Adding your own data

Put one `.3ds` file (and optionally one `.sxm` of the same frame) in a new
folder under `mexpsimu/data/stm_data/`, then convert (with the SciFiReaders
fork installed):

```bash
python scripts/combine_stm_datasets.py               # converts every folder in stm_data/
python scripts/combine_stm_datasets.py --overwrite   # re-convert, replacing sidpy/*.h5
```

or from Python, for a folder anywhere:

```python
from scripts.combine_stm_datasets import convert_nanonis_folder
convert_nanonis_folder("/path/to/my_folder")   # -> /path/to/my_folder/sidpy/<...>_sd.h5
```

Then load it by preset name (folder name) if it's under `stm_data/` or under
a `data_dir` you pass, or by path: `mexpsimu.STMMicroscope("/path/to/my_folder")`.
`combine_sidpy_datasets()` in the same script still combines existing pyNSID
`.h5` files into `topography`/`spectroscopy` sections.

## Package layout

```
mexpsimu/
├── stm_dataset.py       STMScanDataset: loads a combined .h5 into topography + spectroscopy channels
├── stm_microscope.py    STMMicroscope: scan_image / move_tip / spectroscopy
├── nsid_io.py            Shared pyNSID .h5 read/write helpers
├── nanonis_io.py         Raw Nanonis .3ds/.sxm reading (needs the SciFiReaders fork)
├── registry.py           Discovers .h5 presets (STM functions) + legacy .npz presets (AFM)
├── exceptions.py         Shared exception types
└── data/stm_data/        Bundled sample datasets: <folder>/{*.3ds, *.sxm, sidpy/*.h5}
scripts/combine_stm_datasets.py   Convert raw Nanonis folders (.3ds [+ .sxm]) to sidpy .h5; combine .h5 files
tests/test_stm_dataset.py, test_stm_microscope.py
notebooks/stm_sts_dataload_test.ipynb   Raw sidpy/SciFiReaders data-loading tour
notebooks/stm_simulator_demo.ipynb      STMMicroscope.from_nanonis: .sxm + .3ds, and .3ds only
```

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
