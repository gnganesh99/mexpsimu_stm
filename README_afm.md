# MExpSimu

A microscopy experiment simulator. `MExpSimu` plays back recorded
scanning-probe microscopy datasets through the same three operations a real
acquisition control loop uses:

- **image scan** -- perform a full-frame scan and get back a recorded image channel
- **move tip** -- move the probe tip to a coordinate on the sample
- **spectroscopy** -- take a local spectroscopy measurement at the tip's position

It's meant as a stand-in for real instrument hardware when developing or
testing acquisition logic, agents, or analysis pipelines that need something
to talk to.

## Installation

```bash
pip install -e ".[dev]"   # editable install, with pytest for the test suite
```

Requires Python >= 3.10 and numpy.

## Quick start

```python
import mexpsimu

mexpsimu.list_presets()
# ['PTO_BEPS_1d3um', 'PTO_BEPS_1d7um', 'PTO_BEPS_3um']

scope = mexpsimu.Microscope("PTO_BEPS_1d3um")

# 1. Image scan -- pick a channel by name (defaults to the first one)
scope.list_channels()                    # ['image', 'image2']
image = scope.scan_image("image")        # -> (65, 65) array, at that channel's own resolution

# 2. Move the tip to a physical coordinate, in microns
scope.move_tip(0.6, 0.6)
scope.position                            # TipPosition(row=28, col=28, x_um=..., y_um=...)

# 3. Local spectroscopy at the current tip position
voltage, response = scope.spectroscopy()
scope.voltage_source                      # "recorded" if the file had a vstep array, else "synthesized"

# ...or move-and-measure in one call
voltage, response = scope.spectroscopy(0.2, 1.0)
```

Run `examples/quickstart.py` for a plain-script tour (optionally plots with
matplotlib if it's installed), or open `notebooks/quickstart.ipynb` for the
same tour as an executed Jupyter notebook with plots inline.

## Sample data

Four datasets are bundled in `mexpsimu/data/`. Three are Band Excitation
Piezoresponse Spectroscopy (BEPS) scans on a PbTiO3 (PTO) sample, named by
their physical scan size; the fourth is a conductive-AFM (cAFM) scan with a
localized defect:

| preset                    | scan size | pixels  | channels           | spectrum length | recorded voltage |
|---------------------------|-----------|---------|---------------------|------------------|-------------------|
| `PTO_BEPS_1d3um`          | 1.3 um    | 65x65   | `image`, `image2`   | 192              | yes (`vstep`)     |
| `PTO_BEPS_1d7um`          | 1.7 um    | 100x100 | `image`, `image2`   | 256              | yes (`vstep`)     |
| `PTO_BEPS_3um`            | 3.0 um    | 100x100 | `image`, `image2`   | 192              | yes (`vstep`)     |
| `artificial_defect_cAFM`  | unknown*  | 50x50   | `image`             | 128              | yes (`vstep`)     |

\* `artificial_defect_cAFM`'s filename doesn't encode a scan size, so
`ScanDataset.scan_size_um` is `None` for it and `move_tip` requires pixel
units (or a manually-set `scan_size_um`) -- see
`notebooks/artificial_defect_cAFM.ipynb`.

Each `.npz` file holds one or more 2-D image channels (`image` is
topography; `image2` is a derived response map -- it equals the recorded
`spectra` averaged over its last axis), a `spectra` array of shape
`(rows, cols, n_steps)` (the local piezoresponse loop recorded at every
pixel), and a `vstep` array of length `n_steps` (the DC bias actually
applied at each step).

`Microscope` addresses the tip on the dataset's spectroscopy grid
(`ScanDataset.measurement_shape`), which is the same grid as the image
channels in all three bundled datasets, but doesn't have to be --
spectroscopy is slower to acquire than an image scan, so a dataset with a
coarser spectroscopy grid than its image channels is handled correctly too;
`scan_image()` always returns a channel at its own native resolution
regardless of where the tip currently is.

## Loading your own data

Any `.npz` file with the same layout (one or more 2-D image channels, an
optional `spectra` cube, an optional `vstep`/`v_step` array) can be used, in
two ways:

**Point at a directory of your own presets** with `data_dir` -- every
`.npz` file in it becomes available under its filename's stem, without
touching `mexpsimu/data/`:

```python
mexpsimu.list_presets(data_dir="/path/to/my/data")
# -> whatever presets live in that folder

scope = mexpsimu.Microscope("my_scan", data_dir="/path/to/my/data")
```

**Or load one file directly by path** -- no `data_dir` needed:

```python
scope = mexpsimu.Microscope("/path/to/my/data/my_scan.npz")
```

`data_dir` only changes where *preset names* are looked up; an explicit
path to an existing `.npz` file always works regardless of `data_dir`, and
the bundled sample datasets remain reachable by name whenever `data_dir`
isn't passed.

### A note on voltage and coordinates

- **Coordinates** are the pixel grid itself (the spectroscopy grid, when
  present). Physical coordinates (microns) are computed from the scan size
  parsed out of the filename, assuming a square scan with the origin at the
  top-left pixel.
- **Excitation voltage**: all three bundled datasets include a `vstep`
  array (the DC bias actually applied at each spectroscopy step, a +-12V
  bipolar triangular sweep over three cycles), which `Microscope` uses
  verbatim. For a dataset that lacks one, `Microscope` instead synthesizes a
  representative bias waveform (default: a bipolar triangular sweep,
  `0 -> +10V -> 0 -> -10V -> 0`, the same shape typically used in
  switching-spectroscopy PFM / BEPS) sized to match the spectrum's length.
  Check `Microscope.voltage_source` (`"recorded"` or `"synthesized"`) to
  tell which case you're in. The synthesized waveform's shape and amplitude
  can be changed via `waveform_kind`, `waveform_v_max`, and
  `waveform_cycles` when constructing a `Microscope` (see
  `mexpsimu/waveform.py`) -- these are ignored when a real `vstep` is
  present.

## Package layout

```
mexpsimu/
├── __init__.py       Public API
├── registry.py        Discovers .npz presets, parses scan size from filenames
├── dataset.py          ScanDataset: loads a .npz file into channels + spectra
├── waveform.py         Synthetic excitation waveforms (bipolar/unipolar triangle, sine)
├── microscope.py       Microscope: scan_image / move_tip / spectroscopy
├── exceptions.py
└── data/                Bundled sample datasets
tests/                   pytest suite exercising the bundled datasets directly
examples/quickstart.py
notebooks/quickstart.ipynb
notebooks/artificial_defect_cAFM.ipynb
```

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

## Acknowledgment

The domain concepts here (band-excitation PFM, image scan / tip motion /
spectroscopy as the basic building blocks of a probe microscopy experiment)
were informed by the [AEcroscopyWave](https://code.ornl.gov) project, which
drives real AFM hardware. MExpSimu is an independent implementation aimed at
simulating that experience against recorded data rather than live
instruments, and shares no code with it.

## License

MIT
