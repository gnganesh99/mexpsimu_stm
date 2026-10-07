"""A tour of the three core mexpsimu operations: image scan, tip motion, and
local spectroscopy.

Run with:

    uv run --with numpy --with matplotlib python examples/quickstart.py

(matplotlib is optional -- the script still runs and prints results without it.)
"""

from __future__ import annotations

import mexpsimu


def main() -> None:
    print("Available datasets:", mexpsimu.list_presets())

    # Any of the bundled presets works; PTO_BEPS_1d3um has two image channels.
    scope = mexpsimu.Microscope("PTO_BEPS_1d3um")
    print(scope)
    print("Channels:", scope.list_channels())

    # 1. Image scan: grab a full recorded frame, selecting a channel by name.
    topography = scope.scan_image("image")
    response_map = scope.scan_image("image2")
    print(f"Topography scan: shape={topography.shape}, "
          f"min={topography.min():.3g}, max={topography.max():.3g}")
    print(f"Response-map scan: shape={response_map.shape}")

    # 2. Move the tip to a physical coordinate (microns), then check where it landed.
    position = scope.move_tip(0.6, 0.6, units="um")
    print(f"Tip moved to pixel (row={position.row}, col={position.col}), "
          f"physical ({position.x_um:.3f}, {position.y_um:.3f}) um")

    # 3. Local spectroscopy at the tip's current position.
    voltage, response = scope.spectroscopy()
    print(f"Spectroscopy at current position: {len(voltage)} points, "
          f"voltage range [{voltage.min():.2f}, {voltage.max():.2f}] V "
          f"(source: {scope.voltage_source})")

    # spectroscopy() can also move the tip and measure in one call.
    voltage, response = scope.spectroscopy(0.2, 1.0, units="um")
    print(f"Spectroscopy after moving to (0.2, 1.0) um -> "
          f"position now {scope.position}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].imshow(topography, cmap="viridis")
        axes[0].set_title("scan_image()")
        axes[1].plot(voltage, response)
        axes[1].set_xlabel("excitation voltage (V, synthesized)")
        axes[1].set_ylabel("response (a.u.)")
        axes[1].set_title("spectroscopy()")
        fig.tight_layout()
        fig.savefig("quickstart_output.png", dpi=150)
        print("Saved plot to quickstart_output.png")
    except ImportError:
        print("(install matplotlib to also render a plot)")


if __name__ == "__main__":
    main()
