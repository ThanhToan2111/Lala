"""Paper figure: Naturalized IPIB beta versus validation joint advantage."""

from pathlib import Path
import shutil

from paper.scripts.fig_naturalized_ipib import main as _plot


if __name__ == "__main__":
    _plot()
    for suffix in ("pdf", "png"):
        shutil.copyfile(
            Path("results/figures/v60") / f"fig_naturalized_ipib.{suffix}",
            Path("results/figures/v60") / f"fig_naturalized_detection.{suffix}",
        )
