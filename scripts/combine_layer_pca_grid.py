"""
Utility for assembling the raw vs centered PCA subplots into a single figure.

The script reuses the per-layer PNGs already emitted by `template_controlling_final.ipynb`,
so you can tweak the combined figure without re-running any of the heavy PCA/probe code.

Usage:
    python combine_layer_pca_grid.py

This will save `artifacts/layer_raw_vs_centered_grid_clean.png`.
"""
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

# Map each layer to the raw/centered PCA images we already saved.
PANEL_IMAGES: Dict[int, Dict[str, Path]] = {
    20: {
        "raw": ARTIFACT_DIR / "layer_20_raw_pca.png",
        "centered": ARTIFACT_DIR / "layer_20_centered_pca.png",
    },
    26: {
        "raw": ARTIFACT_DIR / "layer_26_raw_pca.png",
        "centered": ARTIFACT_DIR / "layer_26_centered_pca.png",
    },
    28: {
        "raw": ARTIFACT_DIR / "layer_28_raw_pca.png",
        "centered": ARTIFACT_DIR / "layer_28_centered_pca.png",
    },
}


# Tweakable legend entries; adjust the labels or colors here without touching the rest.
LEGEND_LABELS = [
    ("answer_affirm", "#1f77b4"),
    ("answer_deny", "#ff7f0e"),
    ("answer_question_only", "#2ca02c"),
    ("neutral", "#d62728"),
]

OUTPUT_PATH = ARTIFACT_DIR / "layer_raw_vs_centered_grid_clean.png"


def load_image(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing panel image: {path}")
    return Image.open(path)


def main():
    rows = ["raw", "centered"]
    layers = sorted(PANEL_IMAGES)
    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=len(layers),
        figsize=(4.6 * len(layers), 8),
    )
    

    for col, layer in enumerate(layers):
        for row_idx, row_name in enumerate(rows):
            ax = axes[row_idx, col]
            img_path = PANEL_IMAGES[layer][row_name]
            img = load_image(img_path)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

    fig.subplots_adjust(
        left=-0.05, right=1.05, top=0.95, bottom=0.08, wspace=-0.1, hspace=0.0
    )

    handles = [Patch(facecolor=color, label=label) for label, color in LEGEND_LABELS]
    fig.legend(
        handles,
        [label for label, _ in LEGEND_LABELS],
        loc="lower center",
        ncol=len(LEGEND_LABELS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.suptitle('Layer PCA comparisons (raw vs centered)', fontsize=16, y=0.98)
    fig.savefig(OUTPUT_PATH, dpi=350)
    print(f"Saved combined figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
