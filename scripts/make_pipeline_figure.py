"""Render the open-vocabulary change-retrieval engine pipeline (lab report system figure).

Self-authored schematic (matplotlib -> clean PNG; no external/web images). Encodes the real engine:
frozen CLIP-variant encoder -> global or patch-level Delta-similarity scoring -> ranked change events
with side-by-side pairs, query-conditioned heatmap, and a confidence score.

    python scripts/make_pipeline_figure.py --out report/figures/engine_pipeline.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

IN = "#fbe7c2"; ENC = "#cfe3f7"; SCORE = "#e8e8e8"; RANK = "#c9ecc9"; OUT = "#f3d6e8"; EDGE = "#444"


def box(ax, x, y, w, h, text, fc, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=1.1, edgecolor=EDGE, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, p1, p2, color=EDGE, lw=1.3):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, lw=lw,
                                 color=color, zorder=1, shrinkA=2, shrinkB=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="GBDA retrieval-engine pipeline schematic.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # inputs
    box(ax, 0.01, 0.62, 0.17, 0.10, "Natural-language\nchange query", IN, 9)
    box(ax, 0.01, 0.30, 0.17, 0.12, "Multitemporal RS tiles\n$(T_1, T_2)$", IN, 9)

    # frozen encoder (+ optional PEFT)
    box(ax, 0.22, 0.40, 0.21, 0.24,
        "Frozen CLIP-variant encoder\nCLIP ViT-L/14 · GeoRSCLIP ·\nRemoteCLIP\n\n[optional LoRA / PEFT head]", ENC, 8.6, True)
    arrow(ax, (0.18, 0.67), (0.22, 0.58))   # query -> encoder
    arrow(ax, (0.18, 0.36), (0.22, 0.46))   # tiles -> encoder

    # two scoring paths
    box(ax, 0.48, 0.56, 0.22, 0.12, "Global $\\Delta$-similarity\n(zero-shot / naive)", SCORE, 8.6)
    box(ax, 0.48, 0.31, 0.22, 0.14, "Patch-level $\\Delta$ (patch_top3)\nlocalised — best DEN config", SCORE, 8.6, True)
    arrow(ax, (0.43, 0.55), (0.48, 0.62))
    arrow(ax, (0.43, 0.49), (0.48, 0.38))
    ax.text(0.59, 0.70, "text $\\cdot$ image embeddings (global + patch)", fontsize=7.8, ha="center", color="#555")

    # rank
    box(ax, 0.75, 0.42, 0.115, 0.20, "Rank all\npairs →\ntop-$K$\nevents", RANK, 8.8, True)
    arrow(ax, (0.70, 0.62), (0.75, 0.56))
    arrow(ax, (0.70, 0.38), (0.75, 0.48))

    # output (Gradio)
    box(ax, 0.895, 0.30, 0.10, 0.44,
        "Per event\n(Gradio):\nT$_1$/T$_2$\nside-by-side\n+ change\nheatmap\n+ confidence", OUT, 8.0, True)
    arrow(ax, (0.865, 0.52), (0.895, 0.52))

    # seasonal gate note
    box(ax, 0.48, 0.10, 0.22, 0.10, "Seasonal-drift gate\n(stable-pair FP filter)", SCORE, 8.0)
    arrow(ax, (0.59, 0.20), (0.59, 0.31))

    ax.set_title("Open-vocabulary temporal change-retrieval engine", fontsize=11, pad=6)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
