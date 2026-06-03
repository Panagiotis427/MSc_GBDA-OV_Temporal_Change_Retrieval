"""
Figure for the honest DEN result chain (REPORT Appendix B.8-B.10).

Panel A — the headline progression for GeoRSCLIP NRG: the committed 110-pair
test-split mAP (0.426) collapses under 5-fold AOI cross-validation (lucky fold),
then partially recovers via fraction-based relevance (S1, fixes eval starvation)
and patch-level localised scoring (S3, fixes the global-embedding ceiling).
Panel B — per-query CV AP, global vs patch, showing S3 rescues the localised
change-types (buildings / urban / water) that global pooling left at chance.

Reads the committed JSONs; reproduce: ``python -m scripts.make_cv_figure``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
OUT = Path("assets/figures/cv_progression.png")


def _load(name):
    return json.load(open(R / name))


def main() -> None:
    test_split = _load("dynamic_earthnet__georsclip__test__nrg__zero_shot.json")["macro"]["mAP"]
    cv_dom = _load("cv_eval__georsclip__nrg__zero_shot.json")["kfold_zero_shot"]
    cv_s1 = _load("cv_eval__georsclip__nrg__zero_shot__fraction.json")["kfold_zero_shot"]
    s3 = _load("patch_eval__georsclip__nrg__patch_top3.json")["kfold"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A — progression
    labels = ["test split\n(single, dominant-flip)", "5-fold CV\n(dominant-flip)",
              "5-fold CV\n+ fraction (S1)", "5-fold CV\n+ patch top-3 (S3)"]
    vals = [test_split, cv_dom["macro_mAP_mean"], cv_s1["macro_mAP_mean"], s3["macro_mAP_mean"]]
    errs = [0, cv_dom["macro_mAP_std"], cv_s1["macro_mAP_std"], s3["macro_mAP_std"]]
    colors = ["#c0392b", "#7f8c8d", "#2980b9", "#27ae60"]
    x = np.arange(4)
    bars = axA.bar(x, vals, yerr=errs, capsize=5, color=colors, edgecolor="black", linewidth=0.6)
    bars[0].set_hatch("//")
    axA.set_xticks(x)
    axA.set_xticklabels(labels, fontsize=8.5)
    axA.set_ylabel("DEN macro mAP")
    axA.set_title("GeoRSCLIP + NRG: headline 0.426 is a lucky single fold;\n"
                  "CV truth ≈ 0.10, recovered to ≈ 0.20 by fixing eval (S1) + method (S3)",
                  fontsize=9.5)
    for xi, v, e in zip(x, vals, errs):
        axA.text(xi, v + (e if e else 0) + 0.012, f"{v:.3f}" + (f"\n±{e:.3f}" if e else ""),
                 ha="center", va="bottom", fontsize=8)
    axA.axhline(0.083, ls="--", color="grey", lw=1)
    axA.text(3.4, 0.088, "random\nbaseline", fontsize=7, color="grey", ha="right")
    axA.set_ylim(0, max(vals) * 1.25)

    # Panel B — per-query CV AP, global (S1) vs patch (S3)
    g = _load("cv_eval__georsclip__nrg__zero_shot__fraction.json")["full_corpus"]["per_query"]
    p = _load("patch_eval__georsclip__nrg__patch_top3.json")["full_corpus"]["per_query"]
    gmap = {q["query"]: q for q in g}
    short = {"new buildings constructed on former agricultural land": "new buildings",
             "urban expansion replacing vegetation": "urban expansion",
             "deforestation, forest cleared to bare soil": "deforestation",
             "forest loss": "forest loss", "new water body or flooding": "new water",
             "bare soil or land cleared": "bare soil",
             "agricultural land converted to wetland or marsh": "ag→wetland",
             "wetland drained and turned into farmland": "wetland→farmland",
             "land turning into wetland": "→wetland"}
    order = [q["query"] for q in p]
    names = [short.get(q, q[:14]) for q in order]
    gv = [gmap[q]["ap"] for q in order]
    pv = [q["ap"] for q in p]
    psig = [q["perm_p"] < 0.05 for q in p]
    y = np.arange(len(order))
    h = 0.38
    axB.barh(y + h / 2, gv, h, label="global Δ (S1)", color="#2980b9")
    axB.barh(y - h / 2, pv, h, label="patch top-3 (S3)", color="#27ae60")
    for yi, sig, pvv in zip(y, psig, pv):
        if sig:
            axB.text(pvv + 0.004, yi - h / 2, "*", fontsize=12, va="center", color="#1e8449")
    axB.set_yticks(y)
    axB.set_yticklabels(names, fontsize=8.5)
    axB.invert_yaxis()
    axB.set_xlabel("full-corpus AP")
    axB.set_title("Per-query: patch scoring (S3) rescues localised change\n"
                  "(* = beats random at p<0.05; buildings/urban/water lit up)", fontsize=9.5)
    axB.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
