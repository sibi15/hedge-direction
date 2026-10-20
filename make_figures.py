"""Draw the paper's three figures from the per-model results of run.py.

    python make_figures.py                        # looks for the results below this file
    python make_figures.py runs/ paper/figures/   # results folder, output folder

Each model folder (runs/qwen2.5-7b-instruct/, ...) needs its results.json; models
without one are left out. Writes a PDF (for LaTeX) and a PNG preview of each figure:
fig_auroc_depth, fig_test_effects and fig_dose.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# folder, label, colour, marker, line style. Qwen sizes go light -> dark blue;
# the other two families are dashed.
MODELS = [
    ("qwen2.5-1.5b-instruct", "Qwen2.5-1.5B", "#86b6ef", "o", "-"),
    ("qwen2.5-3b-instruct", "Qwen2.5-3B", "#5598e7", "s", "-"),
    ("qwen2.5-7b-instruct", "Qwen2.5-7B", "#2a78d6", "^", "-"),
    ("qwen2.5-14b-instruct", "Qwen2.5-14B", "#1c5cab", "D", "-"),
    ("qwen2.5-32b-instruct", "Qwen2.5-32B", "#0d366b", "X", "-"),
    ("mistral-7b-instruct-v0.3", "Mistral-7B", "#eb6834", "v", "--"),
    ("gemma-2-9b-it", "Gemma-2-9B", "#1baf7a", "P", "--"),
]

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIRECTION_C, RANDOM_C, SHUFFLE_C, FACTUAL_C = "#2a78d6", "#9a9994", "#52514e", "#e34948"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "pdf.fonttype": 42,  # embed TrueType instead of Type 3 fonts
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def find_results_dir(start: Path) -> Path:
    """First folder below `start` that has qwen2.5-7b-instruct/results.json."""
    for hit in sorted(start.rglob("qwen2.5-7b-instruct/results.json")):
        return hit.parent.parent
    raise SystemExit("Could not find the results folder. "
                     "Run: python make_figures.py RESULTS_DIR [OUT_DIR]")


def load_results(results_dir: Path):
    models = [m for m in MODELS if (results_dir / m[0] / "results.json").exists()]
    if not models:
        raise SystemExit(f"No model results found in {results_dir}")
    results = {m[0]: json.loads((results_dir / m[0] / "results.json").read_text()) for m in models}
    return models, results


def rate(block: dict, label: str):
    """(rate, 95% Wilson interval) of `label` in one evaluation block."""
    return block[label]["k"] / block["n"], block[label]["ci95"]


def save(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=220)
    plt.close(fig)


def fig_auroc_depth(models, results, out_dir: Path) -> None:
    """Fig. 1: validation AUROC of each layer's direction vs relative depth."""
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":", zorder=0)  # chance
    ax.grid(axis="y", color=GRID, lw=0.6)
    for folder, label, colour, marker, style in models:
        d = results[folder]["directions"]
        n_layers = d["n_layers"]
        auroc = {int(k): v for k, v in d["auroc_hedge_vs_direct"].items()}
        layers = sorted(auroc)
        ax.plot([l / n_layers for l in layers], [auroc[l] for l in layers],
                color=colour, ls=style, lw=1.2, label=label, zorder=2)
        chosen = results[folder]["final"]["layer"]
        ax.plot([chosen / n_layers], [auroc[chosen]], marker=marker, ms=5, color=colour,
                mec="white", mew=0.8, ls="none", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("relative depth $l/L$")
    ax.set_ylabel("validation AUROC")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              handlelength=1.8, columnspacing=0.8)
    save(fig, out_dir, "fig_auroc_depth")


def fig_test_effects(models, results, out_dir: Path) -> None:
    """Fig. 2: final test, hedging direction vs random and shuffled-label directions."""
    folders = [m[0] for m in models][::-1]  # reversed so the first model is on top
    labels = [m[1] for m in models][::-1]
    panels = [
        ("ablate_on_hedged", "random_ablate_on_hedged", "direct", "null_ablate_direct_rate",
         "Ablation: hedged answers turned direct"),
        ("add_on_direct", "random_add_on_direct", "hedge", "null_add_hedge_rate",
         "Addition: direct answers turned hedged"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), sharey=True)
    for ax, (key, random_key, label, null_key, title) in zip(axes, panels):
        ax.grid(axis="x", color=GRID, lw=0.6)
        for y, folder in enumerate(folders):
            final = results[folder]["final"]
            p, (lo, hi) = rate(final[key], label)
            rp, (rlo, rhi) = rate(final[random_key], label)
            null = final[null_key]
            # shuffled-label nulls: line = range, tick = mean
            ax.plot([min(null), max(null)], [y - 0.22] * 2, color=SHUFFLE_C, lw=0.8)
            ax.plot([sum(null) / len(null)], [y - 0.22], marker="|", ms=6, color=SHUFFLE_C, ls="none")
            ax.errorbar([rp], [y], xerr=[[rp - rlo], [rhi - rp]], fmt="s", ms=4, color=RANDOM_C,
                        mec="white", mew=0.6, elinewidth=1, capsize=0)
            ax.errorbar([p], [y + 0.22], xerr=[[p - lo], [hi - p]], fmt="o", ms=5, color=DIRECTION_C,
                        mec="white", mew=0.6, elinewidth=1.2, capsize=0)
            ax.text(1.05, y, f"{final[key][label]['k']}/{final[key]['n']}", va="center", ha="left",
                    fontsize=6.5, color=INK2, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.set_xlim(-0.02, 1.02)
        for artist in ax.lines + ax.collections:
            artist.set_clip_on(False)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("rate on held-out test split")
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].set_ylim(-0.6, len(labels) - 0.4)
    handles = [
        plt.Line2D([], [], marker="o", color=DIRECTION_C, ls="-", ms=5, label="hedging direction (95% CI)"),
        plt.Line2D([], [], marker="s", color=RANDOM_C, ls="-", ms=4, label="random direction, same norm"),
        plt.Line2D([], [], marker="|", color=SHUFFLE_C, ls="-", ms=6,
                   label="50 label-shuffled directions (range, mean)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.17))
    fig.subplots_adjust(wspace=0.22)
    save(fig, out_dir, "fig_test_effects")


def fig_dose(models, results, out_dir: Path) -> None:
    """Fig. 3: validation dose-response at the layer picked for each model."""
    n_cols = 4 if len(models) > 6 else 3
    fig, axes = plt.subplots(2, n_cols, figsize=(7.0, 3.3), sharex=True, sharey=True)
    for ax, (folder, label, *_) in zip(axes.flat, models):
        chosen = results[folder]["dose"]["chosen"]
        records = sorted((r for r in results[folder]["dose"]["records"] if r["layer"] == chosen["layer"]),
                         key=lambda r: r["coeff"])
        coeffs = [r["coeff"] for r in records]
        ax.grid(color=GRID, lw=0.6)
        for block, colour, marker, name, lw in [("direction", DIRECTION_C, "o", "hedging direction", 1.4),
                                                ("random", RANDOM_C, "s", "random, same norm", 1.2)]:
            rates = [rate(r[block], "hedge") for r in records]
            ps = [p for p, _ in rates]
            err = [[p - ci[0] for p, ci in rates], [ci[1] - p for p, ci in rates]]
            ax.errorbar(coeffs, ps, yerr=err, color=colour, marker=marker, ms=3.5, lw=lw, elinewidth=0.8,
                        capsize=0, mec="white", mew=0.5, label=name)
        not_direct = [1 - rate(r["factual"], "direct")[0] for r in records]
        ax.plot(coeffs, not_direct, color=FACTUAL_C, ls=(0, (2, 1.5)), lw=1.1, marker="x", ms=3.5,
                label="factual prompts not direct")
        ax.axvline(chosen["coeff"], color=INK2, lw=0.7, ls=":")  # dose used for the test split
        ax.set_xscale("log", base=2)
        ax.set_xticks(coeffs)
        ax.set_xticklabels([f"{c:g}" for c in coeffs])
        ax.set_title(f"{label}, layer {chosen['layer']}", loc="left")
        ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel(r"addition coefficient $\alpha$")
    for ax in axes[:, 0]:
        ax.set_ylabel("rate")
    handles, names = axes[0, 0].get_legend_handles_labels()
    spare = list(axes.flat)[len(models):]
    if spare:  # 7 models in a 2x4 grid: the legend goes in the empty panel
        for ax in spare:
            ax.axis("off")
        spare[0].legend(handles, names, loc="center left", bbox_to_anchor=(0.08, 0.5),
                        frameon=False, handlelength=1.6)
    else:
        fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(hspace=0.38, wspace=0.12)
    save(fig, out_dir, "fig_dose")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_dir", nargs="?", type=Path,
                        help="folder with one subfolder per model (default: search below this file)")
    parser.add_argument("out_dir", nargs="?", type=Path, default=here / "figures",
                        help="default: figures/ next to this file")
    args = parser.parse_args()

    results_dir = args.results_dir or find_results_dir(here)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models, results = load_results(results_dir)
    print(f"results: {results_dir}\nfigures: {args.out_dir}")
    print("models: " + ", ".join(m[1] for m in models))

    fig_auroc_depth(models, results, args.out_dir)
    fig_test_effects(models, results, args.out_dir)
    fig_dose(models, results, args.out_dir)
    print("done")


if __name__ == "__main__":
    main()