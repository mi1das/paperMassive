"""CLI: aggregate run JSONs into the report's tables and figure.

Reads ``results/*.json`` (and ``results/extension/*.json``), aggregates over
seeds (mean +/- std), and writes:
  - a CSV summary,
  - LaTeX tabular bodies matching tab:results / tab:extension,
  - a validation-accuracy-vs-data-size figure for the extension.

This is pure post-processing: no model code, so it runs without a GPU.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
EXTENSION_DIR = RESULTS_DIR / "extension"
FIGURES_DIR = REPO_ROOT / "results" / "figures"


def _load(directory: Path) -> list[dict[str, Any]]:
    import json

    out = []
    for path in sorted(directory.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            out.append(json.load(handle))
    return out


def _aggregate_by_name(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group runs by config name and average each metric over seeds."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        grouped[r["config"]["name"]].append(r)

    summary: dict[str, dict[str, Any]] = {}
    for name, runs in grouped.items():
        acc = [r["test"]["accuracy"] for r in runs]
        f1 = [r["test"]["macro_f1"] for r in runs]
        summary[name] = {
            "n_seeds": len(runs),
            "accuracy_mean": mean(acc),
            "accuracy_std": pstdev(acc) if len(acc) > 1 else 0.0,
            "macro_f1_mean": mean(f1),
            "macro_f1_std": pstdev(f1) if len(f1) > 1 else 0.0,
            "train_size": runs[0]["train_size"],
        }
    return summary


def _fmt(value: float, std: float, n: int) -> str:
    return f"{value*100:.2f}" if n <= 1 else f"{value*100:.2f} $\\pm$ {std*100:.2f}"


def _write_latex_table(summary: dict[str, dict[str, Any]], path: Path) -> None:
    lines = []
    for name, s in summary.items():
        acc = _fmt(s["accuracy_mean"], s["accuracy_std"], s["n_seeds"])
        f1 = _fmt(s["macro_f1_mean"], s["macro_f1_std"], s["n_seeds"])
        lines.append(f"    {name} & {acc} & {f1} \\\\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(summary: dict[str, dict[str, Any]], path: Path) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["name", "n_seeds", "train_size", "accuracy_mean", "accuracy_std",
             "macro_f1_mean", "macro_f1_std"]
        )
        for name, s in summary.items():
            writer.writerow(
                [name, s["n_seeds"], s["train_size"],
                 f"{s['accuracy_mean']:.4f}", f"{s['accuracy_std']:.4f}",
                 f"{s['macro_f1_mean']:.4f}", f"{s['macro_f1_std']:.4f}"]
            )


def _plot_scaling(summary: dict[str, dict[str, Any]], path: Path) -> None:
    scaling = {k: v for k, v in summary.items() if k.startswith("scale_")}
    if not scaling:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(scaling.items(), key=lambda kv: kv[1]["train_size"])
    sizes = [v["train_size"] for _, v in items]
    acc = [v["accuracy_mean"] * 100 for _, v in items]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(sizes, acc, marker="o")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Low-resource data scaling")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate results into report assets.")
    parser.parse_args()

    main_results = _load(RESULTS_DIR)
    ext_results = _load(EXTENSION_DIR)

    if main_results:
        summary = _aggregate_by_name(main_results)
        _write_csv(summary, RESULTS_DIR / "summary_main.csv")
        _write_latex_table(summary, RESULTS_DIR / "table_main.tex")
        print(f"Main summary ({len(summary)} configs) -> results/summary_main.csv")

    if ext_results:
        ext_summary = _aggregate_by_name(ext_results)
        _write_csv(ext_summary, RESULTS_DIR / "summary_extension.csv")
        _write_latex_table(ext_summary, RESULTS_DIR / "table_extension.tex")
        _plot_scaling(ext_summary, FIGURES_DIR / "hp_scaling.pdf")
        print(f"Extension summary ({len(ext_summary)} configs) -> results/summary_extension.csv")


if __name__ == "__main__":
    main()
