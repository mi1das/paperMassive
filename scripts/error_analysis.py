"""CLI: qualitative error analysis for a single run (paper Sec. 7.4).

Consumes the saved test predictions/labels of a run (no model reload needed) and
reports the most frequent intent confusions, which surface semantically close
intents and German-specific phenomena.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Top intent confusions for a run.")
    parser.add_argument(
        "--result",
        default=str(RESULTS_DIR / "baseline_seed42.json"),
        help="Path to a saved run result JSON.",
    )
    parser.add_argument("--top", type=int, default=15, help="How many confusions to show.")
    args = parser.parse_args()

    with open(args.result, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    labels = data["test_labels"]
    preds = data["test_predictions"]
    names = data["label_names"]

    confusions = Counter(
        (names[t], names[p]) for t, p in zip(labels, preds) if t != p
    )

    total_errors = sum(confusions.values())
    print(f"Run: {data['config']['name']}  test accuracy={data['test']['accuracy']:.4f}")
    print(f"Total errors: {total_errors}\n")
    print(f"{'true -> predicted':<55} count")
    for (true_name, pred_name), count in confusions.most_common(args.top):
        print(f"{true_name + ' -> ' + pred_name:<55} {count}")


if __name__ == "__main__":
    main()
