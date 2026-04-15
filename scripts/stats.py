#!/usr/bin/env python3
"""
stats.py — MURU-BENCH Dataset Statistics

Scans all problem files and prints summary statistics.
Optionally generates distribution plots saved to paper/figures/.

Usage:
    python scripts/stats.py                 # print stats
    python scripts/stats.py --plots         # print stats + generate figures
    python scripts/stats.py --json          # output stats as JSON
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

CATEGORIES = [
    "bayesian_updating",
    "conditional_probability_chains",
    "distribution_estimation",
    "decision_under_uncertainty",
    "adversarial_ambiguity",
]

CATEGORY_LABELS = {
    "bayesian_updating": "Bayesian Updating",
    "conditional_probability_chains": "Conditional Prob. Chains",
    "distribution_estimation": "Distribution Estimation",
    "decision_under_uncertainty": "Decision Under Uncertainty",
    "adversarial_ambiguity": "Adversarial Ambiguity",
}

DIFFICULTY_RANGE = range(1, 6)


def load_all_problems() -> list[dict]:
    """Load all MURU-*.json problems from the data directory."""
    problems = []
    for filepath in sorted(DATA_DIR.rglob("MURU-*.json")):
        # Avoid double-counting if problems appear in by_category/ or by_difficulty/
        # Only count from train/, validation/, test/ directories
        rel = filepath.relative_to(DATA_DIR)
        top_dir = rel.parts[0] if rel.parts else ""
        if top_dir in ("by_category", "by_difficulty"):
            continue

        try:
            with open(filepath) as f:
                problems.append(json.load(f))
        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARNING: Could not load {filepath}: {e}", file=sys.stderr)
    return problems


def compute_stats(problems: list[dict]) -> dict:
    """Compute comprehensive statistics about the problem set."""
    if not problems:
        return {"total": 0}

    cat_counts = Counter(p["category"] for p in problems)
    diff_counts = Counter(p["difficulty"] for p in problems)
    reviewed_count = sum(1 for p in problems if p.get("metadata", {}).get("reviewed", False))

    # Category × Difficulty matrix
    cat_diff = defaultdict(lambda: Counter())
    for p in problems:
        cat_diff[p["category"]][p["difficulty"]] += 1

    # Confidence interval statistics
    ci_widths = []
    for p in problems:
        ci = p.get("ground_truth", {}).get("confidence_interval", [0, 0])
        if len(ci) == 2:
            ci_widths.append(ci[1] - ci[0])

    avg_ci_width = sum(ci_widths) / len(ci_widths) if ci_widths else 0
    min_ci_width = min(ci_widths) if ci_widths else 0
    max_ci_width = max(ci_widths) if ci_widths else 0

    # Uncertainty type distribution
    unc_counts = Counter(p.get("uncertainty_type", "unknown") for p in problems)

    # Framework distribution
    fw_counts = Counter(p.get("required_framework", "unknown") for p in problems)

    return {
        "total": len(problems),
        "category_counts": dict(cat_counts),
        "difficulty_counts": {str(k): v for k, v in sorted(diff_counts.items())},
        "category_difficulty_matrix": {
            cat: {str(d): cat_diff[cat][d] for d in DIFFICULTY_RANGE}
            for cat in CATEGORIES
        },
        "reviewed": reviewed_count,
        "review_rate": reviewed_count / len(problems) if problems else 0,
        "ci_stats": {
            "mean_width": round(avg_ci_width, 4),
            "min_width": round(min_ci_width, 4),
            "max_width": round(max_ci_width, 4),
        },
        "uncertainty_types": dict(unc_counts),
        "frameworks": dict(fw_counts),
    }


def print_stats(stats: dict):
    """Pretty-print the statistics."""
    if stats["total"] == 0:
        print("No problems found in the dataset.")
        return

    print(f"\n{'═' * 60}")
    print(f"  MURU-BENCH Dataset Statistics")
    print(f"{'═' * 60}")
    print(f"\n  Total problems: {stats['total']}")
    print(f"  Reviewed:       {stats['reviewed']} ({stats['review_rate']:.0%})")

    # Category breakdown
    print(f"\n{'─' * 60}")
    print(f"  Category Distribution")
    print(f"{'─' * 60}")
    cat_data = []
    for cat in CATEGORIES:
        count = stats["category_counts"].get(cat, 0)
        pct = count / stats["total"] * 100
        bar = "█" * int(pct / 2)
        cat_data.append([CATEGORY_LABELS.get(cat, cat), count, f"{pct:.1f}%", bar])

    if tabulate:
        print(tabulate(cat_data, headers=["Category", "Count", "%", ""], tablefmt="simple"))
    else:
        for row in cat_data:
            print(f"    {row[0]:<30} {row[1]:>5}  {row[2]:>6}  {row[3]}")

    # Difficulty breakdown
    print(f"\n{'─' * 60}")
    print(f"  Difficulty Distribution")
    print(f"{'─' * 60}")
    diff_data = []
    for d in DIFFICULTY_RANGE:
        count = stats["difficulty_counts"].get(str(d), 0)
        pct = count / stats["total"] * 100
        bar = "█" * int(pct / 2)
        diff_data.append([f"Level {d}", count, f"{pct:.1f}%", bar])

    if tabulate:
        print(tabulate(diff_data, headers=["Difficulty", "Count", "%", ""], tablefmt="simple"))
    else:
        for row in diff_data:
            print(f"    {row[0]:<30} {row[1]:>5}  {row[2]:>6}  {row[3]}")

    # Category × Difficulty matrix
    print(f"\n{'─' * 60}")
    print(f"  Category × Difficulty Matrix")
    print(f"{'─' * 60}")
    matrix_data = []
    for cat in CATEGORIES:
        row = [CATEGORY_LABELS.get(cat, cat)[:25]]
        for d in DIFFICULTY_RANGE:
            row.append(stats["category_difficulty_matrix"].get(cat, {}).get(str(d), 0))
        matrix_data.append(row)

    headers = ["Category"] + [f"D{d}" for d in DIFFICULTY_RANGE]
    if tabulate:
        print(tabulate(matrix_data, headers=headers, tablefmt="simple"))
    else:
        print(f"    {'Category':<25} " + " ".join(f"D{d:>3}" for d in DIFFICULTY_RANGE))
        for row in matrix_data:
            print(f"    {row[0]:<25} " + " ".join(f"{v:>4}" for v in row[1:]))

    # CI statistics
    print(f"\n{'─' * 60}")
    print(f"  Confidence Interval Statistics")
    print(f"{'─' * 60}")
    ci = stats["ci_stats"]
    print(f"    Mean CI width:  {ci['mean_width']:.4f}")
    print(f"    Min CI width:   {ci['min_width']:.4f}")
    print(f"    Max CI width:   {ci['max_width']:.4f}")
    print()


def generate_plots(stats: dict):
    """Generate distribution plots and save to paper/figures/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("WARNING: matplotlib/numpy not installed. Skipping plots.", file=sys.stderr)
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    # 1. Category distribution (horizontal bar)
    fig, ax = plt.subplots(figsize=(10, 5))
    cats = [CATEGORY_LABELS.get(c, c) for c in CATEGORIES]
    counts = [stats["category_counts"].get(c, 0) for c in CATEGORIES]
    bars = ax.barh(cats, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of Problems")
    ax.set_title("MURU-BENCH: Problems by Category", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "category_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: paper/figures/category_distribution.png")

    # 2. Difficulty distribution (bar chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    diffs = list(DIFFICULTY_RANGE)
    diff_counts = [stats["difficulty_counts"].get(str(d), 0) for d in diffs]
    ax.bar(diffs, diff_counts, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Number of Problems")
    ax.set_title("MURU-BENCH: Problems by Difficulty", fontsize=14, fontweight="bold")
    ax.set_xticks(diffs)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "difficulty_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: paper/figures/difficulty_distribution.png")

    # 3. Category × Difficulty heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    matrix = np.zeros((len(CATEGORIES), len(diffs)))
    for i, cat in enumerate(CATEGORIES):
        for j, d in enumerate(diffs):
            matrix[i, j] = stats["category_difficulty_matrix"].get(cat, {}).get(str(d), 0)

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(diffs)))
    ax.set_xticklabels([f"D{d}" for d in diffs])
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_yticklabels([CATEGORY_LABELS.get(c, c) for c in CATEGORIES])
    ax.set_title("MURU-BENCH: Category × Difficulty", fontsize=14, fontweight="bold")

    # Add text annotations
    for i in range(len(CATEGORIES)):
        for j in range(len(diffs)):
            val = int(matrix[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        color="white" if val > matrix.max() * 0.6 else "black", fontsize=10)

    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "category_difficulty_heatmap.png", dpi=150)
    plt.close()
    print(f"  Saved: paper/figures/category_difficulty_heatmap.png")


def main():
    parser = argparse.ArgumentParser(description="MURU-BENCH dataset statistics.")
    parser.add_argument("--plots", action="store_true", help="Generate distribution plots.")
    parser.add_argument("--json", action="store_true", help="Output stats as JSON.")
    args = parser.parse_args()

    problems = load_all_problems()
    stats = compute_stats(problems)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_stats(stats)

    if args.plots:
        generate_plots(stats)


if __name__ == "__main__":
    main()
