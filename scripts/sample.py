#!/usr/bin/env python3
"""
sample.py — MURU-BENCH Problem Sampler

Randomly sample problems from the dataset for quick inspection.

Usage:
    python scripts/sample.py                          # sample 3 random problems
    python scripts/sample.py --n 5                    # sample 5 problems
    python scripts/sample.py --category bayesian_updating --difficulty 3
    python scripts/sample.py --id MURU-0001           # show a specific problem
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CATEGORY_LABELS = {
    "bayesian_updating": "Bayesian Updating",
    "conditional_probability_chains": "Conditional Prob. Chains",
    "distribution_estimation": "Distribution Estimation",
    "decision_under_uncertainty": "Decision Under Uncertainty",
    "adversarial_ambiguity": "Adversarial Ambiguity",
}


def load_all_problems() -> list[tuple[Path, dict]]:
    """Load all MURU-*.json problems, returning (path, data) tuples."""
    problems = []
    for filepath in sorted(DATA_DIR.rglob("MURU-*.json")):
        rel = filepath.relative_to(DATA_DIR)
        top_dir = rel.parts[0] if rel.parts else ""
        if top_dir in ("by_category", "by_difficulty"):
            continue
        try:
            with open(filepath) as f:
                problems.append((filepath, json.load(f)))
        except (json.JSONDecodeError, IOError):
            continue
    return problems


def pretty_print(problem: dict, filepath: Path | None = None):
    """Pretty-print a single problem."""
    gt = problem["ground_truth"]
    ci = gt["confidence_interval"]
    category = CATEGORY_LABELS.get(problem["category"], problem["category"])

    print(f"\n{'═' * 70}")
    print(f"  {problem['id']}  |  {category}  |  Difficulty: {'★' * problem['difficulty']}{'☆' * (5 - problem['difficulty'])}")
    if filepath:
        print(f"  File: {filepath}")
    print(f"{'═' * 70}")

    print(f"\n  PROBLEM STEM")
    print(f"  {'─' * 50}")
    # Word-wrap the stem at ~70 chars
    words = problem["stem"].split()
    line = "  "
    for word in words:
        if len(line) + len(word) + 1 > 70:
            print(line)
            line = "  " + word
        else:
            line += " " + word if line.strip() else "  " + word
    if line.strip():
        print(line)

    print(f"\n  UNCERTAINTY TYPE: {problem['uncertainty_type'].replace('_', ' ').title()}")
    print(f"  FRAMEWORK: {problem['required_framework'].replace('_', ' ').title()}")

    print(f"\n  GROUND TRUTH")
    print(f"  {'─' * 50}")
    print(f"  Answer:     {gt['answer']}")
    print(f"  Point est:  {gt['point_estimate']}")
    print(f"  CI [{gt['ci_level']:.0%}]:   [{ci[0]}, {ci[1]}]")
    print(f"  CI width:   {ci[1] - ci[0]:.4f}")

    print(f"\n  SOLUTION STEPS")
    print(f"  {'─' * 50}")
    for i, step in enumerate(problem["solution_steps"], 1):
        print(f"  {i}. {step}")

    print(f"\n  COMMON FAILURE MODES")
    print(f"  {'─' * 50}")
    for fm in problem["common_failure_modes"]:
        print(f"  • {fm}")

    meta = problem["metadata"]
    print(f"\n  METADATA")
    print(f"  {'─' * 50}")
    print(f"  Author: {meta['author']}  |  Reviewed: {'Yes' if meta['reviewed'] else 'No'}  |  Source: {meta['source_inspiration']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Sample MURU-BENCH problems.")
    parser.add_argument("--n", type=int, default=3, help="Number of problems to sample.")
    parser.add_argument("--category", "-c", type=str, help="Filter by category.")
    parser.add_argument("--difficulty", "-d", type=int, help="Filter by difficulty level (1-5).")
    parser.add_argument("--id", type=str, help="Show a specific problem by ID.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    all_problems = load_all_problems()

    if not all_problems:
        print("No MURU-*.json problems found in data/")
        sys.exit(1)

    # Filter
    filtered = all_problems
    if args.id:
        filtered = [(p, d) for p, d in filtered if d["id"] == args.id]
        if not filtered:
            print(f"Problem {args.id} not found.")
            sys.exit(1)

    if args.category:
        filtered = [(p, d) for p, d in filtered if d["category"] == args.category]
        if not filtered:
            print(f"No problems found with category '{args.category}'.")
            sys.exit(1)

    if args.difficulty:
        filtered = [(p, d) for p, d in filtered if d["difficulty"] == args.difficulty]
        if not filtered:
            print(f"No problems found with difficulty {args.difficulty}.")
            sys.exit(1)

    # Sample
    n = min(args.n, len(filtered))
    sampled = random.sample(filtered, n)

    print(f"\n  Showing {n} of {len(filtered)} matching problems")

    for filepath, problem in sampled:
        pretty_print(problem, filepath)


if __name__ == "__main__":
    main()
