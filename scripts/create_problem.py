#!/usr/bin/env python3
"""
create_problem.py — MURU-BENCH Problem Creator

Interactive tool that walks you through the 6-step problem writing
workflow from the project plan, generates a valid JSON file, and
validates it automatically.

Usage:
    python scripts/create_problem.py                          # interactive mode
    python scripts/create_problem.py --batch template.csv     # batch from CSV
    python scripts/create_problem.py --next-id                # show next available ID
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "train"

CATEGORIES = {
    "1": "bayesian_updating",
    "2": "conditional_probability_chains",
    "3": "distribution_estimation",
    "4": "decision_under_uncertainty",
    "5": "adversarial_ambiguity",
}

CATEGORY_LABELS = {
    "bayesian_updating": "Bayesian Updating",
    "conditional_probability_chains": "Conditional Probability Chains",
    "distribution_estimation": "Distribution Estimation",
    "decision_under_uncertainty": "Decision Under Uncertainty",
    "adversarial_ambiguity": "Adversarial Ambiguity",
}

UNCERTAINTY_TYPES = {
    "1": "parameter_uncertainty",
    "2": "model_uncertainty",
    "3": "data_uncertainty",
    "4": "structural_uncertainty",
    "5": "epistemic_uncertainty",
}

FRAMEWORKS = {
    "1": "bayesian_inference",
    "2": "frequentist_inference",
    "3": "decision_theory",
    "4": "information_theory",
    "5": "monte_carlo",
}

SOURCES = {
    "1": "original",
    "2": "textbook",
    "3": "paper",
    "4": "real_world",
}


def get_next_id() -> str:
    """Find the next available MURU-XXXX ID."""
    existing = set()
    for f in DATA_DIR.rglob("MURU-*.json"):
        stem = f.stem
        try:
            num = int(stem.split("-")[1])
            existing.add(num)
        except (IndexError, ValueError):
            continue

    # Also check validation and test dirs
    for subdir in ["validation", "test"]:
        for f in (PROJECT_ROOT / "data" / subdir).rglob("MURU-*.json"):
            try:
                num = int(f.stem.split("-")[1])
                existing.add(num)
            except (IndexError, ValueError):
                continue

    next_num = max(existing, default=0) + 1
    return f"MURU-{next_num:04d}"


def prompt_choice(prompt: str, options: dict) -> str:
    """Display options and get user choice."""
    print(f"\n  {prompt}")
    for key, val in options.items():
        label = CATEGORY_LABELS.get(val, val.replace("_", " ").title())
        print(f"    {key}. {label}")
    while True:
        choice = input("  → ").strip()
        if choice in options:
            return options[choice]
        print(f"    Invalid choice. Enter {'/'.join(options.keys())}.")


def prompt_text(prompt: str, min_length: int = 0) -> str:
    """Get text input with optional minimum length."""
    while True:
        print(f"\n  {prompt}")
        text = input("  → ").strip()
        if len(text) >= min_length:
            return text
        print(f"    Minimum {min_length} characters required (got {len(text)}).")


def prompt_number(prompt: str, min_val: float = None, max_val: float = None) -> float:
    """Get numeric input with optional bounds."""
    while True:
        print(f"\n  {prompt}")
        try:
            val = float(input("  → ").strip())
            if min_val is not None and val < min_val:
                print(f"    Must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"    Must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("    Please enter a valid number.")


def prompt_list(prompt: str, min_items: int = 1) -> list[str]:
    """Get a list of text items."""
    print(f"\n  {prompt} (enter each on a separate line, empty line to finish)")
    items = []
    while True:
        item = input(f"    {len(items) + 1}. ").strip()
        if not item:
            if len(items) >= min_items:
                break
            print(f"    At least {min_items} item(s) required.")
        else:
            items.append(item)
    return items


def interactive_create():
    """Walk user through the 6-step problem creation workflow."""
    problem_id = get_next_id()

    print(f"\n{'═' * 60}")
    print(f"  MURU-BENCH Problem Creator")
    print(f"  Creating: {problem_id}")
    print(f"{'═' * 60}")

    # Step 1: Scenario (Category)
    print(f"\n{'─' * 60}")
    print(f"  STEP 1: Choose a Category")
    print(f"  (Pick a real-world scenario where uncertainty is inherent)")
    category = prompt_choice("Category:", CATEGORIES)

    # Difficulty
    difficulty = int(prompt_number("  Difficulty level (1-5):", 1, 5))

    # Step 2: Problem stem
    print(f"\n{'─' * 60}")
    print(f"  STEP 2: Write the Problem Stem")
    print(f"  (Embed specific mathematical structure — Bayes theorem,")
    print(f"   conditional probability, likelihood ratios, etc.)")
    stem = prompt_text("Problem stem (min 50 chars):", 50)

    # Step 3: Uncertainty specification
    print(f"\n{'─' * 60}")
    print(f"  STEP 3: Specify the Uncertainty")
    print(f"  (A parameter range, missing prior, biased sample, etc.)")
    uncertainty_type = prompt_choice("Uncertainty type:", UNCERTAINTY_TYPES)
    framework = prompt_choice("Required framework:", FRAMEWORKS)

    # Step 4: Ground truth
    print(f"\n{'─' * 60}")
    print(f"  STEP 4: Derive the Ground Truth")
    print(f"  (Show all steps; if you can't derive it, the problem isn't ready)")
    answer = prompt_text("Human-readable answer (min 10 chars):", 10)
    point_estimate = prompt_number("Point estimate:")
    ci_lower = prompt_number("Confidence interval lower bound:")
    ci_upper = prompt_number("Confidence interval upper bound:")
    ci_level = prompt_number("CI level (e.g., 0.90 for 90%):", 0.5, 0.99)

    solution_steps = prompt_list("Solution steps (min 2):", 2)

    # Step 5: Failure modes
    print(f"\n{'─' * 60}")
    print(f"  STEP 5: Identify Common Failure Modes")
    print(f"  (What will an overconfident model do wrong?)")
    failure_modes = prompt_list("Failure modes (min 1):", 1)

    # Metadata
    print(f"\n{'─' * 60}")
    print(f"  STEP 6: Metadata")
    author = prompt_text("Author ID:", 1)
    source = prompt_choice("Source inspiration:", SOURCES)

    # Build problem
    problem = {
        "id": problem_id,
        "category": category,
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": uncertainty_type,
        "required_framework": framework,
        "ground_truth": {
            "answer": answer,
            "point_estimate": point_estimate,
            "confidence_interval": [ci_lower, ci_upper],
            "ci_level": ci_level,
        },
        "solution_steps": solution_steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": author,
            "reviewed": False,
            "source_inspiration": source,
        },
    }

    # Save
    filepath = DATA_DIR / f"{problem_id}.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(problem, f, indent=2)

    print(f"\n{'═' * 60}")
    print(f"  ✓ Problem saved to: {filepath.relative_to(PROJECT_ROOT)}")

    # Validate
    print(f"\n  Running validation...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "validate.py"), str(filepath)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print("  ⚠ Validation failed. Please fix the issue and run validate.py again.")
    else:
        print(f"  ✓ Problem is valid and ready!")

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Create MURU-BENCH problems interactively.")
    parser.add_argument("--next-id", action="store_true", help="Print the next available ID and exit.")
    args = parser.parse_args()

    if args.next_id:
        print(get_next_id())
        return

    try:
        interactive_create()
    except KeyboardInterrupt:
        print("\n\n  Cancelled.")
        sys.exit(0)


if __name__ == "__main__":
    main()
