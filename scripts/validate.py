#!/usr/bin/env python3
"""
validate.py — MURU-BENCH Problem Validator

Validates all problem JSON files against the MURU-BENCH schema.
Checks structural compliance, semantic constraints, and unique IDs.

Usage:
    python scripts/validate.py                      # validate all problems in data/
    python scripts/validate.py data/train/           # validate a specific directory
    python scripts/validate.py data/train/MURU-0001.json  # validate a single file

Exit codes:
    0 — all problems valid
    1 — one or more problems invalid
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from jsonschema import validate, ValidationError, Draft7Validator
except ImportError:
    print("ERROR: jsonschema is required. Install with: pip install jsonschema>=4.20.0")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = PROJECT_ROOT / "problem_schema.json"
DATA_DIR = PROJECT_ROOT / "data"

VALID_CATEGORIES = {
    "bayesian_updating",
    "conditional_probability_chains",
    "distribution_estimation",
    "decision_under_uncertainty",
    "adversarial_ambiguity",
}


# ──────────────────────────────────────────────────────────────────────
# Semantic checks (beyond JSON Schema)
# ──────────────────────────────────────────────────────────────────────

def semantic_checks(problem: dict, filepath: str) -> list[str]:
    """Run semantic validation rules that JSON Schema cannot enforce."""
    errors = []

    # CI ordering: lower bound < upper bound
    ci = problem["ground_truth"]["confidence_interval"]
    if ci[0] >= ci[1]:
        errors.append(
            f"confidence_interval lower bound ({ci[0]}) must be less than upper bound ({ci[1]})"
        )

    # Non-trivial CI: not [0, 0] or [1, 1]
    if ci[0] == ci[1]:
        errors.append(
            f"confidence_interval is trivial (both bounds are {ci[0]}). "
            "MURU-BENCH requires non-trivial uncertainty."
        )

    # Point estimate should fall within CI
    pe = problem["ground_truth"]["point_estimate"]
    if not (ci[0] <= pe <= ci[1]):
        errors.append(
            f"point_estimate ({pe}) falls outside confidence_interval [{ci[0]}, {ci[1]}]"
        )

    # Point estimate should not be exactly 0 or 1 for probability problems
    if pe == 0.0 or pe == 1.0:
        errors.append(
            f"point_estimate is {pe}, which suggests a deterministic answer. "
            "MURU-BENCH problems should involve genuine uncertainty."
        )

    # Solution steps should have at least 2 steps
    if len(problem["solution_steps"]) < 2:
        errors.append(
            f"Expected at least 2 solution_steps, got {len(problem['solution_steps'])}"
        )

    # ID should match filename (if we have a filepath)
    if filepath:
        expected_id = Path(filepath).stem
        if problem["id"] != expected_id:
            errors.append(
                f"Problem id '{problem['id']}' does not match filename '{expected_id}'"
            )

    return errors


# ──────────────────────────────────────────────────────────────────────
# Validation engine
# ──────────────────────────────────────────────────────────────────────

def load_schema() -> dict:
    """Load the MURU-BENCH JSON schema."""
    if not SCHEMA_PATH.exists():
        print(f"ERROR: Schema file not found at {SCHEMA_PATH}")
        sys.exit(1)

    with open(SCHEMA_PATH) as f:
        return json.load(f)


def find_problem_files(target: str) -> list[Path]:
    """Find all .json problem files under the given path."""
    target_path = Path(target)

    if target_path.is_file():
        return [target_path]

    if target_path.is_dir():
        return sorted(target_path.rglob("MURU-*.json"))

    print(f"ERROR: Path not found: {target}")
    sys.exit(1)


def validate_file(filepath: Path, schema: dict, seen_ids: set) -> list[str]:
    """Validate a single problem file. Returns a list of error messages."""
    errors = []

    # Parse JSON
    try:
        with open(filepath) as f:
            problem = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    # Schema validation
    try:
        validate(instance=problem, schema=schema)
    except ValidationError as e:
        errors.append(f"Schema violation: {e.message} (at {'.'.join(str(p) for p in e.path)})" if e.path else f"Schema violation: {e.message}")

    # If schema validation passed, run semantic checks
    if not errors:
        errors.extend(semantic_checks(problem, str(filepath)))

    # Unique ID check
    pid = problem.get("id", "UNKNOWN")
    if pid in seen_ids:
        errors.append(f"Duplicate ID: {pid}")
    seen_ids.add(pid)

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate MURU-BENCH problem files against the schema."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=str(DATA_DIR),
        help="Path to a file or directory to validate (default: data/)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print details for each file, including passing ones.",
    )
    args = parser.parse_args()

    schema = load_schema()
    files = find_problem_files(args.target)

    if not files:
        print(f"No MURU-*.json files found under {args.target}")
        sys.exit(1)

    print(f"Validating {len(files)} problem file(s)...\n")

    seen_ids: set[str] = set()
    total_errors = 0
    failed_files = 0

    for filepath in files:
        errors = validate_file(filepath, schema, seen_ids)

        if errors:
            failed_files += 1
            total_errors += len(errors)
            print(f"  FAIL  {filepath.relative_to(PROJECT_ROOT)}")
            for err in errors:
                print(f"        ↳ {err}")
        elif args.verbose:
            print(f"  PASS  {filepath.relative_to(PROJECT_ROOT)}")

    # Summary
    print(f"\n{'─' * 60}")
    passed = len(files) - failed_files
    print(f"  Results: {passed}/{len(files)} passed, {failed_files} failed ({total_errors} errors)")

    if failed_files > 0:
        print(f"\n  ✗ Validation FAILED")
        sys.exit(1)
    else:
        print(f"\n  ✓ All {len(files)} problems are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
