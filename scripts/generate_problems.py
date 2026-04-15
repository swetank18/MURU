#!/usr/bin/env python3
"""
generate_problems.py — MURU-BENCH Parametric Problem Generator

Generates validated problem variations from parameterized templates.
Each template defines a problem structure with randomizable parameters,
producing unique problems with different numerical values while
maintaining mathematical correctness.

Usage:
    python scripts/generate_problems.py --category bayesian_updating --n 50
    python scripts/generate_problems.py --all --n 100
    python scripts/generate_problems.py --template medical_test --n 20
    python scripts/generate_problems.py --list-templates
    python scripts/generate_problems.py --dry-run --n 5

Templates produce problems with:
  - Randomized numerical parameters within valid ranges
  - Automatically computed ground truth (point estimates + CIs)
  - Contextual variation (different domains/scenarios)
  - Difficulty calibrated by parameter complexity
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "train"
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

def get_next_id() -> int:
    """Find the next available MURU ID number."""
    existing = set()
    for subdir in ["train", "validation", "test"]:
        for f in (PROJECT_ROOT / "data" / subdir).rglob("MURU-*.json"):
            try:
                num = int(f.stem.split("-")[1])
                existing.add(num)
            except (IndexError, ValueError):
                continue
    return max(existing, default=0) + 1


def round_sig(x: float, sig: int = 3) -> float:
    """Round to significant figures."""
    if x == 0:
        return 0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def bayes(prior: float, sensitivity: float, specificity: float) -> float:
    """Compute P(H|+) using Bayes' theorem."""
    fp_rate = 1 - specificity
    p_pos = sensitivity * prior + fp_rate * (1 - prior)
    if p_pos == 0:
        return 0
    return sensitivity * prior / p_pos


def bayes_negative(prior: float, sensitivity: float, specificity: float) -> float:
    """Compute P(H|-) using Bayes' theorem for negative evidence."""
    fn_rate = 1 - sensitivity
    p_neg = fn_rate * prior + specificity * (1 - prior)
    if p_neg == 0:
        return 0
    return fn_rate * prior / p_neg


# ──────────────────────────────────────────────────────────────
# Template Registry
# ──────────────────────────────────────────────────────────────

@dataclass
class ProblemTemplate:
    """A parameterized template for generating problem variations."""
    name: str
    category: str
    difficulty_range: tuple[int, int]
    description: str
    generate: Callable[[int, int], dict]  # (problem_id, difficulty) -> problem dict


TEMPLATES: dict[str, ProblemTemplate] = {}


def register_template(template: ProblemTemplate):
    TEMPLATES[template.name] = template


# ──────────────────────────────────────────────────────────────
# Template: Medical Diagnostic Test (Bayesian Updating)
# ──────────────────────────────────────────────────────────────

MEDICAL_CONTEXTS = [
    {"disease": "a rare genetic condition", "test": "a blood screening test", "domain": "genetics"},
    {"disease": "early-stage lung cancer", "test": "a CT scan screening", "domain": "oncology"},
    {"disease": "Type 2 diabetes", "test": "an HbA1c test", "domain": "endocrinology"},
    {"disease": "celiac disease", "test": "a tTG-IgA antibody test", "domain": "gastroenterology"},
    {"disease": "tuberculosis", "test": "a skin tuberculin test", "domain": "infectious disease"},
    {"disease": "HIV", "test": "a rapid antibody test", "domain": "virology"},
    {"disease": "coronary artery disease", "test": "a stress ECG test", "domain": "cardiology"},
    {"disease": "hepatitis C", "test": "an anti-HCV screening test", "domain": "hepatology"},
    {"disease": "prostate cancer", "test": "a PSA blood test", "domain": "urology"},
    {"disease": "iron deficiency anemia", "test": "a serum ferritin test", "domain": "hematology"},
    {"disease": "hypothyroidism", "test": "a TSH blood test", "domain": "endocrinology"},
    {"disease": "rheumatoid arthritis", "test": "an anti-CCP antibody test", "domain": "rheumatology"},
    {"disease": "deep vein thrombosis", "test": "a D-dimer blood test", "domain": "vascular medicine"},
    {"disease": "meningitis", "test": "a lumbar puncture CSF analysis", "domain": "neurology"},
    {"disease": "Lyme disease", "test": "an ELISA screening test", "domain": "infectious disease"},
]


def generate_medical_test(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(MEDICAL_CONTEXTS)

    if difficulty <= 2:
        sensitivity = round(random.uniform(0.85, 0.99), 2)
        specificity = round(random.uniform(0.80, 0.98), 2)
        prev_low = round(random.uniform(0.005, 0.03), 3)
        prev_high = round(prev_low * random.uniform(1.5, 3.0), 3)
        prev_mid = round((prev_low + prev_high) / 2, 3)
    else:
        sensitivity = round(random.uniform(0.75, 0.95), 2)
        specificity = round(random.uniform(0.70, 0.95), 2)
        prev_low = round(random.uniform(0.01, 0.10), 3)
        prev_high = round(prev_low * random.uniform(1.5, 3.0), 3)
        prev_mid = round((prev_low + prev_high) / 2, 3)

    # Compute ground truth
    ppv_low = round_sig(bayes(prev_low, sensitivity, specificity), 3)
    ppv_mid = round_sig(bayes(prev_mid, sensitivity, specificity), 3)
    ppv_high = round_sig(bayes(prev_high, sensitivity, specificity), 3)

    ci = sorted([ppv_low, ppv_high])
    ci_lower = round_sig(ci[0], 3)
    ci_upper = round_sig(ci[1], 3)

    # Ensure valid CI
    if ci_lower >= ci_upper:
        ci_upper = round(ci_lower + 0.01, 3)

    prev_pct_low = f"{prev_low*100:.1f}%"
    prev_pct_high = f"{prev_high*100:.1f}%"

    stem = (
        f"A hospital uses {ctx['test']} to screen for {ctx['disease']}. "
        f"The test has a sensitivity of {sensitivity*100:.0f}% and a specificity of {specificity*100:.0f}%. "
        f"The prevalence of {ctx['disease']} in the patient population is estimated to be between "
        f"{prev_pct_low} and {prev_pct_high}, with uncertainty due to regional variation and "
        f"demographic differences in the hospital's catchment area. "
        f"A patient receives a positive test result. What is the probability that "
        f"the patient actually has {ctx['disease']}? Provide your answer as a probability "
        f"with a confidence interval reflecting the uncertainty in prevalence."
    )

    answer = (
        f"P({ctx['disease']} | positive test) ranges from {ci_lower} to {ci_upper} "
        f"as prevalence varies from {prev_pct_low} to {prev_pct_high}. "
        f"Point estimate at {prev_mid*100:.1f}% prevalence: {ppv_mid}."
    )

    fp_rate = round(1 - specificity, 2)
    steps = [
        f"Apply Bayes' theorem: P(D|+) = P(+|D)×P(D) / [P(+|D)×P(D) + P(+|¬D)×P(¬D)], where P(+|¬D) = 1-specificity = {fp_rate}.",
        f"At prevalence {prev_mid*100:.1f}%: P(D|+) = {sensitivity}×{prev_mid} / ({sensitivity}×{prev_mid} + {fp_rate}×{round(1-prev_mid,3)}) ≈ {ppv_mid}.",
        f"At prevalence {prev_pct_low}: P(D|+) = {sensitivity}×{prev_low} / ({sensitivity}×{prev_low} + {fp_rate}×{round(1-prev_low,3)}) ≈ {ppv_low}.",
        f"At prevalence {prev_pct_high}: P(D|+) = {sensitivity}×{prev_high} / ({sensitivity}×{prev_high} + {fp_rate}×{round(1-prev_high,3)}) ≈ {ppv_high}.",
        f"The confidence interval [{ci_lower}, {ci_upper}] reflects the range of posterior probabilities over plausible prevalence values.",
    ]

    failure_modes = [
        f"Confusing sensitivity ({sensitivity*100:.0f}%) with positive predictive value",
        "Providing only a point estimate without acknowledging the prevalence uncertainty",
        "Ignoring the base rate (prevalence) and using the test sensitivity as the answer",
    ]

    if difficulty >= 3:
        failure_modes.append("Not recognizing the strong base rate effect when prevalence is low")

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "bayesian_updating",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "parameter_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": ppv_mid,
            "confidence_interval": [ci_lower, ci_upper],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": failure_modes,
        "metadata": {
            "author": "generator_medical_test",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="medical_test",
    category="bayesian_updating",
    difficulty_range=(1, 3),
    description="Medical diagnostic test with uncertain prevalence (Bayes' theorem)",
    generate=generate_medical_test,
))


# ──────────────────────────────────────────────────────────────
# Template: Quality Control / Manufacturing (Bayesian Updating)
# ──────────────────────────────────────────────────────────────

QC_CONTEXTS = [
    {"product": "microchips", "source_a": "Fabrication Line A", "source_b": "Fabrication Line B", "domain": "semiconductor"},
    {"product": "pharmaceutical tablets", "source_a": "Batch Process Alpha", "source_b": "Batch Process Beta", "domain": "pharma"},
    {"product": "automotive parts", "source_a": "Robot Arm Station 1", "source_b": "Robot Arm Station 2", "domain": "automotive"},
    {"product": "solar panels", "source_a": "Automated Line East", "source_b": "Automated Line West", "domain": "energy"},
    {"product": "food packages", "source_a": "Packaging Line Morning Shift", "source_b": "Packaging Line Night Shift", "domain": "food"},
    {"product": "aircraft rivets", "source_a": "Forge A", "source_b": "Forge B", "domain": "aerospace"},
    {"product": "PCB boards", "source_a": "SMT Line 1", "source_b": "SMT Line 2", "domain": "electronics"},
    {"product": "glass bottles", "source_a": "Furnace Alpha", "source_b": "Furnace Beta", "domain": "packaging"},
    {"product": "textile rolls", "source_a": "Loom Section East", "source_b": "Loom Section West", "domain": "textile"},
    {"product": "steel beams", "source_a": "Mill A", "source_b": "Mill B", "domain": "construction"},
]


def generate_quality_control(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(QC_CONTEXTS)

    # Source A: known defect rate, source B: uncertain
    prop_a = round(random.uniform(0.50, 0.80), 2)
    prop_b = round(1 - prop_a, 2)
    defect_a = round(random.uniform(0.01, 0.05), 3)
    defect_b_low = round(random.uniform(0.03, 0.08), 3)
    defect_b_high = round(defect_b_low + random.uniform(0.02, 0.06), 3)
    defect_b_mid = round((defect_b_low + defect_b_high) / 2, 3)

    def posterior_b(defect_b):
        p_def = defect_a * prop_a + defect_b * prop_b
        if p_def == 0:
            return 0
        return defect_b * prop_b / p_def

    post_low = round_sig(posterior_b(defect_b_low), 3)
    post_mid = round_sig(posterior_b(defect_b_mid), 3)
    post_high = round_sig(posterior_b(defect_b_high), 3)

    ci = sorted([post_low, post_high])
    if ci[0] >= ci[1]:
        ci[1] = round(ci[0] + 0.01, 3)

    stem = (
        f"A factory produces {ctx['product']} using two sources. {ctx['source_a']} handles "
        f"{prop_a*100:.0f}% of production with a known defect rate of {defect_a*100:.1f}%. "
        f"{ctx['source_b']} handles {prop_b*100:.0f}% of production, but its defect rate "
        f"is uncertain — recent quality audits suggest it lies between {defect_b_low*100:.1f}% "
        f"and {defect_b_high*100:.1f}%, depending on raw material batch quality. "
        f"A randomly selected {ctx['product'].rstrip('s')} is found to be defective. "
        f"What is the probability it came from {ctx['source_b']}?"
    )

    answer = (
        f"P({ctx['source_b']} | defective) ranges from {ci[0]} to {ci[1]} as "
        f"{ctx['source_b']}'s defect rate varies from {defect_b_low*100:.1f}% to {defect_b_high*100:.1f}%. "
        f"Point estimate at {defect_b_mid*100:.1f}%: {post_mid}."
    )

    steps = [
        f"Apply Bayes' theorem: P(B|def) = P(def|B)×P(B) / [P(def|A)×P(A) + P(def|B)×P(B)].",
        f"At defect rate {defect_b_mid*100:.1f}%: P(B|def) = {defect_b_mid}×{prop_b} / ({defect_a}×{prop_a} + {defect_b_mid}×{prop_b}) ≈ {post_mid}.",
        f"At defect rate {defect_b_low*100:.1f}%: P(B|def) ≈ {post_low}.",
        f"At defect rate {defect_b_high*100:.1f}%: P(B|def) ≈ {post_high}.",
        f"The interval [{ci[0]}, {ci[1]}] captures the range of posterior probabilities.",
    ]

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "bayesian_updating",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "parameter_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": post_mid,
            "confidence_interval": [ci[0], ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": [
            f"Using only the midpoint defect rate and providing a single answer",
            f"Ignoring the production proportions ({prop_a*100:.0f}/{prop_b*100:.0f} split)",
            "Comparing defect rates directly without applying Bayes' theorem",
        ],
        "metadata": {
            "author": "generator_quality_control",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="quality_control",
    category="bayesian_updating",
    difficulty_range=(1, 3),
    description="Quality control with uncertain defect rate from one source (Bayes' theorem)",
    generate=generate_quality_control,
))


# ──────────────────────────────────────────────────────────────
# Template: Sequential Pipeline (Conditional Probability Chains)
# ──────────────────────────────────────────────────────────────

PIPELINE_CONTEXTS = [
    {"domain": "hiring", "stages": ["resume screening", "phone interview", "onsite interview", "offer acceptance"],
     "entity": "candidate", "outcome": "hired"},
    {"domain": "manufacturing", "stages": ["raw material inspection", "machining", "assembly", "final QA"],
     "entity": "product", "outcome": "shipped"},
    {"domain": "loan processing", "stages": ["application review", "credit check", "underwriting", "approval"],
     "entity": "loan application", "outcome": "approved"},
    {"domain": "drug development", "stages": ["preclinical testing", "Phase I trial", "Phase II trial", "FDA review"],
     "entity": "drug candidate", "outcome": "approved for market"},
    {"domain": "space mission", "stages": ["design review", "testing", "launch", "orbital insertion"],
     "entity": "mission", "outcome": "successful"},
    {"domain": "software release", "stages": ["code review", "unit testing", "integration testing", "deployment"],
     "entity": "release", "outcome": "deployed without incidents"},
    {"domain": "grant application", "stages": ["departmental review", "external peer review", "panel discussion", "funding decision"],
     "entity": "proposal", "outcome": "funded"},
    {"domain": "immigration", "stages": ["document verification", "background check", "interview", "visa issuance"],
     "entity": "application", "outcome": "approved"},
]


def generate_sequential_pipeline(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(PIPELINE_CONTEXTS)

    if difficulty <= 2:
        n_stages = 3
    elif difficulty <= 3:
        n_stages = 4
    else:
        n_stages = min(len(ctx["stages"]), 4)

    stages = ctx["stages"][:n_stages]

    # Generate pass rates — one stage has uncertainty
    uncertain_stage = random.randint(0, n_stages - 1)
    rates = []
    for i in range(n_stages):
        if i == uncertain_stage:
            low = round(random.uniform(0.55, 0.80), 2)
            high = round(low + random.uniform(0.08, 0.20), 2)
            high = min(high, 0.99)
            mid = round((low + high) / 2, 2)
            rates.append({"low": low, "mid": mid, "high": high, "uncertain": True})
        else:
            rate = round(random.uniform(0.70, 0.98), 2)
            rates.append({"low": rate, "mid": rate, "high": rate, "uncertain": False})

    # Compute overall probabilities
    def chain_prob(scenario):
        p = 1.0
        for r in rates:
            p *= r[scenario]
        return round_sig(p, 3)

    p_low = chain_prob("low")
    p_mid = chain_prob("mid")
    p_high = chain_prob("high")

    ci = sorted([p_low, p_high])
    if ci[0] >= ci[1]:
        ci[1] = round(ci[0] + 0.01, 3)

    # Build stem
    rate_descriptions = []
    for i, (stage, rate) in enumerate(zip(stages, rates)):
        if i == 0:
            prefix = f"A {ctx['entity']} enters the {ctx['domain']} pipeline. The probability of passing {stage}"
        else:
            prefix = f"If the previous stage is passed, the probability of passing {stage}"

        if rate["uncertain"]:
            rate_descriptions.append(
                f"{prefix} is between {rate['low']*100:.0f}% and {rate['high']*100:.0f}% "
                f"(uncertain due to evaluator variability)."
            )
        else:
            rate_descriptions.append(f"{prefix} is {rate['mid']*100:.0f}%.")

    stem = " ".join(rate_descriptions) + (
        f" What is the overall probability that the {ctx['entity']} is {ctx['outcome']}?"
    )

    uncertain_name = stages[uncertain_stage]
    answer = (
        f"P({ctx['outcome']}) ranges from {ci[0]} to {ci[1]} as the {uncertain_name} pass rate "
        f"varies from {rates[uncertain_stage]['low']*100:.0f}% to {rates[uncertain_stage]['high']*100:.0f}%. "
        f"Point estimate: {p_mid}."
    )

    steps = [
        f"The {ctx['entity']} must pass all {n_stages} stages sequentially: P(success) = {'×'.join(f'P(stage {i+1})' for i in range(n_stages))}.",
    ]
    for scenario, label in [("mid", "mid-range"), ("low", "lower bound"), ("high", "upper bound")]:
        val = chain_prob(scenario)
        calc = " × ".join(f"{r[scenario]}" for r in rates)
        steps.append(f"At {label}: P(success) = {calc} = {val}.")

    steps.append(
        f"The uncertainty in the {uncertain_name} stage creates a "
        f"{abs(ci[1]-ci[0])*100:.1f} percentage point range in the overall success probability."
    )

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "conditional_probability_chains",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "parameter_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": p_mid,
            "confidence_interval": [ci[0], ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": [
            "Adding probabilities instead of multiplying for sequential stages",
            f"Using a single value for the {uncertain_name} rate without acknowledging uncertainty",
            "Not recognizing that uncertainty at one stage propagates through all subsequent stages",
        ],
        "metadata": {
            "author": "generator_pipeline",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="sequential_pipeline",
    category="conditional_probability_chains",
    difficulty_range=(1, 4),
    description="Multi-stage sequential pipeline with one uncertain stage",
    generate=generate_sequential_pipeline,
))


# ──────────────────────────────────────────────────────────────
# Template: Search / Detection (Bayesian Negative Evidence)
# ──────────────────────────────────────────────────────────────

SEARCH_CONTEXTS = [
    {"target": "a missing person", "area_a": "the forest zone", "area_b": "the mountain zone", "domain": "search and rescue"},
    {"target": "a submarine", "area_a": "Sector Alpha", "area_b": "Sector Bravo", "domain": "naval search"},
    {"target": "a lost drone", "area_a": "the urban area", "area_b": "the farmland area", "domain": "asset recovery"},
    {"target": "a buried artifact", "area_a": "Dig Site North", "area_b": "Dig Site South", "domain": "archaeology"},
    {"target": "a gas leak", "area_a": "Building Wing A", "area_b": "Building Wing B", "domain": "safety inspection"},
    {"target": "a network intrusion", "area_a": "the web server cluster", "area_b": "the database cluster", "domain": "cybersecurity"},
    {"target": "a tumor", "area_a": "the left lobe", "area_b": "the right lobe", "domain": "radiology"},
    {"target": "an oil deposit", "area_a": "Block X", "area_b": "Block Y", "domain": "petroleum exploration"},
]


def generate_search_detection(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(SEARCH_CONTEXTS)

    prior_a = round(random.uniform(0.30, 0.70), 2)
    prior_b = round(1 - prior_a, 2)
    det_low = round(random.uniform(0.50, 0.70), 2)
    det_high = round(det_low + random.uniform(0.10, 0.25), 2)
    det_high = min(det_high, 0.95)
    det_mid = round((det_low + det_high) / 2, 2)

    def posterior_b(det):
        """P(target in B | not found in A)"""
        p_not_found = (1 - det) * prior_a + 1.0 * prior_b
        if p_not_found == 0:
            return 0
        return prior_b / p_not_found

    post_low = round_sig(posterior_b(det_low), 3)
    post_mid = round_sig(posterior_b(det_mid), 3)
    post_high = round_sig(posterior_b(det_high), 3)

    ci = sorted([post_low, post_high])
    if ci[0] >= ci[1]:
        ci[1] = round(ci[0] + 0.01, 3)

    stem = (
        f"A {ctx['domain']} team is looking for {ctx['target']}. Based on initial analysis, "
        f"they estimate a {prior_a*100:.0f}% probability that {ctx['target']} is in {ctx['area_a']} "
        f"and a {prior_b*100:.0f}% probability it is in {ctx['area_b']}. They search {ctx['area_a']} "
        f"first and do not find {ctx['target']}. However, their search effectiveness is uncertain "
        f"— conditions vary, and they estimate their detection probability in {ctx['area_a']} is "
        f"between {det_low*100:.0f}% and {det_high*100:.0f}%. Given they did not find {ctx['target']} "
        f"in {ctx['area_a']}, what is the updated probability that {ctx['target']} is in {ctx['area_b']}?"
    )

    answer = (
        f"P({ctx['area_b']} | not found in {ctx['area_a']}) ranges from {ci[0]} to {ci[1]} "
        f"as detection probability varies from {det_low*100:.0f}% to {det_high*100:.0f}%. "
        f"Point estimate at {det_mid*100:.0f}% detection: {post_mid}."
    )

    steps = [
        f"Apply Bayes' theorem for negative evidence: P(B|¬found_A) = P(¬found_A|B)×P(B) / P(¬found_A).",
        f"If target is in B: P(¬found in A | in B) = 1 (cannot find what isn't there).",
        f"If target is in A: P(¬found in A | in A) = 1 - detection_probability.",
        f"P(¬found_A) = (1-det)×{prior_a} + 1×{prior_b}.",
        f"At detection {det_mid*100:.0f}%: P(B|¬found) = {prior_b} / ({round(1-det_mid,2)}×{prior_a} + {prior_b}) ≈ {post_mid}.",
        f"At detection {det_low*100:.0f}%: P(B|¬found) ≈ {post_low}.",
        f"At detection {det_high*100:.0f}%: P(B|¬found) ≈ {post_high}.",
    ]

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "bayesian_updating",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "parameter_uncertainty",
        "required_framework": "bayesian_inference",
        "ground_truth": {
            "answer": answer,
            "point_estimate": post_mid,
            "confidence_interval": [ci[0], ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": [
            f"Assuming not finding in {ctx['area_a']} means it is definitely in {ctx['area_b']}",
            "Ignoring the uncertain detection probability and using a single value",
            "Not applying Bayes' theorem correctly for negative evidence (absence of detection)",
        ],
        "metadata": {
            "author": "generator_search",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="search_detection",
    category="bayesian_updating",
    difficulty_range=(1, 2),
    description="Search and detection with uncertain coverage (negative evidence updating)",
    generate=generate_search_detection,
))


# ──────────────────────────────────────────────────────────────
# Template: Decision with Uncertain Payoffs (Decision Theory)
# ──────────────────────────────────────────────────────────────

DECISION_CONTEXTS = [
    {"scenario": "crop choice", "option_a": "planting wheat", "option_b": "planting corn",
     "factor": "rainfall", "domain": "agriculture"},
    {"scenario": "product launch", "option_a": "the premium version", "option_b": "the budget version",
     "factor": "market demand", "domain": "business"},
    {"scenario": "treatment choice", "option_a": "the established treatment", "option_b": "the experimental treatment",
     "factor": "patient response rate", "domain": "medicine"},
    {"scenario": "route selection", "option_a": "the highway route", "option_b": "the scenic route",
     "factor": "traffic conditions", "domain": "logistics"},
    {"scenario": "energy source", "option_a": "solar panels", "option_b": "wind turbines",
     "factor": "weather patterns", "domain": "energy"},
    {"scenario": "server configuration", "option_a": "horizontal scaling", "option_b": "vertical scaling",
     "factor": "traffic load patterns", "domain": "cloud computing"},
]


def generate_decision_payoff(problem_id: int, difficulty: int) -> dict:
    ctx = random.choice(DECISION_CONTEXTS)

    # Option A: safe, predictable
    payoff_a = round(random.uniform(50, 200), 0)

    # Option B: risky, depends on uncertain factor
    p_good_low = round(random.uniform(0.30, 0.45), 2)
    p_good_high = round(p_good_low + random.uniform(0.15, 0.25), 2)
    p_good_mid = round((p_good_low + p_good_high) / 2, 2)

    payoff_b_good = round(random.uniform(payoff_a * 1.5, payoff_a * 3.0), 0)
    payoff_b_bad = round(random.uniform(payoff_a * 0.1, payoff_a * 0.5), 0)

    # Expected values
    def ev_b(p_good):
        return round(p_good * payoff_b_good + (1 - p_good) * payoff_b_bad, 1)

    ev_b_low = ev_b(p_good_low)
    ev_b_mid = ev_b(p_good_mid)
    ev_b_high = ev_b(p_good_high)

    ci = sorted([ev_b_low, ev_b_high])

    stem = (
        f"A decision-maker must choose between two options. "
        f"Option A ({ctx['option_a']}) yields a guaranteed return of ${payoff_a:.0f}K. "
        f"Option B ({ctx['option_b']}) depends on {ctx['factor']}: with probability "
        f"{p_good_low*100:.0f}-{p_good_high*100:.0f}% (uncertain), it yields ${payoff_b_good:.0f}K; "
        f"otherwise it yields ${payoff_b_bad:.0f}K. "
        f"Compute the expected value of each option and determine which is better. "
        f"How does the uncertainty in {ctx['factor']} affect the recommendation?"
    )

    # Determine breakeven
    # payoff_a = p × payoff_b_good + (1-p) × payoff_b_bad
    p_breakeven = (payoff_a - payoff_b_bad) / (payoff_b_good - payoff_b_bad) if payoff_b_good != payoff_b_bad else 0.5
    p_breakeven = round(p_breakeven, 3)

    if ev_b_mid > payoff_a:
        recommendation = f"Option B is preferred at most assumptions (EV_B > EV_A when P(good) > {p_breakeven*100:.1f}%)."
    else:
        recommendation = f"Option A is preferred at most assumptions (EV_A > EV_B when P(good) < {p_breakeven*100:.1f}%)."

    answer = (
        f"EV(A) = ${payoff_a:.0f}K (fixed). EV(B) ranges from ${ci[0]:.1f}K to ${ci[1]:.1f}K. "
        f"Point estimate: ${ev_b_mid:.1f}K. {recommendation}"
    )

    steps = [
        f"EV(A) = ${payoff_a:.0f}K (guaranteed).",
        f"EV(B) = P(good)×${payoff_b_good:.0f}K + (1-P(good))×${payoff_b_bad:.0f}K.",
        f"At P(good)={p_good_mid}: EV(B) = {p_good_mid}×{payoff_b_good:.0f} + {round(1-p_good_mid,2)}×{payoff_b_bad:.0f} = ${ev_b_mid:.1f}K.",
        f"At P(good)={p_good_low}: EV(B) = ${ev_b_low:.1f}K.",
        f"At P(good)={p_good_high}: EV(B) = ${ev_b_high:.1f}K.",
        f"Breakeven probability: P(good) = ({payoff_a:.0f}-{payoff_b_bad:.0f}) / ({payoff_b_good:.0f}-{payoff_b_bad:.0f}) = {p_breakeven}.",
        f"The decision switches at P(good) = {p_breakeven*100:.1f}%, which {'falls within' if p_good_low <= p_breakeven <= p_good_high else 'falls outside'} the uncertainty range.",
    ]

    return {
        "id": f"MURU-{problem_id:04d}",
        "category": "decision_under_uncertainty",
        "difficulty": difficulty,
        "stem": stem,
        "uncertainty_type": "parameter_uncertainty",
        "required_framework": "decision_theory",
        "ground_truth": {
            "answer": answer,
            "point_estimate": ev_b_mid,
            "confidence_interval": [ci[0], ci[1]],
            "ci_level": 0.90,
        },
        "solution_steps": steps,
        "common_failure_modes": [
            "Picking a single probability and giving a definitive recommendation without sensitivity analysis",
            "Not computing the breakeven probability where the optimal decision switches",
            "Ignoring the range of outcomes and focusing only on expected values (risk-neutral assumption)",
        ],
        "metadata": {
            "author": "generator_decision",
            "reviewed": False,
            "source_inspiration": "original",
        },
    }


register_template(ProblemTemplate(
    name="decision_payoff",
    category="decision_under_uncertainty",
    difficulty_range=(1, 3),
    description="Two-option decision with uncertain outcome probability",
    generate=generate_decision_payoff,
))


# ──────────────────────────────────────────────────────────────
# Main: Generate and Save
# ──────────────────────────────────────────────────────────────

def save_problem(problem: dict, dry_run: bool = False) -> bool:
    """Save a problem to disk and validate it."""
    filepath = DATA_DIR / f"{problem['id']}.json"

    if filepath.exists():
        return False  # skip duplicates

    if dry_run:
        print(f"  [DRY RUN] Would create {problem['id']} ({problem['category']}, D{problem['difficulty']})")
        return True

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(problem, f, indent=2)
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate MURU-BENCH problems from templates.")
    parser.add_argument("--n", type=int, default=10, help="Number of problems to generate.")
    parser.add_argument("--category", "-c", type=str, help="Filter templates by category.")
    parser.add_argument("--template", "-t", type=str, help="Use a specific template.")
    parser.add_argument("--all", action="store_true", help="Generate from all templates equally.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving.")
    parser.add_argument("--list-templates", action="store_true", help="List available templates.")
    parser.add_argument("--validate", action="store_true", help="Run validation after generation.")
    args = parser.parse_args()

    if args.list_templates:
        print(f"\n  Available Templates ({len(TEMPLATES)}):\n")
        for name, tmpl in sorted(TEMPLATES.items()):
            print(f"    {name:<25} {tmpl.category:<35} D{tmpl.difficulty_range[0]}-{tmpl.difficulty_range[1]}  {tmpl.description}")
        return

    if args.seed is not None:
        random.seed(args.seed)

    # Select templates
    if args.template:
        if args.template not in TEMPLATES:
            print(f"Unknown template: {args.template}. Use --list-templates.")
            sys.exit(1)
        templates = [TEMPLATES[args.template]]
    elif args.category:
        templates = [t for t in TEMPLATES.values() if t.category == args.category]
        if not templates:
            print(f"No templates for category: {args.category}")
            sys.exit(1)
    else:
        templates = list(TEMPLATES.values())

    print(f"\n{'═' * 60}")
    print(f"  MURU-BENCH Problem Generator")
    print(f"  Templates: {', '.join(t.name for t in templates)}")
    print(f"  Target: {args.n} problems")
    print(f"{'═' * 60}\n")

    next_id = get_next_id()
    generated = 0
    failures = 0

    for i in range(args.n):
        template = templates[i % len(templates)]
        d_low, d_high = template.difficulty_range
        difficulty = random.randint(d_low, d_high)
        pid = next_id + i

        try:
            problem = template.generate(pid, difficulty)
            if save_problem(problem, dry_run=args.dry_run):
                generated += 1
                if not args.dry_run:
                    print(f"  ✓ {problem['id']}  {template.name:<20} D{difficulty}  {problem['category']}")
            else:
                failures += 1
        except Exception as e:
            failures += 1
            print(f"  ✗ MURU-{pid:04d}  Error: {e}")

    print(f"\n{'─' * 60}")
    print(f"  Generated: {generated}  |  Skipped/Failed: {failures}")

    if args.validate and not args.dry_run and generated > 0:
        print(f"\n  Running validation...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "validate.py")],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)


if __name__ == "__main__":
    main()
