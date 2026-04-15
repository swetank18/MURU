#!/usr/bin/env python3
"""
run_eval.py — MURU-BENCH Model Evaluator

Runs language models against MURU-BENCH problems via their APIs
and scores them using the metrics module.

Usage:
    python evaluation/run_eval.py --model gpt-4o --subset data/test/
    python evaluation/run_eval.py --model claude-3.5-sonnet --n 50
    python evaluation/run_eval.py --model gemini-1.5-pro --subset data/test/ --save

API keys should be set as environment variables:
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY

Supported models:
    OpenAI:     gpt-4o, gpt-4-turbo, gpt-3.5-turbo
    Anthropic:  claude-3.5-sonnet, claude-3-opus, claude-3-haiku
    Google:     gemini-1.5-pro, gemini-1.5-flash
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import MURUMetrics, Prediction


# ──────────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a mathematical reasoning expert. You will be given a problem involving mathematical uncertainty. You must:

1. Identify the correct mathematical framework (bayesian_inference, frequentist_inference, decision_theory, information_theory, or monte_carlo).
2. Show your reasoning step-by-step.
3. Provide your final answer as a single number (point estimate).
4. Provide a confidence interval [lower, upper] for your answer.
5. State your confidence in your answer as a probability between 0 and 1.

Format your response EXACTLY as follows at the end:

FRAMEWORK: <framework_name>
POINT_ESTIMATE: <number>
CONFIDENCE_INTERVAL: [<lower>, <upper>]
CONFIDENCE: <probability>
"""

USER_PROMPT_TEMPLATE = """Problem:
{stem}

Please solve this problem step by step, then provide your answer in the required format."""


# ──────────────────────────────────────────────────────────────────────
# API Clients
# ──────────────────────────────────────────────────────────────────────

class ModelClient:
    """Base class for model API clients."""

    def query(self, prompt: str, system: str = "") -> str:
        raise NotImplementedError


class OpenAIClient(ModelClient):
    def __init__(self, model: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai>=1.12.0")
        self.client = OpenAI()
        self.model = model

    def query(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""


class AnthropicClient(ModelClient):
    def __init__(self, model: str):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic>=0.18.0")
        self.client = anthropic.Anthropic()
        self.model = model

    def query(self, prompt: str, system: str = "") -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""


class GoogleClient(ModelClient):
    def __init__(self, model: str):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai>=0.4.0")
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model)

    def query(self, prompt: str, system: str = "") -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = self.model.generate_content(full_prompt)
        return response.text or ""


def get_client(model_name: str) -> ModelClient:
    """Create the appropriate API client for the given model name."""
    model_lower = model_name.lower()

    if any(x in model_lower for x in ["gpt", "o1", "o3"]):
        return OpenAIClient(model_name)
    elif any(x in model_lower for x in ["claude"]):
        return AnthropicClient(model_name)
    elif any(x in model_lower for x in ["gemini"]):
        return GoogleClient(model_name)
    else:
        print(f"ERROR: Unknown model provider for '{model_name}'.")
        print("Supported prefixes: gpt/o1/o3 (OpenAI), claude (Anthropic), gemini (Google)")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# Response parser
# ──────────────────────────────────────────────────────────────────────

def parse_response(response: str) -> dict:
    """Extract structured answer from model response."""
    result = {
        "framework": None,
        "point_estimate": None,
        "confidence_interval": None,
        "confidence": 0.5,  # default
    }

    # Extract FRAMEWORK
    fw_match = re.search(r"FRAMEWORK:\s*(\w+)", response, re.IGNORECASE)
    if fw_match:
        result["framework"] = fw_match.group(1).lower()

    # Extract POINT_ESTIMATE
    pe_match = re.search(r"POINT_ESTIMATE:\s*([-+]?\d*\.?\d+)", response, re.IGNORECASE)
    if pe_match:
        result["point_estimate"] = float(pe_match.group(1))

    # Extract CONFIDENCE_INTERVAL
    ci_match = re.search(
        r"CONFIDENCE_INTERVAL:\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]",
        response, re.IGNORECASE
    )
    if ci_match:
        result["confidence_interval"] = (float(ci_match.group(1)), float(ci_match.group(2)))

    # Extract CONFIDENCE
    conf_match = re.search(r"CONFIDENCE:\s*([-+]?\d*\.?\d+)", response, re.IGNORECASE)
    if conf_match:
        result["confidence"] = float(conf_match.group(1))

    return result


# ──────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────

def load_problems(subset_dir: str) -> list[dict]:
    """Load problems from a directory."""
    problems = []
    path = Path(subset_dir)
    for filepath in sorted(path.rglob("MURU-*.json")):
        try:
            with open(filepath) as f:
                problems.append(json.load(f))
        except (json.JSONDecodeError, IOError):
            continue
    return problems


def run_evaluation(
    model_name: str,
    problems: list[dict],
    max_n: int | None = None,
    delay: float = 0.5,
) -> tuple[list[Prediction], list[dict]]:
    """Run model on all problems and collect predictions."""
    client = get_client(model_name)

    if max_n and max_n < len(problems):
        import random
        random.seed(42)
        problems = random.sample(problems, max_n)

    predictions = []
    raw_results = []

    for i, problem in enumerate(problems):
        print(f"  [{i + 1}/{len(problems)}] {problem['id']} ... ", end="", flush=True)

        prompt = USER_PROMPT_TEMPLATE.format(stem=problem["stem"])

        try:
            response = client.query(prompt, system=SYSTEM_PROMPT)
            parsed = parse_response(response)

            if parsed["point_estimate"] is not None:
                pred = Prediction(
                    problem_id=problem["id"],
                    predicted_answer=parsed["point_estimate"],
                    predicted_confidence=parsed["confidence"],
                    predicted_interval=parsed["confidence_interval"],
                    predicted_framework=parsed["framework"],
                    raw_response=response,
                )
                predictions.append(pred)
                print(f"✓ (est={parsed['point_estimate']:.3f}, conf={parsed['confidence']:.2f})")
            else:
                print(f"✗ (could not parse response)")

            raw_results.append({
                "problem_id": problem["id"],
                "response": response,
                "parsed": parsed,
                "success": parsed["point_estimate"] is not None,
            })

        except Exception as e:
            print(f"✗ (error: {e})")
            raw_results.append({
                "problem_id": problem["id"],
                "response": "",
                "parsed": {},
                "success": False,
                "error": str(e),
            })

        time.sleep(delay)

    return predictions, raw_results


def main():
    parser = argparse.ArgumentParser(description="Run MURU-BENCH evaluation.")
    parser.add_argument("--model", "-m", required=True, help="Model name (e.g., gpt-4o, claude-3.5-sonnet)")
    parser.add_argument("--subset", "-s", default=str(PROJECT_ROOT / "data" / "test"), help="Problem directory.")
    parser.add_argument("--n", type=int, help="Max number of problems to evaluate.")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (seconds).")
    parser.add_argument("--save", action="store_true", help="Save results to evaluation/baselines/.")
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print(f"  MURU-BENCH Evaluation")
    print(f"  Model: {args.model}")
    print(f"  Subset: {args.subset}")
    print(f"{'═' * 60}\n")

    problems = load_problems(args.subset)
    if not problems:
        print(f"No problems found in {args.subset}")
        sys.exit(1)

    print(f"  Loaded {len(problems)} problems\n")

    predictions, raw_results = run_evaluation(
        args.model, problems, max_n=args.n, delay=args.delay
    )

    if not predictions:
        print("\n  No valid predictions obtained. Cannot compute metrics.")
        sys.exit(1)

    # Compute metrics
    metrics = MURUMetrics(problems, predictions)
    print(metrics.summary())

    # Save results
    if args.save:
        baselines_dir = PROJECT_ROOT / "evaluation" / "baselines"
        baselines_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = args.model.replace("/", "_").replace(".", "_")

        results_data = {
            "model": args.model,
            "timestamp": timestamp,
            "n_problems": len(problems),
            "n_predictions": len(predictions),
            "metrics": metrics.compute_all(),
            "raw_results": raw_results,
        }

        outpath = baselines_dir / f"{model_slug}_{timestamp}.json"
        with open(outpath, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        print(f"\n  Results saved to: {outpath.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
