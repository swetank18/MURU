# MURU-BENCH: Mathematical Reasoning Under Uncertainty Benchmark

> A rigorous benchmark for evaluating LLM calibration and probabilistic reasoning — targeting NeurIPS 2026 Datasets & Benchmarks.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## The Problem

Every major math benchmark (GSM8K, MATH, MMLU, BIG-Bench) tests models on problems with **single, deterministic answers**. But real-world reasoning requires handling **genuine mathematical uncertainty** — Bayesian updating with unknown priors, decisions under parameter ambiguity, and calibrated confidence in interval estimates.

**MURU-BENCH fills this gap.** It is the first benchmark that requires models to:
1. Identify the correct probabilistic framework
2. Execute mathematical reasoning under uncertainty
3. Produce **calibrated confidence intervals**, not just point answers
4. Articulate which assumptions drive the uncertainty

## Dataset Overview

| Property | Value |
|----------|-------|
| **Total Problems** | 3,000 |
| **Categories** | 5 |
| **Difficulty Levels** | 1–5 |
| **Answer Format** | Point estimate + confidence interval |
| **License** | CC BY 4.0 |

### Problem Categories

| Category | Count | Description |
|----------|-------|-------------|
| **Bayesian Updating** | 700 | Prior beliefs revised by evidence with ambiguous likelihoods |
| **Conditional Probability Chains** | 600 | Multi-step conditioning with uncertain intermediates |
| **Distribution Estimation** | 600 | Inferring distributions from incomplete/biased samples |
| **Decision Under Uncertainty** | 550 | Expected value with uncertain utility functions |
| **Adversarial Ambiguity** | 550 | Problems designed to fool overconfident models |

## Quick Start

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/muru-bench.git
cd muru-bench
pip install -r requirements.txt
```

### Validate Problems

```bash
python scripts/validate.py              # validate all problems
python scripts/validate.py data/train/  # validate a subset
```

### Inspect Problems

```bash
python scripts/sample.py                              # 3 random problems
python scripts/sample.py --n 5 --category bayesian_updating --difficulty 3
```

### View Dataset Statistics

```bash
python scripts/stats.py          # text summary
python scripts/stats.py --plots  # + generate figures
```

### Run Evaluation

```bash
export OPENAI_API_KEY="your-key"
python evaluation/run_eval.py --model gpt-4o --subset data/test/ --save
```

## Sample Problem

**Category:** Bayesian Updating | **Difficulty:** ★★★☆☆

> A factory produces widgets using one of two machines. Machine A produces 60% of widgets with a 2% defect rate. Machine B produces 40% of widgets with an unknown defect rate estimated to be between 4% and 8% based on recent maintenance reports. You sample 20 widgets and find 3 defective ones. What is the probability that your sample came from Machine B, and what is your confidence in this estimate given the uncertainty in Machine B's defect rate?

**Ground Truth:** P(Machine B | 3 defects) ∈ [0.71, 0.89] at 90% credibility. Point estimate: 0.81.

**What a wrong model does:** Collapses the uncertainty in Machine B's defect rate and produces a single point answer without acknowledging the range.

## Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Accuracy@exact** | Answer within ground truth CI |
| **ECE** | Expected Calibration Error |
| **Category Breakdown** | Per-category accuracy |
| **Difficulty Scaling** | Accuracy drop with difficulty |
| **Overconfidence Rate** | Confidence exceeds accuracy |
| **Reasoning Chain Quality** | Correct framework selection |

## Leaderboard

| Model | Accuracy | ECE ↓ | Overconf. Rate ↓ | Framework Match |
|-------|----------|-------|-------------------|-----------------|
| *Evaluations coming soon* | — | — | — | — |

## Repository Structure

```
muru-bench/
├── data/
│   ├── train/              # 80% of problems
│   ├── validation/         # 10% of problems
│   ├── test/               # 10% held-out
│   ├── by_category/        # category views
│   └── by_difficulty/      # difficulty views
├── evaluation/
│   ├── run_eval.py         # evaluation runner
│   ├── metrics.py          # ECE, accuracy, calibration
│   └── baselines/          # model results
├── scripts/
│   ├── validate.py         # schema validation
│   ├── stats.py            # dataset statistics
│   ├── sample.py           # problem sampler
│   └── split_data.py       # train/val/test splitter
├── paper/
│   └── figures/            # generated figures
├── problem_schema.json     # JSON schema
├── requirements.txt
├── CONTRIBUTING.md
└── LICENSE
```

## Citation

```bibtex
@misc{muru-bench-2026,
  title={MURU-BENCH: A Benchmark for Mathematical Reasoning Under Uncertainty},
  author={TODO},
  year={2026},
  url={https://github.com/YOUR_USERNAME/muru-bench}
}
```

## License

This dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt the material with appropriate attribution.
