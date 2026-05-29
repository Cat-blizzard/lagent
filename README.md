# MathSolve-Agent

A single-agent math solving pipeline built on [Lagent](https://github.com/InternLM/lagent), designed for batch evaluation of mathematical reasoning with structured JSON output, multi-model answer verification, and accuracy benchmarking.

## Overview

MathSolve-Agent extends the Lagent agent framework with `math_prove/`, a multi-stage solver that processes math problems through classification, candidate generation with retries, self-verification, and answer extraction. It produces judgeable JSON suitable for automated evaluation and supports comparison against a simple MathCoder baseline.

**Key features:**

- Multi-stage solver pipeline: classify → solve → verify → retry → extract
- Rule-first problem diagnosis with domain-aware tool routing
- Multi-model majority-vote evaluation (local equivalence + up to 3 LLM judges)
- Batch solving with resume support, per-problem logs, and structured output
- Baseline solver wrapping the original Lagent MathCoder for direct comparison
- Benchmark conversion (UGMathBench, TheoremQA, MathBench)

## Quick Start

```bash
# Install
uv pip install -r requirements.txt
uv pip install -e .
uv pip install sympy scipy numpy pandas pyarrow pydantic

# API configuration (Intern-S1)
export OPENAI_API_KEY="your-api-token"
export LLM_API_BASE="https://chat.intern-ai.org.cn/api/v1/chat/completions"

# Single demo
python -m math_prove.main --demo --model intern-s1 --ablation safe

# Batch solve
python -m math_prove.main \
  -i problems.jsonl \
  -o outputs/results.jsonl \
  --results-json outputs/results.json \
  --log-dir outputs/logs \
  --model intern-s1 \
  --ablation safe \
  --resume

# Evaluate with multi-model judge
python -m math_prove.evaluate \
  --results outputs/results.jsonl \
  --expected problems.jsonl \
  --report outputs/report.json \
  --llm-judge \
  --judge-model deepseek-chat --judge-api-key "$KEY1" --judge-api-base "https://api.deepseek.com/chat/completions" \
  --judge-model2 gpt-4o-mini --judge-api-key2 "$KEY2" --judge-api-base2 "https://api.openai.com/v1/chat/completions" \
  --judge-model3 intern-s1 --judge-api-key3 "$KEY3" --judge-api-base3 "https://chat.intern-ai.org.cn/api/v1/chat/completions"
```

## Solver Pipeline

The `MathSolverAgent` processes each problem through six stages:

1. **Preprocess** — normalize whitespace, strip empty lines
2. **Classify** — heuristic rule router determines domain, answer type, difficulty, and tool policy (`direct|sympy|ortools|python|hybrid`). The LLM refines this classification.
3. **Solve** — LLM generates a candidate solution with answer, reasoning, checkable claims, and optional verification code. Failed verifications drive retries with targeted repair instructions.
4. **Verify** — LLM performs layered checks (format, question target, conditions, result, judgeability) plus per-claim validation. Local equivalence against sandbox output when available.
5. **Select** (hard problems) — LLM compares multiple candidates and picks the most reliable.
6. **Extract** — LLM produces the final judgeable JSON. The accepted candidate answer is protected from unsafe rewrites.

All stages produce strict JSON. Model output is cleaned of `<think>` tags, BOM, and Markdown fences before parsing.

## Evaluation

The evaluation pipeline uses a two-tier approach:

| Tier | Method | When |
|------|--------|------|
| Local equivalence | Canonical string match → numeric tolerance → SymPy simplification | First pass; zero API cost |
| Multi-model LLM judge | Up to 3 models vote on answer correctness against the problem statement | Falls back when local equivalence fails |

Unlike reference-answer comparison, the LLM judges evaluate the answer directly against the problem, eliminating the long-proof truncation issue. Majority vote determines the final verdict.

## Output Schema

Each problem produces a `MathSolution` JSON:

```json
{
  "problem_id": "001",
  "domain": "linear_algebra",
  "answer": "(0, -6, -4)",
  "answer_type": "tuple",
  "reasoning_summary": "Compute powers of A, set up linear system, solve for p, q, r.",
  "key_steps": ["Compute A² and A³", "Form the matrix equation", "Extract and solve linear system"],
  "learning_hint": "For Cayley-Hamilton problems, the characteristic polynomial provides p, q, r directly.",
  "verification": {
    "passed": true,
    "confidence": 0.95,
    "issues": [],
    "error_type": "none",
    "repair_instruction": ""
  }
}
```

## Project Structure

```
math_prove/
├── agent.py                    # MathSolverAgent pipeline
├── config.py                   # SolverConfig + ablation presets
├── main.py                     # CLI: demo, batch solving, resume
├── evaluate.py                 # CLI: validation with multi-judge support
├── validator.py                # Schema validation, answer equivalence, LLM judge voting
├── parser.py                   # Pydantic schemas (MathSolution, VerificationResult, etc.)
├── prompts.py                  # LLM prompt templates with domain-specific strategies
├── normalizer.py               # Answer normalization (LaTeX→canonical) and equivalence checks
├── sandbox.py                  # IPython sandbox with SymPy/NumPy/SciPy/OR-Tools
├── baseline_solver.py          # Minimal MathCoder wrapper for baseline comparison
├── run_parallel_batch.py       # ThreadPool + RPM-limited parallel runner
├── run_ablation_experiments.py # Batch ablation scheduler
├── convert_benchmarks.py       # UGMathBench / TheoremQA / MathBench converters
└── validation/
    └── core_18_sample.jsonl
```

## Ablation Presets

`SolverConfig` controls every pipeline stage. Presets layer overrides on `BASE_PRESET`:

| Preset | Sandbox | LLM Verify | Equivalence | Extract | Use case |
|--------|---------|------------|-------------|---------|----------|
| `base` | off | off | off | off | Minimal baseline |
| `safe` | off | on | off | off | Conservative production runs |
| `safe_plus` | off | on | on | on | Enhanced local checks |
| `strong` | on | on | on (strict) | on | Hard problem stress tests |
| `official_stable` | off | on | off | off | Fail-fast Intern-S1 mode |

## Baseline Comparison

To compare the full pipeline against the minimal Lagent MathCoder:

```bash
# Pipeline (multi-stage solver)
python -m math_prove.main -i problems.jsonl -o outputs/pipeline/results.jsonl \
  --model intern-s1 --ablation safe --resume

# Baseline (simple think→code→execute loop)  
python -m math_prove.baseline_solver -i problems.jsonl -o outputs/baseline/results.jsonl \
  --model intern-s1 --max-turn 6 --resume

# Evaluate both with identical judges
python -m math_prove.evaluate --results outputs/pipeline/results.jsonl \
  --expected problems.jsonl --report outputs/pipeline/report.json --llm-judge ...
python -m math_prove.evaluate --results outputs/baseline/results.jsonl \
  --expected problems.jsonl --report outputs/baseline/report.json --llm-judge ...
```

## License

This project is built on [Lagent](https://github.com/InternLM/lagent) (Apache 2.0).
