# MathSolve-Agent

Chinese guide: [README_CN.md](README_CN.md).

`math_prove/` is a single-agent math-solving system built on top of `lagent`.
It is designed for stable batch evaluation rather than a web demo or a
multi-agent showcase. The system focuses on judgeable JSON output, per-problem
logs, local validation, accuracy reporting, and benchmark conversion.

## What It Does

- Reads JSON, JSONL, CSV, and XLSX problem files.
- Runs a single `MathSolverAgent` with a multi-stage internal pipeline.
- Performs Problem Diagnosis before solving.
- Uses domain-aware solving and verification prompts.
- Adds a rule-first router prior with explicit `tool_policy`
  (`direct`, `sympy`, `ortools`, `python`, `hybrid`, `none`).
- Records verifier-friendly intermediate solution structure:
  assumptions, target, derivation steps, and checkable claims.
- Extracts short final answers for automatic judging.
- Keeps accepted candidate answers protected from unsafe verifier, extract, or
  normalizer rewrites.
- Produces strict structured JSON for every problem.
- Saves per-problem logs and batch summaries.
- Validates schema, answer equivalence, low-quality answers, and log presence.
- Converts UGMathBench, TheoremQA, and MathBench into the local JSONL format.
- Runs ablation experiments and reports accuracy.
- Can run batches serially or through a sidecar parallel runner with a global
  RPM limiter.

## Project Layout

```text
math_prove/
├── agent.py                    # MathSolverAgent pipeline
├── config.py                   # Runtime config and ablation presets
├── convert_benchmarks.py       # UGMathBench / TheoremQA / MathBench converters
├── evaluate.py                 # Run, validate, and report accuracy
├── main.py                     # Demo, batch solving, resume, logs
├── normalizer.py               # Answer normalization and equivalence checks
├── parser.py                   # Pydantic schema, JSON parsing, fallback
├── prompts.py                  # Diagnosis, solve, verify, extract prompts
├── run_ablation_experiments.py # One-command ablation scheduler
├── run_parallel_batch.py       # Sidecar concurrent batch runner
├── sandbox.py                  # SymPy / NumPy / SciPy / OR-Tools helpers
├── validator.py                # Schema, equivalence, and preflight checks
├── validation/
│   └── core_18_sample.jsonl
└── README.md
```

## Install With uv

Run from the lagent repository root:

```powershell
cd D:\lagent-main\lagent
```

Install uv:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

Create and activate the virtual environment:

```powershell
uv venv ..\.venv
..\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
uv pip install -r requirements.txt
uv pip install -e .
uv pip install sympy scipy numpy pandas pyarrow openpyxl pydantic ortools
```

`pandas` and `pyarrow` are needed for converting TheoremQA parquet files.

## Intern-S1 API Setup

The current implementation uses the InternLM OpenAI-compatible Chat
Completions endpoint through `lagent.llms.GPTAPI`.

```powershell
$env:OPENAI_API_KEY = "your-internlm-api-token"
$env:LLM_API_BASE = "https://chat.intern-ai.org.cn/api/v1/chat/completions"
```

Use the model name at runtime:

```powershell
--model intern-s1
```

Notes:

- Put only the token in `OPENAI_API_KEY`; do not include `Bearer`.
- This project does not use the Claude-like `/v1/messages` API.
- Intern-S1 may emit `<think>...</think>` or Markdown JSON wrappers. The agent
  cleans these before JSON parsing.
- Sandbox output is used for local equivalence checks only when it contains the
  `FINAL_RESULT_FOR_CHECK:` marker.

For formal competition-style runs, use fail-fast official mode so an accidental
OpenAI/other-model configuration is rejected before the batch starts:

```powershell
uv run python -m math_prove.main `
  --demo `
  --model intern-s1 `
  --ablation official_stable `
  --official
```

## Correctness Guardrails

The default stable path is conservative: helper stages may warn, normalize, or
log alternatives, but they should not silently damage a mathematically correct
answer.

- Rule-first diagnosis now produces a local routing prior before the LLM
  diagnosis. The prior covers obvious arithmetic, matrix, calculus,
  optimization, graph/discrete, topology, and proof-like questions, then
  Intern-S1 can correct or enrich it.
- `tool_policy` is recorded in `classification` and controls whether generated
  verification code is allowed to run. Proof/topology-style diagnoses stay on
  the direct reasoning path unless the model gives a stronger reason.
- Final JSON is assembled from the accepted `CandidateSolution` by code. The
  extract stage may improve `reasoning_summary`, `key_steps`, and
  `learning_hint`, but stable presets keep the final `answer` from the accepted
  candidate.
- Stable presets use a 240-second problem timeout by default. If the accepted
  candidate has already been produced but the final extract stage would exceed
  the timeout, extraction is skipped and the verified candidate answer is kept.
- `verifier_can_overwrite_answer=false`: the verifier's `corrected_answer` is
  logged by default instead of replacing the candidate answer.
- `extract_must_match_candidate=true`: the extract stage can compress/reformat
  an answer, but if the extracted answer is not equivalent to the accepted
  candidate, the system reverts to the candidate answer.
- `normalizer_overwrite_answer=false`: answer normalization records raw, LaTeX,
  and canonical forms without overwriting the final `answer`.
- The normalizer supports common judge formats such as choices, sets, tuples,
  vectors, intervals, matrices, fractions, square roots, and elementary
  functions. These forms are used for comparison and reporting, not for
  silently changing the final answer in stable presets.
- Cosmetic format-only verifier failures are downgraded when target, condition,
  result, and judgeability checks all pass.
- Local equivalence failures are warnings by default
  (`equivalence_can_fail_candidate=false`) unless a strict ablation enables
  hard failure.
- `official_stable` intentionally inherits the conservative `safe` preset. Use
  `strong`, `strict_equivalence`, or sandbox presets for local stress tests, not
  as the first formal-submission configuration.

Supported `answer_type` values are:

```text
formula, numeric, proof, choice, set, interval, matrix, vector, tuple, text, other
```

## Verifiable Reasoning Trace

The solver borrows the generator-verifier-refiner idea from multi-agent
reasoning work, but implements it inside one `MathSolverAgent`; it does not run
multi-agent training.

- `solve_candidate` acts as the generator and returns a candidate answer plus
  `assumptions`, `target`, `derivation_steps`, and `checkable_claims`.
- `verify_candidate` acts as the verifier and performs layered checks plus
  per-claim checks with `passed`, `failed`, or `uncertain` status.
- Retry feedback acts as the refiner: failed or uncertain claims are included in
  the next attempt's repair instruction.
- `select_best` is used only when the configured candidate-selection path is
  enabled for harder questions.

These fields are kept in per-problem logs and candidate records. The final
submission JSON remains short and judgeable.

## Single Demo

```powershell
uv run python -m math_prove.main --demo --model intern-s1 --ablation safe
```

## Batch Solving

Recommended JSONL input:

```jsonl
{"problem_id":"001","problem_text":"Find all real roots of x^4 - 5x^2 + 4 = 0."}
{"problem_id":"002","problem_text":"Compute the residue of f(z)=(z^2+1)/(z-i) at z=i."}
```

Run a batch:

```powershell
uv run python -m math_prove.main `
  -i data\problems.jsonl `
  -o outputs\results.jsonl `
  --results-json outputs\results.json `
  --log-dir outputs\logs `
  --summary outputs\run_summary.json `
  --model intern-s1 `
  --ablation safe `
  --resume
```

Common arguments:

| Argument | Description |
| --- | --- |
| `-i, --input` | Input file, supporting JSON / JSONL / CSV / XLSX |
| `-o, --output` | Incremental JSONL result file |
| `--results-json` | Merged JSON array result file |
| `--log-dir` | Per-problem log directory |
| `--summary` | Batch summary path |
| `--limit` | Run only the first N problems |
| `--resume` | Skip IDs already present in the output JSONL |
| `--ablation` | Runtime preset |

## Parallel Batch Runner

Use the sidecar runner when your Intern-S1 quota allows concurrency. It keeps
the serial `math_prove.main` path unchanged, creates one agent per worker, and
uses a global RPM limiter around every LLM request.

Recommended starting point for a 100 RPM / 1,000,000 TPM quota:

```powershell
uv run python -m math_prove.run_parallel_batch `
  -i D:\dataset\converted\mathbench_all.jsonl `
  -o outputs\mathbench_parallel\results.jsonl `
  --model intern-s1 `
  --ablation safe `
  --workers 3 `
  --rpm-limit 80 `
  --resume
```

If this is stable, try `--workers 5 --rpm-limit 90`. Avoid very high worker
counts unless the API quota is raised again.

The parallel runner writes the same JSONL / JSON / per-problem log structure as
the serial path. It only changes scheduling, not the solver pipeline.

## Output Schema

Each problem produces one strict JSON object:

```json
{
  "problem_id": "001",
  "domain": "complex_analysis",
  "answer": "\\frac{\\pi}{2}",
  "answer_type": "formula",
  "reasoning_summary": "Identify the relevant poles, compute residues, then apply the residue theorem.",
  "key_steps": ["Locate poles", "Compute residues", "Apply the residue theorem"],
  "learning_hint": "For contour-integral problems, verify the contour orientation and pole locations before applying the residue theorem.",
  "verification": {
    "passed": true,
    "confidence": 0.86,
    "issues": [],
    "error_type": "none",
    "repair_instruction": ""
  }
}
```

The `answer` field should be short and judge-friendly. It should not contain
the full derivation.

## Convert External Benchmarks

The converter writes JSONL files that can be used directly by both `main.py`
and `evaluate.py`.

### UGMathBench

```powershell
uv run python -m math_prove.convert_benchmarks ugmathbench `
  --input D:\dataset\ugmathbench\data `
  --output D:\dataset\converted\ugmathbench_all.jsonl
```

Small sample:

```powershell
uv run python -m math_prove.convert_benchmarks ugmathbench `
  --input D:\dataset\ugmathbench\data `
  --output D:\dataset\converted\ugmathbench_50.jsonl `
  --limit 50
```

### TheoremQA

By default, image-dependent rows are skipped because this system is text-only:

```powershell
uv run python -m math_prove.convert_benchmarks theoremqa `
  --input D:\dataset\TheoremQA\data\test-00000-of-00001.parquet `
  --output D:\dataset\converted\theoremqa_text_only.jsonl
```

### MathBench

Full conversion:

```powershell
uv run python -m math_prove.convert_benchmarks mathbench `
  --input D:\dataset\MathBench\mathbench_v1 `
  --output D:\dataset\converted\mathbench_all.jsonl
```

English only:

```powershell
uv run python -m math_prove.convert_benchmarks mathbench `
  --input D:\dataset\MathBench\mathbench_v1 `
  --output D:\dataset\converted\mathbench_en.jsonl `
  --language en
```

## Accuracy Evaluation

Validate an existing result file:

```powershell
uv run python -m math_prove.evaluate `
  --results outputs\mathbench_en_safe\safe\results.jsonl `
  --expected D:\dataset\converted\mathbench_en.jsonl `
  --report outputs\mathbench_en_safe\validation_report.json `
  --ignore-missing-expected
```

Example summary:

```text
Accuracy=85.00% (17/20 checked) | schema_valid=100.00% | preflight_issues=0
```

Use `--ignore-missing-expected` for limited runs, such as `--limit 20`, so
unrun problems in the full expected file are not counted as missing.

### Optional DeepSeek Judge

For a format-tolerant correctness estimate, enable an OpenAI-compatible
DeepSeek judge. Local equivalence is still used first; by default, DeepSeek is
called only when local equivalence cannot prove the answer correct.

```powershell
$env:DEEPSEEK_API_KEY = "your-deepseek-api-key"

uv run python -m math_prove.evaluate `
  --results outputs\mathbench_en_safe\safe\results.jsonl `
  --expected D:\dataset\converted\mathbench_en.jsonl `
  --report outputs\mathbench_en_safe\validation_report_deepseek.json `
  --ignore-missing-expected `
  --llm-judge `
  --judge-model deepseek-v4-flash
```

The report will include both local equivalence accuracy and
`llm_judge_accuracy`. The LLM judge is instructed to ignore superficial
formatting differences such as `[-2, -1, 1, 2]` versus `{-2, -1, 1, 2}`.

## Run And Evaluate

MathBench sample run:

```powershell
uv run python -m math_prove.evaluate `
  --run `
  --expected D:\dataset\converted\mathbench_en.jsonl `
  --output-dir outputs\mathbench_en_safe `
  --model intern-s1 `
  --ablation safe `
  --limit 20 `
  --ignore-missing-expected
```

TheoremQA sample run:

```powershell
uv run python -m math_prove.evaluate `
  --run `
  --expected D:\dataset\converted\theoremqa_text_only.jsonl `
  --output-dir outputs\theoremqa_safe `
  --model intern-s1 `
  --ablation safe `
  --limit 20 `
  --ignore-missing-expected
```

UGMathBench sample run:

```powershell
uv run python -m math_prove.evaluate `
  --run `
  --expected D:\dataset\converted\ugmathbench_all.jsonl `
  --output-dir outputs\ugmath_safe `
  --model intern-s1 `
  --ablation safe `
  --limit 20 `
  --ignore-missing-expected
```

## Ablation Experiments

```powershell
uv run python -m math_prove.run_ablation_experiments `
  --suite smoke `
  --expected D:\dataset\converted\mathbench_en.jsonl `
  --model intern-s1 `
  --limit 20 `
  --ignore-missing-expected `
  --ablation base,safe,safe_plus
```

Common presets:

```text
base
base_verify
base_normalizer
base_extract
base_multi
safe
safe_plus
strong
base_sandbox_observe
base_sandbox_verify
base_ortools_verify
```

Suggested use:

- `base`: minimal baseline.
- `safe`: conservative candidate for formal batch runs; helper stages warn but
  do not directly overwrite accepted answers.
- `safe_plus`: low-risk enhanced local comparison.
- `strong`: harder-problem mode with sandbox, OR-Tools, strict equivalence, and
  verifier correction enabled; use it for hard sets or ablation, not as the
  default formal run.
- `official_stable`: fail-fast Intern-S1 configuration based on `safe`.

## Local Checks

```powershell
uv run python -m py_compile math_prove\*.py
uv run python -m math_prove.main --help
uv run python -m math_prove.evaluate --help
uv run python -m math_prove.convert_benchmarks --help
uv run python -m math_prove.run_parallel_batch --help
uv run --with pytest python -m pytest tests\test_math_prove\test_accuracy_guards.py -q
```

## GitHub Notes

Do not commit:

```text
.venv/
outputs/
__pycache__/
*.pyc
.env
API keys
official private evaluation data
large external datasets
```

Safe to commit:

```text
math_prove/*.py
math_prove/README.md
math_prove/README_CN.md
math_prove/validation/core_18_sample.jsonl
```
