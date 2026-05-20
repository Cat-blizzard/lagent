"""One-command ablation experiment runner for MathSolve-Agent.

This module is intentionally a thin scheduler around ``math_prove.evaluate``.
It does not duplicate solver logic and does not mutate the main pipeline.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from .config import ABLATION_PRESETS
from .evaluate import run_regression


PACKAGE_ROOT = Path(__file__).resolve().parent
VALIDATION_ROOT = PACKAGE_ROOT / "validation"
DEFAULT_OUTPUT_ROOT = Path("outputs") / "ablation_runs"

SUITES: Dict[str, Dict[str, str]] = {
    "smoke": {
        "dataset": "core_18_sample.jsonl",
        "ablation": "base,base_verify,base_normalizer,base_extract,safe",
    },
    "core": {
        "dataset": "core_18_sample.jsonl",
        "ablation": "base,base_verify,base_normalizer,base_extract,base_multi,safe,safe_plus",
    },
    "format": {
        "dataset": "format_stress.jsonl",
        "ablation": "base,base_normalizer,base_extract,safe,safe_plus",
    },
    "hard": {
        "dataset": "hard_20.jsonl",
        "ablation": "base,base_verify,base_multi,safe,safe_plus,strong",
    },
    "tool": {
        "dataset": "calc_tool_20.jsonl",
        "ablation": "base,base_sandbox_observe,base_sandbox_verify,strong",
    },
    "opt": {
        "dataset": "optimization_10.jsonl",
        "ablation": "base,base_sandbox_verify,base_ortools_verify,strong",
    },
    "final": {
        "dataset": "larger_eval.jsonl",
        "ablation": "base,safe,safe_plus,strong",
    },
}

FALLBACK_DATASET = VALIDATION_ROOT / "core_18_sample.jsonl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MathSolve-Agent ablation suites")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="smoke",
        help="Experiment suite to run",
    )
    parser.add_argument("--expected", type=str, default=None, help="Override suite dataset")
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for experiment outputs",
    )
    parser.add_argument("--model", type=str, default="intern-s1")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume within preset directories")
    parser.add_argument(
        "--ignore-missing-expected",
        action="store_true",
        help="Do not count expected IDs missing from a limited/subset run as preflight issues.",
    )
    parser.add_argument("--llm-judge", action="store_true", help="Enable DeepSeek judge")
    parser.add_argument("--llm-judge-all", action="store_true", help="Judge all answers with LLM")
    parser.add_argument("--judge-api-key", type=str, default=None)
    parser.add_argument(
        "--judge-api-base",
        type=str,
        default="https://api.deepseek.com/chat/completions",
    )
    parser.add_argument("--judge-model", type=str, default="deepseek-v4-flash")
    parser.add_argument("--judge-timeout", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true", help="Print the planned command only")
    parser.add_argument("--ablation", type=str, default=None, help="Override suite preset list")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    started_at = datetime.now().isoformat(timespec="seconds")
    suite = SUITES[args.suite]

    dataset, dataset_warning = resolve_dataset(args.suite, args.expected, args.dry_run)
    ablations = parse_ablations(args.ablation or suite["ablation"])
    validate_presets(ablations)

    run_dir = build_run_dir(Path(args.output_root), args.suite, started_at)
    equivalent_command = build_equivalent_command(args, dataset, run_dir, ablations)

    manifest = {
        "suite": args.suite,
        "dataset": str(dataset),
        "dataset_exists": dataset.exists(),
        "dataset_warning": dataset_warning,
        "model": args.model,
        "ablations": ablations,
        "output_dir": str(run_dir),
        "limit": args.limit,
        "resume": bool(args.resume),
        "ignore_missing_expected": bool(args.ignore_missing_expected),
        "llm_judge": bool(args.llm_judge),
        "llm_judge_all": bool(args.llm_judge_all),
        "judge_model": args.judge_model,
        "dry_run": bool(args.dry_run),
        "started_at": started_at,
        "equivalent_command": equivalent_command,
    }

    print_plan(manifest)
    if args.dry_run:
        if dataset_warning:
            print(f"[dry-run warning] {dataset_warning}")
        return

    if not dataset.exists():
        raise SystemExit(dataset_warning or f"Dataset does not exist: {dataset}")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    regression_args = SimpleNamespace(
        expected=str(dataset),
        output_dir=str(run_dir),
        api_key=args.api_key,
        api_base=args.api_base,
        ablation=",".join(ablations),
        model=args.model,
        limit=args.limit,
        resume=args.resume,
        config=None,
        ignore_missing_expected=args.ignore_missing_expected,
        llm_judge=args.llm_judge,
        llm_judge_all=args.llm_judge_all,
        judge_api_key=args.judge_api_key,
        judge_api_base=args.judge_api_base,
        judge_model=args.judge_model,
        judge_timeout=args.judge_timeout,
    )
    run_regression(regression_args)

    finished_at = datetime.now().isoformat(timespec="seconds")
    write_suite_summary(run_dir, manifest, finished_at)
    print(f"Suite summary saved to {run_dir / 'suite_summary.json'}")


def resolve_dataset(
    suite_name: str,
    override: Optional[str],
    dry_run: bool,
) -> tuple[Path, str]:
    if override:
        path = Path(override)
        if path.exists() or dry_run:
            warning = "" if path.exists() else f"Override dataset does not exist yet: {path}"
            return path, warning
        raise SystemExit(f"Dataset does not exist: {path}")

    default = VALIDATION_ROOT / SUITES[suite_name]["dataset"]
    if default.exists():
        return default, ""

    if suite_name in {"smoke", "core"} and FALLBACK_DATASET.exists():
        return FALLBACK_DATASET, (
            f"Default dataset {default} is missing; falling back to {FALLBACK_DATASET}."
        )

    warning = (
        f"Default dataset is missing: {default}. Create it or pass --expected <path>."
    )
    if dry_run:
        return default, warning
    raise SystemExit(warning)


def parse_ablations(raw: str) -> List[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("No ablation presets were provided.")
    return values


def validate_presets(ablations: List[str]) -> None:
    missing = [name for name in ablations if name not in ABLATION_PRESETS]
    if missing:
        known = ", ".join(sorted(ABLATION_PRESETS))
        raise SystemExit(
            "Missing ablation presets in math_prove.config: "
            + ", ".join(missing)
            + f"\nKnown presets: {known}"
        )


def build_run_dir(output_root: Path, suite: str, started_at: str) -> Path:
    stamp = started_at.replace(":", "").replace("-", "").replace("T", "_")
    return output_root / f"{stamp}_{suite}"


def build_equivalent_command(
    args: argparse.Namespace,
    dataset: Path,
    run_dir: Path,
    ablations: List[str],
) -> str:
    parts = [
        "uv run python -m math_prove.evaluate",
        "  --run",
        f"  --expected {quote_path(dataset)}",
        f"  --output-dir {quote_path(run_dir)}",
        f"  --model {args.model}",
        f"  --ablation {','.join(ablations)}",
    ]
    if args.limit is not None:
        parts.append(f"  --limit {args.limit}")
    if args.resume:
        parts.append("  --resume")
    if args.ignore_missing_expected:
        parts.append("  --ignore-missing-expected")
    if args.llm_judge:
        parts.append("  --llm-judge")
    if args.llm_judge_all:
        parts.append("  --llm-judge-all")
    if args.judge_api_key:
        parts.append("  --judge-api-key <provided>")
    if args.judge_api_base != "https://api.deepseek.com/chat/completions":
        parts.append(f"  --judge-api-base {args.judge_api_base}")
    if args.judge_model != "deepseek-v4-flash":
        parts.append(f"  --judge-model {args.judge_model}")
    if args.api_key:
        parts.append("  --api-key <provided>")
    if args.api_base:
        parts.append(f"  --api-base {args.api_base}")
    return " `\n".join(parts)


def quote_path(path: Path) -> str:
    text = str(path)
    return f'"{text}"' if " " in text else text


def print_plan(manifest: Dict[str, Any]) -> None:
    print("=" * 72)
    print(f"Suite: {manifest['suite']}")
    print(f"Dataset: {manifest['dataset']}")
    print(f"Dataset exists: {manifest['dataset_exists']}")
    print(f"Model: {manifest['model']}")
    print(f"Ablations: {', '.join(manifest['ablations'])}")
    print(f"Output dir: {manifest['output_dir']}")
    print("-" * 72)
    print(manifest["equivalent_command"])
    print("=" * 72)


def write_suite_summary(
    run_dir: Path,
    manifest: Dict[str, Any],
    finished_at: str,
) -> None:
    summary_path = run_dir / "ablation_summary.json"
    rows = load_summary_rows(summary_path)
    compact_rows = [compact_summary_row(row) for row in rows]
    suite_summary = {
        "suite": manifest["suite"],
        "dataset": manifest["dataset"],
        "model": manifest["model"],
        "ablations": manifest["ablations"],
        "started_at": manifest["started_at"],
        "finished_at": finished_at,
        "rows": compact_rows,
    }
    (run_dir / "suite_summary.json").write_text(
        json.dumps(suite_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_summary_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def compact_summary_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ablation": row.get("ablation"),
        "total": row.get("total"),
        "schema_valid_rate": row.get("schema_valid_rate"),
        "answer_accuracy": row.get("answer_accuracy"),
        "preflight_issue_count": row.get("preflight_issue_count"),
        "log_missing_count": row.get("log_missing_count"),
        "answer_checked": row.get("answer_checked"),
        "answer_correct": row.get("answer_correct"),
        "answer_incorrect": row.get("answer_incorrect"),
        "result_jsonl": row.get("result_jsonl"),
        "validation_report": row.get("validation_report"),
    }


if __name__ == "__main__":
    main()
