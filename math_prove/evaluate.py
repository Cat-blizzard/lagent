"""Regression and ablation runner for MathSolve-Agent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

from .main import run_batch
from .validator import validate_results, write_validation_report


DEFAULT_EXPECTED = Path(__file__).parent / "validation" / "core_18_sample.jsonl"


def run_validation_only(
    results: str,
    expected: Optional[str],
    report: str,
    log_dir: Optional[str] = None,
    strict_expected_ids: bool = True,
) -> None:
    validation = validate_results(
        results,
        expected,
        log_dir=log_dir,
        strict_expected_ids=strict_expected_ids,
    )
    write_validation_report(validation, report)
    payload = validation.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print_score_summary(payload)
    print(f"Validation report saved to {report}")


def run_regression(args: argparse.Namespace) -> None:
    expected = str(Path(args.expected or DEFAULT_EXPECTED))
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    api_base = args.api_base or os.environ.get("LLM_API_BASE")
    ablations = [name.strip() for name in args.ablation.split(",") if name.strip()]
    summary = []

    for ablation in ablations:
        run_dir = output_root / ablation
        run_dir.mkdir(parents=True, exist_ok=True)
        result_jsonl = run_dir / "results.jsonl"
        result_json = run_dir / "results.json"
        logs = run_dir / "logs"
        run_summary = run_dir / "run_summary.json"
        validation_report = run_dir / "validation_report.json"

        print("=" * 72)
        print(f"Running ablation: {ablation}")
        run_batch(
            input_path=expected,
            output_path=str(result_jsonl),
            model_type=args.model,
            api_key=api_key,
            api_base=api_base,
            limit=args.limit,
            resume=args.resume,
            results_json_path=str(result_json),
            log_dir=str(logs),
            summary_path=str(run_summary),
            config_path=args.config,
            ablation=ablation,
        )
        validation = validate_results(
            str(result_jsonl),
            expected,
            log_dir=str(logs),
            strict_expected_ids=not args.ignore_missing_expected,
        )
        write_validation_report(validation, str(validation_report))
        row = validation.to_dict()
        row["ablation"] = ablation
        row["result_jsonl"] = str(result_jsonl)
        row["validation_report"] = str(validation_report)
        summary.append(row)
        print_score_summary(row, prefix=f"[{ablation}] ")

    summary_path = output_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"Ablation summary saved to {summary_path}")


def print_score_summary(row: dict, prefix: str = "") -> None:
    accuracy = row.get("answer_accuracy")
    accuracy_text = "n/a" if accuracy is None else f"{accuracy:.2%}"
    schema_rate = row.get("schema_valid_rate", 0.0)
    print(
        f"{prefix}Accuracy={accuracy_text} "
        f"({row.get('answer_correct', 0)}/{row.get('answer_checked', 0)} checked) | "
        f"schema_valid={schema_rate:.2%} | "
        f"preflight_issues={row.get('preflight_issue_count', 0)}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate or regress MathSolve-Agent")
    parser.add_argument("--results", type=str, default=None, help="Existing results.json/jsonl to validate")
    parser.add_argument("--expected", type=str, default=str(DEFAULT_EXPECTED), help="Expected-answer JSONL")
    parser.add_argument("--report", type=str, default="outputs/validation_report.json")
    parser.add_argument("--log-dir", type=str, default=None, help="Per-problem log directory to check")
    parser.add_argument(
        "--ignore-missing-expected",
        action="store_true",
        help="Do not count expected IDs missing from a limited/subset run as preflight issues.",
    )
    parser.add_argument("--run", action="store_true", help="Run the solver before validating")
    parser.add_argument("--output-dir", type=str, default="outputs/regression")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        help="Comma-separated presets, e.g. full,single_candidate,no_normalizer",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.run:
        run_regression(args)
        return
    if not args.results:
        raise SystemExit("Pass --results to validate an existing file, or --run to run regression.")
    run_validation_only(
        args.results,
        args.expected,
        args.report,
        args.log_dir,
        strict_expected_ids=not args.ignore_missing_expected,
    )


if __name__ == "__main__":
    main()
