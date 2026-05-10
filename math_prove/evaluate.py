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
) -> None:
    validation = validate_results(results, expected, log_dir=log_dir)
    write_validation_report(validation, report)
    print(json.dumps(validation.to_dict(), ensure_ascii=False, indent=2))
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
        validation = validate_results(str(result_jsonl), expected, log_dir=str(logs))
        write_validation_report(validation, str(validation_report))
        row = validation.to_dict()
        row["ablation"] = ablation
        row["result_jsonl"] = str(result_jsonl)
        row["validation_report"] = str(validation_report)
        summary.append(row)

    summary_path = output_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"Ablation summary saved to {summary_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate or regress MathSolve-Agent")
    parser.add_argument("--results", type=str, default=None, help="Existing results.json/jsonl to validate")
    parser.add_argument("--expected", type=str, default=str(DEFAULT_EXPECTED), help="Expected-answer JSONL")
    parser.add_argument("--report", type=str, default="outputs/validation_report.json")
    parser.add_argument("--log-dir", type=str, default=None, help="Per-problem log directory to check")
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
    run_validation_only(args.results, args.expected, args.report, args.log_dir)


if __name__ == "__main__":
    main()
