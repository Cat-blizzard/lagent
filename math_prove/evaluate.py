"""Regression and ablation runner for MathSolve-Agent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

from .main import run_batch
from .validator import LLMJudgeConfig, validate_results, write_validation_report


DEFAULT_EXPECTED = Path(__file__).parent / "validation" / "core_18_sample.jsonl"


def run_validation_only(
    results: str,
    expected: Optional[str],
    report: str,
    log_dir: Optional[str] = None,
    strict_expected_ids: bool = True,
    llm_judge: Optional[LLMJudgeConfig] = None,
) -> None:
    validation = validate_results(
        results,
        expected,
        log_dir=log_dir,
        strict_expected_ids=strict_expected_ids,
        llm_judge=llm_judge,
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
    llm_judge = build_llm_judge_config(args)
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
            llm_judge=llm_judge,
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
    judge_accuracy = row.get("llm_judge_accuracy")
    judge_text = "" if judge_accuracy is None else f" | llm_judge={judge_accuracy:.2%}"
    schema_rate = row.get("schema_valid_rate", 0.0)
    print(
        f"{prefix}Accuracy={accuracy_text} "
        f"({row.get('answer_correct', 0)}/{row.get('answer_checked', 0)} checked) | "
        f"schema_valid={schema_rate:.2%} | "
        f"preflight_issues={row.get('preflight_issue_count', 0)}"
        f"{judge_text}"
    )


def build_llm_judge_config(args: argparse.Namespace) -> Optional[LLMJudgeConfig]:
    if not getattr(args, "llm_judge", False):
        return None
    config = LLMJudgeConfig(
        enabled=True,
        timeout=args.judge_timeout,
        judge_all=args.llm_judge_all,
    )
    # Primary judge
    key1 = args.judge_api_key or os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("MODEL_API_KEY", "")
    base1 = args.judge_api_base or os.environ.get("LLM_API_BASE", "https://api.deepseek.com/chat/completions")
    if key1:
        config.add_judge(args.judge_model, key1, base1)
    # Second judge
    model2 = getattr(args, "judge_model2", None)
    if model2:
        key2 = getattr(args, "judge_api_key2", None) or os.environ.get("MODEL2_API_KEY", "")
        base2 = getattr(args, "judge_api_base2", None) or os.environ.get("LLM_API_BASE", "https://api.openai.com/v1/chat/completions")
        if key2:
            config.add_judge(model2, key2, base2)
    # Third judge
    model3 = getattr(args, "judge_model3", None)
    if model3:
        key3 = getattr(args, "judge_api_key3", None) or os.environ.get("MODEL3_API_KEY", "")
        base3 = getattr(args, "judge_api_base3", None) or os.environ.get("LLM_API_BASE", "https://api.openai.com/v1/chat/completions")
        if key3:
            config.add_judge(model3, key3, base3)
    if not config.judges:
        return None
    return config


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
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use a DeepSeek/OpenAI-compatible judge for format-tolerant correctness checks.",
    )
    parser.add_argument(
        "--llm-judge-all",
        action="store_true",
        help="Call the LLM judge even when local equivalence already passed.",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default=None,
        help="Judge 1 API key. Defaults to DEEPSEEK_API_KEY or MODEL_API_KEY.",
    )
    parser.add_argument(
        "--judge-api-base",
        type=str,
        default=None,
        help="Judge 1 API base URL.",
    )
    parser.add_argument("--judge-model", type=str, default="deepseek-chat", help="Judge 1 model name")
    parser.add_argument("--judge-timeout", type=int, default=60)
    parser.add_argument("--judge-model2", type=str, default=None, help="Judge 2 model name (optional)")
    parser.add_argument("--judge-api-key2", type=str, default=None, help="Judge 2 API key")
    parser.add_argument("--judge-api-base2", type=str, default=None, help="Judge 2 API base URL")
    parser.add_argument("--judge-model3", type=str, default=None, help="Judge 3 model name (optional)")
    parser.add_argument("--judge-api-key3", type=str, default=None, help="Judge 3 API key")
    parser.add_argument("--judge-api-base3", type=str, default=None, help="Judge 3 API base URL")
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
        llm_judge=build_llm_judge_config(args),
    )


if __name__ == "__main__":
    main()
