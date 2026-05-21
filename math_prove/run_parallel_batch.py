"""Parallel batch runner for MathSolve-Agent.

This is a sidecar entrypoint. It does not replace ``math_prove.main``; it only
adds a guarded way to run multiple problems concurrently under a global RPM
limit.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from .main import (
    _load_and_validate_results,
    _read_existing_results,
    _resolve_api_config,
    _write_problem_log,
    load_problems,
)
from .parser import fallback_solution, solution_to_json


def _safe_print(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    try:
        print(*args, **kwargs)
    except ValueError:
        # Some hosted shells close stdout early after very noisy child output.
        try:
            sys.stderr.write(" ".join(str(arg) for arg in args) + "\n")
        except Exception:
            pass


class RpmLimiter:
    """Thread-safe sliding-window request limiter.

    The limiter is injected around each LLM stage call, so the limit applies to
    API requests rather than just problem-level tasks.
    """

    def __init__(self, rpm_limit: int) -> None:
        self.rpm_limit = max(1, int(rpm_limit))
        self._events: Deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0] >= 60.0:
                    self._events.popleft()
                if len(self._events) < self.rpm_limit:
                    self._events.append(now)
                    return
                wait_seconds = max(0.05, 60.0 - (now - self._events[0]))
            time.sleep(wait_seconds)


class RateLimitedAgent:
    """Small composition wrapper around MathSolverAgent."""

    def __init__(
        self,
        limiter: RpmLimiter,
        model_type: str,
        api_key: Optional[str],
        api_base: Optional[str],
        config_path: Optional[str],
        ablation: str,
        official_mode: bool,
    ) -> None:
        from .agent import MathSolverAgent

        self._agent = MathSolverAgent(
            model_type=model_type,
            api_key=api_key,
            api_base=api_base,
            config_path=config_path,
            ablation=ablation,
            official_mode=official_mode,
        )
        self._limiter = limiter
        original_call_llm = self._agent._call_llm

        def limited_call_llm(*args, **kwargs):  # type: ignore[no-untyped-def]
            self._limiter.acquire()
            return original_call_llm(*args, **kwargs)

        self._agent._call_llm = limited_call_llm  # type: ignore[method-assign]

    @property
    def last_run_log(self) -> Dict[str, Any]:
        return self._agent.last_run_log

    def solve(self, problem: str, problem_id: str, raw_metadata: Dict[str, Any]):
        return self._agent.solve(problem, problem_id, raw_metadata=raw_metadata)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)


def run_parallel_batch(
    input_path: str,
    output_path: str,
    model_type: str = "intern-s1",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    limit: Optional[int] = None,
    resume: bool = False,
    results_json_path: Optional[str] = None,
    log_dir: Optional[str] = None,
    summary_path: Optional[str] = None,
    config_path: Optional[str] = None,
    ablation: str = "safe",
    official_mode: bool = False,
    workers: int = 3,
    rpm_limit: int = 80,
    dry_run: bool = False,
) -> Dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    results_json = Path(results_json_path) if results_json_path else output.with_suffix(".json")
    logs = Path(log_dir) if log_dir else output.parent / "logs"
    summary = Path(summary_path) if summary_path else output.parent / "run_summary.json"
    logs.mkdir(parents=True, exist_ok=True)

    problems = load_problems(input_path)
    if limit is not None:
        problems = problems[:limit]

    existing_results: Dict[str, Dict[str, Any]] = {}
    if resume and output.exists():
        existing_results = _read_existing_results(output)
    todo = [item for item in problems if not (resume and item["problem_id"] in existing_results)]
    skipped = len(problems) - len(todo)

    mode = "a" if resume and output.exists() else "w"
    write_lock = threading.Lock()
    progress_lock = threading.Lock()
    limiter = RpmLimiter(rpm_limit)
    started_at = time.time()
    processed = 0
    fallback_count = 0

    _safe_print(f"Loaded {len(problems)} problems from {input_path}")
    _safe_print(
        f"Parallel todo={len(todo)}, skipped={skipped}, workers={workers}, rpm_limit={rpm_limit}"
    )
    _safe_print(f"Writing JSONL to {output}")
    _safe_print(f"Writing per-problem logs to {logs}")
    if dry_run:
        preview = [item["problem_id"] for item in todo[: min(10, len(todo))]]
        _safe_print("Dry run only; no API calls will be made.")
        _safe_print(f"First todo IDs: {preview}")
        return {
            "input_path": str(input_path),
            "output_jsonl": str(output),
            "total_loaded": len(problems),
            "todo": len(todo),
            "skipped_by_resume": skipped,
            "workers": workers,
            "rpm_limit": rpm_limit,
            "dry_run": True,
        }

    def solve_one(index: int, record: Dict[str, Any]) -> Tuple[int, str, bool]:
        nonlocal processed, fallback_count
        pid = record["problem_id"]
        text = record["problem_text"]
        metadata = record.get("raw_metadata", {})
        item_start = time.time()
        used_exception_fallback = False
        agent = RateLimitedAgent(
            limiter=limiter,
            model_type=model_type,
            api_key=api_key,
            api_base=api_base,
            config_path=config_path,
            ablation=ablation,
            official_mode=official_mode,
        )
        try:
            solution = agent.solve(text, pid, raw_metadata=metadata)
            run_log = dict(agent.last_run_log)
        except Exception as exc:
            used_exception_fallback = True
            solution = fallback_solution(pid, f"{type(exc).__name__}: {exc}")
            run_log = {
                "problem_id": pid,
                "raw_problem": text,
                "raw_metadata": metadata,
                "api_status": "parallel_batch_fallback_exception",
                "exception": repr(exc),
                "latency_seconds": round(time.time() - item_start, 3),
                "final_json": solution.model_dump(mode="json"),
            }

        is_fallback = (
            used_exception_fallback
            or solution.answer == "unable_to_determine"
            or not solution.verification.passed
        )
        line = solution_to_json(solution, indent=None)
        elapsed = time.time() - item_start
        with write_lock:
            with output.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
            _write_problem_log(logs, pid, run_log)
        with progress_lock:
            processed += 1
            if is_fallback:
                fallback_count += 1
            _safe_print(
                f"[{processed}/{len(todo)}] done {pid} in {elapsed:.1f}s "
                f"| passed={solution.verification.passed} "
                f"| conf={solution.verification.confidence:.2f} "
                f"| answer={solution.answer[:80]}"
            )
        return index, pid, is_fallback

    if mode == "w":
        output.write_text("", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(solve_one, index, record): record["problem_id"]
            for index, record in enumerate(todo, start=1)
        }
        for future in as_completed(futures):
            pid = futures[future]
            try:
                future.result()
            except Exception as exc:
                with progress_lock:
                    processed += 1
                    fallback_count += 1
                solution = fallback_solution(pid, f"{type(exc).__name__}: {exc}")
                run_log = {
                    "problem_id": pid,
                    "api_status": "parallel_future_fallback_exception",
                    "exception": repr(exc),
                    "final_json": solution.model_dump(mode="json"),
                }
                with write_lock:
                    with output.open("a", encoding="utf-8") as handle:
                        handle.write(solution_to_json(solution, indent=None) + "\n")
                        handle.flush()
                    _write_problem_log(logs, pid, run_log)
                _safe_print(f"[{processed}/{len(todo)}] fallback {pid}: {type(exc).__name__}: {exc}")

    all_results, schema_errors = _load_and_validate_results(output)
    results_json.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_summary = {
        "input_path": str(input_path),
        "output_jsonl": str(output),
        "output_json": str(results_json),
        "log_dir": str(logs),
        "total_loaded": len(problems),
        "processed_this_run": processed,
        "skipped_by_resume": skipped,
        "results_in_jsonl": len(all_results),
        "schema_error_count": len(schema_errors),
        "schema_errors": schema_errors[:20],
        "fallback_or_unpassed_count_this_run": fallback_count,
        "config_path": config_path,
        "ablation": ablation,
        "workers": workers,
        "rpm_limit": rpm_limit,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    summary.write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _safe_print("=" * 72)
    _safe_print(
        f"Parallel batch complete. Processed={processed}, skipped={skipped}, "
        f"results={len(all_results)}"
    )
    _safe_print(f"Schema errors={len(schema_errors)}")
    _safe_print(f"Final JSON: {results_json}")
    _safe_print(f"Summary: {summary}")
    return run_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel MathSolve-Agent batch runner")
    parser.add_argument("--input", "-i", required=True, help="Input JSON/JSONL/CSV/XLSX")
    parser.add_argument("--output", "-o", default="outputs/parallel/results.jsonl")
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--model", "-m", default="intern-s1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--limit", "-n", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ablation", default="safe")
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rpm-limit", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true", help="Show planned work without API calls")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    api_key, api_base = _resolve_api_config(args)
    run_parallel_batch(
        input_path=args.input,
        output_path=args.output,
        model_type=args.model,
        api_key=api_key,
        api_base=api_base,
        limit=args.limit,
        resume=args.resume,
        results_json_path=args.results_json,
        log_dir=args.log_dir,
        summary_path=args.summary,
        config_path=args.config,
        ablation=args.ablation,
        official_mode=args.official,
        workers=args.workers,
        rpm_limit=args.rpm_limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
