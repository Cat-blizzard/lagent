"""Bare-prompt baseline runner for Intern-S1 math experiments.

This script intentionally bypasses MathSolverAgent, lagent wrappers, verifier,
normalizer, sandbox, extraction, repair, and candidate selection. It is a clean
control-variable runner for measuring raw model + prompt behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .main import load_problems


DEFAULT_API_BASE = "https://chat.intern-ai.org.cn/api/v1/chat/completions"


PROMPTS: Dict[str, Dict[str, str]] = {
    "baseline_v0": {
        "system": (
            "You are a careful mathematics solver. Solve the problem and give "
            "the final answer clearly."
        ),
        "user": "Problem:\n{problem}\n\nSolve it and include the final answer.",
    },
    "concise_v1": {
        "system": (
            "You are a careful mathematics solver. Keep the solution concise. "
            "The last line must be exactly in the form: Final Answer: <answer>."
        ),
        "user": (
            "Problem:\n{problem}\n\n"
            "Solve the problem. Put only the shortest judgeable result after "
            "the final 'Final Answer:' label."
        ),
    },
    "json_v1": {
        "system": (
            "You are a careful mathematics solver. Return only one valid JSON "
            "object and no Markdown."
        ),
        "user": (
            "Problem:\n{problem}\n\n"
            "Return exactly this JSON shape:\n"
            '{"answer":"<short final answer>","reasoning_summary":"<brief method>"}'
        ),
    },
    "cot_v1": {
        "system": (
            "You are a careful mathematics solver. Reason step by step, but make "
            "the final answer short and judgeable. The last line must be exactly "
            "in the form: Final Answer: <answer>."
        ),
        "user": (
            "Problem:\n{problem}\n\n"
            "Work through the solution, then put only the final result after "
            "the final 'Final Answer:' label."
        ),
    },
}


def run_baseline(
    input_path: str,
    output_path: str,
    model: str = "intern-s1",
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    prompt_name: str = "baseline_v0",
    log_dir: Optional[str] = None,
    limit: Optional[int] = None,
    resume: bool = False,
    timeout: int = 120,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    if prompt_name not in PROMPTS:
        known = ", ".join(sorted(PROMPTS))
        raise ValueError(f"Unknown prompt '{prompt_name}'. Known prompts: {known}")

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or --api-key is required.")
    endpoint = normalize_chat_endpoint(api_base or os.environ.get("LLM_API_BASE") or DEFAULT_API_BASE)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = Path(log_dir) if log_dir else output.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    summary_path = output.parent / "run_summary.json"

    problems = load_problems(input_path)
    if limit is not None:
        problems = problems[:limit]

    existing = read_existing_results(output) if resume and output.exists() else {}
    mode = "a" if resume and output.exists() else "w"

    started_at = time.time()
    processed = 0
    skipped_resume = 0
    skipped_empty = 0
    success_count = 0
    error_count = 0

    print(f"Loaded {len(problems)} problems from {input_path}")
    print(f"Writing JSONL to {output}")
    print(f"Writing per-problem logs to {logs}")
    print(f"Prompt: {prompt_name}")

    with output.open(mode, encoding="utf-8") as handle:
        for index, record in enumerate(problems, start=1):
            pid = str(record["problem_id"])
            problem_text = str(record.get("problem_text") or "").strip()
            raw_metadata = record.get("raw_metadata", {})

            if resume and pid in existing:
                skipped_resume += 1
                print(f"[{index}/{len(problems)}] skip {pid} (resume)")
                continue

            if not problem_text:
                skipped_empty += 1
                log_record = build_log_record(
                    pid=pid,
                    problem_text=problem_text,
                    raw_metadata=raw_metadata,
                    prompt_name=prompt_name,
                    messages=[],
                    model=model,
                    raw_response="",
                    cleaned_response="",
                    parsed_answer="unable_to_determine",
                    api_status="skipped_empty_text",
                    error="empty text problem; image-only rows are skipped by baseline",
                    latency_seconds=0.0,
                )
                write_problem_log(logs, pid, log_record)
                print(f"[{index}/{len(problems)}] skip {pid} (empty text)")
                continue

            print(f"[{index}/{len(problems)}] baseline solving {pid} ...")
            item_start = time.time()
            messages = build_messages(prompt_name, problem_text)

            try:
                raw_response = call_chat_completion(
                    endpoint=endpoint,
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout,
                    max_retries=max_retries,
                )
                cleaned = clean_model_output(raw_response)
                answer = extract_answer(cleaned)
                status = "success"
                error = None
                success_count += 1
            except Exception as exc:
                raw_response = ""
                cleaned = ""
                answer = "unable_to_determine"
                status = "api_error"
                error = f"{type(exc).__name__}: {exc}"
                error_count += 1

            latency = round(time.time() - item_start, 3)
            result = {
                "problem_id": pid,
                "answer": answer,
                "raw_response": raw_response,
                "prompt_name": prompt_name,
                "latency_seconds": latency,
                "api_status": status,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()

            log_record = build_log_record(
                pid=pid,
                problem_text=problem_text,
                raw_metadata=raw_metadata,
                prompt_name=prompt_name,
                messages=messages,
                model=model,
                raw_response=raw_response,
                cleaned_response=cleaned,
                parsed_answer=answer,
                api_status=status,
                error=error,
                latency_seconds=latency,
            )
            write_problem_log(logs, pid, log_record)
            processed += 1
            print(f"  done in {latency:.1f}s | status={status} | answer={answer[:100]}")

    summary = {
        "input_path": str(input_path),
        "output_jsonl": str(output),
        "log_dir": str(logs),
        "model": model,
        "api_base": endpoint,
        "prompt_name": prompt_name,
        "total_loaded": len(problems),
        "processed_this_run": processed,
        "skipped_by_resume": skipped_resume,
        "skipped_empty_text": skipped_empty,
        "success_count": success_count,
        "error_count": error_count,
        "timeout": timeout,
        "max_retries": max_retries,
        "temperature": temperature,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 72)
    print(f"Baseline complete. Processed={processed}, success={success_count}, errors={error_count}")
    print(f"Summary: {summary_path}")
    return summary


def build_messages(prompt_name: str, problem: str) -> List[Dict[str, str]]:
    prompt = PROMPTS[prompt_name]
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"].format(problem=problem)},
    ]


def call_chat_completion(
    endpoint: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    timeout: int,
    max_retries: int,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(payload, ensure_ascii=False),
                timeout=timeout,
            )
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
                time.sleep(retry_after or min(2 ** attempt, 20))
                continue
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(2 ** attempt, 20))

    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")


def clean_model_output(raw: str) -> str:
    text = str(raw or "").lstrip("\ufeff").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.IGNORECASE | re.DOTALL)
    fenced = re.fullmatch(r"```(?:json|text)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    return text.strip()


def extract_answer(cleaned_response: str) -> str:
    text = str(cleaned_response or "").strip()
    if not text:
        return "unable_to_determine"

    json_answer = extract_answer_from_json(text)
    if json_answer:
        return json_answer

    label_patterns = [
        r"(?:Final\s+Answer|Final answer|Answer)\s*[:：]\s*(.+)",
        r"(?:最终答案|答案)\s*[:：]\s*(.+)",
    ]
    for pattern in label_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return trim_answer(matches[-1])

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "unable_to_determine"
    return trim_answer(lines[-1])


def extract_answer_from_json(text: str) -> str:
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        value = data.get("answer", data.get("final_answer"))
        if value is not None and str(value).strip():
            return trim_answer(value)
    return ""


def trim_answer(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^```(?:json|text)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    if (text.startswith("$") and text.endswith("$")) or (
        text.startswith("\\(") and text.endswith("\\)")
    ):
        text = text.strip("$")
        if text.startswith("\\(") and text.endswith("\\)"):
            text = text[2:-2].strip()
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text or "unable_to_determine"


def read_existing_results(path: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return results
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            pid = str(row.get("problem_id") or "").strip()
            if pid:
                results[pid] = row
    return results


def build_log_record(
    pid: str,
    problem_text: str,
    raw_metadata: Dict[str, Any],
    prompt_name: str,
    messages: List[Dict[str, str]],
    model: str,
    raw_response: str,
    cleaned_response: str,
    parsed_answer: str,
    api_status: str,
    error: Optional[str],
    latency_seconds: float,
) -> Dict[str, Any]:
    return {
        "problem_id": pid,
        "problem_text": problem_text,
        "raw_metadata": raw_metadata,
        "prompt_name": prompt_name,
        "messages": messages,
        "raw_response": raw_response,
        "cleaned_response": cleaned_response,
        "parsed_answer": parsed_answer,
        "api_status": api_status,
        "error": error,
        "latency_seconds": latency_seconds,
        "model": model,
    }


def write_problem_log(log_dir: Path, problem_id: str, log_record: Dict[str, Any]) -> None:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(problem_id)) or "unknown"
    path = log_dir / f"{safe_id}.json"
    path.write_text(json.dumps(log_record, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_chat_endpoint(api_base: str) -> str:
    base = str(api_base or DEFAULT_API_BASE).strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, min(float(value), 60.0))
    except Exception:
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bare prompt + log baseline runner for Intern-S1 math tests"
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSON/JSONL/CSV/XLSX")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL path")
    parser.add_argument("--log-dir", default=None, help="Per-problem log directory")
    parser.add_argument("--model", default="intern-s1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--prompt", default="baseline_v0", choices=sorted(PROMPTS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_baseline(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        prompt_name=args.prompt,
        log_dir=args.log_dir,
        limit=args.limit,
        resume=args.resume,
        timeout=args.timeout,
        max_retries=args.max_retries,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
