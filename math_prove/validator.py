"""Local schema and answer-equivalence validation utilities."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from .normalizer import EquivalenceResult, equivalent_answers, normalize_answer
from .parser import MathSolution, parse_and_validate


LOW_QUALITY_ANSWER_PATTERNS = (
    "unable_to_determine",
    "i don't know",
    "i do not know",
    "not sure",
    "cannot solve",
    "can't solve",
    "unknown",
    "无法",
    "不能确定",
    "不确定",
    "不会",
)

MOJIBAKE_MARKERS = (
    "\ufffd",
    "Ã",
    "â",
    "鈥",
    "涓",
    "鐨",
    "棰",
    "瑙",
    "鍒",
)


@dataclass
class ValidationItem:
    problem_id: str
    schema_valid: bool
    preflight_passed: bool = True
    log_present: Optional[bool] = None
    answer_equivalent: Optional[bool] = None
    equivalence_method: str = ""
    llm_judge_correct: Optional[bool] = None
    llm_judge_confidence: Optional[float] = None
    llm_judge_reason: str = ""
    llm_judge_method: str = ""
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    total: int = 0
    schema_valid: int = 0
    schema_invalid: int = 0
    preflight_issue_count: int = 0
    duplicate_ids: List[str] = field(default_factory=list)
    missing_expected_ids: List[str] = field(default_factory=list)
    extra_result_ids: List[str] = field(default_factory=list)
    log_missing_count: int = 0
    answer_checked: int = 0
    answer_correct: int = 0
    answer_incorrect: int = 0
    llm_judge_checked: int = 0
    llm_judge_correct: int = 0
    llm_judge_incorrect: int = 0
    items: List[ValidationItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "schema_valid": self.schema_valid,
            "schema_invalid": self.schema_invalid,
            "preflight_issue_count": self.preflight_issue_count,
            "duplicate_ids": self.duplicate_ids,
            "missing_expected_ids": self.missing_expected_ids,
            "extra_result_ids": self.extra_result_ids,
            "log_missing_count": self.log_missing_count,
            "answer_checked": self.answer_checked,
            "answer_correct": self.answer_correct,
            "answer_incorrect": self.answer_incorrect,
            "llm_judge_checked": self.llm_judge_checked,
            "llm_judge_correct": self.llm_judge_correct,
            "llm_judge_incorrect": self.llm_judge_incorrect,
            "schema_valid_rate": self.schema_valid / self.total if self.total else 0.0,
            "answer_accuracy": (
                self.answer_correct / self.answer_checked if self.answer_checked else None
            ),
            "llm_judge_accuracy": (
                self.llm_judge_correct / self.llm_judge_checked
                if self.llm_judge_checked
                else None
            ),
            "items": [item.to_dict() for item in self.items],
        }


@dataclass
class LLMJudgeConfig:
    enabled: bool = False
    timeout: int = 60
    judge_all: bool = False
    judges: List[Dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_single(cls, model: str, api_key: str, api_base: str, **kwargs) -> "LLMJudgeConfig":
        return cls(
            enabled=True,
            judges=[{"model": model, "api_key": api_key, "api_base": api_base}],
            **kwargs,
        )

    def add_judge(self, model: str, api_key: str, api_base: str) -> None:
        self.judges.append({"model": model, "api_key": api_key, "api_base": api_base})


class _GenericLLM:
    """Minimal OpenAI-compatible chat client, no vendor assumptions."""

    def __init__(self, model: str, api_key: str, api_base: str, timeout: int = 120):
        self.model = model
        self.api_key = api_key
        self.url = api_base or "https://api.openai.com/v1/chat/completions"
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]]) -> str:
        data: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "n": 1,
            "temperature": 0.0,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = requests.post(
            self.url, headers=headers, data=json.dumps(data), timeout=self.timeout
        )
        body = resp.json()
        if "choices" in body:
            return str(body["choices"][0]["message"]["content"])
        if "error" in body:
            msg = str(body["error"].get("message", body["error"]))
            if "response_format" in msg.lower() or "json_object" in msg.lower():
                return self._chat_no_json_mode(messages)
            raise RuntimeError(msg)
        raise RuntimeError(resp.text[:200])

    def _chat_no_json_mode(self, messages: List[Dict[str, str]]) -> str:
        data: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "n": 1,
            "temperature": 0.0,
            "max_tokens": 512,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = requests.post(
            self.url, headers=headers, data=json.dumps(data), timeout=self.timeout
        )
        body = resp.json()
        if "choices" in body:
            return str(body["choices"][0]["message"]["content"])
        raise RuntimeError(str(body.get("error", body)))


JUDGE_SYSTEM_PROMPT = (
    "You are a strict mathematical answer judge. Given a math problem and a "
    "proposed answer, determine whether the answer is mathematically correct. "
    "For proof problems: check whether the answer states the correct conclusion. "
    "Ignore superficial formatting (LaTeX vs plain text, bracket styles, "
    "whitespace, Unicode). Do not penalize the answer for being concise — a "
    "one-sentence conclusion is acceptable even for a proof problem, as long "
    "as it correctly identifies what was to be proved. "
    "Return only JSON with keys: correct, confidence, reason."
)


def load_result_file(path: str) -> List[Dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON result file must contain a list")
        return data
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            rows.append({"problem_id": f"line_{line_no}", "_parse_error": str(exc)})
    return rows


def load_expected_file(path: str) -> Dict[str, Dict[str, Any]]:
    rows = load_result_file(path)
    expected: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        pid = str(row.get("problem_id") or row.get("id") or "").strip()
        if pid:
            expected[pid] = row
    return expected


def validate_results(
    result_path: str,
    expected_path: Optional[str] = None,
    log_dir: Optional[str] = None,
    strict_expected_ids: bool = True,
    llm_judge: Optional[LLMJudgeConfig] = None,
) -> ValidationReport:
    rows = load_result_file(result_path)
    expected = load_expected_file(expected_path) if expected_path else {}
    report = ValidationReport(total=len(rows))
    result_ids = [_row_problem_id(row, index) for index, row in enumerate(rows, start=1)]
    counts = Counter(pid for pid in result_ids if pid)
    duplicate_ids = {pid for pid, count in counts.items() if count > 1}
    report.duplicate_ids = sorted(duplicate_ids)

    if expected:
        expected_ids = set(expected.keys())
        result_id_set = set(result_ids)
        if strict_expected_ids:
            report.missing_expected_ids = sorted(expected_ids - result_id_set)
        else:
            report.missing_expected_ids = []
        report.extra_result_ids = sorted(result_id_set - expected_ids)
        report.preflight_issue_count += len(report.missing_expected_ids) + len(
            report.extra_result_ids
        )

    log_root = Path(log_dir) if log_dir else None

    for index, row in enumerate(rows, start=1):
        pid = result_ids[index - 1]
        item = ValidationItem(problem_id=pid, schema_valid=False)
        preflight_issues: List[str] = []

        if pid in duplicate_ids:
            preflight_issues.append("duplicate problem_id in result file")

        if log_root is not None:
            log_path = log_root / f"{_safe_log_name(pid)}.json"
            item.log_present = log_path.exists()
            if not item.log_present:
                preflight_issues.append(f"missing per-problem log: {log_path}")
                report.log_missing_count += 1

        if "_parse_error" in row:
            item.issues.append(f"JSON parse error: {row['_parse_error']}")
            item.issues.extend(preflight_issues)
            item.preflight_passed = not preflight_issues
            report.preflight_issue_count += len(preflight_issues)
            report.schema_invalid += 1
            report.items.append(item)
            continue

        try:
            solution = parse_and_validate(json.dumps(row, ensure_ascii=False), pid)
            item.schema_valid = True
            report.schema_valid += 1
        except Exception as exc:
            item.issues.append(f"schema error: {type(exc).__name__}: {exc}")
            item.issues.extend(preflight_issues)
            item.preflight_passed = not preflight_issues
            report.preflight_issue_count += len(preflight_issues)
            report.schema_invalid += 1
            report.items.append(item)
            continue

        answer_issues = _answer_preflight_issues(solution)
        preflight_issues.extend(answer_issues)
        item.issues.extend(preflight_issues)
        item.preflight_passed = not preflight_issues
        report.preflight_issue_count += len(preflight_issues)

        exp = expected.get(solution.problem_id)
        if exp is not None:
            expected_answer = exp.get("expected_answer", exp.get("answer"))
            answer_type = str(exp.get("answer_type") or solution.answer_type or "other")
            eq = equivalent_answers(solution.answer, expected_answer, answer_type)
            item.answer_equivalent = eq.equivalent
            item.equivalence_method = eq.method
            report.answer_checked += 1
            if eq.equivalent:
                report.answer_correct += 1
            else:
                report.answer_incorrect += 1
                item.issues.extend(eq.issues or ["answer mismatch"])
                item.issues.append(
                    f"pred={eq.normalized_prediction}; expected={eq.normalized_expected}"
                )

            if llm_judge and llm_judge.enabled:
                judge = judge_answer_with_llm(
                    problem=str(exp.get("problem_text") or exp.get("question") or exp.get("problem") or ""),
                    prediction=solution.answer,
                    expected=expected_answer,
                    answer_type=answer_type,
                    config=llm_judge,
                    local_equivalent=eq.equivalent,
                )
                item.llm_judge_correct = judge.get("correct")
                item.llm_judge_confidence = judge.get("confidence")
                item.llm_judge_reason = judge.get("reason", "")
                item.llm_judge_method = judge.get("method", "")
                if item.llm_judge_correct is not None:
                    report.llm_judge_checked += 1
                    if item.llm_judge_correct:
                        report.llm_judge_correct += 1
                    else:
                        report.llm_judge_incorrect += 1
                        item.issues.append("LLM judge marked answer incorrect")
                        if item.llm_judge_reason:
                            item.issues.append(f"LLM judge reason: {item.llm_judge_reason}")

        # Progress
        if llm_judge and llm_judge.enabled:
            jc = "✓" if item.llm_judge_correct else ("?" if item.llm_judge_correct is None else "✗")
        else:
            jc = "✓" if item.answer_equivalent else "✗"
        print(f"  [{index}/{report.total}] {pid}  local={'✓' if item.answer_equivalent else '✗'}  judge={jc}")

        report.items.append(item)

    return report


def _call_one_judge(
    problem: str,
    prediction: Any,
    llm: _GenericLLM,
) -> Optional[Dict[str, Any]]:
    """Call a single judge model, returning its verdict or None on error."""
    try:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "problem": problem,
                        "answer": str(prediction or ""),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        raw = llm.chat(messages)
        parsed = _parse_judge_json(raw)
        return {
            "correct": bool(parsed.get("correct")),
            "confidence": _clamp_float(parsed.get("confidence", 0.0)),
            "reason": str(parsed.get("reason", ""))[:500],
        }
    except Exception as exc:
        return None


def _multi_judge(
    problem: str,
    prediction: Any,
    config: LLMJudgeConfig,
    local_equivalent: bool = False,
) -> Dict[str, Any]:
    """Judge with local equivalence shortcut + optional multi-model voting."""
    if local_equivalent and not config.judge_all:
        return {
            "correct": True,
            "confidence": 1.0,
            "reason": "Accepted by local equivalence check.",
            "method": "local_equivalence_shortcut",
        }

    if not config.judges:
        return {
            "correct": None,
            "confidence": 0.0,
            "reason": "No LLM judge configured.",
            "method": "skipped_no_judge",
        }

    llms = [
        _GenericLLM(model=j["model"], api_key=j["api_key"], api_base=j.get("api_base", ""), timeout=config.timeout)
        for j in config.judges
    ]

    results: List[Optional[Dict[str, Any]]] = []
    for llm in llms:
        results.append(_call_one_judge(problem, prediction, llm))

    correct_votes = sum(1 for r in results if r is not None and r["correct"])
    valid = sum(1 for r in results if r is not None)
    passed = correct_votes > valid / 2 if valid > 0 else False
    confidences = [r["confidence"] for r in results if r is not None]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    reasons = [r["reason"] for r in results if r is not None and r["reason"]]
    models_used = [j["model"] for j in config.judges]

    individual = []
    for idx, (judge_cfg, r) in enumerate(zip(config.judges, results)):
        individual.append({
            "judge": judge_cfg["model"],
            "correct": r["correct"] if r else None,
            "confidence": r["confidence"] if r else None,
            "error": None if r else "judge_call_failed",
        })

    return {
        "correct": passed,
        "confidence": round(avg_conf, 4),
        "reason": "; ".join(reasons[:3]) if reasons else "multi-judge vote",
        "method": f"multi_judge:{','.join(models_used)}",
        "voting_detail": {
            "judges": len(llms),
            "valid": valid,
            "correct_votes": correct_votes,
            "majority_correct": passed,
            "individual": individual,
        },
    }


def judge_answer_with_llm(
    problem: str,
    prediction: Any,
    expected: Any,
    answer_type: str,
    config: LLMJudgeConfig,
    local_equivalent: bool = False,
) -> Dict[str, Any]:
    return _multi_judge(problem, prediction, config, local_equivalent)


def _parse_judge_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _clamp_float(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        number = 0.0
    return max(0.0, min(1.0, number))


def _row_problem_id(row: Dict[str, Any], index: int) -> str:
    return str(row.get("problem_id") or row.get("id") or f"row_{index}").strip()


def _safe_log_name(problem_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(problem_id)) or "unknown"


def _answer_preflight_issues(solution: MathSolution) -> List[str]:
    issues: List[str] = []
    answer = str(solution.answer or "").strip()
    lower_answer = answer.lower()

    if not answer:
        issues.append("empty answer")
    if any(pattern in lower_answer for pattern in LOW_QUALITY_ANSWER_PATTERNS):
        issues.append("low-quality or fallback answer")
    if _has_markdown_pollution(answer):
        issues.append("answer contains Markdown/code-fence pollution")
    if _looks_garbled(answer):
        issues.append("answer may contain garbled text/mojibake")

    explanation_text = " ".join(
        [
            str(solution.reasoning_summary or ""),
            str(solution.learning_hint or ""),
            " ".join(str(step) for step in solution.key_steps),
        ]
    )
    if _looks_garbled(explanation_text):
        issues.append("explanation fields may contain garbled text/mojibake")
    if len(solution.key_steps) > 5:
        issues.append("key_steps exceeds 5 items")
    if len(answer) > 500:
        issues.append("answer is too long for judge-friendly output")
    return issues


def _has_markdown_pollution(text: str) -> bool:
    stripped = text.strip()
    lines = stripped.splitlines()
    has_table_separator = any(
        re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", line)
        for line in lines
    )
    return (
        "```" in stripped
        or stripped.startswith("#")
        or bool(re.search(r"^\s*[-*]\s+", stripped))
        or has_table_separator
    )


def _looks_garbled(text: str) -> bool:
    if not text:
        return False
    marker_hits = sum(1 for marker in MOJIBAKE_MARKERS if marker in text)
    return marker_hits >= 1


def enrich_solution_log(solution: MathSolution) -> Dict[str, Any]:
    forms = normalize_answer(solution.answer, solution.answer_type)
    return {
        "solution": solution.model_dump(mode="json"),
        "answer_forms": forms.to_dict(),
    }


def write_validation_report(report: ValidationReport, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
