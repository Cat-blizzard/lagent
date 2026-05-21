"""Strict JSON parsing and Pydantic schemas for MathSolve-Agent outputs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator


DOMAIN_VALUES = {
    "linear_algebra",
    "calculus_real_analysis",
    "complex_analysis",
    "ordinary_differential_equations",
    "partial_differential_equations",
    "probability_statistics",
    "topology",
    "functional_analysis",
    "operations_research_optimization",
    "number_theory",
    "combinatorics",
    "discrete_mathematics",
    "geometry",
    "graph_theory",
    "numerical_analysis",
    "mathematical_modeling",
    "control_dynamical_systems",
    "other",
}

DOMAIN_ALIASES = {
    "higher_algebra": "linear_algebra",
    "advanced_algebra": "linear_algebra",
    "algebra": "linear_algebra",
    "linear algebra": "linear_algebra",
    "real_analysis": "calculus_real_analysis",
    "calculus": "calculus_real_analysis",
    "analysis": "calculus_real_analysis",
    "complex analysis": "complex_analysis",
    "ode": "ordinary_differential_equations",
    "ordinary_differential_equation": "ordinary_differential_equations",
    "ordinary differential equations": "ordinary_differential_equations",
    "pde": "partial_differential_equations",
    "partial_differential_equation": "partial_differential_equations",
    "partial differential equations": "partial_differential_equations",
    "probability": "probability_statistics",
    "statistics": "probability_statistics",
    "optimization": "operations_research_optimization",
    "operations_research": "operations_research_optimization",
    "or": "operations_research_optimization",
    "discrete math": "discrete_mathematics",
    "dynamical_systems": "control_dynamical_systems",
    "control": "control_dynamical_systems",
}

ANSWER_TYPES = {
    "formula",
    "numeric",
    "proof",
    "choice",
    "set",
    "interval",
    "matrix",
    "vector",
    "tuple",
    "text",
    "other",
}
DIFFICULTIES = {"easy", "medium", "hard"}
TOOL_POLICIES = {"direct", "sympy", "ortools", "python", "hybrid", "none"}
ERROR_TYPES = {
    "none",
    "missing_condition",
    "wrong_theorem_condition",
    "calculation_error",
    "missing_case_split",
    "answer_not_simplified",
    "not_answering_question",
    "boundary_condition_error",
    "domain_error",
    "proof_gap",
    "format_error",
    "unknown",
}
CLAIM_STATUSES = {"passed", "failed", "uncertain"}
CLAIM_CHECK_TYPES = {"symbolic", "numeric", "logical", "definition", "format", "tool", "other"}

T = TypeVar("T", bound=BaseModel)


def _trim(value: str, limit: int) -> str:
    value = str(value or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def normalize_domain(value: Any) -> str:
    raw = str(value or "other").strip()
    key = raw.lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    key = re.sub(r"_+", "_", key)
    if key in DOMAIN_VALUES:
        return key
    spaced = raw.lower().strip()
    if spaced in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[spaced]
    return DOMAIN_ALIASES.get(key, "other")


def normalize_answer_type(value: Any) -> str:
    raw = str(value or "other").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "number": "numeric",
        "integer": "numeric",
        "float": "numeric",
        "latex": "formula",
        "expression": "formula",
        "boolean": "choice",
        "multiple_choice": "choice",
        "range": "interval",
        "array": "matrix",
        "list": "tuple",
        "ordered_pair": "tuple",
        "ordered_tuple": "tuple",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in ANSWER_TYPES else "other"


def normalize_difficulty(value: Any) -> str:
    raw = str(value or "medium").strip().lower()
    aliases = {"simple": "easy", "normal": "medium", "difficult": "hard"}
    raw = aliases.get(raw, raw)
    return raw if raw in DIFFICULTIES else "medium"


def normalize_tool_policy(value: Any) -> str:
    raw = str(value or "direct").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "no_tool": "none",
        "none_needed": "none",
        "manual": "direct",
        "logic": "direct",
        "symbolic": "sympy",
        "numeric": "sympy",
        "numpy": "python",
        "scipy": "python",
        "or_tools": "ortools",
        "operations_research": "ortools",
        "mixed": "hybrid",
        "tool": "hybrid",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in TOOL_POLICIES else "direct"


def normalize_error_type(value: Any) -> str:
    raw = str(value or "none").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "missing_boundary_condition": "boundary_condition_error",
        "boundary_error": "boundary_condition_error",
        "wrong_domain": "domain_error",
        "format": "format_error",
        "not_answering": "not_answering_question",
        "not_answered": "not_answering_question",
        "wrong_calculation": "calculation_error",
        "compute_error": "calculation_error",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in ERROR_TYPES else "unknown"


def normalize_claim_status(value: Any) -> str:
    raw = str(value or "uncertain").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "pass": "passed",
        "ok": "passed",
        "valid": "passed",
        "fail": "failed",
        "invalid": "failed",
        "unknown": "uncertain",
        "not_checked": "uncertain",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in CLAIM_STATUSES else "uncertain"


def normalize_claim_check_type(value: Any) -> str:
    raw = str(value or "other").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "math": "logical",
        "logic": "logical",
        "definition_check": "definition",
        "executable": "tool",
        "program": "tool",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in CLAIM_CHECK_TYPES else "other"


def _stringify(value: Any, limit: int = 1200) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _trim(value, limit)
    if isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            text = str(value)
        return _trim(text, limit)
    return _trim(value, limit)


def _list_text(value: Any, limit: int = 300, max_items: int = 6) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_stringify(item, limit) for item in value if _stringify(item, limit)][:max_items]
    return [_stringify(value, limit)] if _stringify(value, limit) else []


class LayerCheck(BaseModel):
    passed: bool = True
    issues: List[str] = Field(default_factory=list)

    @field_validator("issues", mode="before")
    @classmethod
    def issues_as_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [_trim(item, 240) for item in value if str(item).strip()]
        return [_trim(value, 240)] if str(value).strip() else []


class ClaimCheck(BaseModel):
    claim: str = ""
    status: str = "uncertain"
    check_type: str = "other"
    reason: str = ""

    @field_validator("claim")
    @classmethod
    def claim_short(cls, value: Any) -> str:
        return _stringify(value, 300)

    @field_validator("status")
    @classmethod
    def status_allowed(cls, value: Any) -> str:
        return normalize_claim_status(value)

    @field_validator("check_type")
    @classmethod
    def check_type_allowed(cls, value: Any) -> str:
        return normalize_claim_check_type(value)

    @field_validator("reason")
    @classmethod
    def reason_short(cls, value: Any) -> str:
        return _stringify(value, 300)


class VerificationResult(BaseModel):
    passed: bool = False
    confidence: float = 0.0
    issues: List[str] = Field(default_factory=list)
    format_check: LayerCheck = Field(default_factory=LayerCheck)
    question_target_check: LayerCheck = Field(default_factory=LayerCheck)
    condition_check: LayerCheck = Field(default_factory=LayerCheck)
    result_check: LayerCheck = Field(default_factory=LayerCheck)
    judgeability_check: LayerCheck = Field(default_factory=LayerCheck)
    claim_checks: List[ClaimCheck] = Field(default_factory=list)
    error_type: str = "none"
    repair_instruction: str = ""
    corrected_answer: str = Field(default="", exclude=True)

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, value: float) -> float:
        try:
            value = float(value)
        except Exception:
            value = 0.0
        return max(0.0, min(1.0, value))

    @field_validator("issues", mode="before")
    @classmethod
    def issues_as_list(cls, value: Any) -> List[str]:
        return _list_text(value, 300, 8)

    @field_validator("claim_checks", mode="before")
    @classmethod
    def claim_checks_list(cls, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value[:8]
        if isinstance(value, dict):
            return [value]
        return [
            {
                "claim": _stringify(value, 300),
                "status": "uncertain",
                "check_type": "other",
                "reason": "",
            }
        ] if _stringify(value, 300) else []

    @field_validator("error_type")
    @classmethod
    def error_type_allowed(cls, value: Any) -> str:
        return normalize_error_type(value)

    @field_validator("repair_instruction")
    @classmethod
    def repair_instruction_short(cls, value: str) -> str:
        return _stringify(value, 500)

    @field_validator("corrected_answer")
    @classmethod
    def corrected_answer_short(cls, value: Any) -> str:
        return _stringify(value, 1200)


class ClassificationResult(BaseModel):
    domain: str = "other"
    subtype: str = ""
    goal: str = ""
    difficulty: str = "medium"
    answer_type: str = "other"
    required_methods: List[str] = Field(default_factory=list)
    solution_plan: List[str] = Field(default_factory=list)
    possible_pitfalls: List[str] = Field(default_factory=list)
    constraints_to_check: List[str] = Field(default_factory=list)
    risk_points: List[str] = Field(default_factory=list)
    needs_case_split: bool = False
    needs_tool_verification: bool = False
    tool_policy: str = "direct"
    expected_answer_shape: str = ""

    @field_validator("domain")
    @classmethod
    def domain_allowed(cls, value: Any) -> str:
        return normalize_domain(value)

    @field_validator("difficulty")
    @classmethod
    def difficulty_allowed(cls, value: Any) -> str:
        return normalize_difficulty(value)

    @field_validator("answer_type")
    @classmethod
    def answer_type_allowed(cls, value: Any) -> str:
        return normalize_answer_type(value)

    @field_validator("tool_policy")
    @classmethod
    def tool_policy_allowed(cls, value: Any) -> str:
        return normalize_tool_policy(value)

    @field_validator(
        "required_methods",
        "solution_plan",
        "possible_pitfalls",
        "constraints_to_check",
        "risk_points",
        mode="before",
    )
    @classmethod
    def list_fields(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [_trim(item, 300) for item in value if str(item).strip()]
        return [_trim(value, 300)] if str(value).strip() else []

    @field_validator("subtype", "goal", "expected_answer_shape")
    @classmethod
    def text_short(cls, value: str) -> str:
        return _trim(value, 120)


class CandidateSolution(BaseModel):
    candidate_id: str = "A"
    method: str = ""
    reasoning_summary: str = ""
    key_steps: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    target: str = ""
    derivation_steps: List[str] = Field(default_factory=list)
    checkable_claims: List[str] = Field(default_factory=list)
    final_answer: str = ""
    answer_type: str = "other"
    verification_code: str = ""

    @field_validator("answer_type")
    @classmethod
    def answer_type_allowed(cls, value: Any) -> str:
        return normalize_answer_type(value)

    @field_validator("key_steps", mode="before")
    @classmethod
    def key_steps_list(cls, value: Any) -> List[str]:
        return _list_text(value, 260, 5)

    @field_validator("assumptions", "derivation_steps", "checkable_claims", mode="before")
    @classmethod
    def checkable_lists(cls, value: Any) -> List[str]:
        return _list_text(value, 300, 6)

    @field_validator("final_answer", mode="before")
    @classmethod
    def final_answer_short(cls, value: Any) -> str:
        return _stringify(value, 1200)

    @field_validator("reasoning_summary", "target")
    @classmethod
    def reasoning_short(cls, value: Any) -> str:
        return _stringify(value, 800)

    @field_validator("verification_code")
    @classmethod
    def code_short(cls, value: Any) -> str:
        return _stringify(value, 4000)


class SelectionResult(BaseModel):
    selected_candidate_id: str = "A"
    answer: str = ""
    reasoning_summary: str = ""
    key_steps: List[str] = Field(default_factory=list)
    learning_hint: str = ""
    verification: VerificationResult = Field(default_factory=VerificationResult)

    @field_validator("key_steps", mode="before")
    @classmethod
    def key_steps_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [_trim(item, 260) for item in value if str(item).strip()][:5]
        return [_trim(value, 260)] if str(value).strip() else []

    @field_validator("answer")
    @classmethod
    def answer_short(cls, value: Any) -> str:
        return _stringify(value, 1200)

    @field_validator("reasoning_summary", "learning_hint")
    @classmethod
    def text_short(cls, value: Any) -> str:
        return _stringify(value, 800)


class MathSolution(BaseModel):
    """Final judgeable JSON object for one math problem."""

    problem_id: str
    domain: str = "other"
    answer: str = "unable_to_determine"
    answer_type: str = "other"
    reasoning_summary: str = ""
    key_steps: List[str] = Field(default_factory=list)
    learning_hint: str = ""
    verification: VerificationResult = Field(default_factory=VerificationResult)

    @field_validator("problem_id")
    @classmethod
    def problem_id_not_empty(cls, value: Any) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("problem_id cannot be empty")
        return value

    @field_validator("domain")
    @classmethod
    def domain_allowed(cls, value: Any) -> str:
        return normalize_domain(value)

    @field_validator("answer_type")
    @classmethod
    def answer_type_allowed(cls, value: Any) -> str:
        return normalize_answer_type(value)

    @field_validator("answer", mode="before")
    @classmethod
    def answer_not_empty(cls, value: Any) -> str:
        value = _stringify(value, 1200)
        return value or "unable_to_determine"

    @field_validator("reasoning_summary", "learning_hint")
    @classmethod
    def summary_short(cls, value: Any) -> str:
        return _stringify(value, 800)

    @field_validator("key_steps", mode="before")
    @classmethod
    def key_steps_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [_trim(item, 260) for item in value if str(item).strip()][:5]
        return [_trim(value, 260)] if str(value).strip() else []

    @property
    def final_answer(self) -> str:
        return self.answer

    @property
    def is_solved(self) -> bool:
        return bool(self.verification.passed and self.answer != "unable_to_determine")

    @property
    def logs(self) -> List[Any]:
        return []


class LogEntry(BaseModel):
    """Backward-compatible compact log entry for callers that still import it."""

    step: int = 1
    thought: str = ""
    action: str = ""
    observation: str = ""


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from free-form text."""

    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    return None


def parse_json_object(raw_text: str) -> Dict[str, Any]:
    json_str = extract_json_from_text(raw_text)
    if json_str is None:
        raise ValueError("No JSON object found in model output")
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse failed: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not an object")
    return data


def parse_model(raw_text: str, model: Type[T]) -> T:
    data = parse_json_object(raw_text)
    return model(**data)


def _coerce_solution_payload(data: Dict[str, Any], problem_id: str) -> Dict[str, Any]:
    data = dict(data)
    data["problem_id"] = str(data.get("problem_id") or problem_id)
    if "answer" not in data and "final_answer" in data:
        data["answer"] = data.get("final_answer")
    if "reasoning_summary" not in data and "reasoning_process" in data:
        data["reasoning_summary"] = data.get("reasoning_process")
    if "verification" not in data:
        data["verification"] = {
            "passed": bool(data.get("is_solved", False)),
            "confidence": 0.0,
            "issues": [],
            "corrected_answer": data.get("answer", ""),
        }
    data.setdefault("domain", "other")
    data.setdefault("answer", "unable_to_determine")
    data.setdefault("answer_type", "other")
    data.setdefault("reasoning_summary", "")
    data.setdefault("key_steps", [])
    data.setdefault("learning_hint", "")
    return data


def parse_and_validate(raw_text: str, problem_id: str, max_retries: int = 2) -> MathSolution:
    """Parse a model response or JSON string into the final MathSolution schema."""

    del max_retries
    data = _coerce_solution_payload(parse_json_object(raw_text), problem_id)
    try:
        solution = MathSolution(**data)
    except ValidationError as exc:
        raise ValueError(f"Schema validation failed: {exc}") from exc
    if solution.problem_id != str(problem_id):
        solution.problem_id = str(problem_id)
    return solution


def validate_solution_dict(data: Dict[str, Any], problem_id: str) -> MathSolution:
    payload = _coerce_solution_payload(data, problem_id)
    return MathSolution(**payload)


def fallback_solution(
    problem_id: str,
    reason: str = "",
    domain: str = "other",
    answer_type: str = "other",
) -> MathSolution:
    issue = _trim(reason, 300) if reason else "Unable to determine a reliable answer"
    return MathSolution(
        problem_id=str(problem_id),
        domain=domain,
        answer="unable_to_determine",
        answer_type=answer_type,
        reasoning_summary="The system could not produce a reliable final answer.",
        key_steps=[],
        learning_hint="Check the problem conditions and rerun with a stricter method.",
        verification=VerificationResult(
            passed=False,
            confidence=0.0,
            issues=[issue],
            corrected_answer="unable_to_determine",
        ),
    )


def solution_to_json(solution: MathSolution, indent: Optional[int] = 2) -> str:
    """Serialize a final solution as strict JSON."""

    return json.dumps(solution.model_dump(mode="json"), ensure_ascii=False, indent=indent)


def model_to_json(data: BaseModel, indent: Optional[int] = 2) -> str:
    return json.dumps(data.model_dump(mode="json"), ensure_ascii=False, indent=indent)


def build_json_prompt(schema: Type[BaseModel] = MathSolution) -> str:
    fields = []
    for name, field in schema.model_fields.items():
        desc = field.description or field.annotation
        fields.append(f'- "{name}": {desc}')
    return "Return exactly one JSON object with these fields:\n" + "\n".join(fields)
