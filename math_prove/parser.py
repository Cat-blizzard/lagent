"""Strict JSON parsing and Pydantic schemas for preliminary-round outputs."""

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

ANSWER_TYPES = {"formula", "numeric", "proof", "choice", "set", "text", "other"}
DIFFICULTIES = {"easy", "medium", "hard"}

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
    }
    raw = aliases.get(raw, raw)
    return raw if raw in ANSWER_TYPES else "other"


def normalize_difficulty(value: Any) -> str:
    raw = str(value or "medium").strip().lower()
    aliases = {"simple": "easy", "normal": "medium", "difficult": "hard"}
    raw = aliases.get(raw, raw)
    return raw if raw in DIFFICULTIES else "medium"


class VerificationResult(BaseModel):
    passed: bool = False
    confidence: float = 0.0
    issues: List[str] = Field(default_factory=list)
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
        if value is None:
            return []
        if isinstance(value, list):
            return [_trim(item, 300) for item in value if str(item).strip()]
        return [_trim(value, 300)] if str(value).strip() else []

    @field_validator("corrected_answer")
    @classmethod
    def corrected_answer_short(cls, value: str) -> str:
        return _trim(value, 1200)


class ClassificationResult(BaseModel):
    domain: str = "other"
    subtype: str = ""
    difficulty: str = "medium"
    answer_type: str = "other"
    required_methods: List[str] = Field(default_factory=list)
    solution_plan: List[str] = Field(default_factory=list)
    possible_pitfalls: List[str] = Field(default_factory=list)

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

    @field_validator("required_methods", "solution_plan", "possible_pitfalls", mode="before")
    @classmethod
    def list_fields(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [_trim(item, 300) for item in value if str(item).strip()]
        return [_trim(value, 300)] if str(value).strip() else []

    @field_validator("subtype")
    @classmethod
    def subtype_short(cls, value: str) -> str:
        return _trim(value, 120)


class CandidateSolution(BaseModel):
    candidate_id: str = "A"
    method: str = ""
    reasoning_summary: str = ""
    key_steps: List[str] = Field(default_factory=list)
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
        if value is None:
            return []
        if isinstance(value, list):
            return [_trim(item, 260) for item in value if str(item).strip()][:5]
        return [_trim(value, 260)] if str(value).strip() else []

    @field_validator("final_answer")
    @classmethod
    def final_answer_short(cls, value: str) -> str:
        return _trim(value, 1200)

    @field_validator("reasoning_summary")
    @classmethod
    def reasoning_short(cls, value: str) -> str:
        return _trim(value, 800)

    @field_validator("verification_code")
    @classmethod
    def code_short(cls, value: str) -> str:
        return _trim(value, 4000)


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
    def answer_short(cls, value: str) -> str:
        return _trim(value, 1200)

    @field_validator("reasoning_summary", "learning_hint")
    @classmethod
    def text_short(cls, value: str) -> str:
        return _trim(value, 800)


class MathSolution(BaseModel):
    """Final judgeable JSON object for one preliminary-round problem."""

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

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, value: Any) -> str:
        value = _trim(value, 1200)
        return value or "unable_to_determine"

    @field_validator("reasoning_summary", "learning_hint")
    @classmethod
    def summary_short(cls, value: str) -> str:
        return _trim(value, 800)

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
