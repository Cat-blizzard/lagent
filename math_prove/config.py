"""Runtime configuration for MathSolve-Agent."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SolverConfig:
    confidence_threshold: float = 0.70
    problem_timeout: float = 240.0
    sandbox_timeout: int = 10
    max_api_retries: int = 5
    max_attempts_easy: int = 1
    max_attempts_medium: int = 2
    max_attempts_hard: int = 3
    enable_sandbox: bool = True
    enable_ortools: bool = True
    enable_normalizer: bool = True
    normalizer_overwrite_answer: bool = False
    enable_equivalence_check: bool = True
    equivalence_can_fail_candidate: bool = False
    enable_llm_verify: bool = True
    enable_extract_stage: bool = True
    enable_candidate_selection: bool = True
    verifier_can_overwrite_answer: bool = False
    verifier_correction_min_confidence: float = 0.80
    extract_must_match_candidate: bool = True
    official_mode: bool = False
    force_max_attempts: Optional[int] = None

    def attempts_for(self, difficulty: str) -> int:
        if self.force_max_attempts is not None:
            return max(1, int(self.force_max_attempts))
        mapping = {
            "easy": self.max_attempts_easy,
            "medium": self.max_attempts_medium,
            "hard": self.max_attempts_hard,
        }
        return max(1, int(mapping.get(difficulty, self.max_attempts_medium)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


BASE_PRESET: Dict[str, Any] = {
    "confidence_threshold": 0.70,
    "problem_timeout": 240.0,
    "sandbox_timeout": 8,
    "max_api_retries": 3,
    "max_attempts_easy": 1,
    "max_attempts_medium": 1,
    "max_attempts_hard": 1,
    "enable_sandbox": False,
    "enable_ortools": False,
    "enable_normalizer": False,
    "normalizer_overwrite_answer": False,
    "enable_equivalence_check": False,
    "equivalence_can_fail_candidate": False,
    "enable_llm_verify": False,
    "enable_extract_stage": False,
    "enable_candidate_selection": False,
    "verifier_can_overwrite_answer": False,
    "force_max_attempts": 1,
    "extract_must_match_candidate": True,
    "official_mode": False,
}

SAFE_PRESET: Dict[str, Any] = {
    **BASE_PRESET,
    "max_attempts_hard": 2,
    "enable_normalizer": True,
    "enable_llm_verify": True,
    "force_max_attempts": None,
}

SAFE_PLUS_PRESET: Dict[str, Any] = {
    **SAFE_PRESET,
    "problem_timeout": 240.0,
    "max_attempts_medium": 2,
    "enable_equivalence_check": True,
    "enable_extract_stage": True,
    "enable_candidate_selection": True,
}

STRONG_PRESET: Dict[str, Any] = {
    **SAFE_PLUS_PRESET,
    "confidence_threshold": 0.75,
    "problem_timeout": 240.0,
    "sandbox_timeout": 10,
    "max_api_retries": 5,
    "max_attempts_hard": 3,
    "enable_sandbox": True,
    "enable_ortools": True,
    "equivalence_can_fail_candidate": True,
    "verifier_can_overwrite_answer": True,
}


ABLATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "full": {},
    "base": BASE_PRESET,
    "base_verify": {**BASE_PRESET, "enable_llm_verify": True},
    "base_normalizer": {**BASE_PRESET, "enable_normalizer": True},
    "base_extract": {**BASE_PRESET, "enable_extract_stage": True},
    "base_normalizer_extract": {
        **BASE_PRESET,
        "enable_normalizer": True,
        "enable_extract_stage": True,
    },
    "base_multi": {
        **BASE_PRESET,
        "max_attempts_easy": 1,
        "max_attempts_medium": 2,
        "max_attempts_hard": 3,
        "enable_candidate_selection": True,
        "force_max_attempts": None,
    },
    "base_equivalence_observe": {
        **BASE_PRESET,
        "enable_normalizer": True,
        "enable_equivalence_check": True,
    },
    "base_equivalence_strict": {
        **BASE_PRESET,
        "enable_normalizer": True,
        "enable_equivalence_check": True,
        "equivalence_can_fail_candidate": True,
    },
    "base_sandbox_observe": {**BASE_PRESET, "enable_sandbox": True},
    "base_sandbox_verify": {
        **BASE_PRESET,
        "enable_sandbox": True,
        "enable_llm_verify": True,
    },
    "base_ortools_verify": {
        **BASE_PRESET,
        "enable_sandbox": True,
        "enable_ortools": True,
        "enable_llm_verify": True,
    },
    "safe": SAFE_PRESET,
    "safe_plus": SAFE_PLUS_PRESET,
    "strong": STRONG_PRESET,
    "official_stable": {
        **SAFE_PRESET,
        "official_mode": True,
    },
    "no_sandbox": {"enable_sandbox": False, "enable_ortools": False},
    "no_ortools": {"enable_ortools": False},
    "no_normalizer": {"enable_normalizer": False, "enable_equivalence_check": False},
    "no_equivalence": {"enable_equivalence_check": False},
    "strict_equivalence": {"equivalence_can_fail_candidate": True},
    "no_llm_verify": {"enable_llm_verify": False},
    "no_extract": {"enable_extract_stage": False},
    "single_candidate": {
        "force_max_attempts": 1,
        "enable_candidate_selection": False,
    },
}


def load_config(path: Optional[str] = None, ablation: str = "full") -> SolverConfig:
    data: Dict[str, Any] = {}
    if path:
        data.update(_load_config_file(Path(path)))
    if ablation:
        if ablation not in ABLATION_PRESETS:
            known = ", ".join(sorted(ABLATION_PRESETS))
            raise ValueError(f"Unknown ablation preset '{ablation}'. Known: {known}")
        data.update(ABLATION_PRESETS[ablation])
    return SolverConfig(**data)


def _load_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        value = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        value = _load_yaml_like(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    if not isinstance(value, dict):
        raise ValueError("Config file must contain an object")
    return value


def _load_yaml_like(text: str) -> Dict[str, Any]:
    """Load a small flat YAML subset without adding PyYAML as a hard dependency."""

    result: Dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        result[key] = _parse_scalar(value)
    return result


def _parse_scalar(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
