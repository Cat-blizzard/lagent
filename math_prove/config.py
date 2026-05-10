"""Runtime configuration for MathSolve-Agent."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SolverConfig:
    confidence_threshold: float = 0.70
    problem_timeout: float = 180.0
    sandbox_timeout: int = 10
    max_api_retries: int = 5
    max_attempts_easy: int = 1
    max_attempts_medium: int = 2
    max_attempts_hard: int = 3
    enable_sandbox: bool = True
    enable_ortools: bool = True
    enable_normalizer: bool = True
    enable_equivalence_check: bool = True
    enable_llm_verify: bool = True
    enable_extract_stage: bool = True
    enable_candidate_selection: bool = True
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


ABLATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "full": {},
    "no_sandbox": {"enable_sandbox": False, "enable_ortools": False},
    "no_ortools": {"enable_ortools": False},
    "no_normalizer": {"enable_normalizer": False, "enable_equivalence_check": False},
    "no_equivalence": {"enable_equivalence_check": False},
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
