"""MathSolve-Agent package."""

from .parser import (
    CandidateSolution,
    ClaimCheck,
    ClassificationResult,
    LogEntry,
    MathSolution,
    SelectionResult,
    VerificationResult,
    fallback_solution,
    parse_and_validate,
    solution_to_json,
)
from .config import SolverConfig, load_config
from .normalizer import AnswerForms, EquivalenceResult, equivalent_answers, normalize_answer
from .validator import ValidationReport, validate_results


def __getattr__(name):
    if name == "MathSolverAgent":
        from .agent import MathSolverAgent

        return MathSolverAgent
    if name == "MathSandbox":
        from .sandbox import MathSandbox

        return MathSandbox
    raise AttributeError(name)


__all__ = [
    "MathSandbox",
    "MathSolverAgent",
    "MathSolution",
    "ClassificationResult",
    "CandidateSolution",
    "ClaimCheck",
    "VerificationResult",
    "SelectionResult",
    "LogEntry",
    "fallback_solution",
    "parse_and_validate",
    "solution_to_json",
    "SolverConfig",
    "load_config",
    "AnswerForms",
    "EquivalenceResult",
    "normalize_answer",
    "equivalent_answers",
    "ValidationReport",
    "validate_results",
]
