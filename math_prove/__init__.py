"""Preliminary-round MathSolve-Agent package."""

from .parser import (
    CandidateSolution,
    ClassificationResult,
    LogEntry,
    MathSolution,
    SelectionResult,
    VerificationResult,
    fallback_solution,
    parse_and_validate,
    solution_to_json,
)


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
    "VerificationResult",
    "SelectionResult",
    "LogEntry",
    "fallback_solution",
    "parse_and_validate",
    "solution_to_json",
]
