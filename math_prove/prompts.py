"""Prompt templates for the preliminary-round math solver.

The prompts keep one external agent identity while making the internal
workflow explicit: classify, solve, verify, extract.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional


DOMAINS = [
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
]


DOMAIN_STRATEGIES: Dict[str, str] = {
    "complex_analysis": (
        "Check analytic regions, singularity types, residues, contour "
        "orientation, theorem assumptions, principal values, and whether the "
        "answer needs real/imaginary parts."
    ),
    "partial_differential_equations": (
        "Classify the PDE, identify initial/boundary conditions, choose among "
        "separation of variables, characteristics, transforms, Green functions, "
        "or energy methods, and verify all conditions."
    ),
    "operations_research_optimization": (
        "State decision variables, objective, constraints, feasible region, "
        "optimality proof, and check whether LP, DP, KKT, duality, or graph "
        "algorithms are appropriate."
    ),
    "topology": (
        "Use definitions precisely. Check open/closed sets, compactness, "
        "connectedness, quotient spaces, homeomorphisms, fundamental groups, "
        "counterexamples, and both directions of equivalences."
    ),
    "functional_analysis": (
        "Check normed-space assumptions, completeness, boundedness, compactness, "
        "duality, weak convergence, and theorem hypotheses before applying them."
    ),
    "probability_statistics": (
        "Identify the random variables, distributions, independence assumptions, "
        "conditioning, support, estimators, and whether exact or asymptotic "
        "claims are required."
    ),
    "number_theory": (
        "Check divisibility, congruence classes, coprimality, parity, modular "
        "conditions, and whether constructive or impossibility proof is needed."
    ),
    "graph_theory": (
        "Identify vertices, edges, weights, connectivity, matching/flow/coloring "
        "structure, extremal constraints, and proof of optimality or uniqueness."
    ),
    "numerical_analysis": (
        "Check discretization, convergence, stability, truncation error, "
        "conditioning, and whether numerical evidence needs symbolic backing."
    ),
}


COMMON_SYSTEM = """\
You are MathSolve-Agent, a single Intern-S1 based mathematical reasoning agent.
Your priority is correctness and a judgeable structured result.

Rules:
- Do not invent conditions that are not in the problem.
- If cases are needed, discuss all relevant cases.
- Keep the final answer concise and easy to parse.
- Use local symbolic/numeric tools only as verification support when helpful.
- Output valid JSON only when a JSON schema is requested.
"""


CLASSIFY_SYSTEM = COMMON_SYSTEM + """\

Classify the problem and plan the solution. Output ONLY one JSON object:
{
  "domain": "one of the allowed domain ids",
  "subtype": "short subtype",
  "difficulty": "easy|medium|hard",
  "answer_type": "formula|numeric|proof|choice|set|text|other",
  "required_methods": ["method 1", "method 2"],
  "solution_plan": ["step 1", "step 2", "step 3"],
  "possible_pitfalls": ["pitfall 1", "pitfall 2"]
}
"""


SOLVE_SYSTEM = COMMON_SYSTEM + """\

Solve the problem according to the plan. Output ONLY one JSON object:
{
  "candidate_id": "A",
  "method": "short method name",
  "reasoning_summary": "concise explanation of the core reasoning",
  "key_steps": ["step 1", "step 2", "step 3"],
  "final_answer": "short final answer only",
  "answer_type": "formula|numeric|proof|choice|set|text|other",
  "verification_code": "optional short Python code for SymPy/NumPy/SciPy verification, or empty string"
}

Keep key_steps to at most 5 items. Put no long derivation in final_answer.
If the problem is a proof or topology-style task, verification_code may be empty.
"""


VERIFY_SYSTEM = COMMON_SYSTEM + """\

Verify the proposed solution. Output ONLY one JSON object:
{
  "passed": true,
  "confidence": 0.0,
  "issues": [],
  "corrected_answer": "short corrected answer, or same as candidate answer"
}

Use confidence from 0 to 1. Mark passed=false if assumptions, theorem conditions,
calculation, special cases, or answer format are doubtful.
"""


SELECT_SYSTEM = COMMON_SYSTEM + """\

Compare candidate solutions for the same problem and select the most reliable one.
Output ONLY one JSON object:
{
  "selected_candidate_id": "A",
  "answer": "short final answer only",
  "reasoning_summary": "concise reason for selection",
  "key_steps": ["step 1", "step 2", "step 3"],
  "learning_hint": "one sentence learning hint",
  "verification": {
    "passed": true,
    "confidence": 0.0,
    "issues": []
  }
}
"""


EXTRACT_SYSTEM = COMMON_SYSTEM + """\

Extract the final preliminary-round JSON. Output ONLY one JSON object:
{
  "problem_id": "string",
  "domain": "one of the allowed domain ids",
  "answer": "short final answer only",
  "answer_type": "formula|numeric|proof|choice|set|text|other",
  "reasoning_summary": "one concise sentence",
  "key_steps": ["step 1", "step 2", "step 3"],
  "learning_hint": "one concise learning hint",
  "verification": {
    "passed": true,
    "confidence": 0.0,
    "issues": []
  }
}

The answer field must not contain the full reasoning process.
"""


JSON_FIX_SYSTEM = """\
You repair invalid JSON without changing the mathematical meaning.
Output ONLY one valid JSON object. No markdown, no commentary.
"""


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def strategy_for(domain: str) -> str:
    return DOMAIN_STRATEGIES.get(
        domain,
        "Use rigorous definitions, check theorem assumptions, compute carefully, "
        "and make the final answer concise.",
    )


def classification_messages(problem: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {
            "role": "user",
            "content": (
                "Allowed domain ids:\n"
                + "\n".join(f"- {domain}" for domain in DOMAINS)
                + f"\n\nProblem:\n{problem}"
            ),
        },
    ]


def solve_messages(
    problem: str,
    classification: Dict[str, Any],
    attempt: int,
    previous_feedback: Optional[str] = None,
) -> List[Dict[str, str]]:
    domain = str(classification.get("domain", "other"))
    style = "primary method"
    if attempt == 2:
        style = "alternative method; do not repeat the first reasoning path"
    elif attempt >= 3:
        style = "direct answer correction and edge-case focused method"

    feedback = previous_feedback or "No previous feedback."
    return [
        {"role": "system", "content": SOLVE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Problem:\n{problem}\n\n"
                f"Classification and plan:\n{_json(classification)}\n\n"
                f"Domain-specific checks:\n{strategy_for(domain)}\n\n"
                f"Attempt: {attempt} ({style}).\n"
                f"Previous verifier feedback:\n{feedback}"
            ),
        },
    ]


def verify_messages(
    problem: str,
    classification: Dict[str, Any],
    candidate: Dict[str, Any],
    tool_result: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": VERIFY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Problem:\n{problem}\n\n"
                f"Classification:\n{_json(classification)}\n\n"
                f"Candidate solution:\n{_json(candidate)}\n\n"
                f"Tool verification result:\n{_json(tool_result or {})}\n\n"
                "Check whether the final answer is correct, concise, and judgeable."
            ),
        },
    ]


def select_messages(
    problem: str,
    classification: Dict[str, Any],
    candidates: Iterable[Dict[str, Any]],
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SELECT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Problem:\n{problem}\n\n"
                f"Classification:\n{_json(classification)}\n\n"
                f"Candidates:\n{_json(list(candidates))}\n\n"
                "Select the most reliable candidate."
            ),
        },
    ]


def extract_messages(
    problem_id: str,
    problem: str,
    classification: Dict[str, Any],
    candidate: Dict[str, Any],
    verification: Dict[str, Any],
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Problem ID: {problem_id}\n\n"
                f"Problem:\n{problem}\n\n"
                f"Classification:\n{_json(classification)}\n\n"
                f"Accepted candidate:\n{_json(candidate)}\n\n"
                f"Verification:\n{_json(verification)}"
            ),
        },
    ]


def json_fix_messages(raw_text: str, error: str, schema_hint: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": JSON_FIX_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Schema hint:\n{schema_hint}\n\n"
                f"Parser error:\n{error}\n\n"
                f"Invalid output:\n{raw_text}"
            ),
        },
    ]
