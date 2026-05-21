"""Prompt templates for MathSolve-Agent.

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
        "algorithms are appropriate. When the problem is LP/IP/CP or scheduling, "
        "use OR-Tools style verification code if it is concise."
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


DOMAIN_VERIFIER_RUBRICS: Dict[str, str] = {
    "complex_analysis": (
        "Complex analysis rubric: verify all singularities, whether poles are "
        "inside the contour, residue calculations, contour orientation, real-axis "
        "or principal-value issues, Jordan lemma / arc estimates, and parameter ranges."
    ),
    "partial_differential_equations": (
        "PDE rubric: verify PDE type, all initial/boundary conditions, substitution "
        "back into the equation, boundary satisfaction, uniqueness/regularity claims, "
        "and domain assumptions."
    ),
    "ordinary_differential_equations": (
        "ODE rubric: substitute the solution into the ODE, check initial/boundary "
        "conditions, constants, domains, singular points, and uniqueness assumptions."
    ),
    "operations_research_optimization": (
        "Optimization rubric: verify variable definitions, objective, all constraints, "
        "candidate feasibility, objective value, KKT/duality/DP recurrence when relevant, "
        "and proof of global optimality."
    ),
    "topology": (
        "Topology rubric: verify definitions, both directions of equivalences, whether "
        "a counterexample is needed, distinction between general topological and metric "
        "spaces, and correct use of compactness/connectedness/countability."
    ),
}


DOMAIN_SOLVE_ADDENDA: Dict[str, str] = {
    "complex_analysis": (
        "Complex analysis focus: explicitly check analytic domain, poles and "
        "singularities, contour orientation, residues, real-axis or principal-value "
        "issues, and parameter ranges before giving the final answer."
    ),
    "partial_differential_equations": (
        "PDE focus: identify equation type, initial/boundary conditions, domain, "
        "solution method, and verify by substitution into the PDE plus every "
        "condition. Mention uniqueness or regularity only when justified."
    ),
    "operations_research_optimization": (
        "Optimization focus: define variables, objective, constraints, feasibility, "
        "objective value, and a global optimality certificate. For concise LP/IP/CP "
        "or scheduling checks, verification_code may use OR-Tools."
    ),
    "topology": (
        "Topology focus: reason from definitions, separate general topological "
        "spaces from metric-space assumptions, prove both directions when needed, "
        "and construct counterexamples when the statement is false."
    ),
}


ERROR_TYPE_HINTS = [
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
]


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

Diagnose the problem and plan the solution. Output ONLY one JSON object:
{
  "domain": "one of the allowed domain ids",
  "subtype": "short subtype",
  "goal": "what the problem asks for",
  "difficulty": "easy|medium|hard",
  "answer_type": "formula|numeric|proof|choice|set|interval|matrix|vector|tuple|text|other",
  "required_methods": ["method 1", "method 2"],
  "solution_plan": ["step 1", "step 2", "step 3"],
  "possible_pitfalls": ["pitfall 1", "pitfall 2"],
  "constraints_to_check": ["condition, boundary, parameter, domain, or theorem assumption"],
  "risk_points": ["likely error point 1", "likely error point 2"],
  "needs_case_split": false,
  "needs_tool_verification": true,
  "tool_policy": "direct|sympy|ortools|python|hybrid|none",
  "expected_answer_shape": "scalar|set|interval|matrix|proof conclusion|choice|..."
}
"""


SOLVE_SYSTEM = COMMON_SYSTEM + """\

Solve the problem according to the plan. Output ONLY one JSON object:
{
  "candidate_id": "A",
  "method": "short method name",
  "reasoning_summary": "concise explanation of the core reasoning",
  "key_steps": ["step 1", "step 2", "step 3"],
  "assumptions": ["condition explicitly used from the problem"],
  "target": "the exact quantity, statement, or object to compute/prove",
  "derivation_steps": ["short checkable derivation step 1", "short step 2"],
  "checkable_claims": ["claim that a verifier or tool can check"],
  "final_answer": "short final answer only",
  "answer_type": "formula|numeric|proof|choice|set|interval|matrix|vector|tuple|text|other",
  "verification_code": "optional short Python code for SymPy/NumPy/SciPy verification, or empty string"
}

Keep key_steps to at most 5 items. Put no long derivation in final_answer.
Use assumptions/target/derivation_steps/checkable_claims as a compact,
verifier-friendly intermediate form. For easy numeric questions these may be
short; for proof, topology, abstract algebra, or hard questions, fill them with
the actual logical structure of the solution.
If the problem is a proof or topology-style task, verification_code may be empty.
If you write verification_code, keep it short and make the last relevant output a
single clean line exactly like:
print("FINAL_RESULT_FOR_CHECK:", clean_value)
The clean_value must be the comparable final value only, without Eq(...), debug
text, derivation, or wrappers.
"""


VERIFY_SYSTEM = COMMON_SYSTEM + """\

Verify the proposed solution. Output ONLY one JSON object:
{
  "passed": true,
  "confidence": 0.0,
  "issues": [],
  "format_check": {"passed": true, "issues": []},
  "question_target_check": {"passed": true, "issues": []},
  "condition_check": {"passed": true, "issues": []},
  "result_check": {"passed": true, "issues": []},
  "judgeability_check": {"passed": true, "issues": []},
  "claim_checks": [
    {
      "claim": "one checkable claim from the candidate",
      "status": "passed|failed|uncertain",
      "check_type": "symbolic|numeric|logical|definition|format|tool|other",
      "reason": "short reason"
    }
  ],
  "error_type": "none",
  "repair_instruction": "",
  "corrected_answer": "short corrected answer, or same as candidate answer"
}

Use confidence from 0 to 1. Mark passed=false if assumptions, theorem conditions,
calculation, special cases, or the ability to judge the answer are doubtful.
Cosmetic formatting issues should go in format_check.issues and issues, but do
not make passed=false when question_target_check, condition_check, result_check,
and judgeability_check all pass.
Check the candidate's assumptions, target, derivation_steps, and
checkable_claims explicitly. A failed claim should produce a concrete
repair_instruction; an uncertain claim is a warning unless it blocks judging the
answer.
Allowed error_type values: none, missing_condition, wrong_theorem_condition,
calculation_error, missing_case_split, answer_not_simplified,
not_answering_question, boundary_condition_error, domain_error, proof_gap,
format_error, unknown.
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

Extract the final judgeable JSON. Output ONLY one JSON object:
{
  "problem_id": "string",
  "domain": "one of the allowed domain ids",
  "answer": "short final answer only",
  "answer_type": "formula|numeric|proof|choice|set|interval|matrix|vector|tuple|text|other",
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
Do not change the mathematical content of the accepted candidate answer. Only
compress or reformat it when the result is clearly equivalent.
The learning_hint must be specific to this problem. Base it on the actual method,
possible_pitfalls, risk_points, verification.issues, or error_type. Do not use a
generic hint such as "check theorem conditions" unless it names the concrete
condition or pitfall in this problem.
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


def verifier_rubric_for(domain: str) -> str:
    return DOMAIN_VERIFIER_RUBRICS.get(
        domain,
        "General verifier rubric: check answer format, question target, all stated "
        "conditions, theorem assumptions, computations, missing cases, proof gaps, "
        "and whether the final answer is judgeable.",
    )


def solve_system_for(domain: str) -> str:
    addendum = DOMAIN_SOLVE_ADDENDA.get(str(domain or "other"), "")
    if not addendum:
        return SOLVE_SYSTEM
    return SOLVE_SYSTEM + "\nDomain-specific system focus:\n" + addendum


def classification_messages(
    problem: str, rule_prior: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    prior_text = ""
    if rule_prior:
        prior_text = (
            "\n\nRule-first router prior:\n"
            + _json(rule_prior)
            + "\nUse this as a strong baseline. Correct it only when the problem "
            "statement clearly supports a better domain, answer_type, difficulty, "
            "or tool_policy."
        )
    return [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {
            "role": "user",
            "content": (
                "Allowed domain ids:\n"
                + "\n".join(f"- {domain}" for domain in DOMAINS)
                + "\n\nAllowed tool_policy values:\n"
                "- direct: no tool needed; logical or short reasoning\n"
                "- sympy: symbolic/numeric algebra, calculus, matrices, equations\n"
                "- ortools: LP/IP/CP/scheduling/operations research optimization\n"
                "- python: discrete search, graph checks, brute-force enumeration\n"
                "- hybrid: use reasoning plus possible symbolic/numeric support\n"
                "- none: tools are not useful or unsafe for this problem\n"
                + f"\n\nProblem:\n{problem}"
                + prior_text
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
        {"role": "system", "content": solve_system_for(domain)},
        {
            "role": "user",
            "content": (
                f"Problem:\n{problem}\n\n"
                f"Problem diagnosis and plan:\n{_json(classification)}\n\n"
                "Treat the diagnosis as a strong but revisable hypothesis. "
                "If the problem statement supports a better domain, method, or "
                "answer shape, correct the diagnosis in your reasoning and solve "
                "according to the actual conditions. Respect tool_policy as a "
                "tool-use preference, not as a substitute for mathematical "
                "reasoning.\n\n"
                f"Domain-specific checks:\n{strategy_for(domain)}\n\n"
                f"Attempt: {attempt} ({style}).\n"
                f"Previous verifier feedback and repair instruction:\n{feedback}"
            ),
        },
    ]


def verify_messages(
    problem: str,
    classification: Dict[str, Any],
    candidate: Dict[str, Any],
    tool_result: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    domain = str(classification.get("domain", "other"))
    return [
        {"role": "system", "content": VERIFY_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Problem:\n{problem}\n\n"
                f"Classification:\n{_json(classification)}\n\n"
                f"Domain verifier rubric:\n{verifier_rubric_for(domain)}\n\n"
                f"Candidate solution:\n{_json(candidate)}\n\n"
                f"Tool verification result:\n{_json(tool_result or {})}\n\n"
                "Perform layered checks: format_check, question_target_check, "
                "condition_check, result_check, and judgeability_check. Then inspect "
                "each checkable_claim and return claim_checks. If any layer or claim "
                "fails, set a specific error_type and a concrete repair_instruction."
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
