import time

from math_prove.agent import MathSolverAgent
from math_prove.config import load_config, SolverConfig
from math_prove.normalizer import equivalent_answers
from math_prove.parser import CandidateSolution, ClassificationResult, VerificationResult


def test_common_math_formats_are_equivalent():
    cases = [
        (r"\sqrt{2}", "2**0.5", "numeric"),
        (r"\sin{x}", "sin(x)", "formula"),
        (r"\begin{pmatrix}1&2\\3&4\end{pmatrix}", "[[1,2],[3,4]]", "matrix"),
        ("B. 3/2", "B", "choice"),
        ("Option B", "B", "choice"),
        ("[2,1]", "{1,2}", "set"),
        ("1, 2", "(1,2)", "tuple"),
    ]

    for prediction, expected, answer_type in cases:
        result = equivalent_answers(prediction, expected, answer_type)
        assert result.equivalent, result


def test_tool_mismatch_can_fail_candidate_when_strict_equivalence_enabled():
    agent = object.__new__(MathSolverAgent)
    agent._config = SolverConfig(
        enable_llm_verify=False,
        enable_equivalence_check=True,
        equivalence_can_fail_candidate=True,
    )

    classification = ClassificationResult(
        domain="calculus_real_analysis",
        answer_type="numeric",
    )
    candidate = CandidateSolution(
        candidate_id="A",
        final_answer="1",
        answer_type="numeric",
    )
    tool_result = {
        "passed": True,
        "check_output": "2",
        "has_check_marker": True,
    }
    run_log = {}

    verification = agent._verify_candidate(
        problem="Compute 1+1.",
        classification=classification,
        candidate=candidate,
        tool_result=tool_result,
        run_log=run_log,
    )

    assert verification.passed is False
    assert verification.error_type == "calculation_error"
    assert run_log["local_equivalence_checks"][0]["equivalent"] is False


def test_extract_stage_does_not_upgrade_failed_candidate_verification():
    agent = object.__new__(MathSolverAgent)
    agent._config = SolverConfig(enable_extract_stage=False)
    agent._problem_timeout = 240.0

    classification = ClassificationResult(
        domain="calculus_real_analysis",
        answer_type="numeric",
    )
    candidate = CandidateSolution(
        candidate_id="A",
        final_answer="4",
        answer_type="numeric",
        reasoning_summary="Computed directly.",
        key_steps=["Add the terms."],
    )
    verification = VerificationResult(
        passed=False,
        confidence=0.9,
        issues=["tool verification failed"],
        error_type="calculation_error",
        corrected_answer="5",
    )

    solution = agent._extract_answer(
        problem_id="p1",
        problem="Compute 2+3.",
        classification=classification,
        candidate=candidate,
        verification=verification,
        run_log={},
        start_time=0.0,
    )

    assert solution.answer == "4"
    assert solution.verification.passed is False


def test_extract_stage_keeps_candidate_answer_and_uses_metadata():
    agent = object.__new__(MathSolverAgent)
    agent._config = SolverConfig(enable_extract_stage=True)
    agent._problem_timeout = 240.0

    def fake_call_stage(stage, messages, run_log):
        return """
        {
          "problem_id": "p2",
          "domain": "calculus_real_analysis",
          "answer": "5",
          "answer_type": "numeric",
          "reasoning_summary": "Extracted concise summary.",
          "key_steps": ["Extracted step"],
          "learning_hint": "Check the arithmetic target before simplifying.",
          "verification": {"passed": true, "confidence": 1.0, "issues": []}
        }
        """

    agent._call_stage = fake_call_stage
    classification = ClassificationResult(
        domain="calculus_real_analysis",
        answer_type="numeric",
    )
    candidate = CandidateSolution(
        candidate_id="A",
        final_answer="4",
        answer_type="numeric",
        reasoning_summary="Candidate summary.",
        key_steps=["Candidate step"],
    )
    verification = VerificationResult(passed=True, confidence=0.92, corrected_answer="4")
    run_log = {}

    solution = agent._extract_answer(
        problem_id="p2",
        problem="Compute 2+2.",
        classification=classification,
        candidate=candidate,
        verification=verification,
        run_log=run_log,
        start_time=time.time(),
    )

    assert solution.answer == "4"
    assert solution.reasoning_summary == "Extracted concise summary."
    assert solution.learning_hint == "Check the arithmetic target before simplifying."
    assert run_log["extract_answer_adapter"][0]["answer_source"] == "candidate"


def test_rule_router_sets_tool_policy_for_optimization_and_proof():
    opt = MathSolverAgent._heuristic_classification(
        "Maximize 3x + 2y subject to x + y <= 4 and x,y are nonnegative."
    )
    assert opt.domain == "operations_research_optimization"
    assert opt.tool_policy == "ortools"
    assert opt.needs_tool_verification is True

    proof = MathSolverAgent._heuristic_classification(
        "Prove that every compact subset of a Hausdorff space is closed."
    )
    assert proof.domain == "topology"
    assert proof.answer_type == "proof"
    assert proof.tool_policy == "direct"
    assert proof.needs_tool_verification is False

    combinatorics = MathSolverAgent._heuristic_classification(
        "How many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?"
    )
    assert combinatorics.domain == "combinatorics"
    assert combinatorics.answer_type == "numeric"
    assert combinatorics.tool_policy == "python"


def test_rule_prior_fills_missing_tool_policy():
    classification = ClassificationResult(
        domain="other",
        answer_type="other",
        tool_policy="direct",
        needs_tool_verification=False,
    )
    rule_prior = ClassificationResult(
        domain="linear_algebra",
        answer_type="matrix",
        tool_policy="sympy",
        needs_tool_verification=True,
        constraints_to_check=["matrix shape"],
        risk_points=["row order"],
    )

    merged = MathSolverAgent._merge_rule_prior(classification, rule_prior)

    assert merged.domain == "linear_algebra"
    assert merged.answer_type == "matrix"
    assert merged.tool_policy == "sympy"
    assert merged.needs_tool_verification is True
    assert merged.constraints_to_check == ["matrix shape"]


def test_official_stable_keeps_accuracy_guards_conservative():
    config = load_config(ablation="official_stable")

    assert config.official_mode is True
    assert config.enable_sandbox is False
    assert config.enable_equivalence_check is False
    assert config.equivalence_can_fail_candidate is False
    assert config.verifier_can_overwrite_answer is False
    assert config.enable_llm_verify is True
