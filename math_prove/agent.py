"""Single-agent MathSolve-Agent solving pipeline."""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

from lagent.hooks import MessageLogger
from lagent.llms import GPTAPI
from lagent.memory import Memory
from lagent.schema import AgentMessage

from . import prompts
from .config import SolverConfig, load_config
from .normalizer import equivalent_answers, normalize_answer
from .parser import (
    CandidateSolution,
    ClassificationResult,
    MathSolution,
    SelectionResult,
    VerificationResult,
    build_json_prompt,
    fallback_solution,
    model_to_json,
    parse_and_validate,
    parse_model,
    validate_solution_dict,
)
from .sandbox import MathSandbox, Status


PROBLEM_TIMEOUT = 180.0
SANDBOX_TIMEOUT = 10
MAX_API_RETRIES = 5
CONFIDENCE_THRESHOLD = 0.70

TOOL_FRIENDLY_DOMAINS = {
    "linear_algebra",
    "calculus_real_analysis",
    "complex_analysis",
    "ordinary_differential_equations",
    "partial_differential_equations",
    "probability_statistics",
    "operations_research_optimization",
    "number_theory",
    "combinatorics",
    "discrete_mathematics",
    "geometry",
    "graph_theory",
    "numerical_analysis",
    "mathematical_modeling",
    "control_dynamical_systems",
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _compact(value: Any, limit: int = 2000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


class MathSolverAgent:
    """Intern-S1 based single math agent with explicit solve/verify stages."""

    def __init__(
        self,
        model_type: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 4096,
        retry: int = 3,
        sandbox_timeout: int = SANDBOX_TIMEOUT,
        problem_timeout: float = PROBLEM_TIMEOUT,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        config: Optional[SolverConfig] = None,
        config_path: Optional[str] = None,
        ablation: str = "full",
    ) -> None:
        self._config = config or load_config(config_path, ablation)
        if config is None and config_path is None:
            self._config.sandbox_timeout = sandbox_timeout
            self._config.problem_timeout = problem_timeout
            self._config.confidence_threshold = confidence_threshold

        key = api_key or os.environ.get("OPENAI_API_KEY", "ENV")
        base = api_base or os.environ.get(
            "LLM_API_BASE", "https://api.openai.com/v1/chat/completions"
        )
        self._llm = GPTAPI(
            model_type=model_type,
            key=key,
            api_base=base,
            retry=retry,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        self._sandbox = MathSandbox(timeout=self._config.sandbox_timeout)
        self._memory = Memory(recent_n=30)
        self._msg_logger = MessageLogger(name="math_prove", add_file_handler=True)
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._problem_timeout = self._config.problem_timeout
        self._confidence_threshold = self._config.confidence_threshold
        self.last_run_log: Dict[str, Any] = {}

    def solve(
        self,
        problem: str,
        problem_id: str = "0",
        raw_metadata: Optional[Dict[str, Any]] = None,
    ) -> MathSolution:
        """Solve one problem and always return a valid judgeable JSON object."""

        start = time.time()
        run_log: Dict[str, Any] = {
            "problem_id": str(problem_id),
            "timestamp": _now(),
            "raw_problem": problem,
            "raw_metadata": raw_metadata or {},
            "preprocessed_problem": "",
            "classification": {},
            "stages": [],
            "candidates": [],
            "retry_count": 0,
            "api_status": "success",
            "latency_seconds": 0.0,
            "final_json": {},
            "config": self._config.to_dict(),
        }
        self.last_run_log = run_log
        self._memory = Memory(recent_n=30)

        try:
            clean_problem = self._preprocess(problem)
            run_log["preprocessed_problem"] = clean_problem
            self._msg_logger.logger.info(
                f"[{problem_id}] start solving: {_compact(clean_problem, 120)}"
            )

            if not clean_problem:
                solution = fallback_solution(problem_id, "Empty problem text")
                run_log["api_status"] = "fallback_empty_problem"
                return self._finish(run_log, solution, start)

            classification = self._classify_and_plan(clean_problem, run_log)
            run_log["classification"] = classification.model_dump(mode="json")

            candidate, verification = self._solve_with_retries(
                clean_problem,
                classification,
                run_log,
                start,
            )

            solution = self._extract_answer(
                problem_id=str(problem_id),
                problem=clean_problem,
                classification=classification,
                candidate=candidate,
                verification=verification,
                run_log=run_log,
                start_time=start,
            )
            return self._finish(run_log, solution, start)

        except Exception as exc:
            run_log["api_status"] = "fallback_exception"
            run_log["exception"] = traceback.format_exc()
            domain = run_log.get("classification", {}).get("domain", "other")
            answer_type = run_log.get("classification", {}).get("answer_type", "other")
            solution = fallback_solution(
                problem_id=str(problem_id),
                reason=f"{type(exc).__name__}: {exc}",
                domain=domain,
                answer_type=answer_type,
            )
            return self._finish(run_log, solution, start)

    def _finish(
        self, run_log: Dict[str, Any], solution: MathSolution, start_time: float
    ) -> MathSolution:
        if self._config.enable_normalizer:
            forms = normalize_answer(solution.answer, solution.answer_type)
            solution.answer = forms.latex or solution.answer
            run_log["answer_forms"] = forms.to_dict()
        run_log["latency_seconds"] = round(time.time() - start_time, 3)
        run_log["final_json"] = solution.model_dump(mode="json")
        self.last_run_log = run_log
        return solution

    @staticmethod
    def _preprocess(problem: str) -> str:
        text = str(problem or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    def _classify_and_plan(
        self, problem: str, run_log: Dict[str, Any]
    ) -> ClassificationResult:
        messages = prompts.classification_messages(problem)
        try:
            raw = self._call_stage("classify_and_plan", messages, run_log)
            return self._parse_or_fix(raw, ClassificationResult, "classify_and_plan", run_log)
        except Exception as exc:
            run_log.setdefault("warnings", []).append(f"classification fallback: {exc}")
            return self._heuristic_classification(problem)

    def _solve_with_retries(
        self,
        problem: str,
        classification: ClassificationResult,
        run_log: Dict[str, Any],
        start_time: float,
    ) -> Tuple[CandidateSolution, VerificationResult]:
        max_attempts = self._config.attempts_for(classification.difficulty)
        previous_feedback = ""
        best_candidate: Optional[CandidateSolution] = None
        best_verification: Optional[VerificationResult] = None
        best_score = -1.0

        for attempt in range(1, max_attempts + 1):
            self._check_timeout(start_time)
            if attempt > 1:
                run_log["retry_count"] += 1

            candidate = self._solve_candidate(
                problem, classification, attempt, previous_feedback, run_log
            )
            if self._config.enable_normalizer:
                forms = normalize_answer(candidate.final_answer, candidate.answer_type)
                candidate.final_answer = forms.latex or candidate.final_answer
                run_log.setdefault("answer_forms_by_candidate", {})[
                    candidate.candidate_id
                ] = forms.to_dict()
            tool_result = self._maybe_run_sandbox(candidate, classification, run_log)
            verification = self._verify_candidate(
                problem, classification, candidate, tool_result, run_log
            )

            if verification.corrected_answer:
                candidate.final_answer = verification.corrected_answer

            candidate_record = candidate.model_dump(mode="json")
            candidate_record["tool_result"] = tool_result or {}
            candidate_record["verification"] = verification.model_dump(mode="json")
            run_log["candidates"].append(candidate_record)

            score = verification.confidence + (0.15 if verification.passed else 0.0)
            if candidate.final_answer and score > best_score:
                best_candidate = candidate
                best_verification = verification
                best_score = score

            if (
                candidate.final_answer
                and verification.passed
                and verification.confidence >= self._confidence_threshold
            ):
                return candidate, verification

            previous_feedback = "; ".join(verification.issues) or (
                "Verifier confidence was below threshold; retry with a different method."
            )
            previous_feedback = self._repair_feedback(verification, previous_feedback)

        if (
            self._config.enable_candidate_selection
            and classification.difficulty == "hard"
            and len(run_log["candidates"]) > 1
        ):
            selected = self._select_best(problem, classification, run_log)
            if selected is not None:
                candidate = CandidateSolution(
                    candidate_id=selected.selected_candidate_id,
                    method="candidate_comparison",
                    reasoning_summary=selected.reasoning_summary,
                    key_steps=selected.key_steps,
                    final_answer=selected.answer,
                    answer_type=classification.answer_type,
                    verification_code="",
                )
                return candidate, selected.verification

        if best_candidate is None:
            best_candidate = CandidateSolution(
                candidate_id="fallback",
                method="fallback",
                reasoning_summary="No reliable candidate was produced.",
                key_steps=[],
                final_answer="unable_to_determine",
                answer_type=classification.answer_type,
            )
        if best_verification is None:
            best_verification = VerificationResult(
                passed=False,
                confidence=0.0,
                issues=["No verifier result was available"],
                corrected_answer=best_candidate.final_answer,
            )
        return best_candidate, best_verification

    def _solve_candidate(
        self,
        problem: str,
        classification: ClassificationResult,
        attempt: int,
        previous_feedback: str,
        run_log: Dict[str, Any],
    ) -> CandidateSolution:
        messages = prompts.solve_messages(
            problem=problem,
            classification=classification.model_dump(mode="json"),
            attempt=attempt,
            previous_feedback=previous_feedback,
        )
        raw = self._call_stage(f"solve_candidate_{attempt}", messages, run_log)
        candidate = self._parse_or_fix(
            raw, CandidateSolution, f"solve_candidate_{attempt}", run_log
        )
        candidate.candidate_id = candidate.candidate_id or chr(ord("A") + attempt - 1)
        if not candidate.final_answer:
            candidate.final_answer = "unable_to_determine"
        if candidate.answer_type == "other":
            candidate.answer_type = classification.answer_type
        return candidate

    def _verify_candidate(
        self,
        problem: str,
        classification: ClassificationResult,
        candidate: CandidateSolution,
        tool_result: Optional[Dict[str, Any]],
        run_log: Dict[str, Any],
    ) -> VerificationResult:
        messages = prompts.verify_messages(
            problem=problem,
            classification=classification.model_dump(mode="json"),
            candidate=candidate.model_dump(mode="json"),
            tool_result=tool_result,
        )
        if not self._config.enable_llm_verify:
            has_judgeable_answer = bool(
                candidate.final_answer and candidate.final_answer != "unable_to_determine"
            )
            return VerificationResult(
                passed=has_judgeable_answer,
                confidence=0.6 if has_judgeable_answer else 0.0,
                issues=[] if has_judgeable_answer else ["empty or fallback candidate answer"],
                format_check={
                    "passed": has_judgeable_answer,
                    "issues": [] if has_judgeable_answer else ["empty or fallback answer"],
                },
                question_target_check={"passed": True, "issues": []},
                condition_check={"passed": True, "issues": []},
                result_check={"passed": True, "issues": []},
                judgeability_check={
                    "passed": has_judgeable_answer,
                    "issues": [] if has_judgeable_answer else ["answer is not judgeable"],
                },
                error_type="none" if has_judgeable_answer else "format_error",
                repair_instruction="" if has_judgeable_answer else "Return a concise non-empty final answer.",
                corrected_answer=candidate.final_answer,
            )

        try:
            raw = self._call_stage(f"verify_{candidate.candidate_id}", messages, run_log)
            verification = self._parse_or_fix(
                raw, VerificationResult, f"verify_{candidate.candidate_id}", run_log
            )
        except Exception as exc:
            issues = [f"Verifier fallback: {type(exc).__name__}: {exc}"]
            if tool_result and not tool_result.get("passed", True):
                issues.append("Tool verification failed")
            has_judgeable_answer = bool(
                candidate.final_answer and candidate.final_answer != "unable_to_determine"
            )
            verification = VerificationResult(
                passed=has_judgeable_answer,
                confidence=0.55 if has_judgeable_answer else 0.0,
                issues=issues,
                format_check={
                    "passed": has_judgeable_answer,
                    "issues": [] if has_judgeable_answer else ["empty or fallback answer"],
                },
                question_target_check={"passed": True, "issues": []},
                condition_check={"passed": True, "issues": []},
                result_check={
                    "passed": not (tool_result and not tool_result.get("passed", True)),
                    "issues": ["tool verification failed"]
                    if tool_result and not tool_result.get("passed", True)
                    else [],
                },
                judgeability_check={
                    "passed": has_judgeable_answer,
                    "issues": [] if has_judgeable_answer else ["answer is not judgeable"],
                },
                error_type="unknown" if issues else "none",
                repair_instruction="Review verifier fallback issues and produce a corrected concise answer.",
                corrected_answer=candidate.final_answer,
            )
        if not verification.corrected_answer:
            verification.corrected_answer = candidate.final_answer
        if self._config.enable_equivalence_check and tool_result and tool_result.get("passed"):
            local_eq = equivalent_answers(
                candidate.final_answer,
                tool_result.get("output", ""),
                candidate.answer_type or classification.answer_type,
            )
            run_log.setdefault("local_equivalence_checks", []).append(
                {
                    "candidate_id": candidate.candidate_id,
                    "against": "tool_output",
                    **local_eq.to_dict(),
                }
            )
            if local_eq.equivalent:
                verification.confidence = max(verification.confidence, 0.85)
            elif local_eq.method != "none":
                verification.passed = False
                verification.error_type = "calculation_error"
                verification.repair_instruction = (
                    "The candidate answer differs from executable verification output. "
                    "Recompute and reconcile the final answer with the tool result."
                )
                verification.issues.append("Candidate answer differs from tool output")
        return verification

    def _select_best(
        self,
        problem: str,
        classification: ClassificationResult,
        run_log: Dict[str, Any],
    ) -> Optional[SelectionResult]:
        messages = prompts.select_messages(
            problem=problem,
            classification=classification.model_dump(mode="json"),
            candidates=run_log["candidates"],
        )
        try:
            raw = self._call_stage("select_best_candidate", messages, run_log)
            return self._parse_or_fix(raw, SelectionResult, "select_best_candidate", run_log)
        except Exception as exc:
            run_log.setdefault("warnings", []).append(f"selection fallback: {exc}")
            return None

    def _extract_answer(
        self,
        problem_id: str,
        problem: str,
        classification: ClassificationResult,
        candidate: CandidateSolution,
        verification: VerificationResult,
        run_log: Dict[str, Any],
        start_time: float,
    ) -> MathSolution:
        self._check_timeout(start_time)
        if not self._config.enable_extract_stage:
            payload = {
                "problem_id": problem_id,
                "domain": classification.domain,
                "answer": candidate.final_answer or "unable_to_determine",
                "answer_type": candidate.answer_type or classification.answer_type,
                "reasoning_summary": candidate.reasoning_summary,
                "key_steps": candidate.key_steps,
                "learning_hint": self._fallback_learning_hint(classification.domain),
                "verification": verification.model_dump(mode="json"),
            }
            return validate_solution_dict(payload, problem_id)

        messages = prompts.extract_messages(
            problem_id=problem_id,
            problem=problem,
            classification=classification.model_dump(mode="json"),
            candidate=candidate.model_dump(mode="json"),
            verification=verification.model_dump(mode="json"),
        )
        try:
            raw = self._call_stage("extract_answer", messages, run_log)
            solution = parse_and_validate(raw, problem_id)
        except Exception as exc:
            run_log.setdefault("warnings", []).append(f"extract fallback: {exc}")
            payload = {
                "problem_id": problem_id,
                "domain": classification.domain,
                "answer": candidate.final_answer or "unable_to_determine",
                "answer_type": candidate.answer_type or classification.answer_type,
                "reasoning_summary": candidate.reasoning_summary,
                "key_steps": candidate.key_steps,
                "learning_hint": self._fallback_learning_hint(classification.domain),
                "verification": verification.model_dump(mode="json"),
            }
            solution = validate_solution_dict(payload, problem_id)

        if solution.domain == "other" and classification.domain != "other":
            solution.domain = classification.domain
        if solution.answer_type == "other" and classification.answer_type != "other":
            solution.answer_type = classification.answer_type
        solution.verification.confidence = max(
            solution.verification.confidence, verification.confidence
        )
        solution.verification.passed = solution.verification.passed or verification.passed
        if verification.issues:
            merged = list(solution.verification.issues)
            for issue in verification.issues:
                if issue not in merged:
                    merged.append(issue)
            solution.verification.issues = merged[:5]
        return solution

    def _maybe_run_sandbox(
        self,
        candidate: CandidateSolution,
        classification: ClassificationResult,
        run_log: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        code = (candidate.verification_code or "").strip()
        if not code:
            return None
        if not self._config.enable_sandbox:
            return {"skipped": True, "reason": "Sandbox disabled by config"}
        if not self._config.enable_ortools and "ortools" in code.lower():
            return {"skipped": True, "reason": "OR-Tools disabled by config"}
        if classification.domain not in TOOL_FRIENDLY_DOMAINS:
            return {"skipped": True, "reason": "Domain is not tool-friendly"}

        record = {
            "stage": f"sandbox_{candidate.candidate_id}",
            "started_at": _now(),
            "code": _compact(code, 4000),
        }
        try:
            result = self._sandbox.exec(code)
            passed = result.status == Status.SUCCESS
            payload = {
                "passed": passed,
                "status": str(result.status),
                "output": _compact(result.value if passed else result.msg, 4000),
            }
            record.update(payload)
            return payload
        except Exception as exc:
            payload = {
                "passed": False,
                "status": "exception",
                "output": _compact(f"{type(exc).__name__}: {exc}", 4000),
            }
            record.update(payload)
            return payload
        finally:
            record["finished_at"] = _now()
            run_log.setdefault("tool_runs", []).append(record)

    def _call_stage(
        self,
        stage: str,
        messages: List[Dict[str, str]],
        run_log: Dict[str, Any],
    ) -> str:
        record: Dict[str, Any] = {
            "stage": stage,
            "started_at": _now(),
            "messages": messages,
            "response": "",
            "error": "",
        }
        self._memory.add(
            [AgentMessage(sender=msg["role"], content=msg["content"]) for msg in messages]
        )
        try:
            response = self._call_llm(messages)
            record["response"] = response
            self._memory.add(AgentMessage(sender="assistant", content=response))
            return response
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["finished_at"] = _now()
            run_log.setdefault("stages", []).append(record)

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = MAX_API_RETRIES,
    ) -> str:
        last_error = ""
        max_retries = self._config.max_api_retries if max_retries == MAX_API_RETRIES else max_retries
        for attempt in range(max_retries):
            try:
                response = self._llm.chat(
                    messages,
                    temperature=self._temperature,
                    max_new_tokens=self._max_new_tokens,
                )
                return str(response).strip()
            except Exception as exc:
                last_error = str(exc)
                if attempt < max_retries - 1:
                    time.sleep(min(2**attempt + 1, 30))
        raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_error}")

    def _parse_or_fix(
        self,
        raw: str,
        model_cls: Type[Any],
        stage: str,
        run_log: Dict[str, Any],
    ) -> Any:
        try:
            return parse_model(raw, model_cls)
        except Exception as exc:
            fix_messages = prompts.json_fix_messages(
                raw_text=raw,
                error=str(exc),
                schema_hint=build_json_prompt(model_cls),
            )
            fixed = self._call_stage(f"{stage}_json_fix", fix_messages, run_log)
            return parse_model(fixed, model_cls)

    @staticmethod
    def _fallback_learning_hint(domain: str) -> str:
        if domain == "complex_analysis":
            return "First verify the singularities and theorem assumptions before computing."
        if domain == "partial_differential_equations":
            return "Check that the proposed solution satisfies both the equation and all conditions."
        if domain == "operations_research_optimization":
            return "State variables, constraints, and an optimality certificate explicitly."
        if domain == "topology":
            return "Work from the definitions and check boundary cases or counterexamples."
        return "Identify the applicable theorem conditions before applying a formula."

    @staticmethod
    def _repair_feedback(verification: VerificationResult, fallback: str) -> str:
        parts: List[str] = []
        if verification.error_type and verification.error_type != "none":
            parts.append(f"error_type={verification.error_type}")
        if verification.repair_instruction:
            parts.append(f"repair_instruction={verification.repair_instruction}")
        if verification.issues:
            parts.append("issues=" + "; ".join(verification.issues[:5]))
        for name in (
            "format_check",
            "question_target_check",
            "condition_check",
            "result_check",
            "judgeability_check",
        ):
            layer = getattr(verification, name)
            if not layer.passed or layer.issues:
                parts.append(
                    f"{name}: passed={layer.passed}; issues={'; '.join(layer.issues)}"
                )
        return "\n".join(parts) if parts else fallback

    @staticmethod
    def _heuristic_classification(problem: str) -> ClassificationResult:
        text = problem.lower()
        checks = [
            ("partial_differential_equations", ["pde", "partial differential", "heat equation", "wave equation", "laplace"]),
            ("ordinary_differential_equations", ["ode", "differential equation", "initial value"]),
            ("complex_analysis", ["complex", "residue", "contour", "holomorphic", "analytic", "cauchy"]),
            ("topology", ["topology", "compact", "connected", "homeomorphic", "open cover", "quotient"]),
            ("operations_research_optimization", ["linear programming", "maximize", "minimize", "constraint", "kkt", "optimal"]),
            ("probability_statistics", ["probability", "random variable", "distribution", "expectation", "variance"]),
            ("graph_theory", ["graph", "vertex", "edge", "matching", "coloring", "path"]),
            ("number_theory", ["integer", "prime", "mod", "congruence", "divisible"]),
            ("linear_algebra", ["matrix", "eigen", "vector", "rank", "linear transformation"]),
            ("calculus_real_analysis", ["integral", "derivative", "limit", "series", "continuous"]),
        ]
        domain = "other"
        for candidate, keywords in checks:
            if any(keyword in text for keyword in keywords):
                domain = candidate
                break
        answer_type = "proof" if any(word in text for word in ["prove", "show that", "证明"]) else "formula"
        difficulty = "hard" if len(problem) > 1200 else "medium"
        if len(problem) < 240:
            difficulty = "easy"
        return ClassificationResult(
            domain=domain,
            subtype="heuristic",
            goal="solve the stated problem",
            difficulty=difficulty,
            answer_type=answer_type,
            required_methods=[],
            solution_plan=["Understand the target", "Apply a suitable method", "Check the result"],
            possible_pitfalls=["Classification was produced by fallback heuristics"],
            constraints_to_check=["all stated conditions", "answer format"],
            risk_points=["heuristic diagnosis may miss a specific theorem condition"],
            needs_case_split=any(
                word in text for word in ["case", "parameter", "depending", "分类", "参数"]
            ),
            needs_tool_verification=domain in TOOL_FRIENDLY_DOMAINS and answer_type != "proof",
            expected_answer_shape=answer_type,
        )

    def _check_timeout(self, start_time: float) -> None:
        if time.time() - start_time > self._problem_timeout:
            raise TimeoutError(f"Problem exceeded {self._problem_timeout:.1f}s timeout")

    def _memory_to_openai(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        for msg in self._memory.get_memory():
            role = "assistant"
            if msg.sender == "system":
                role = "system"
            elif msg.sender in ("user", "environment"):
                role = "user"
            messages.append({"role": role, "content": str(msg.content)})
        return messages

    @staticmethod
    def dump_solution(solution: MathSolution) -> str:
        return model_to_json(solution)
