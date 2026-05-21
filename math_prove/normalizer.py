"""Answer normalization and lightweight equivalence checks."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, List, Optional


@dataclass
class AnswerForms:
    raw: str
    latex: str
    canonical: str
    answer_type: str = "other"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EquivalenceResult:
    equivalent: bool
    method: str
    confidence: float
    normalized_prediction: str
    normalized_expected: str
    issues: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


UNICODE_REPLACEMENTS = {
    "−": "-",
    "–": "-",
    "—": "-",
    "×": "*",
    "·": "*",
    "÷": "/",
    "π": "pi",
    "∞": "oo",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "∈": " in ",
    "，": ",",
    "；": ";",
    "：": ":",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "｛": "{",
    "｝": "}",
}


def normalize_answer(answer: Any, answer_type: str = "other") -> AnswerForms:
    raw = str(answer or "").strip()
    cleaned = strip_answer_wrappers(raw)
    latex = normalize_latex(cleaned)
    canonical = canonicalize(latex, answer_type=answer_type)
    return AnswerForms(raw=raw, latex=latex, canonical=canonical, answer_type=answer_type)


def strip_answer_wrappers(answer: str) -> str:
    text = answer.strip()
    text = re.sub(r"^```(?:json|latex|text|math)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    text = re.sub(
        r"^(final\s+answer|answer|答案|最终答案)\s*[:：]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    boxed = extract_boxed(text)
    if boxed:
        text = boxed
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    return text.strip()


def extract_boxed(text: str) -> str:
    for command in (r"\boxed", r"\fbox"):
        idx = text.find(command)
        if idx == -1:
            continue
        brace = text.find("{", idx)
        if brace == -1:
            continue
        extracted = _extract_braced(text, brace)
        if extracted:
            return extracted.strip()
    return ""


def _extract_braced(text: str, start: int) -> str:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    return ""


def normalize_latex(text: str) -> str:
    value = str(text or "").strip()
    for src, dst in UNICODE_REPLACEMENTS.items():
        value = value.replace(src, dst)
    value = value.replace("\\[", "").replace("\\]", "")
    value = value.replace("\\(", "").replace("\\)", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    for src, dst in {
        "\\cdot": "*",
        "\\times": "*",
        "\\div": "/",
        "\\leq": "<=",
        "\\le": "<=",
        "\\geq": ">=",
        "\\ge": ">=",
        "\\neq": "!=",
    }.items():
        value = value.replace(src, dst)
    value = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\operatorname\s*\{([^{}]*)\}", r"\1", value)
    value = value.replace("\\pi", "pi").replace("\\infty", "oo")
    value = _replace_latex_matrices(value)
    value = _replace_latex_frac(value)
    value = _replace_latex_sqrt(value)
    value = _replace_latex_functions(value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _replace_latex_matrices(text: str) -> str:
    def matrix_repl(match: re.Match[str]) -> str:
        body = match.group("body").strip()
        rows = [row.strip() for row in re.split(r"\\\\", body) if row.strip()]
        normalized_rows = []
        for row in rows:
            cells = [cell.strip() for cell in row.split("&")]
            normalized_rows.append(",".join(cells))
        return "[[" + "],[".join(normalized_rows) + "]]"

    matrix_pattern = re.compile(
        r"\\begin\{(?P<env>p?matrix|bmatrix|Bmatrix|vmatrix|Vmatrix|smallmatrix)\}"
        r"(?P<body>.*?)"
        r"\\end\{(?P=env)\}",
        flags=re.DOTALL,
    )
    value = matrix_pattern.sub(matrix_repl, text)

    array_pattern = re.compile(
        r"\\begin\{array\}(?:\{[^{}]*\})?"
        r"(?P<body>.*?)"
        r"\\end\{array\}",
        flags=re.DOTALL,
    )
    return array_pattern.sub(matrix_repl, value)


def _replace_latex_frac(text: str) -> str:
    pattern = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    value = text
    while previous != value:
        previous = value
        value = pattern.sub(r"(\1)/(\2)", value)
    return value


def _replace_latex_sqrt(text: str) -> str:
    pattern = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
    previous = None
    value = text
    while previous != value:
        previous = value
        value = pattern.sub(r"sqrt(\1)", value)
    value = re.sub(r"\\sqrt\s+([A-Za-z0-9_.]+)", r"sqrt(\1)", value)
    return value


_LATEX_FUNCTIONS = (
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "log",
    "ln",
    "exp",
)


def _replace_latex_functions(text: str) -> str:
    value = text
    for name in _LATEX_FUNCTIONS:
        value = re.sub(rf"\\{name}\s*\{{([^{{}}]+)\}}", rf"{name}(\1)", value)
        value = re.sub(rf"\\{name}\s+([A-Za-z0-9_.]+)", rf"{name}(\1)", value)
        value = value.replace(f"\\{name}", name)
    return value


def canonicalize(text: str, answer_type: str = "other") -> str:
    value = normalize_latex(text)
    value = value.strip()
    value = _drop_units(value)
    value = value.replace(" ", "")
    if not value:
        return ""

    if answer_type == "choice":
        return _canonical_choice(value)
    if answer_type == "numeric":
        numeric = _canonical_number(value)
        if numeric is not None:
            return numeric
    if answer_type == "matrix":
        matrix = _canonical_matrix(value)
        if matrix is not None:
            return matrix
    if answer_type in {"tuple", "vector"}:
        brackets = ("[", "]") if answer_type == "vector" else ("(", ")")
        sequence = _canonical_sequence(value, brackets=brackets)
        if sequence is not None:
            return sequence
    if answer_type == "set":
        set_value = _canonical_set(value, force=True)
        if set_value is not None:
            return set_value
    interval = _canonical_interval(value)
    if interval is not None:
        return interval
    matrix = _canonical_matrix(value)
    if matrix is not None:
        return matrix
    if answer_type in {"formula", "other"}:
        set_value = _canonical_set(value)
        if set_value is not None:
            return set_value

    expr = _canonical_sympy(value)
    return expr if expr is not None else value.lower()


def _canonical_choice(value: str) -> str:
    text = value.strip().upper()
    text = re.sub(r"^(OPTION|CHOICE|ANSWER|ANS)\s*[:.\u3001\uff1a]?\s*", "", text)
    match = re.match(r"^\(?([A-Z])\)?(?:[.\u3001:\uff1a\s]|$)", text)
    if match:
        return match.group(1)
    letters = re.findall(r"\b([A-Z])\b", text)
    if len(letters) == 1:
        return letters[0]
    return text


def equivalent_answers(
    prediction: Any,
    expected: Any,
    answer_type: str = "other",
    numeric_tol: float = 1e-8,
) -> EquivalenceResult:
    pred = normalize_answer(prediction, answer_type)
    exp = normalize_answer(expected, answer_type)
    issues: List[str] = []

    if pred.canonical == exp.canonical:
        return EquivalenceResult(True, "canonical_exact", 1.0, pred.canonical, exp.canonical, [])

    pred_num = _to_float(pred.canonical)
    exp_num = _to_float(exp.canonical)
    if pred_num is not None and exp_num is not None:
        ok = math.isclose(pred_num, exp_num, rel_tol=numeric_tol, abs_tol=numeric_tol)
        return EquivalenceResult(
            ok,
            "numeric_tolerance",
            0.98 if ok else 0.0,
            pred.canonical,
            exp.canonical,
            [] if ok else ["numeric values differ"],
        )

    sympy_result = _sympy_equivalent(pred.canonical, exp.canonical)
    if sympy_result is not None:
        return EquivalenceResult(
            sympy_result,
            "sympy_simplify",
            0.95 if sympy_result else 0.0,
            pred.canonical,
            exp.canonical,
            [] if sympy_result else ["sympy expressions are not equivalent"],
        )

    issues.append("no equivalence method matched")
    return EquivalenceResult(False, "none", 0.0, pred.canonical, exp.canonical, issues)


def _drop_units(value: str) -> str:
    return re.sub(r"(?<=\d)\s*(cm|mm|m|kg|g|s|sec|seconds?|units?)$", "", value, flags=re.I)


def _canonical_number(value: str) -> Optional[str]:
    number = _to_float(value)
    if number is None:
        return None
    if math.isfinite(number) and abs(number - round(number)) < 1e-12:
        return str(int(round(number)))
    return f"{number:.12g}"


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        pass
    safe = _safe_numeric_eval(value)
    if safe is not None:
        return safe
    try:
        import sympy as sp

        expr = sp.sympify(_sympy_ready(value))
        if expr.is_number:
            return float(expr.evalf())
    except Exception:
        return None
    return None


def _safe_numeric_eval(value: str) -> Optional[float]:
    text = value.replace("^", "**").replace("pi", str(math.pi)).replace("oo", "inf")
    if not re.fullmatch(r"[0-9eE+\-*/()., inf]+", text):
        return None
    try:
        tree = ast.parse(text, mode="eval")
        return float(_eval_numeric_ast(tree.body))
    except Exception:
        return None


def _eval_numeric_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id == "inf":
        return math.inf
    if isinstance(node, ast.UnaryOp):
        value = _eval_numeric_ast(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        left = _eval_numeric_ast(node.left)
        right = _eval_numeric_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    raise ValueError("unsupported numeric expression")


def _canonical_set(value: str, force: bool = False) -> Optional[str]:
    brackets = [("{", "}")]
    if force:
        brackets.extend([("[", "]"), ("(", ")")])
    wrapped = next(
        ((left, right) for left, right in brackets if value.startswith(left) and value.endswith(right)),
        None,
    )
    if not (wrapped or "," in value):
        return None
    inner = value[1:-1] if wrapped else value
    parts = _split_top_level(inner, ",")
    if len(parts) <= 1:
        return None
    canonical_parts = [canonicalize(part, "formula") for part in parts if part]
    return "{" + ",".join(sorted(canonical_parts)) + "}"


def _canonical_sequence(value: str, brackets: tuple[str, str] = ("(", ")")) -> Optional[str]:
    left, right = brackets
    inner = value
    for start, end in (("(", ")"), ("[", "]"), ("{", "}")):
        if value.startswith(start) and value.endswith(end):
            inner = value[1:-1]
            break
    parts = _split_top_level(inner, ",")
    if len(parts) <= 1:
        return None
    canonical_parts = [canonicalize(part, "formula") for part in parts if part]
    return left + ",".join(canonical_parts) + right


def _canonical_interval(value: str) -> Optional[str]:
    if len(value) < 5:
        return None
    if value[0] not in "[(" or value[-1] not in "])":
        return None
    inner = value[1:-1]
    parts = _split_top_level(inner, ",")
    if len(parts) != 2:
        return None
    left = canonicalize(parts[0], "formula")
    right = canonicalize(parts[1], "formula")
    return f"{value[0]}{left},{right}{value[-1]}"


def _canonical_matrix(value: str) -> Optional[str]:
    if not (value.startswith("[[") and value.endswith("]]")):
        return None
    rows = value[2:-2].split("],[")
    normalized_rows = []
    for row in rows:
        cells = _split_top_level(row, ",")
        normalized_rows.append(",".join(canonicalize(cell, "formula") for cell in cells))
    return "[[" + "],[".join(normalized_rows) + "]]"


def _canonical_sympy(value: str) -> Optional[str]:
    try:
        import sympy as sp

        expr = sp.sympify(_sympy_ready(value))
        return str(sp.factor(sp.simplify(expr)))
    except Exception:
        return None


def _sympy_equivalent(left: str, right: str) -> Optional[bool]:
    try:
        import sympy as sp

        a = sp.sympify(_sympy_ready(left))
        b = sp.sympify(_sympy_ready(right))
        return bool(sp.simplify(a - b) == 0)
    except Exception:
        return None


def _sympy_ready(value: str) -> str:
    text = value.replace("^", "**")
    text = text.replace("oo", "sp.oo")
    # sympify does not know the sp namespace here, so map back after protecting words.
    text = text.replace("sp.oo", "oo")
    text = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", text)
    text = re.sub(r"(?<=[A-Za-z)])(?=\d)", "*", text)
    return text


def _split_top_level(text: str, sep: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == sep and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts
