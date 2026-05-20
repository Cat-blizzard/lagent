"""Convert external math benchmarks into MathSolve-Agent JSONL.

The generated JSONL can be used both as solver input and as an expected-answer
file for ``math_prove.evaluate`` because each row contains ``problem_text`` and
``expected_answer``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


UGMATH_TYPE_MAP = {
    "NV": "numeric",
    "INT": "numeric",
    "EX": "formula",
    "EQ": "formula",
    "OE": "formula",
    "TF": "choice",
    "MCS": "choice",
    "MCM": "set",
    "OL": "text",
    "UOL": "set",
}

THEOREMQA_TYPE_MAP = {
    "integer": "numeric",
    "float": "numeric",
    "bool": "choice",
    "boolean": "choice",
    "option": "choice",
    "list of integer": "text",
    "list of float": "text",
    "list": "text",
}

MATHBENCH_TYPE_MAP = {
    "single_choice": "choice",
    "single-choice": "choice",
    "multiple_choice": "choice",
    "multiple-choice": "choice",
    "choice": "choice",
    "cloze": "formula",
    "fill-in-the-blank": "formula",
    "fill_in_the_blank": "formula",
    "problem-solving": "formula",
    "problem_solving": "formula",
    "proof": "proof",
}

UGMATH_DOMAIN_MAP = {
    "abstract_algebra": "linear_algebra",
    "algebra": "linear_algebra",
    "arithmetic": "calculus_real_analysis",
    "calculus_-_multivariable": "calculus_real_analysis",
    "calculus_-_single_variable": "calculus_real_analysis",
    "combinatorics": "combinatorics",
    "complex_analysis": "complex_analysis",
    "differential_equations": "ordinary_differential_equations",
    "financial_mathematics": "mathematical_modeling",
    "geometry": "geometry",
    "linear_algebra": "linear_algebra",
    "number_theory": "number_theory",
    "probability": "probability_statistics",
    "set_theory_and_logic": "discrete_mathematics",
    "statistics": "probability_statistics",
    "trigonometry": "calculus_real_analysis",
}


def convert_ugmathbench(
    input_dir: Path,
    output_path: Path,
    limit: Optional[int] = None,
    versions: Iterable[int] = (1, 2, 3),
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for source_file in sorted(input_dir.glob("*.json")):
        data = json.loads(source_file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {source_file}")
        for item in data:
            for version in versions:
                problem = str(item.get(f"problem_v{version}") or "").strip()
                if not problem:
                    continue
                answer = item.get(f"answer_v{version}")
                answer_types = item.get(f"answer_type_v{version}")
                options = item.get(f"options_v{version}")
                row = _ugmath_row(item, source_file, version, problem, answer, answer_types, options)
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    return _write_rows(output_path, rows, "ugmathbench")
    return _write_rows(output_path, rows, "ugmathbench")


def _ugmath_row(
    item: Dict[str, Any],
    source_file: Path,
    version: int,
    problem: str,
    answer: Any,
    answer_types: Any,
    options: Any,
) -> Dict[str, Any]:
    original_id = str(item.get("id") or source_file.stem).strip()
    subject = str(item.get("subject") or source_file.stem).strip()
    answer_type = _map_ugmath_answer_type(answer, answer_types)
    prompt = _append_options(problem, options)
    return {
        "problem_id": f"ugmath_{original_id}_v{version}",
        "problem_text": prompt,
        "domain": _domain_from_ugmath_subject(subject),
        "answer_type": answer_type,
        "expected_answer": _format_answer(answer, answer_type),
        "raw_metadata": {
            "benchmark": "UGMathBench",
            "source_file": source_file.name,
            "original_id": original_id,
            "variant": f"v{version}",
            "subject": subject,
            "topic": item.get("topic", ""),
            "subtopic": item.get("subtopic", ""),
            "level": item.get("level", ""),
            "keywords": item.get("keywords", []),
            "original_answer_type": answer_types,
        },
    }


def convert_theoremqa(
    input_path: Path,
    output_path: Path,
    limit: Optional[int] = None,
    skip_images: bool = True,
) -> Dict[str, Any]:
    records = _read_parquet_records(input_path)
    rows: List[Dict[str, Any]] = []
    skipped_images = 0
    for index, item in enumerate(records):
        picture = item.get("Picture")
        if skip_images and _has_picture(picture):
            skipped_images += 1
            continue
        question = str(item.get("Question") or "").strip()
        if not question:
            continue
        answer_type = _map_theoremqa_answer_type(item.get("Answer_type"))
        rows.append(
            {
                "problem_id": f"theoremqa_{index:04d}",
                "problem_text": question,
                "domain": "other",
                "answer_type": answer_type,
                "expected_answer": _format_answer(item.get("Answer"), answer_type),
                "raw_metadata": {
                    "benchmark": "TheoremQA",
                    "row_index": index,
                    "original_answer_type": item.get("Answer_type", ""),
                    "has_picture": _has_picture(picture),
                },
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    summary = _write_rows(output_path, rows, "theoremqa")
    summary["skipped_image_rows"] = skipped_images
    return summary


def convert_mathbench(
    input_dir: Path,
    output_path: Path,
    limit: Optional[int] = None,
    language: str = "all",
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    scanned_files = 0
    for source_file in sorted(_iter_mathbench_files(input_dir)):
        scanned_files += 1
        for index, item in enumerate(_read_json_records(source_file)):
            question = _first_text(item, ("question", "problem", "prompt", "input", "query"))
            answer = _first_text(item, ("answer", "target", "label", "gold", "final_answer"))
            if not question or not answer:
                continue
            if language != "all" and not _mathbench_language_matches(item, question, language):
                continue
            rows.append(_mathbench_row(item, source_file, input_dir, index, question, answer))
            if limit is not None and len(rows) >= limit:
                summary = _write_rows(output_path, rows, "mathbench")
                summary["scanned_files"] = scanned_files
                return summary

    if not rows:
        raise SystemExit(
            "No MathBench question rows were found. The current directory looks like the "
            "MathBench source repository, not the release data. Download/extract the "
            "release dataset so that a folder such as 'mathbench_v1' containing "
            "cloze.json and/or single_choice.json exists, then rerun this command."
        )
    summary = _write_rows(output_path, rows, "mathbench")
    summary["scanned_files"] = scanned_files
    return summary


def _iter_mathbench_files(input_dir: Path) -> Iterator[Path]:
    if input_dir.is_file():
        if input_dir.suffix.lower() in {".json", ".jsonl"}:
            yield input_dir
        return
    for pattern in ("*.jsonl", "*.json"):
        for path in input_dir.rglob(pattern):
            if ".git" in path.parts:
                continue
            yield path


def _read_json_records(path: Path) -> Iterator[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return
    if text.startswith("["):
        data = json.loads(text)
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict):
                yield item
        return
    if text.startswith("{") and "\n" not in text:
        data = json.loads(text)
        if isinstance(data, dict):
            for item in _records_from_json_dict(data):
                yield item
        return
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path} line {line_no}: {exc}") from exc
        if isinstance(item, dict):
            yield item


def _records_from_json_dict(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    if any(key in data for key in ("question", "problem", "answer", "target")):
        yield data
        return
    for key in ("data", "items", "examples", "questions", "test"):
        value = data.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item


def _mathbench_row(
    item: Dict[str, Any],
    source_file: Path,
    root: Path,
    index: int,
    question: str,
    answer: str,
) -> Dict[str, Any]:
    options = item.get("options", item.get("choices", []))
    question_type = str(item.get("question_type") or item.get("type") or source_file.stem).strip()
    answer_type = _map_mathbench_answer_type(question_type, options, answer)
    rel = source_file.relative_to(root) if root.is_dir() else source_file.name
    rel_stem = str(rel).replace("\\", "_").replace("/", "_").replace(".", "_")
    original_id = str(item.get("id") or item.get("question_id") or item.get("qid") or index).strip()
    prompt = _append_mathbench_options(question, options)
    return {
        "problem_id": f"mathbench_{rel_stem}_{original_id}",
        "problem_text": prompt,
        "domain": _domain_from_mathbench_item(item, source_file),
        "answer_type": answer_type,
        "expected_answer": _format_scalar(answer),
        "raw_metadata": {
            "benchmark": "MathBench",
            "source_file": str(rel),
            "row_index": index,
            "original_id": original_id,
            "question_type": question_type,
            "stage": item.get("stage", item.get("level", "")),
            "category": item.get("category", item.get("subject", "")),
            "sub_category": item.get("sub_category", item.get("subfield", "")),
            "language": item.get("language", item.get("lang", "")),
        },
    }


def _map_mathbench_answer_type(question_type: str, options: Any, answer: Any = "") -> str:
    if isinstance(options, list) and options:
        return "choice"
    if _looks_numeric(_format_scalar(answer)):
        return "numeric"
    raw = question_type.lower().strip().replace(" ", "_")
    return MATHBENCH_TYPE_MAP.get(raw, "formula")


def _domain_from_mathbench_item(item: Dict[str, Any], source_file: Path) -> str:
    text = " ".join(
        str(item.get(key, ""))
        for key in ("subject", "category", "sub_category", "field", "topic", "question_type")
    )
    text = f"{text} {source_file}".lower()
    mapping = [
        ("ordinary differential", "ordinary_differential_equations"),
        ("differential equations", "ordinary_differential_equations"),
        ("partial differential", "partial_differential_equations"),
        ("complex", "complex_analysis"),
        ("graph", "graph_theory"),
        ("discrete", "discrete_mathematics"),
        ("set theory", "discrete_mathematics"),
        ("probability", "probability_statistics"),
        ("statistics", "probability_statistics"),
        ("arithmetic", "calculus_real_analysis"),
        ("linear", "linear_algebra"),
        ("algebra", "linear_algebra"),
        ("calculus", "calculus_real_analysis"),
        ("analysis", "calculus_real_analysis"),
        ("combinator", "combinatorics"),
        ("number", "number_theory"),
        ("geometry", "geometry"),
        ("topology", "topology"),
        ("optimization", "operations_research_optimization"),
    ]
    for marker, domain in mapping:
        if marker in text:
            return domain
    return "other"


def _append_mathbench_options(question: str, options: Any) -> str:
    if not isinstance(options, list) or not options:
        return question
    lines = []
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for index, option in enumerate(options):
        label = labels[index] if index < len(labels) else str(index + 1)
        lines.append(f"{label}. {_format_scalar(option)}")
    return question.rstrip() + "\nOptions:\n" + "\n".join(lines)


def _mathbench_language_matches(item: Dict[str, Any], question: str, language: str) -> bool:
    lang = str(item.get("language") or item.get("lang") or "").lower()
    if lang:
        return lang.startswith(language.lower())
    has_cjk = any("\u4e00" <= char <= "\u9fff" for char in question)
    if language.lower() in {"zh", "cn", "chinese"}:
        return has_cjk
    if language.lower() in {"en", "english"}:
        return not has_cjk
    return True


def _first_text(item: Dict[str, Any], keys: Iterable[str]) -> str:
    lower_map = {str(key).lower(): key for key in item.keys()}
    for key in keys:
        actual = lower_map.get(key.lower())
        if actual is not None:
            value = item.get(actual)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _read_parquet_records(input_path: Path) -> List[Dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Reading TheoremQA parquet requires pandas + pyarrow. Install with:\n"
            "  uv pip install pandas pyarrow\n"
            "Then rerun this converter."
        ) from exc
    try:
        frame = pd.read_parquet(input_path)
    except ImportError as exc:
        raise SystemExit(
            "Reading TheoremQA parquet requires a parquet engine. Install with:\n"
            "  uv pip install pyarrow\n"
            "Then rerun this converter."
        ) from exc
    return frame.fillna("").to_dict(orient="records")


def _map_ugmath_answer_type(answer: Any, answer_types: Any) -> str:
    flat_types = [str(value).strip().upper() for value in _flatten(answer_types) if str(value).strip()]
    answer_count = len([value for value in _flatten(answer) if str(value).strip()])
    if answer_count > 1 and not any(value in {"UOL", "MCM"} for value in flat_types):
        return "text"
    mapped = [UGMATH_TYPE_MAP.get(value, "other") for value in flat_types]
    if not mapped:
        return "other"
    if "text" in mapped:
        return "text"
    if "set" in mapped:
        return "set"
    if "formula" in mapped:
        return "formula"
    if "choice" in mapped:
        return "choice"
    if all(value == "numeric" for value in mapped):
        return "numeric"
    return mapped[0]


def _map_theoremqa_answer_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return THEOREMQA_TYPE_MAP.get(raw, "other")


def _domain_from_ugmath_subject(subject: str) -> str:
    key = subject.strip().lower().replace(" ", "_")
    return UGMATH_DOMAIN_MAP.get(key, "other")


def _append_options(problem: str, options: Any) -> str:
    normalized_options = []
    for index, option_group in enumerate(options if isinstance(options, list) else [], start=1):
        if isinstance(option_group, list) and option_group:
            normalized_options.append(f"Blank {index} options: {', '.join(map(str, option_group))}")
    if not normalized_options:
        return problem
    return problem.rstrip() + "\n" + "\n".join(normalized_options)


def _format_answer(answer: Any, answer_type: str) -> str:
    if isinstance(answer, list):
        values = [_format_scalar(value) for value in _flatten_one_level(answer)]
        if answer_type == "set":
            return "{" + ",".join(values) + "}"
        return "[" + ", ".join(values) + "]"
    return _format_scalar(answer)


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return ""
    return str(value).strip()


def _looks_numeric(value: str) -> bool:
    text = value.strip().replace(",", "")
    if not text:
        return False
    try:
        float(text)
        return True
    except Exception:
        return False


def _flatten(value: Any) -> Iterator[Any]:
    if isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def _flatten_one_level(value: List[Any]) -> List[Any]:
    if len(value) == 1 and isinstance(value[0], list):
        return list(value[0])
    return value


def _has_picture(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return bool(value)
    return True


def _write_rows(output_path: Path, rows: List[Dict[str, Any]], dataset: str) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {"dataset": dataset, "output_path": str(output_path), "rows": len(rows)}
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert UGMathBench/TheoremQA to MathSolve-Agent JSONL"
    )
    sub = parser.add_subparsers(dest="dataset", required=True)

    ug = sub.add_parser("ugmathbench", help="Convert UGMathBench JSON files")
    ug.add_argument("--input", required=True, type=Path, help="UGMathBench data directory")
    ug.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    ug.add_argument("--limit", type=int, default=None)
    ug.add_argument(
        "--versions",
        default="1,2,3",
        help="Comma-separated variants to export, default: 1,2,3",
    )

    theorem = sub.add_parser("theoremqa", help="Convert TheoremQA parquet")
    theorem.add_argument("--input", required=True, type=Path, help="TheoremQA parquet path")
    theorem.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    theorem.add_argument("--limit", type=int, default=None)
    theorem.add_argument(
        "--include-images",
        action="store_true",
        help="Keep image-dependent rows. Default skips them because MathSolve-Agent is text-only.",
    )

    mathbench = sub.add_parser("mathbench", help="Convert MathBench release JSON/JSONL")
    mathbench.add_argument(
        "--input",
        required=True,
        type=Path,
        help="MathBench root or mathbench_v1 directory containing JSON/JSONL files",
    )
    mathbench.add_argument("--output", required=True, type=Path, help="Output JSONL path")
    mathbench.add_argument("--limit", type=int, default=None)
    mathbench.add_argument(
        "--language",
        choices=("all", "en", "zh"),
        default="all",
        help="Optional language filter when language metadata is absent or present",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.dataset == "ugmathbench":
        versions = [int(value.strip()) for value in args.versions.split(",") if value.strip()]
        summary = convert_ugmathbench(args.input, args.output, args.limit, versions)
    elif args.dataset == "theoremqa":
        summary = convert_theoremqa(args.input, args.output, args.limit, not args.include_images)
    elif args.dataset == "mathbench":
        summary = convert_mathbench(args.input, args.output, args.limit, args.language)
    else:
        raise SystemExit(f"Unsupported dataset: {args.dataset}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
