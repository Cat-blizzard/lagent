# MathSolve-Agent

`math_prove/` 是基于 `lagent` 构建的单智能体数学解题系统。目标不是做 Web Demo，也不是多智能体框架，而是面向批量数学题评测，稳定完成：

- 题目读取与预处理
- Problem Diagnosis
- 领域化求解
- 自检与修正
- 答案抽取
- 规范 JSON 输出
- 每题独立日志
- 本地 schema / 等价验证 / 准确率统计
- 消融实验与外部 benchmark 转换

系统对外仍是一个 `MathSolverAgent`，内部采用多阶段流水线。

## 目录结构

```text
math_prove/
├── agent.py                    # MathSolverAgent 主流程
├── config.py                   # 运行配置与 ablation preset
├── convert_benchmarks.py       # UGMathBench / TheoremQA / MathBench 转换脚本
├── evaluate.py                 # 运行 + 验证 + 准确率统计
├── main.py                     # demo / 批量求解 / resume / 日志
├── normalizer.py               # 答案规范化与等价判断
├── parser.py                   # Pydantic schema、JSON 解析、fallback
├── prompts.py                  # 诊断、求解、验证、抽取 prompt
├── run_ablation_experiments.py # 一键消融实验调度器
├── sandbox.py                  # SymPy / NumPy / SciPy / OR-Tools 辅助验证
├── validator.py                # schema、等价验证、提交前体检
├── validation/
│   └── core_18_sample.jsonl
└── README.md
```

## uv 安装

建议在包含 `setup.py` 的 lagent 根目录运行：

```powershell
cd D:\lagent-main\lagent
```

安装 uv：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

创建并激活虚拟环境：

```powershell
uv venv ..\.venv
..\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
uv pip install -r requirements.txt
uv pip install -e .
uv pip install sympy scipy numpy pandas pyarrow openpyxl pydantic ortools
```

其中 `pandas pyarrow` 主要用于读取 TheoremQA 的 parquet 文件。

## Intern-S1 API 配置

当前代码走书生 API 的 OpenAI-compatible Chat Completions 接口，继续使用 `lagent.llms.GPTAPI` 即可。

```powershell
$env:OPENAI_API_KEY = "your-internlm-api-token"
$env:LLM_API_BASE = "https://chat.intern-ai.org.cn/api/v1/chat/completions"
```

运行时指定模型：

```powershell
--model intern-s1
```

注意：

- `OPENAI_API_KEY` 只填 token，不手写 `Bearer`。
- 当前不使用 Claude-like `/v1/messages` 接口。
- Intern-S1 可能输出 `<think>...</think>` 或 Markdown JSON 外壳，代码会在进入 JSON parser 前清洗。
- sandbox 输出只有带 `FINAL_RESULT_FOR_CHECK:` 标记时才进入本地等价检查，避免把调试 stdout 当成答案。

## 单题 Demo

```powershell
uv run python -m math_prove.main --demo --model intern-s1 --ablation safe
```

## 批量运行

输入支持 JSON / JSONL / CSV / XLSX。推荐 JSONL：

```jsonl
{"problem_id":"001","problem_text":"Find all real roots of x^4 - 5x^2 + 4 = 0."}
{"problem_id":"002","problem_text":"Compute the residue of f(z)=(z^2+1)/(z-i) at z=i."}
```

批量求解：

```powershell
uv run python -m math_prove.main `
  -i data\problems.jsonl `
  -o outputs\results.jsonl `
  --results-json outputs\results.json `
  --log-dir outputs\logs `
  --summary outputs\run_summary.json `
  --model intern-s1 `
  --ablation safe `
  --resume
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `-i, --input` | 输入文件 |
| `-o, --output` | 增量 JSONL 结果 |
| `--results-json` | 合并后的 JSON 数组 |
| `--log-dir` | 每题独立日志目录 |
| `--summary` | 运行摘要 |
| `--limit` | 只跑前 N 题，适合调试 |
| `--resume` | 跳过已经存在于结果 JSONL 的题号 |
| `--ablation` | 指定配置 preset |

## 输出 Schema

每题输出一个严格 JSON 对象：

```json
{
  "problem_id": "001",
  "domain": "complex_analysis",
  "answer": "\\frac{\\pi}{2}",
  "answer_type": "formula",
  "reasoning_summary": "识别奇点并计算留数，再应用留数定理得到结果。",
  "key_steps": ["确定奇点位置", "计算留数", "应用留数定理"],
  "learning_hint": "这类题要先确认围道方向和奇点位置，再套留数定理。",
  "verification": {
    "passed": true,
    "confidence": 0.86,
    "issues": [],
    "format_check": {"passed": true, "issues": []},
    "question_target_check": {"passed": true, "issues": []},
    "condition_check": {"passed": true, "issues": []},
    "result_check": {"passed": true, "issues": []},
    "judgeability_check": {"passed": true, "issues": []},
    "error_type": "none",
    "repair_instruction": ""
  }
}
```

`answer` 字段只放短答案，不放完整推理过程。

## 外部数据集转换

转换脚本统一输出可直接给 `main.py` 和 `evaluate.py` 使用的 JSONL：

```json
{
  "problem_id": "...",
  "problem_text": "...",
  "domain": "...",
  "answer_type": "...",
  "expected_answer": "...",
  "raw_metadata": {...}
}
```

### UGMathBench

原始目录：

```text
D:\dataset\ugmathbench\data
```

全量转换：

```powershell
uv run python -m math_prove.convert_benchmarks ugmathbench `
  --input D:\dataset\ugmathbench\data `
  --output D:\dataset\converted\ugmathbench_all.jsonl
```

小样本：

```powershell
uv run python -m math_prove.convert_benchmarks ugmathbench `
  --input D:\dataset\ugmathbench\data `
  --output D:\dataset\converted\ugmathbench_50.jsonl `
  --limit 50
```

### TheoremQA

原始文件：

```text
D:\dataset\TheoremQA\data\test-00000-of-00001.parquet
```

默认跳过图片题，只保留文本题：

```powershell
uv run python -m math_prove.convert_benchmarks theoremqa `
  --input D:\dataset\TheoremQA\data\test-00000-of-00001.parquet `
  --output D:\dataset\converted\theoremqa_text_only.jsonl
```

如需保留图片题：

```powershell
uv run python -m math_prove.convert_benchmarks theoremqa `
  --input D:\dataset\TheoremQA\data\test-00000-of-00001.parquet `
  --output D:\dataset\converted\theoremqa_all.jsonl `
  --include-images
```

当前 MathSolve-Agent 是文本单智能体，正式测试建议先用 text-only。

### MathBench

原始目录：

```text
D:\dataset\MathBench\mathbench_v1
```

全量转换：

```powershell
uv run python -m math_prove.convert_benchmarks mathbench `
  --input D:\dataset\MathBench\mathbench_v1 `
  --output D:\dataset\converted\mathbench_all.jsonl
```

只转英文：

```powershell
uv run python -m math_prove.convert_benchmarks mathbench `
  --input D:\dataset\MathBench\mathbench_v1 `
  --output D:\dataset\converted\mathbench_en.jsonl `
  --language en
```

## 准确率评测

如果已经有结果文件：

```powershell
uv run python -m math_prove.evaluate `
  --results outputs\mathbench_en_safe\safe\results.jsonl `
  --expected D:\dataset\converted\mathbench_en.jsonl `
  --report outputs\mathbench_en_safe\validation_report.json `
  --ignore-missing-expected
```

输出会包含：

```text
Accuracy=85.00% (17/20 checked) | schema_valid=100.00% | preflight_issues=0
```

`--ignore-missing-expected` 很重要：当你用 `--limit 20` 跑全量 expected 文件时，它会只按已经生成的 20 道结果算准确率，不把未跑的几千题算作缺失。

## 运行并评测

一条命令跑 MathBench 小批量并计算准确率：

```powershell
uv run python -m math_prove.evaluate `
  --run `
  --expected D:\dataset\converted\mathbench_en.jsonl `
  --output-dir outputs\mathbench_en_safe `
  --model intern-s1 `
  --ablation safe `
  --limit 20 `
  --ignore-missing-expected
```

TheoremQA：

```powershell
uv run python -m math_prove.evaluate `
  --run `
  --expected D:\dataset\converted\theoremqa_text_only.jsonl `
  --output-dir outputs\theoremqa_safe `
  --model intern-s1 `
  --ablation safe `
  --limit 20 `
  --ignore-missing-expected
```

UGMathBench：

```powershell
uv run python -m math_prove.evaluate `
  --run `
  --expected D:\dataset\converted\ugmathbench_all.jsonl `
  --output-dir outputs\ugmath_safe `
  --model intern-s1 `
  --ablation safe `
  --limit 20 `
  --ignore-missing-expected
```

## 消融实验

一键消融入口：

```powershell
uv run python -m math_prove.run_ablation_experiments `
  --suite smoke `
  --expected D:\dataset\converted\mathbench_en.jsonl `
  --model intern-s1 `
  --limit 20 `
  --ignore-missing-expected `
  --ablation base,safe,safe_plus
```

输出目录形如：

```text
outputs/ablation_runs/<timestamp>_<suite>/
├── command_manifest.json
├── suite_summary.json
├── ablation_summary.json
├── base/
├── safe/
└── safe_plus/
```

常用 preset：

```text
base
base_verify
base_normalizer
base_extract
base_multi
safe
safe_plus
strong
base_sandbox_observe
base_sandbox_verify
base_ortools_verify
```

建议使用方式：

- `base`：最小系统，对照组。
- `safe`：正式提交候选，偏保守。
- `safe_plus`：低风险增强，用于本地对比。
- `strong`：难题增强，耗时更高。

## 提交前体检

验证器会检查：

- JSON 是否可解析。
- schema 是否完整。
- 是否存在重复题号、缺题、多题。
- answer 是否为空、过长、含 Markdown 污染。
- 是否出现明显乱码。
- 是否存在低质量兜底答案。
- 每题日志是否存在。
- 有 expected answer 时，计算本地答案等价率。

## 本地检查

语法检查：

```powershell
uv run python -m py_compile math_prove\*.py
```

查看 CLI：

```powershell
uv run python -m math_prove.main --help
uv run python -m math_prove.evaluate --help
uv run python -m math_prove.convert_benchmarks --help
```

## GitHub 注意事项

不要提交：

```text
.venv/
outputs/
__pycache__/
*.pyc
.env
API key
真实比赛评测集
大体积外部数据集
```

可以提交：

```text
math_prove/*.py
math_prove/README.md
math_prove/validation/core_18_sample.jsonl
```

如果发布为 lagent fork 的分支，建议根目录 README 只加一个 MathSolve-Agent 入口说明，保留原 lagent README 和 LICENSE。详细使用方式放在本文件里。
