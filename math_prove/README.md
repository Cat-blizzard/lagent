# MathSolve-Agent

面向初赛的 Intern-S1 数学解题单智能体系统。当前代码位于 `math_prove/` 包内，对外系统名使用 **MathSolve-Agent**。

本系统目标不是做复杂 Demo，而是稳定完成组委会 112 道多领域数学题的批量求解，并保证每道题都能输出可判分的结构化 JSON。

## 核心能力

- 基于 Intern-S1 或兼容 OpenAI Chat Completions 的 API 调用。
- 单智能体多阶段流水线：题目预处理、题型识别、解题规划、主解答、自检修正、答案抽取、JSON 校验、日志保存。
- 支持 JSON / JSONL / CSV / XLSX 批量输入。
- 支持逐题异常隔离，单题失败不会中断整批任务。
- 支持 `--resume` 断点续跑。
- 每题输出一份独立日志，便于复盘错题和撰写技术报告。
- 保留 `MathSandbox`，用于 SymPy / NumPy / SciPy 局部计算校验，但不强制所有题都写代码验证。

## 当前实现结构

```text
math_prove/
├── __init__.py      # 包导出
├── agent.py         # MathSolverAgent 主流水线
├── main.py          # CLI：demo / 批量处理 / resume / 日志
├── parser.py        # Pydantic schema、JSON 解析、fallback
├── prompts.py       # 分类、求解、自检、抽取、JSON 修复 prompt
├── sandbox.py       # 数学计算沙箱
└── README.md
```

## 工作流

```text
题目输入
  ↓
轻量预处理
  ↓
领域识别与解题规划
  ↓
按领域策略生成候选解
  ↓
可选数学工具校验
  ↓
自我校验与置信度判断
  ↓
低置信度重试 / 困难题候选比较
  ↓
最终答案抽取
  ↓
Pydantic JSON 校验
  ↓
结果与日志落盘
```

## 输出格式

每道题输出一个严格 JSON 对象：

```json
{
  "problem_id": "001",
  "domain": "complex_analysis",
  "answer": "\\frac{\\pi}{2}",
  "answer_type": "formula",
  "reasoning_summary": "先识别奇点并计算留数，再应用留数定理得到结果。",
  "key_steps": [
    "确定奇点位置",
    "计算围道内留数",
    "应用留数定理"
  ],
  "learning_hint": "复积分题要先确认围道选择和奇点位置。",
  "verification": {
    "passed": true,
    "confidence": 0.86,
    "issues": []
  }
}
```

`answer` 字段只放简洁最终答案，不放完整推理过程，降低自动判分解析风险。

## 使用 uv 安装环境

建议在仓库根目录 `D:\lagent-main\lagent` 下运行。这里的 `lagent` 指包含 `setup.py`、`requirements.txt` 和 `math_prove/` 的目录。

### 1. 安装 uv

Windows PowerShell：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

安装后重新打开终端，确认可用：

```powershell
uv --version
```

### 2. 创建虚拟环境

```powershell
cd D:\lagent-main\lagent
uv venv ..\.venv
```

激活环境：

```powershell
..\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止激活脚本，可以直接用 `uv run` 运行命令，不一定要手动激活。

### 3. 安装依赖

安装 lagent 运行依赖：

```powershell
uv pip install -r requirements.txt
```

安装数学工具依赖：

```powershell
uv pip install sympy scipy numpy pandas openpyxl pydantic
```

可选安装 Z3：

```powershell
uv pip install z3-solver
```

如果希望以可编辑模式安装当前 lagent 包：

```powershell
uv pip install -e .
```

## 配置 Intern-S1 API

设置兼容 OpenAI Chat Completions 的 API 信息：

```powershell
$env:OPENAI_API_KEY = "sk-your-api-key"
$env:LLM_API_BASE = "https://your-endpoint/v1/chat/completions"
```

模型名通过 `--model` 传入，例如：

```powershell
--model intern-s1
```

不要把真实 API Key 写进代码或提交到 GitHub。

## 运行 Demo

```powershell
cd D:\lagent-main\lagent
uv run python -m math_prove.main --demo --model intern-s1
```

也可以直接使用已有 venv：

```powershell
..\.venv\Scripts\python.exe -m math_prove.main --demo --model intern-s1
```

## 批量运行

JSONL 输入示例：

```jsonl
{"problem_id": "001", "problem_text": "Find all real roots of x^4 - 5x^2 + 4 = 0."}
{"problem_id": "002", "problem_text": "Given f(z)=(z^2+1)/(z-i), find the residue at z=i."}
```

运行：

```powershell
uv run python -m math_prove.main `
  -i data\problems.jsonl `
  -o outputs\results.jsonl `
  --results-json outputs\results.json `
  --log-dir outputs\logs `
  --summary outputs\run_summary.json `
  --model intern-s1 `
  --resume
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `-i, --input` | 输入文件，支持 JSON / JSONL / CSV / XLSX |
| `-o, --output` | 增量写入的 JSONL 结果文件 |
| `--results-json` | 合并后的 JSON 数组结果文件 |
| `--log-dir` | 每题独立日志目录 |
| `--summary` | 本次运行摘要 |
| `--limit` | 只跑前 N 题，调试用 |
| `--resume` | 跳过已存在于 JSONL 的题号 |
| `--model` | 模型名，例如 `intern-s1` |
| `--api-key` | API Key，也可用环境变量 |
| `--api-base` | API 地址，也可用环境变量 |

## 输入字段兼容

程序会自动识别常见字段：

- 题号字段：`problem_id`、`id`、`question_id`、`qid`、`uid`、`index`
- 题面字段：`problem_text`、`problem`、`question`、`text`、`content`、`题目`

其他字段会保存在 `raw_metadata` 中，并进入每题日志。

## 结果文件

批量运行后默认生成：

```text
outputs/
├── results.jsonl       # 增量结果，每行一个 JSON
├── results.json        # 合并后的 JSON 数组
├── run_summary.json    # 运行统计与 schema 检查结果
└── logs/
    ├── 001.json
    ├── 002.json
    └── ...
```

提交评测时通常优先使用 `results.json` 或官方要求的结果文件格式。

## 本地检查

语法检查：

```powershell
uv run python -m py_compile math_prove\agent.py math_prove\main.py math_prove\parser.py math_prove\prompts.py math_prove\sandbox.py math_prove\__init__.py
```

查看 CLI：

```powershell
uv run python -m math_prove.main --help
```

小批量联调：

```powershell
uv run python -m math_prove.main -i data\problems.jsonl -o outputs\results.jsonl --model intern-s1 --limit 3
```

## GitHub 推送建议

当前代码可以推送到 GitHub，但建议先做以下清理：

1. 不提交 `.venv/`、`__pycache__/`、`.pytest_cache/` 等本地缓存。
2. 不提交 `outputs/`、日志、真实比赛数据、API Key。
3. 如果推送整个 lagent fork，保留原项目 LICENSE，并在 README 里说明本项目基于 lagent 扩展。
4. 如果只想提交比赛代码，可以只提交 `math_prove/`，并在仓库说明依赖 lagent。
5. 推送前运行一次 `py_compile` 和 `--help` 检查。

推荐 `.gitignore` 至少包含：

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
outputs/
*.log
.env
```

## 注意事项

- 当前包名仍是 `math_prove`，命令也是 `python -m math_prove.main`。
- 系统展示名可写作 `MathSolve-Agent`。
- 如果后续想把包名也改成 `math_solve`，需要同步修改目录名、导入路径、运行命令和文档。
