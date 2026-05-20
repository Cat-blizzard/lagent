# MathSolve-Agent

基于 `lagent` 的稳定型单智能体数学解题系统。当前代码位于 `math_prove/` 包内，对外展示名使用 **MathSolve-Agent**。

系统目标是批量处理多领域数学题，并稳定输出可判分的结构化 JSON。它不是 Web Demo，也不是多智能体框架，而是一个强调正确率、容错性、日志可复现和答案规范化的单智能体流水线。

## Core Features

- 单智能体多阶段状态控制：预处理、Problem Diagnosis、候选求解、工具辅助验证、分层自检、错误类型驱动修复、答案抽取、JSON 校验。
- Problem Diagnosis 不只判断题型，还输出 `goal`、`constraints_to_check`、`risk_points`、`needs_case_split`、`needs_tool_verification`、`expected_answer_shape`。
- 分层 verifier：`format_check`、`question_target_check`、`condition_check`、`result_check`、`judgeability_check`。
- 领域级 verifier rubric：已重点覆盖复分析、ODE/PDE、优化/运筹、拓扑。
- 动态候选策略：简单题少调用，困难题可生成多个候选并比较选择。
- Answer Normalizer：保留原答案、LaTeX 形式和 canonical 形式，默认不覆盖最终答案。
- 本地 schema + 等价验证器：可检查结果文件是否可解析、字段是否完整、答案是否等价。
- 提交前体检：检查重复题号、缺题、空答案、Markdown 污染、乱码、低质量答案、日志缺失等工程风险。
- 批量容错：单题异常不会中断整批任务，支持 `--resume` 断点续跑。
- 可选数学工具校验：SymPy / NumPy / SciPy / OR-Tools 可用于局部验证，但主推理仍由 LLM 完成。

## Project Layout

```text
math_prove/
├── __init__.py
├── agent.py              # MathSolverAgent 主流程
├── config.py             # 运行配置与消融 preset
├── configs/
│   └── default.yaml
├── evaluate.py           # 本地验证、回归与消融入口
├── main.py               # CLI：demo / 批量处理 / resume / 日志
├── normalizer.py         # 答案规范化与等价判断
├── parser.py             # Pydantic schema、JSON 解析、fallback
├── prompts.py            # 诊断、求解、验证、抽取、JSON 修复 prompt
├── sandbox.py            # 数学计算沙箱
├── validator.py          # schema、等价验证、提交前体检
├── validation/
│   └── core_18_sample.jsonl
└── README.md
```

## Install With uv

建议在包含 `setup.py`、`requirements.txt` 和 `math_prove/` 的目录下运行：

```powershell
cd D:\lagent-main\lagent
```

安装 uv：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

创建虚拟环境：

```powershell
uv venv ..\.venv
```

激活环境：

```powershell
..\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
uv pip install -r requirements.txt
uv pip install sympy scipy numpy pandas openpyxl pydantic ortools
uv pip install -e .
```

可选安装 Z3：

```powershell
uv pip install z3-solver
```

## Configure API

系统使用兼容 OpenAI Chat Completions 的接口。可以通过环境变量配置：

```powershell
$env:OPENAI_API_KEY = "sk-your-api-key"
$env:LLM_API_BASE = "https://your-endpoint/v1/chat/completions"
```

运行时通过 `--model` 指定模型名：

```powershell
--model intern-s1
```

不要把真实 API Key 写入代码、日志或提交到 GitHub。

## InternLM / Intern-S1 API Strategy

参考官方文档：

- [多轮对话 Chat API](https://internlm.intern-ai.org.cn/doc/docs/Chat/)
- [用户鉴权](https://internlm.intern-ai.org.cn/doc/docs/%E7%94%A8%E6%88%B7%E9%89%B4%E6%9D%83/)
- [模型列表](https://internlm.intern-ai.org.cn/doc/docs/%E6%A8%A1%E5%9E%8B%E5%88%97%E8%A1%A8/)
- [Claude-like API](https://internlm.intern-ai.org.cn/doc/docs/API_DOCUMENTATION_ZH/)

本项目当前基于 `lagent.llms.GPTAPI`，它会直接向 `api_base` 指定的完整 URL 发起 `POST` 请求，所以推荐使用 OpenAI-compatible Chat API：

```powershell
$env:OPENAI_API_KEY = "your-internlm-api-token"
$env:LLM_API_BASE = "https://chat.intern-ai.org.cn/api/v1/chat/completions"
```

然后运行：

```powershell
uv run python -m math_prove.main `
  -i data\problems.jsonl `
  -o outputs\results.jsonl `
  --results-json outputs\results.json `
  --log-dir outputs\logs `
  --model intern-s1 `
  --resume
```

注意两种写法的区别：

| 使用方式 | 地址写法 | 鉴权方式 |
| --- | --- | --- |
| 本项目 `GPTAPI` | `https://chat.intern-ai.org.cn/api/v1/chat/completions` | 环境变量里只填 token，代码会自动加 `Authorization: Bearer ...` |
| OpenAI Python SDK | `base_url="https://chat.intern-ai.org.cn/api/v1/"` | `api_key` 填 token，不手写 `Bearer` |
| 原生 requests/curl | `https://chat.intern-ai.org.cn/api/v1/chat/completions` | 请求头写 `Authorization: Bearer YOUR_API_TOKEN` |

官方策略中和本项目最相关的点：

- 默认每用户每分钟限制约 30 次请求；本项目批量跑题时建议保持顺序执行，不要并发压测。
- `intern-s1-pro`、`intern-s1`、`intern-s1-mini` 支持 `thinking_mode` 控制深度思考模式；当前 `GPTAPI` 路径未额外注入该字段，通常使用模型默认行为即可。
- Chat API 支持的常用参数包括 `model`、`messages`、`temperature`、`top_p`、`stream`、`max_tokens`、`tools`；本项目主要使用非流式 `messages` 调用。
- API 侧 120 秒仍未输出完成时可能返回当前已生成结果；因此本项目保留本地 JSON 修复、fallback 和单题异常隔离。
- API Token 有效期约 6 个月，且只在创建时完整展示；请用环境变量或安全密钥管理，不要写入仓库。
- 常见错误需要重点处理：鉴权失败、token 过期、模型不存在、messages 格式错误、频率或 token 限制超限。
- Intern-S1 可能输出 `<think>...</think>` 或 Markdown JSON 外壳；本项目会在进入 JSON parser 前清洗这些外层内容。
- 工具验证代码如需输出可比较结果，应使用 `FINAL_RESULT_FOR_CHECK:` 标记；没有该标记的 sandbox stdout 只作为 verifier 参考，不参与本地等价判死。

Claude-like `/v1/messages` 也是官方支持的接口，但它使用 `x-api-key` 鉴权、`system` 独立字段和 `content[0].text` 响应结构；这和当前 `GPTAPI` 的 OpenAI-compatible 返回格式不同。除非后续专门新增一个 Claude-like client，否则不建议直接把 `LLM_API_BASE` 改成 `/v1/messages`。

## Run Demo

```powershell
uv run python -m math_prove.main --demo --model intern-s1
```

如果直接使用已创建的虚拟环境：

```powershell
..\.venv\Scripts\python.exe -m math_prove.main --demo --model intern-s1
```

## Batch Run

输入支持 JSON、JSONL、CSV、XLSX。JSONL 示例：

```jsonl
{"problem_id": "001", "problem_text": "Find all real roots of x^4 - 5x^2 + 4 = 0."}
{"problem_id": "002", "problem_text": "Given f(z)=(z^2+1)/(z-i), find the residue at z=i."}
```

批量运行：

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
| `--limit` | 只跑前 N 题，适合调试 |
| `--resume` | 跳过 JSONL 中已经存在的题号 |
| `--config` | 指定运行配置 |
| `--ablation` | 指定消融 preset |

## Conservative Answer Control

为了避免 verifier、normalizer 或本地等价验证器把原本正确的答案改坏，默认策略偏保守：

- Problem Diagnosis 是强提示，不是硬约束；solver 可以根据题目条件修正诊断和方法。
- Verifier 的 `corrected_answer` 不会无条件覆盖候选答案。只有在 `verification.passed=true`、`confidence >= 0.80`、答案非空、不是 `unable_to_determine`，并且改动足够小或通过工具输出等价验证时才会覆盖。
- Normalizer 默认只生成 `answer_forms` 日志，不改写最终 `answer`。如确实要让它覆盖答案，可在配置中设置 `normalizer_overwrite_answer: true`。
- 本地等价验证器默认只提升通过验证的置信度；失败时只写 risk warning，不直接否决候选。若调试时想让可靠工具验证否决候选，可设置 `equivalence_can_fail_candidate: true` 或使用 `strict_equivalence` preset。

## Output Schema

每道题输出一个严格 JSON 对象：

```json
{
  "problem_id": "001",
  "domain": "complex_analysis",
  "answer": "\\frac{\\pi}{2}",
  "answer_type": "formula",
  "reasoning_summary": "识别奇点并计算留数，再应用留数定理得到结果。",
  "key_steps": ["确定奇点位置", "计算留数", "应用留数定理"],
  "learning_hint": "复积分题要先确认围道方向和奇点位置。",
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

## Validate Results

仅做 schema、等价和提交前体检：

```powershell
uv run python -m math_prove.evaluate `
  --results outputs\results.jsonl `
  --expected math_prove\validation\core_18_sample.jsonl `
  --report outputs\validation_report.json `
  --log-dir outputs\logs
```

体检内容包括：

- JSON 是否可解析，schema 是否完整。
- 是否缺题、重复题号或多出题号。
- `answer` 是否为空、过长、包含 Markdown 代码块或低质量兜底文本。
- 是否出现明显乱码。
- 每题日志是否存在，并能与结果题号对应。
- 有 expected answer 时，检查基础答案等价性。

## Regression And Ablation

内置一个轻量样例集：

```text
math_prove/validation/core_18_sample.jsonl
```

运行回归：

```powershell
uv run python -m math_prove.evaluate `
  --run `
  --expected math_prove\validation\core_18_sample.jsonl `
  --output-dir outputs\regression `
  --model intern-s1 `
  --ablation full
```

运行多组消融：

```powershell
uv run python -m math_prove.evaluate `
  --run `
  --expected math_prove\validation\core_18_sample.jsonl `
  --output-dir outputs\regression `
  --model intern-s1 `
  --ablation full,single_candidate,no_normalizer,no_sandbox
```

支持的 preset：

```text
full
official_stable
strong
no_sandbox
no_ortools
no_normalizer
no_equivalence
strict_equivalence
no_llm_verify
no_extract
single_candidate
```

建议正式批量运行优先试：

```powershell
uv run python -m math_prove.main `
  -i data\problems.jsonl `
  -o outputs\results.jsonl `
  --model intern-s1 `
  --ablation official_stable `
  --resume
```

需要更强求解但耗时更高时使用：

```powershell
--ablation strong
```

## Local Checks

语法检查：

```powershell
uv run python -m py_compile math_prove\*.py
```

查看 CLI：

```powershell
uv run python -m math_prove.main --help
uv run python -m math_prove.evaluate --help
```

真实 API 小批量联调：

```powershell
uv run python -m math_prove.main `
  -i data\problems.jsonl `
  -o outputs\results.jsonl `
  --model intern-s1 `
  --limit 3
```

## GitHub Notes

推送前建议确认：

- 不提交 `.venv/`、`__pycache__/`、`.pytest_cache/`。
- 不提交 `outputs/`、真实比赛数据、完整运行日志、API Key。
- 如果以 lagent fork 形式发布，保留原项目 LICENSE，并说明本目录是基于 lagent 的扩展。
- 推送前至少运行一次 `py_compile` 和 `evaluate --results ...`。

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
