# MathSolve-Agent 中文说明

`math_prove/` 是基于 `lagent` 构建的单智能体数学解题系统。它的目标不是展示界面，而是稳定批量处理数学题，输出可判分的结构化 JSON，并支持本地验证、准确率统计和消融实验。

## 核心能力

- 单智能体多阶段流水线：预处理、Problem Diagnosis、领域化求解、自检、修正、答案抽取、JSON 输出。
- 面向 Intern-S1 的 OpenAI-compatible Chat API 调用。
- 规则优先 router 会先给出本地诊断先验，并显式记录 `tool_policy`：`direct`、`sympy`、`ortools`、`python`、`hybrid`、`none`。
- 记录可验证中间结构：assumptions、target、derivation_steps、checkable_claims。
- 输出清洗：剥离 `<think>...</think>`、Markdown JSON 外壳和多余空白。
- 保守答案控制：verifier、normalizer、sandbox 默认不轻易覆盖最终答案。
- 已接受的候选答案会受到保护，extract / verifier / normalizer 在稳定配置下不能随意改写最终答案。
- 本地验证：schema 检查、答案等价检查、低质量答案检查、日志完整性检查。
- 准确率统计：可直接对 MathBench、UGMathBench、TheoremQA 转换后的数据计算准确率。
- 外部数据集转换：支持 UGMathBench、TheoremQA、MathBench。
- 消融实验：支持 base、safe、safe_plus、strong 等配置对照。
- 支持串行批量运行，也支持带全局 RPM 限流的旁路并发批量脚本。

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
├── run_parallel_batch.py       # 旁路并发批量脚本
├── sandbox.py                  # SymPy / NumPy / SciPy / OR-Tools 辅助验证
├── validator.py                # schema、等价验证、提交前体检
├── validation/
│   └── core_18_sample.jsonl
└── README_CN.md
```

## uv 安装

在 lagent 根目录执行：

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

`pandas pyarrow` 主要用于读取 TheoremQA 的 parquet 文件。

## Intern-S1 API 配置

当前项目使用书生 API 的 OpenAI-compatible Chat Completions 接口：

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
- Intern-S1 可能输出 `<think>...</think>`，代码会在 JSON 解析前清洗。
- sandbox 只有输出 `FINAL_RESULT_FOR_CHECK:` 时才参与本地等价检查。

正式比赛风格运行建议开启 fail-fast official mode，避免误用 OpenAI 或其他非 Intern-S1 模型：

```powershell
uv run python -m math_prove.main `
  --demo `
  --model intern-s1 `
  --ablation official_stable `
  --official
```

## 正确率保护机制

当前稳定配置偏保守：辅助阶段可以给出 warning、规范化记录和候选修正建议，但默认不直接改坏已经接受的数学答案。

- 规则优先诊断会在 LLM 诊断前先生成本地 routing prior，覆盖明显的算术、矩阵、微积分、优化、图论/离散、拓扑和证明类问题；Intern-S1 负责修正或补充它。
- `tool_policy` 会写入 `classification`，并控制候选解里的 `verification_code` 是否允许运行。证明/拓扑类诊断默认走直接推理路径，避免被工具误导。
- 最终 JSON 由代码从已接受的 `CandidateSolution` 组装。extract 阶段可以改进 `reasoning_summary`、`key_steps` 和 `learning_hint`，但稳定配置保留候选答案作为最终 `answer`。
- 稳定配置默认使用 240 秒单题 timeout。如果候选答案已经产生，但最终 extract 阶段会超时，系统会跳过 extract 并保留已验证候选答案。
- `verifier_can_overwrite_answer=false`：verifier 的 `corrected_answer` 默认只写日志，不替换候选答案。
- `extract_must_match_candidate=true`：extract 阶段只能压缩或等价改写答案；如果抽取答案和候选答案不等价，会自动回退到候选答案。
- `normalizer_overwrite_answer=false`：normalizer 只记录 raw / latex / canonical 三层形式，不覆盖最终 `answer`。
- normalizer 已增强常见判分格式：选择题、集合、tuple、vector、interval、matrix、分数、根式和初等函数。稳定配置只把这些规范形式用于比较和报告，不静默改写最终答案。
- 如果只是表面格式问题，但题意、条件、结果和可判分性都通过，verifier 不再把整体 `passed` 判为 false。
- 本地等价检查失败默认只是风险提示，`equivalence_can_fail_candidate=false`；只有 strict ablation 才会把它作为硬失败。
- `official_stable` 继承保守的 `safe` 配置。`strong`、`strict_equivalence`、sandbox 类 preset 更适合本地压力测试或难题重跑，不建议作为第一版正式提交配置。

支持的 `answer_type`：

```text
formula, numeric, proof, choice, set, interval, matrix, vector, tuple, text, other
```

## 可验证推理轨迹

系统借鉴 generator-verifier-refiner 的推理闭环思想，但仍然只在一个 `MathSolverAgent` 内部实现，不做多智能体训练。

- `solve_candidate` 相当于 generator，输出候选答案以及 `assumptions`、`target`、`derivation_steps`、`checkable_claims`。
- `verify_candidate` 相当于 verifier，除了分层检查，还会逐条检查 claim，状态为 `passed`、`failed` 或 `uncertain`。
- retry feedback 相当于 refiner，会把失败或不确定的 claim 写入下一轮修复提示。
- `select_best` 只在配置允许、且难题多候选路径触发时使用。

这些字段主要进入每题日志和候选记录；最终提交 JSON 仍保持短答案和可判分格式。

## 单题 Demo

```powershell
uv run python -m math_prove.main --demo --model intern-s1 --ablation safe
```

## 批量求解

输入推荐 JSONL：

```jsonl
{"problem_id":"001","problem_text":"Find all real roots of x^4 - 5x^2 + 4 = 0."}
{"problem_id":"002","problem_text":"Compute the residue of f(z)=(z^2+1)/(z-i) at z=i."}
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
  --ablation safe `
  --resume
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `-i, --input` | 输入文件，支持 JSON / JSONL / CSV / XLSX |
| `-o, --output` | 增量 JSONL 结果 |
| `--results-json` | 合并后的 JSON 数组 |
| `--log-dir` | 每题独立日志目录 |
| `--summary` | 运行摘要 |
| `--limit` | 只跑前 N 题，适合调试 |
| `--resume` | 跳过已经存在于结果 JSONL 的题号 |
| `--ablation` | 指定配置 preset |

## 并发批量脚本

当 Intern-S1 额度允许并发时，可以使用旁路脚本。它不替换串行 `math_prove.main`，而是每个 worker 创建一个 agent，并用全局 RPM limiter 限制所有 LLM 请求。

对于 100 RPM / 1,000,000 TPM 的额度，建议从下面配置开始：

```powershell
uv run python -m math_prove.run_parallel_batch `
  -i D:\dataset\converted\mathbench_all.jsonl `
  -o outputs\mathbench_parallel\results.jsonl `
  --model intern-s1 `
  --ablation safe `
  --workers 3 `
  --rpm-limit 80 `
  --resume
```

如果稳定，再尝试 `--workers 5 --rpm-limit 90`。不建议在没有更高额度前开很大的 worker 数。

并发脚本输出的 JSONL / JSON / 每题日志结构与串行 `main.py` 一致；它只改变调度方式，不改变解题流水线。

## 输出格式

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
    "error_type": "none",
    "repair_instruction": ""
  }
}
```

`answer` 字段只放短答案，不放完整推理过程。

## 外部数据集转换

转换脚本会生成 `main.py` 和 `evaluate.py` 都能直接使用的 JSONL。

### UGMathBench

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

默认跳过图片题，只保留文本题：

```powershell
uv run python -m math_prove.convert_benchmarks theoremqa `
  --input D:\dataset\TheoremQA\data\test-00000-of-00001.parquet `
  --output D:\dataset\converted\theoremqa_text_only.jsonl
```

### MathBench

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

已有结果文件时：

```powershell
uv run python -m math_prove.evaluate `
  --results outputs\mathbench_en_safe\safe\results.jsonl `
  --expected D:\dataset\converted\mathbench_en.jsonl `
  --report outputs\mathbench_en_safe\validation_report.json `
  --ignore-missing-expected
```

输出示例：

```text
Accuracy=85.00% (17/20 checked) | schema_valid=100.00% | preflight_issues=0
```

`--ignore-missing-expected` 用于小批量测试，例如只跑 `--limit 20` 时，不把未跑的几千题算成缺失。

### 可选 DeepSeek 裁判

如果想先不管格式，只判断数学结果是否正确，可以开启 DeepSeek 裁判。系统会先跑本地等价检查；默认情况下，只有本地无法证明正确时才调用 DeepSeek，节省 API 费用。

```powershell
$env:DEEPSEEK_API_KEY = "your-deepseek-api-key"

uv run python -m math_prove.evaluate `
  --results outputs\mathbench_en_safe\safe\results.jsonl `
  --expected D:\dataset\converted\mathbench_en.jsonl `
  --report outputs\mathbench_en_safe\validation_report_deepseek.json `
  --ignore-missing-expected `
  --llm-judge `
  --judge-model deepseek-v4-flash
```

报告里会同时保留本地等价准确率和 `llm_judge_accuracy`。DeepSeek 裁判会被提示忽略表面格式差异，例如 `[-2, -1, 1, 2]` 和 `{-2, -1, 1, 2}`。

## 运行并评测

MathBench 小批量：

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

TheoremQA 小批量：

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

UGMathBench 小批量：

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

```powershell
uv run python -m math_prove.run_ablation_experiments `
  --suite smoke `
  --expected D:\dataset\converted\mathbench_en.jsonl `
  --model intern-s1 `
  --limit 20 `
  --ignore-missing-expected `
  --ablation base,safe,safe_plus
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

建议：

- `base`：最小系统，对照组。
- `safe`：正式提交候选，偏保守；辅助阶段给 warning，但不直接覆盖已接受答案。
- `safe_plus`：低风险增强，用于本地对比。
- `strong`：难题增强，会打开 sandbox、OR-Tools、strict equivalence 和 verifier correction，适合难题集或消融实验，不建议默认用于正式跑全量。
- `official_stable`：基于 `safe` 的 Intern-S1 fail-fast 正式运行配置。

## 本地检查

```powershell
uv run python -m py_compile math_prove\*.py
uv run python -m math_prove.main --help
uv run python -m math_prove.evaluate --help
uv run python -m math_prove.convert_benchmarks --help
uv run python -m math_prove.run_parallel_batch --help
uv run --with pytest python -m pytest tests\test_math_prove\test_accuracy_guards.py -q
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
math_prove/README_CN.md
math_prove/validation/core_18_sample.jsonl
```
