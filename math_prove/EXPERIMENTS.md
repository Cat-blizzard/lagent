# MathSolve-Agent Ablation Experiments

这份文档只说明如何做消融实验，不改变主求解流程。实验脚本会调用现有的 `math_prove.evaluate`，输出仍然落在 `outputs/` 下。

## 目标

用控制变量法判断这些模块是否真的有用：

- LLM verifier
- answer normalizer
- answer extract stage
- multi-candidate
- local equivalence check
- sandbox
- OR-Tools

原则很简单：先跑最小系统 `base`，再一次只打开一个模块，最后比较 `safe`、`safe_plus`、`strong` 三个组合配置。

## 前置条件

在仓库目录运行：

```powershell
cd D:\lagent-main\lagent
```

确认命令可用：

```powershell
uv run python -m math_prove.main --help
uv run python -m math_prove.evaluate --help
```

配置 Intern-S1 API：

```powershell
$env:OPENAI_API_KEY = "your-internlm-api-token"
$env:LLM_API_BASE = "https://chat.intern-ai.org.cn/api/v1/chat/completions"
```

先用 `--dry-run` 看命令，不会调用 API：

```powershell
uv run python -m math_prove.run_ablation_experiments --suite smoke --model intern-s1 --dry-run
```

## 一键运行

最小 smoke 实验：

```powershell
uv run python -m math_prove.run_ablation_experiments `
  --suite smoke `
  --model intern-s1
```

限制题数调试：

```powershell
uv run python -m math_prove.run_ablation_experiments `
  --suite smoke `
  --model intern-s1 `
  --limit 2
```

只跑指定配置：

```powershell
uv run python -m math_prove.run_ablation_experiments `
  --suite core `
  --model intern-s1 `
  --ablation base,safe,safe_plus
```

指定自己的验证集：

```powershell
uv run python -m math_prove.run_ablation_experiments `
  --suite hard `
  --expected math_prove\validation\hard_20.jsonl `
  --model intern-s1
```

## Suite 说明

默认 suite：

| suite | 默认数据集 | 默认 ablation |
| --- | --- | --- |
| `smoke` | `core_18_sample.jsonl` | `base,base_verify,base_normalizer,base_extract,safe` |
| `core` | `core_18_sample.jsonl` | `base,base_verify,base_normalizer,base_extract,base_multi,safe,safe_plus` |
| `format` | `format_stress.jsonl` | `base,base_normalizer,base_extract,safe,safe_plus` |
| `hard` | `hard_20.jsonl` | `base,base_verify,base_multi,safe,safe_plus,strong` |
| `tool` | `calc_tool_20.jsonl` | `base,base_sandbox_observe,base_sandbox_verify,strong` |
| `opt` | `optimization_10.jsonl` | `base,base_sandbox_verify,base_ortools_verify,strong` |
| `final` | `larger_eval.jsonl` | `base,safe,safe_plus,strong` |

当前仓库至少内置：

```text
math_prove/validation/core_18_sample.jsonl
```

如果 `format/hard/tool/opt/final` 的默认数据集还没准备好，脚本会提示你先创建对应 JSONL，或者用 `--expected` 指定现有文件。

## 输出目录

默认输出：

```text
outputs/ablation_runs/<timestamp>_<suite>/
├── command_manifest.json
├── suite_summary.json
├── ablation_summary.json
├── base/
│   ├── results.jsonl
│   ├── results.json
│   ├── run_summary.json
│   ├── validation_report.json
│   └── logs/
├── safe/
└── ...
```

其中：

- `command_manifest.json` 记录本次实验的模型、数据集、preset、命令。
- `ablation_summary.json` 是现有 `evaluate.py` 输出的完整汇总。
- `suite_summary.json` 是更轻量的表格化汇总，方便快速比较。

## 判断规则

模块可以进入正式配置，建议满足：

- `core` 正确率不下降。
- `format` 的 schema / preflight 问题不增加。
- 没有明显把正确答案改坏的案例。
- 平均耗时不超过 `base` 的 1.5 倍；`strong` 可以放宽。
- `hard` 至少提升 1 题，或明显减少 `unable_to_determine` / 低质量答案。

推荐结论：

- 正式候选优先看 `safe`。
- 本地增强对比看 `safe_plus`。
- 难题和错题重跑看 `strong`。

## GitHub 注意

不要提交：

```text
outputs/
.env
真实 API key
真实评测数据
完整运行日志
```

建议只提交：

```text
math_prove/EXPERIMENTS.md
math_prove/run_ablation_experiments.py
math_prove/config.py
```

