# Lagent —— 轻量级 LLM Agent 框架

## 项目概述

**Lagent**（版本 `0.5.0rc3`）是由 InternLM 团队开发的一个轻量级开源框架，旨在帮助开发者高效构建基于大语言模型（LLM）的智能体（Agent）应用。该框架的设计哲学借鉴了 PyTorch，将 Agent 组件类比为神经网络层，用户只需专注于创建"层"（组件）并定义它们之间的消息传递，即可以 Pythonic 的方式搭建复杂的多 Agent 工作流。

- **许可证**：Apache 2.0
- **仓库地址**：https://github.com/InternLM/lagent
- **文档地址**：https://lagent.readthedocs.io

---

## 顶级目录结构

```
lagent-main/
├── .github/                  # GitHub CI/CD 工作流与 Dependabot 配置
├── docs/                     # 项目文档（英文 & 中文，基于 Sphinx + ReadTheDocs）
├── examples/                 # 示例脚本（Agent 的同步/异步/分布式运行）
├── lagent/                   # 核心源代码包（框架主体）
├── requirements/             # 依赖文件（运行时 / 可选 / 文档）
├── tests/                    # 单元测试
├── setup.py                  # 安装打包配置
├── setup.cfg                 # setuptools 配置
├── requirements.txt          # 依赖入口（引用 runtime + optional）
├── README.md                 # 项目英文介绍
├── LICENSE                   # Apache 2.0 许可证
├── MANIFEST.in               # 打包时包含的非 Python 文件清单
├── .gitignore                # Git 忽略规则
├── .pre-commit-config.yaml   # pre-commit 钩子配置（代码风格检查）
├── .pylintrc                 # Pylint 代码检查配置
└── .readthedocs.yaml         # ReadTheDocs 构建配置
```

---

## 核心包 `lagent/` 详细说明

`lagent/` 是框架的核心，采用模块化设计，每个子目录（模块）承担独立职责。

### 1. 顶层文件

| 文件 | 作用 |
|------|------|
| `__init__.py` | 包入口，导出 `__version__` 和 `version_info` |
| `version.py` | 版本号定义（`0.5.0rc3`）及版本解析函数 |
| `schema.py` | **核心数据结构定义**，包含 `AgentMessage`（Agent 间通信的消息体）、`ActionReturn`（工具调用返回）、`FunctionCall`（函数调用描述）、以及各类状态码枚举（`ActionStatusCode`、`AgentStatusCode`、`ModelStatusCode`） |

### 2. `agents/` —— Agent 组件

Agent 是框架的核心执行单元，负责与大模型通信、管理记忆、聚合消息和解析输出。

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出所有 Agent 类：同步/异步/流式的 `Agent`、`Sequential`、`ReAct`、`AgentForInternLM`、`MathCoder` 等 |
| `agent.py` | **基础 Agent 实现**，定义了 `Agent`（核心基类）、`AsyncAgent`、`StreamingAgent`、`AsyncStreamingAgent` 四种同步/异步/流式变体；还包含 `Sequential`（顺序执行多个 Agent 的容器）、`AgentList` 和 `AgentDict`（Agent 集合容器） |
| `react.py` | **ReAct 模式 Agent**，实现了思考→行动→观察的循环（Reasoning + Acting），支持最多 `max_turn` 轮迭代调用工具，直到满足结束条件 |
| `stream.py` | **InternLM 专用流式 Agent**，包括 `AgentForInternLM`（支持插件解释器混合调用）、`MathCoder`（数学解题 Agent）、以及对应的异步版本。内置了工具调用提示模板（中文/英文） |
| `fc_agent.py` | 支持 OpenAI Function Calling 风格的 Agent |
| `aggregator/` | **消息聚合器子模块**，负责将 `AgentMessage` 列表转换为 LLM 可接受的 OpenAI 消息格式 |
| `aggregator/__init__.py` | 导出聚合器类 |
| `aggregator/default_aggregator.py` | `DefaultAggregator`：默认聚合器，将记忆中的消息按 role 转为标准对话格式 |
| `aggregator/tool_aggregator.py` | `InternLMToolAggregator`：适配 InternLM 工具调用格式的聚合器，处理工具调用和结果反馈的消息拼接 |

### 3. `actions/` —— 工具 / 动作模块

Actions 是 Agent 可以调用的外部工具，通过 `ActionExecutor` 统一管理和执行。

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出所有 Action 类及 ActionExecutor |
| `base_action.py` | **工具基类** `BaseAction`：通过元类 `ToolMeta` 自动解析工具描述（type hints、docstring），提供 `tool_api` 装饰器用于标注单个 API，支持工具包（toolkit）模式（一个类包含多个 API） |
| `action_executor.py` | **工具执行器** `ActionExecutor`：管理工具注册、路由（`name` → `action.api`）、执行，以及调用前后的 Hook 处理。支持同步和异步（`AsyncActionExecutor`） |
| `parser.py` | **参数解析器**：`BaseParser`、`JsonParser`、`TupleParser`，负责将 LLM 输出的文本解析为工具调用参数 |
| `builtin_actions.py` | 内置特殊 Action：`InvalidAction`（无效调用处理）、`FinishAction`（正常结束）、`NoAction`（无需工具） |
| `python_interpreter.py` | Python 代码解释器（无状态，每次新建进程执行） |
| `ipython_interpreter.py` | IPython 解释器（无状态，单次执行） |
| `ipython_interactive.py` | IPython 交互式解释器（有状态，可在会话内维护变量） |
| `ipython_manager.py` | IPython 解释器管理器（管理多个交互式会话） |
| `web_browser.py` | 网页浏览器工具（支持 Bing 搜索等） |
| `web_visitor.py` | 网页访问器工具 |
| `google_search.py` | Google 搜索工具 |
| `google_scholar_search.py` | Google Scholar 学术搜索工具 |
| `arxiv_search.py` | Arxiv 论文搜索工具 |
| `bing_map.py` | Bing 地图工具 |
| `ppt.py` | PPT 生成工具 |
| `mcp_client.py` | MCP（Model Context Protocol）客户端，可接入外部 MCP 服务 |

### 4. `llms/` —— 大语言模型接口

提供统一的 LLM 调用抽象，支持多种后端。

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出所有 LLM 类 |
| `base_llm.py` | **LLM 基类** `BaseLLM`：定义了 `generate`、`chat`、`stream_chat` 等标准接口，以及 `LMTemplateParser`（元模板解析器，用于包装对话模板）。`AsyncBaseLLM` 为其异步版本 |
| `base_api.py` | **API 型 LLM 基类** `BaseAPILLM`：继承自 `BaseLLM`，为 OpenAI 兼容 API 提供通用生成逻辑 |
| `openai.py` | OpenAI API 封装（`GPTAPI`、`AsyncGPTAPI`） |
| `anthropic_llm.py` | Anthropic Claude API 封装（`ClaudeAPI`、`AsyncClaudeAPI`） |
| `sensenova.py` | 商汤 SenseNova API 封装 |
| `lmdeploy_wrapper.py` | LMDeploy 推理引擎支持：`LMDeployPipeline`（本地 Pipeline）、`LMDeployClient`（客户端）、`LMDeployServer`（服务端），均有异步版本 |
| `vllm_wrapper.py` | vLLM 推理引擎支持（`VllmModel`、`AsyncVllmModel`） |
| `huggingface.py` | HuggingFace Transformers 本地模型支持（`HFTransformer`、`HFTransformerCasualLM`、`HFTransformerChat`） |
| `meta_template.py` | 模型元模板定义，如 `INTERNLM2_META`（InternLM2 模型的对话格式模板） |

### 5. `memory/` —— 记忆管理模块

负责存储和管理 Agent 对话历史。

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出 `Memory` 和 `MemoryManager` |
| `base_memory.py` | **记忆类** `Memory`：提供消息的增删查改、持久化（`save` / `load`）、最近 N 条截断（`recent_n`）、过滤等功能。消息以 `AgentMessage` 列表形式存储 |
| `manager.py` | **记忆管理器** `MemoryManager`：管理多个会话（session）的独立记忆实例，通过 `session_id` 隔离，支持动态创建与重置 |

### 6. `hooks/` —— 钩子系统

提供 AOP（面向切面编程）风格的插件机制，可在 Agent / ActionExecutor 调用前后插入自定义逻辑。

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出所有 Hook 类 |
| `hook.py` | **Hook 基类**：定义了四个钩子点——`before_agent`、`after_agent`、`before_action`、`after_action`。`RemovableHandle` 用于注册后可移除的 Hook |
| `action_preprocessor.py` | **Action 预处理器**：`ActionPreprocessor`（通用消息预处理基类）和 `InternLMActionProcessor`（专门适配 InternLM 的工具调用格式转换——将 LLM 输出的 `formatted` 字段转为 `ActionExecutor` 可识别的 `FunctionCall`） |
| `logger.py` | **消息日志 Hook** `MessageLogger`：用于记录 Agent 交互过程中的消息 |

### 7. `prompts/` —— 提示词模板与解析

负责提示词模板管理和 LLM 输出解析。

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出 `PromptTemplate` 和所有 Parser |
| `prompt_template.py` | **提示词模板类** `PromptTemplate`：支持 JSON 格式和 Jinja2 格式的模板渲染，可注入 action_info 和 agents_info |
| `parsers/__init__.py` | 导出所有解析器 |
| `parsers/str_parser.py` | `StrParser`：字符串解析器（不做特殊处理） |
| `parsers/json_parser.py` | `JSONParser`：JSON 格式解析器（基于 Pydantic 模型） |
| `parsers/tool_parser.py` | **工具调用解析器**：`ToolParser`（通用工具调用解析，如解析 Python 代码块）、`PluginParser`（插件调用解析）、`InterpreterParser`（解释器调用解析）、`MixedToolParser`（混合工具解析，同时支持插件和解释器） |
| `parsers/custom_parser.py` | `CustomFormatParser`：自定义格式解析器 |

### 8. `distributed/` —— 分布式部署模块

支持将 Agent 部署为 HTTP 服务或 Ray Actor，实现分布式调用。

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出所有分布式组件 |
| `http_serve/api_server.py` | Agent API 服务器 |
| `http_serve/app.py` | HTTP 服务应用 |
| `ray_serve/ray_warpper.py` | Ray Actor 封装（`AgentRayActor`、`AsyncAgentRayActor`） |

### 9. `utils/` —— 工具函数

| 文件 | 作用 |
|------|------|
| `__init__.py` | 导出工具函数 |
| `util.py` | **核心工具集**：`create_object`（根据配置字典动态创建对象）、`load_class_from_string`（从字符串路径加载类）、`get_logger`（日志工具）、`filter_suffix`（响应文本后处理）、`truncate_text`（中英文混合截断）、`async_as_completed`（异步生成器包装） |
| `gen_key.py` | 密钥生成工具 |
| `package.py` | 包管理工具 |

---

## 其他目录说明

### `docs/` —— 文档

| 路径 | 说明 |
|------|------|
| `docs/en/` | 英文文档（Sphinx + ReadTheDocs），包含安装指南（`get_started/install.md`）、快速上手（`get_started/quickstart.md`）、教程（`tutorials/action.md`） |
| `docs/zh_cn/` | 中文文档，结构与英文文档对称 |
| `docs/imgs/` | 文档图片（Logo 等） |

### `examples/` —— 示例

| 文件 | 说明 |
|------|------|
| `model_cli_demo.py` | 模型命令行交互 Demo |
| `run_agent_lmdeploy.py` | 使用 LMDeploy 后端的同步 Agent 示例 |
| `run_async_agent_lmdeploy.py` | 使用 LMDeploy 后端的异步 Agent 示例 |
| `run_async_agent_lmdeploy_server.py` | LMDeploy Server 模式异步 Agent 示例 |
| `run_async_agent_openai.py` | 使用 OpenAI API 的异步 Agent 示例 |
| `run_async_agent_vllm.py` | 使用 vLLM 后端的异步 Agent 示例 |
| `run_agent_services.py` | Agent 服务化部署示例 |
| `run_ray_async_agent_lmdeploy.py` | 基于 Ray 的分布式异步 Agent 示例 |

### `tests/` —— 测试

| 路径 | 说明 |
|------|------|
| `tests/test_actions/` | Action 工具模块的单元测试（内置 Action、Google 搜索、Python 解释器等） |
| `tests/test_agents/` | Agent 模块的单元测试（如 ReWOO Agent） |
| `tests/data/` | 测试数据（如 `search.json`） |

### `requirements/` —— 依赖管理

| 文件 | 说明 |
|------|------|
| `runtime.txt` | 运行时核心依赖（openai、pydantic、jupyter、ray、griffe 等） |
| `optional.txt` | 可选依赖 |
| `docs.txt` | 文档构建依赖 |

---

## 架构概览

```
┌─────────────────────────────────────────────────────┐
│                    用户输入                          │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                  Agent / ReAct / MathCoder           │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Memory  │  │  Aggregator │  │ OutputFormat    │  │
│  │ (记忆)   │  │ (消息聚合)  │  │ (输出解析器)    │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│  ┌──────────────────────────────────────────────┐   │
│  │                  LLM (大模型)                 │   │
│  │  GPT / Claude / vLLM / LMDeploy / HF         │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│               ActionExecutor (工具执行器)            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Python   │ │ 搜索工具  │ │ Web工具  │  ...       │
│  │ 解释器   │ │ (Google等)│ │ (浏览器)  │            │
│  └──────────┘ └──────────┘ └──────────┘            │
└─────────────────────────────────────────────────────┘
```

**核心流程**：
1. 用户发送 `AgentMessage`
2. Agent 通过 Hook 预处理消息 → 存入 Memory
3. Aggregator 将 Memory 中的消息聚合为 LLM 可接受的格式
4. LLM 生成回复
5. OutputFormat Parser 解析回复（提取工具调用或最终答案）
6. 如需调用工具 → ActionExecutor 执行 → 结果反馈给 Agent → 回到步骤 3 继续循环
7. 满足结束条件后返回最终回复

**设计亮点**：
- **双接口设计**：几乎所有组件都提供同步/异步版本（如 `Agent`/`AsyncAgent`、`ActionExecutor`/`AsyncActionExecutor`）
- **流式支持**：`StreamingAgent` 支持逐 token 输出
- **Session 隔离**：通过 `session_id` 实现多会话并发隔离
- **Hook 机制**：可在 Agent 和 ActionExecutor 的前后插入自定义逻辑
- **动态对象创建**：通过 `create_object` 支持使用字典配置动态实例化任意组件

---

## 安装方式

```bash
# 从源码安装
git clone https://github.com/InternLM/lagent.git
cd lagent
pip install -e .

# 安装全部可选依赖
pip install -e ".[all]"

# 仅安装运行时依赖
pip install -e .
```

---

## 引用

如果该项目对你的研究有帮助，请引用：

```latex
@misc{lagent2023,
    title={{Lagent: InternLM} a lightweight open-source framework that
           allows users to efficiently build large language model(LLM)-based agents},
    author={Lagent Developer Team},
    howpublished = {\url{https://github.com/InternLM/lagent}},
    year={2023}
}
```
