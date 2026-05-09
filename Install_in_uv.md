# Lagent 安装指南（使用 uv）

[uv](https://github.com/astral-sh/uv) 是 Rust 编写的高性能 Python 包管理器，可替代 `pip` + `venv`，速度更快且完全兼容 pip 生态。

---

## 1. 安装 uv

### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Linux / macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装后重启终端，验证版本：
```bash
uv --version
```

---

## 2. 创建虚拟环境 & 安装 Lagent

```bash
# 克隆项目
git clone https://github.com/InternLM/lagent.git
cd lagent

# 在当前目录创建 .venv 虚拟环境
uv venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 以可编辑模式安装（开发模式，代码修改立即生效）
uv pip install -e .

# 如需全部可选依赖（文档构建等）
uv pip install -e ".[all]"
```

---

## 3. 一步到位（无需手动激活）

也可以不显式激活环境，直接用 `uv run` 前缀执行命令：

```bash
# 创建环境并安装
uv venv
uv pip install -e .

# 直接运行脚本（自动使用 .venv 中的依赖）
uv run python examples/run_async_agent_openai.py
```

---

## 4. 常用操作对照表

| 操作用途 | pip 命令 | uv 命令 |
|---------|---------|--------|
| 创建虚拟环境 | `python -m venv .venv` | `uv venv` |
| 安装可编辑包 | `pip install -e .` | `uv pip install -e .` |
| 安装额外依赖 | `pip install -e ".[all]"` | `uv pip install -e ".[all]"` |
| 列出已安装包 | `pip list` | `uv pip list` |
| 冻结依赖版本 | `pip freeze` | `uv pip freeze` |
| 安装 requirements.txt | `pip install -r requirements.txt` | `uv pip install -r requirements.txt` |
| 同步锁文件（uv 特有） | — | `uv pip sync requirements.txt` |
| 生成锁文件（uv 特有） | — | `uv pip compile requirements.txt -o requirements.lock` |

---

## 5. 安装后验证

```bash
# 进入 Python 交互环境
uv run python -c "from lagent.version import __version__; print(__version__)"
# 应输出: 0.5.0rc3

# 或运行示例（需先设置 OPENAI_API_KEY）
uv run python examples/run_async_agent_openai.py
```

---

## 常见问题

### Q: uv 和 pip 能共存吗？
**可以。** uv 创建的是标准 Python 虚拟环境，你随时可以切回 `pip` 操作，两种工具管理的 `.venv` 完全一样。

### Q: 会不会和其他项目的 conda/pip 环境冲突？
**不会。** `uv venv` 在当前项目目录下创建 `.venv`，与系统全局 Python 和其他项目的环境相互隔离。

### Q: 如果之前已经用 pip 装了，需要重装吗？
**不需要。** 直接在当前 `.venv` 里用 `uv pip install -e .` 即可，uv 会识别已安装的包进行增量安装。
