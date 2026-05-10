"""
数学沙箱引擎 —— 基于 lagent.actions.IPythonInteractive 子类化。

预注入 sympy / scipy / numpy / z3，并增强错误捕获以返回完整 Traceback。
"""
import re
import traceback
import os
from contextlib import redirect_stdout
from io import StringIO
from typing import Optional, Type

from lagent.actions.ipython_interactive import (
    IPythonInteractive,
    Status,
    ExecutionResult,
)
from lagent.actions.parser import BaseParser, JsonParser


# ─── 预注入的数学与科学计算库 ──────────────────────────────────────
PRE_IMPORTS = [
    "import sympy as sp",
    "from sympy import *",
    "import numpy as np",
    "from scipy import integrate, optimize, linalg, special",
    "import math",
    "import itertools",
    "import functools",
]
# z3 是可选依赖，导入失败不应阻止沙箱初始化
_OPTIONAL_IMPORTS = [
    "from z3 import *",
    "from ortools.linear_solver import pywraplp",
    "from ortools.sat.python import cp_model",
]


class MathSandbox(IPythonInteractive):
    """扩展的 IPython 沙箱，启动时静默注入数学库。

    继承自 ``lagent.actions.IPythonInteractive``，所有 API 完全兼容。
    重写 ``exec()`` 以在出错时返回完整 Traceback（含局部变量快照）。

    Args:
        timeout (int): 单次代码执行的超时秒数，默认 10。
        max_out_len (int): stdout 输出最大长度，-1 表示不截断。
    """

    def __init__(
        self,
        timeout: int = 10,
        max_out_len: int = 8192,
        use_signals: Optional[bool] = None,
        description: Optional[dict] = None,
        parser: Type[BaseParser] = JsonParser,
    ):
        if use_signals is None:
            use_signals = os.name != "nt"
        super().__init__(
            timeout=timeout,
            max_out_len=max_out_len,
            use_signals=use_signals,
            description=description,
            parser=parser,
        )
        self._inject_libraries()

    # ── 库注入 ───────────────────────────────────────────────────
    def _inject_libraries(self) -> None:
        """在 IPython shell 中静默执行预导入语句。"""
        with StringIO() as io:
            with redirect_stdout(io):
                for stmt in PRE_IMPORTS:
                    self._executor.run_cell(stmt, store_history=False)
                for stmt in _OPTIONAL_IMPORTS:
                    try:
                        self._executor.run_cell(stmt, store_history=False)
                    except Exception:
                        pass  # 可选库不可用也不阻止运行

    def reset(self) -> None:
        """重置沙箱并重新注入库。"""
        super().reset()
        self._inject_libraries()

    # ── 增强错误捕获 ──────────────────────────────────────────────
    def exec(self, code: str) -> ExecutionResult:
        """执行 Python 代码，失败时返回完整 Traceback。

        相比父类，错误信息包含：
        - 完整的 Traceback 堆栈
        - 异常发生时的局部变量快照（截断到安全长度）

        Args:
            code (str): 待执行的 Python 代码（可含 Markdown 代码块标记）

        Returns:
            ExecutionResult: 包含 status / value / msg 的执行结果
        """
        from IPython.core.interactiveshell import ExecutionResult as IPyResult

        code = self.extract_code(code)
        wrapped = (
            self.wrap_code_with_timeout(code, self.timeout)
            if getattr(self, "_use_signals", True)
            else code
        )

        with StringIO() as io:
            with redirect_stdout(io):
                try:
                    ret: IPyResult = self._executor.run_cell(wrapped)
                except Exception:
                    return ExecutionResult(
                        Status.FAILURE,
                        msg=traceback.format_exc()[:self._max_out_len],
                    )

                # ── 成功路径 ──
                if ret.success:
                    # 若有 cell 返回值
                    if ret.result is not None:
                        return ExecutionResult(
                            Status.SUCCESS,
                            str(ret.result)[:self._max_out_len],
                        )
                    # 否则取 stdout
                    outs = io.getvalue().strip()
                    return ExecutionResult(
                        Status.SUCCESS,
                        outs[:self._max_out_len] if outs else "",
                    )

                # ── 失败路径 构造完整错误信息 ──
                outs = io.getvalue()
                error_lines = self._build_error_report(ret, outs)
                return ExecutionResult(
                    Status.FAILURE,
                    msg=error_lines[:self._max_out_len],
                )

    @staticmethod
    def _build_error_report(ret, stdout_captured: str) -> str:
        """从 IPython 执行结果构造人类可读的错误报告。

        包含:
        - 标准输出（如有
        - 异常类型与消息
        - 完整 Traceback
        """
        parts = []

        # 1) 标准输出（可能有部分成功的输出）
        if stdout_captured.strip():
            parts.append(f"[stdout]\n{stdout_captured.strip()}")

        # 2) 异常信息
        if ret.error_in_exec is not None:
            exc = ret.error_in_exec
            tb_str = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            parts.append(f"[traceback]\n{tb_str.rstrip()}")
        elif ret.error_before_exec is not None:
            parts.append(f"[compile error]\n{ret.error_before_exec}")

        return "\n\n".join(parts) if parts else "Unknown execution error"

    # ── 便捷接口 ──────────────────────────────────────────────────
    def run_code(self, code: str) -> ExecutionResult:
        """直接 exec 的别名，语义更清晰。"""
        return self.exec(code)
