"""
单智能体数学解题示例 —— 使用 OpenAI API（无需 GPU）
改编自 Lagent README 中的 Coder 示例
"""
import os
from lagent.agents import Agent
from lagent.agents.aggregator import InternLMToolAggregator
from lagent.actions import ActionExecutor, IPythonInteractive
from lagent.hooks import Hook
from lagent.llms import GPTAPI
from lagent.prompts.parsers import ToolParser
from lagent.schema import ActionReturn, ActionStatusCode, AgentMessage


# ============================================================
# 1. 定义 CodeProcessor Hook（来自 README，用于消息格式转换）
# ============================================================
class CodeProcessor(Hook):
    """在 Agent 输出和 ActionExecutor 之间转换消息格式"""

    def before_action(self, executor, message, session_id):
        """Agent 输出 → 提取代码 → 封装为 IPythonInteractive 可调用的参数"""
        message = message.model_copy(deep=True)
        message.content = dict(
            name='IPythonInteractive',
            parameters={'command': message.formatted['action']},
        )
        return message

    def after_action(self, executor, message, session_id):
        """IPython 执行结果 → 提取文本 → 还给 Agent"""
        action_return = message.content
        if isinstance(action_return, ActionReturn):
            if action_return.state == ActionStatusCode.SUCCESS:
                response = action_return.format_result()
            else:
                response = action_return.errmsg
        else:
            response = str(action_return)
        message.content = response
        return message


# ============================================================
# 2. 定义 Coder 智能体
# ============================================================
class Coder(Agent):
    def __init__(self, model_type='gpt-4o-mini', system_prompt='', max_turn=3):
        super().__init__()

        # 用 OpenAI API 替代 vLLM
        llm = GPTAPI(
            model_type=model_type,
            key=os.environ['OPENAI_API_KEY'],
            retry=3,
            max_new_tokens=2048,
            temperature=0.0,          # 数学题用 0，输出更稳定
            stop_words=['\n```\n'],
        )

        self.agent = Agent(
            llm,
            system_prompt,
            output_format=ToolParser(
                tool_type='code interpreter',
                begin='```python\n',
                end='\n```\n',
            ),
            aggregator=InternLMToolAggregator(),
        )
        self.executor = ActionExecutor(
            [IPythonInteractive()],
            hooks=[CodeProcessor()],
        )
        self.max_turn = max_turn

    def forward(self, message, session_id=0):
        for _ in range(self.max_turn):
            # 1. Agent 思考 → 输出 Python 代码或最终答案
            message = self.agent(message, session_id=session_id)

            # 2. 如果没调用工具（tool_type 为 None），说明已经得出答案
            if message.formatted['tool_type'] is None:
                return message

            # 3. 有代码 → 执行它 → 结果返回给 Agent 继续思考
            message = self.executor(message, session_id=session_id)

        return message


# ============================================================
# 3. 运行
# ============================================================
if __name__ == '__main__':
    system_prompt = (
        'Solve math problems step by step. '
        'First analyze, then write Python code to compute the answer. '
        'Wrap your code in ```python ... ``` blocks.'
    )

    coder = Coder(
        model_type='gpt-4o-mini',    # 也可以换成 gpt-4o / gpt-3.5-turbo
        system_prompt=system_prompt,
        max_turn=3,
    )

    question = (
        r"Find the projection of $\mathbf{a}$ onto "
        r"$\mathbf{b} = \begin{pmatrix} 1 \\ -3 \end{pmatrix}$ "
        r"if $\mathbf{a} \cdot \mathbf{b} = 2.$"
    )

    print("=" * 60)
    print(f"问题: {question}")
    print("=" * 60)

    answer = coder(AgentMessage(sender='user', content=question))
    print(f"\n最终答案:\n{answer.content}")

    # 打印完整对话历史
    print('\n' + '-' * 60)
    print('对话历史:')
    print('-' * 60)
    for msg in coder.state_dict()['agent.memory']:
        print(f"\n[{msg['sender']}]:")
        print(msg['content'])
