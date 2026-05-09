# 安装

从源码安装：

```bash
git clone https://github.com/InternLM/lagent.git
cd lagent
pip install -e .
```

# 用法

Lagent 的设计理念受 PyTorch 启发。我们期望这种类似于神经网络层的类比能使工作流更清晰直观，用户只需关注创建层以及以 Python 方式定义层之间的消息传递。下面是一个简单的教程，帮助你快速上手构建多智能体应用。

## 模型作为智能体

智能体使用 `AgentMessage` 进行通信。

```python
from typing import Dict, List
from lagent.agents import Agent
from lagent.schema import AgentMessage
from lagent.llms import VllmModel, INTERNLM2_META

llm = VllmModel(
    path='Qwen/Qwen2-7B-Instruct',
    meta_template=INTERNLM2_META,
    tp=1,
    top_k=1,
    temperature=1.0,
    stop_words=['<|im_end|>'],
    max_new_tokens=1024,
)
system_prompt = '你的回答只能从“典”、“孝”、“急”三个字中选一个。'
agent = Agent(llm, system_prompt)

user_msg = AgentMessage(sender='user', content='今天天气情况')
bot_msg = agent(user_msg)
print(bot_msg)
# 输出: content='急' sender='Agent' formatted=None extra_info=None type=None receiver=None stream_state=<AgentStatusCode.END: 0>
```

## 记忆作为状态

每次前向传播时，输入和输出消息都会被添加到 Agent 的记忆中。这是在 `__call__` 而不是 `forward` 中执行的。参见以下伪代码：

```python
def __call__(self, *message):
    message = pre_hooks(message)
    add_memory(message)
    message = self.forward(*message)
    add_memory(message)
    message = post_hooks(message)
    return message
```

通过两种方式查看记忆：

```python
memory: List[AgentMessage] = agent.memory.get_memory()
print(memory)
print('-' * 120)
dumped_memory: Dict[str, List[dict]] = agent.state_dict()
print(dumped_memory['memory'])
```

清除当前会话（默认 session_id=0）的记忆：

```python
agent.reset()
```

## 自定义消息聚合

`DefaultAggregator` 在底层被调用，用于组装并将 `AgentMessage` 转换为 OpenAI 消息格式。

```python
def forward(self, *message: AgentMessage, session_id=0, **kwargs) -> Union[AgentMessage, str]:
    formatted_messages = self.aggregator.aggregate(
        self.memory.get(session_id),
        self.name,
        self.output_format,
        self.template,
    )
    llm_response = self.llm.chat(formatted_messages, **kwargs)
    ...
```

实现一个能接收 few-shot 的简单聚合器：

```python
from typing import List, Union
from lagent.memory import Memory
from lagent.prompts import StrParser
from lagent.agents.aggregator import DefaultAggregator

class FewshotAggregator(DefaultAggregator):
    def __init__(self, few_shot: List[dict] = None):
        self.few_shot = few_shot or []

    def aggregate(self,
                  messages: Memory,
                  name: str,
                  parser: StrParser = None,
                  system_instruction: Union[str, dict, List[dict]] = None) -> List[dict]:
        _message = []
        if system_instruction:
            _message.extend(
                self.aggregate_system_intruction(system_instruction))
        _message.extend(self.few_shot)
        messages = messages.get_memory()
        for message in messages:
            if message.sender == name:
                _message.append(
                    dict(role='assistant', content=str(message.content)))
            else:
                user_message = message.content
                if len(_message) > 0 and _message[-1]['role'] == 'user':
                    _message[-1]['content'] += user_message
                else:
                    _message.append(dict(role='user', content=user_message))
        return _message

agent = Agent(
    llm,
    aggregator=FewshotAggregator(
        [
            {"role": "user", "content": "今天天气"},
            {"role": "assistant", "content": "【晴】"},
        ]
    )
)
user_msg = AgentMessage(sender='user', content='昨天天气')
bot_msg = agent(user_msg)
print(bot_msg)
# 输出: content='【多云转晴，夜间有轻微降温】' sender='Agent' ...
```

## 灵活的输出格式化

在 `AgentMessage` 中，`formatted` 字段用于存储通过 `output_format` 从模型输出中解析得到的信息。

```python
def forward(self, *message: AgentMessage, session_id=0, **kwargs) -> Union[AgentMessage, str]:
    ...
    llm_response = self.llm.chat(formatted_messages, **kwargs)
    if self.output_format:
        formatted_messages = self.output_format.parse_response(llm_response)
        return AgentMessage(
            sender=self.name,
            content=llm_response,
            formatted=formatted_messages,
        )
    ...
```

使用工具解析器：

```python
from lagent.prompts.parsers import ToolParser

system_prompt = "逐步分析并编写Python代码解决以下问题。"
parser = ToolParser(tool_type='code interpreter', begin='```python\n', end='\n```\n')
llm.gen_params['stop_words'].append('\n```\n')
agent = Agent(llm, system_prompt, output_format=parser)

user_msg = AgentMessage(
    sender='user',
    content='Marie is thinking of a multiple of 63, while Jay is thinking of a '
    'factor of 63. They happen to be thinking of the same number. There are '
    'two possibilities for the number that each of them is thinking of, one '
    'positive and one negative. Find the product of these two numbers.')
bot_msg = agent(user_msg)
print(bot_msg.model_dump_json(indent=4))
```

## 工具调用的一致性

`ActionExecutor` 使用与 Agent 相同的数据结构进行通信，但要求输入的 `AgentMessage` 的 `content` 是一个字典，包含：

- `name`: 工具名称，例如 `'IPythonInterpreter'`, `'WebBrowser.search'`
- `parameters`: 工具 API 的关键字参数，例如 `{'command': 'import math;math.sqrt(2)'}`, `{'query': ['recent progress in AI']}`

你可以注册自定义钩子来转换消息。

```python
from lagent.hooks import Hook
from lagent.schema import ActionReturn, ActionStatusCode, AgentMessage
from lagent.actions import ActionExecutor, IPythonInteractive

class CodeProcessor(Hook):
    def before_action(self, executor, message, session_id):
        message = message.copy(deep=True)
        message.content = dict(
            name='IPythonInteractive', parameters={'command': message.formatted['action']}
        )
        return message

    def after_action(self, executor, message, session_id):
        action_return = message.content
        if isinstance(action_return, ActionReturn):
            if action_return.state == ActionStatusCode.SUCCESS:
                response = action_return.format_result()
            else:
                response = action_return.errmsg
        else:
            response = action_return
        message.content = response
        return message

executor = ActionExecutor(actions=[IPythonInteractive()], hooks=[CodeProcessor()])
bot_msg = AgentMessage(
    sender='Agent',
    content='首先，我们需要...',
    formatted={
        'tool_type': 'code interpreter',
        'thought': '首先，我们需要...',
        'action': 'def find_numbers(): ...',
        'status': 1
    })
executor_msg = executor(bot_msg)
print(executor_msg)
# 输出: content='3969.0' sender='ActionExecutor' ...
```

为了方便，Lagent 提供了 `InternLMActionProcessor`，它与 `ToolParser` 格式化后的消息适配。

## 双接口

Lagent 采用双接口设计，几乎每个组件（LLM、动作、动作执行器等）都有对应的异步变体，只需在标识符前加上 `Async` 前缀。建议调试时使用同步智能体，大规模推理时使用异步智能体，以充分利用空闲的 CPU 和 GPU 资源。

但是，请确保智能体内部的组件一致性，即异步智能体应配备异步 LLM 和驱动异步工具的异步动作执行器。

```python
from lagent.llms import VllmModel, AsyncVllmModel, LMDeployPipeline, AsyncLMDeployPipeline
from lagent.actions import ActionExecutor, AsyncActionExecutor, WebBrowser, AsyncWebBrowser
from lagent.agents import Agent, AsyncAgent, AgentForInternLM, AsyncAgentForInternLM
```

## 实践

- 除非必要，尽量实现子类的 `forward` 而不是 `__call__`。
- 始终显式包含 `session_id` 参数，该参数用于在并发中隔离记忆、LLM 请求和工具调用（例如维护多个独立的 IPython 环境）。

### 单智能体

通过编程解决问题的数学智能体：

```python
from lagent.agents.aggregator import InternLMToolAggregator

class Coder(Agent):
    def __init__(self, model_path, system_prompt, max_turn=3):
        super().__init__()
        llm = VllmModel(
            path=model_path,
            meta_template=INTERNLM2_META,
            tp=1,
            top_k=1,
            temperature=1.0,
            stop_words=['\n```\n', '<|im_end|>'],
            max_new_tokens=1024,
        )
        self.agent = Agent(
            llm,
            system_prompt,
            output_format=ToolParser(
                tool_type='code interpreter', begin='```python\n', end='\n```\n'
            ),
            # `InternLMToolAggregator` 与 `ToolParser` 适配，用于聚合带有工具调用和执行结果的消息
            aggregator=InternLMToolAggregator(),
        )
        self.executor = ActionExecutor([IPythonInteractive()], hooks=[CodeProcessor()])
        self.max_turn = max_turn

    def forward(self, message: AgentMessage, session_id=0) -> AgentMessage:
        for _ in range(self.max_turn):
            message = self.agent(message, session_id=session_id)
            if message.formatted['tool_type'] is None:
                return message
            message = self.executor(message, session_id=session_id)
        return message

coder = Coder('Qwen/Qwen2-7B-Instruct', 'Solve the problem step by step with assistance of Python code')
query = AgentMessage(
    sender='user',
    content='Find the projection of $\\mathbf{a}$ onto $\\mathbf{b} = '
    '\\begin{pmatrix} 1 \\\\ -3 \\end{pmatrix}$ if $\\mathbf{a} \\cdot \\mathbf{b} = 2.$'
)
answer = coder(query)
print(answer.content)
print('-' * 120)
for msg in coder.state_dict()['agent.memory']:
    print('*' * 80)
    print(f'{msg["sender"]}:\n\n{msg["content"]}')
```

### 多智能体

通过自我优化来提升写作质量的异步博客智能体（原始 AutoGen 示例）：

```python
import asyncio
import os
from lagent.llms import AsyncGPTAPI
from lagent.agents import AsyncAgent
os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

class PrefixedMessageHook(Hook):
    def __init__(self, prefix: str, senders: list = None):
        self.prefix = prefix
        self.senders = senders or []

    def before_agent(self, agent, messages, session_id):
        for message in messages:
            if message.sender in self.senders:
                message.content = self.prefix + message.content

class AsyncBlogger(AsyncAgent):
    def __init__(self, model_path, writer_prompt, critic_prompt, critic_prefix='', max_turn=3):
        super().__init__()
        llm = AsyncGPTAPI(model_type=model_path, retry=5, max_new_tokens=2048)
        self.writer = AsyncAgent(llm, writer_prompt, name='writer')
        self.critic = AsyncAgent(
            llm, critic_prompt, name='critic', hooks=[PrefixedMessageHook(critic_prefix, ['writer'])]
        )
        self.max_turn = max_turn

    async def forward(self, message: AgentMessage, session_id=0) -> AgentMessage:
        for _ in range(self.max_turn):
            message = await self.writer(message, session_id=session_id)
            message = await self.critic(message, session_id=session_id)
        return await self.writer(message, session_id=session_id)

blogger = AsyncBlogger(
    'gpt-4o-2024-05-13',
    writer_prompt="You are an writing assistant tasked to write engaging blogpost. You try to generate the best blogpost possible for the user's request. "
    "If the user provides critique, then respond with a revised version of your previous attempts",
    critic_prompt="Generate critique and recommendations on the writing. Provide detailed recommendations, including requests for length, depth, style, etc..",
    critic_prefix='Reflect and provide critique on the following writing. \n\n',
)
user_prompt = (
    "Write an engaging blogpost on the recent updates in {topic}. "
    "The blogpost should be engaging and understandable for general audience. "
    "Should have more than 3 paragraphes but no longer than 1000 words.")
bot_msgs = asyncio.get_event_loop().run_until_complete(
    asyncio.gather(
        *[
            blogger(AgentMessage(sender='user', content=user_prompt.format(topic=topic)), session_id=i)
            for i, topic in enumerate(['AI', 'Biotechnology', 'New Energy', 'Video Games', 'Pop Music'])
        ]
    )
)
print(bot_msgs[0].content)
print('-' * 120)
for msg in blogger.state_dict(session_id=0)['writer.memory']:
    print('*' * 80)
    print(f'{msg["sender"]}:\n\n{msg["content"]}')
print('-' * 120)
for msg in blogger.state_dict(session_id=0)['critic.memory']:
    print('*' * 80)
    print(f'{msg["sender"]}:\n\n{msg["content"]}')
```

一个执行信息检索、数据收集和图表绘制等多智能体工作流（原始 LangGraph 示例）：

```python
import json
from lagent.actions import IPythonInterpreter, WebBrowser, ActionExecutor
from lagent.agents.stream import get_plugin_prompt
from lagent.llms import GPTAPI
from lagent.hooks import InternLMActionProcessor

TOOL_TEMPLATE = (
    "You are a helpful AI assistant, collaborating with other assistants. Use the provided tools to progress"
    " towards answering the question. If you are unable to fully answer, that's OK, another assistant with"
    " different tools will help where you left off. Execute what you can to make progress. If you or any of"
    " the other assistants have the final answer or deliverable, prefix your response with {finish_pattern}"
    " so the team knows to stop. You have access to the following tools:\n{tool_description}\nPlease provide"
    " your thought process when you need to use a tool, followed by the call statement in this format:"
    "\n{invocation_format}\\\\n**{system_prompt}**"
)

class DataVisualizer(Agent):
    def __init__(self, model_path, research_prompt, chart_prompt, finish_pattern="Final Answer", max_turn=10):
        super().__init__()
        llm = GPTAPI(model_path, key='YOUR_OPENAI_API_KEY', retry=5, max_new_tokens=1024, stop_words=["```\n"])
        interpreter, browser = IPythonInterpreter(), WebBrowser("BingSearch", api_key="YOUR_BING_API_KEY")
        self.researcher = Agent(
            llm,
            TOOL_TEMPLATE.format(
                finish_pattern=finish_pattern,
                tool_description=get_plugin_prompt(browser),
                invocation_format='```json\n{"name": {{tool name}}, "parameters": {{keyword arguments}}}\n```\n',
                system_prompt=research_prompt,
            ),
            output_format=ToolParser(
                "browser",
                begin="```json\n",
                end="\n```\n",
                validate=lambda x: json.loads(x.rstrip('`')),
            ),
            aggregator=InternLMToolAggregator(),
            name="researcher",
        )
        self.charter = Agent(
            llm,
            TOOL_TEMPLATE.format(
                finish_pattern=finish_pattern,
                tool_description=interpreter.name,
                invocation_format='```python\n{{code}}\n```\n',
                system_prompt=chart_prompt,
            ),
            output_format=ToolParser(
                "interpreter",
                begin="```python\n",
                end="\n```\n",
                validate=lambda x: x.rstrip('`'),
            ),
            aggregator=InternLMToolAggregator(),
            name="charter",
        )
        self.executor = ActionExecutor([interpreter, browser], hooks=[InternLMActionProcessor()])
        self.finish_pattern = finish_pattern
        self.max_turn = max_turn

    def forward(self, message, session_id=0):
        for _ in range(self.max_turn):
            message = self.researcher(message, session_id=session_id, stop_words=["```\n", "```python"]) # 覆盖 llm 停止词
            while message.formatted["tool_type"]:
                message = self.executor(message, session_id=session_id)
                message = self.researcher(message, session_id=session_id, stop_words=["```\n", "```python"])
            if self.finish_pattern in message.content:
                return message
            message = self.charter(message)
            while message.formatted["tool_type"]:
                message = self.executor(message, session_id=session_id)
                message = self.charter(message, session_id=session_id)
            if self.finish_pattern in message.content:
                return message
        return message

visualizer = DataVisualizer(
    "gpt-4o-2024-05-13",
    research_prompt="You should provide accurate data for the chart generator to use.",
    chart_prompt="Any charts you display will be visible by the user.",
)
user_msg = AgentMessage(
    sender='user',
    content="Fetch the China's GDP over the past 5 years, then draw a line graph of it. Once you code it up, finish.")
bot_msg = visualizer(user_msg)
print(bot_msg.content)
json.dump(visualizer.state_dict(), open('visualizer.json', 'w'), ensure_ascii=False, indent=4)
```