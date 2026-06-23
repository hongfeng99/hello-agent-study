# chapter7/agents/react_agent.py

import re
from typing import Any, Optional, Tuple, List

from chapter7.core import Agent, Message, Config
from chapter7.tools import ToolRegistry, CalculatorTool


MY_REACT_PROMPT = """
你是一个具备推理和工具调用能力的 AI 助手。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式回答：

Thought: 你的思考过程
Action: 工具名[工具输入]

或者当你已经得到最终答案时，使用：

Thought: 我已经得到最终答案
Action: Finish[最终答案]

## 规则
1. 每次只能执行一个 Action。
2. 工具调用必须使用：工具名[工具输入]。
3. 最终回答必须使用：Finish[最终答案]。
4. 如果工具返回 Observation，请基于 Observation 继续推理。

## 用户问题
{question}

## 历史执行记录
{history}

现在请开始：
"""


class MyReActAgent(Agent):
    """
    使用 ToolRegistry 的 ReAct Agent。

    流程：
    Thought -> Action -> Observation -> Thought -> Action -> Finish
    """

    def __init__(
        self,
        name: str,
        llm: Any,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or "你是一个会推理并能调用工具的 AI 助手。",
            config=config,
        )

        self.max_steps = max_steps
        self.prompt_template = custom_prompt or MY_REACT_PROMPT

        if tool_registry is None:
            tool_registry = ToolRegistry()
            tool_registry.register_tool(CalculatorTool())

        self.tool_registry = tool_registry
        self.current_history: List[str] = []

    def _ask_llm(self, prompt: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.llm.invoke(messages, **kwargs)

    def _parse_action(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        action_match = re.search(r"Action\s*:\s*(.*)", text, re.S)

        if not action_match:
            return None, None

        action_text = action_match.group(1).strip()

        finish_match = re.search(r"Finish\[(.*)\]", action_text, re.S)
        if finish_match:
            return "Finish", finish_match.group(1).strip()

        tool_match = re.search(r"(\w+)\[(.*)\]", action_text, re.S)
        if tool_match:
            return tool_match.group(1).strip(), tool_match.group(2).strip()

        return None, None

    def run(self, input_text: str, **kwargs) -> str:
        self.current_history = []

        for step in range(self.max_steps):
            print(f"{self.name} 正在执行 ReAct 步骤 {step + 1}...")

            prompt = self.prompt_template.format(
                tools=self.tool_registry.get_tools_description(),
                question=input_text,
                history="\n".join(self.current_history) if self.current_history else "暂无",
            )

            response = self._ask_llm(prompt, **kwargs)

            self.current_history.append(response)

            print("模型输出：")
            print(response)

            tool_name, tool_input = self._parse_action(response)

            if tool_name == "Finish":
                final_answer = tool_input

                self.add_message(Message(content=input_text, role="user"))
                self.add_message(Message(content=final_answer, role="assistant"))

                return final_answer

            if tool_name:
                observation = self.tool_registry.execute_tool(tool_name, tool_input)
                observation_text = f"Observation: {observation}"

                print(observation_text)

                self.current_history.append(observation_text)
                continue

            error_text = "Observation: 未能解析 Action，请严格使用 Action: 工具名[输入] 或 Action: Finish[答案]。"
            print(error_text)
            self.current_history.append(error_text)

        final_answer = "抱歉，我无法在限定步数内完成这个任务。"

        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=final_answer, role="assistant"))

        return final_answer