# chapter7/agents/reflection_agent.py

from typing import Any, Optional, Dict

from chapter7.core import Agent, Message, Config


DEFAULT_PROMPTS = {
    "initial": """
请根据以下要求完成任务：

任务：{task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间：

# 原始任务：
{task}

# 当前回答：
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答“无需改进”。
""",
    "refine": """
请根据反馈意见改进你的回答：

# 原始任务：
{task}

# 上一轮回答：
{last_attempt}

# 反馈意见：
{feedback}

请提供一个改进后的回答。
"""
}


class MyReflectionAgent(Agent):
    """
    Reflection Agent。

    执行流程：
    1. initial：先生成初始回答；
    2. reflect：反思初始回答；
    3. refine：根据反思意见改进回答。
    """

    def __init__(
        self,
        name: str,
        llm: Any,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
        max_reflections: int = 1,
    ):
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or "你是一个善于自我反思和改进答案的AI助手。",
            config=config
        )

        self.prompts = DEFAULT_PROMPTS.copy()

        if custom_prompts:
            self.prompts.update(custom_prompts)

        self.max_reflections = max_reflections

    def _ask_llm(self, prompt: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.llm.invoke(messages, **kwargs)

    def run(self, input_text: str, **kwargs) -> str:
        print(f"{self.name} 正在生成初始回答...")

        initial_prompt = self.prompts["initial"].format(task=input_text)
        current_answer = self._ask_llm(initial_prompt, **kwargs)

        for i in range(self.max_reflections):
            print(f"{self.name} 正在进行第 {i + 1} 轮反思...")

            reflect_prompt = self.prompts["reflect"].format(
                task=input_text,
                content=current_answer
            )

            feedback = self._ask_llm(reflect_prompt, **kwargs)

            if "无需改进" in feedback:
                break

            print(f"{self.name} 正在根据反思意见改进回答...")

            refine_prompt = self.prompts["refine"].format(
                task=input_text,
                last_attempt=current_answer,
                feedback=feedback
            )

            current_answer = self._ask_llm(refine_prompt, **kwargs)

        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=current_answer, role="assistant"))

        return current_answer