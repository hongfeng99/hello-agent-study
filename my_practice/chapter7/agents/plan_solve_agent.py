# chapter7/agents/plan_solve_agent.py

import ast
import re
from typing import Any, Optional, Dict, List

from chapter7.core import Agent, Message, Config


DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。

请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。

你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题：{question}

请严格按照以下格式输出你的计划：

["步骤1", "步骤2", "步骤3"]
"""


DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。

# 原始问题：
{question}

# 完整计划：
{plan}

# 历史步骤与结果：
{history}

# 当前步骤：
{current_step}

请仅输出针对“当前步骤”的回答，不要输出额外解释。
"""


DEFAULT_SUMMARIZER_PROMPT = """
请根据以下问题、计划和每一步的执行结果，给出最终答案。

# 原始问题：
{question}

# 计划：
{plan}

# 执行结果：
{history}

请给出清晰、准确的最终答案。
"""


class MyPlanAndSolveAgent(Agent):
    """
    Plan-and-Solve Agent。

    执行流程：
    1. Planner：生成步骤列表；
    2. Executor：逐步执行；
    3. Summarizer：汇总最终答案。
    """

    def __init__(
        self,
        name: str,
        llm: Any,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or "你是一个擅长规划和逐步解决问题的AI助手。",
            config=config
        )

        self.prompts = {
            "planner": DEFAULT_PLANNER_PROMPT,
            "executor": DEFAULT_EXECUTOR_PROMPT,
            "summarizer": DEFAULT_SUMMARIZER_PROMPT,
        }

        if custom_prompts:
            self.prompts.update(custom_prompts)

    def _ask_llm(self, prompt: str, **kwargs) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.llm.invoke(messages, **kwargs)

    def _extract_plan(self, text: str) -> List[str]:
        """
        从模型输出中提取 Python list。
        """
        text = text.strip()

        try:
            result = ast.literal_eval(text)
            if isinstance(result, list):
                return [str(item) for item in result]
        except Exception:
            pass

        pattern = r"\[.*?\]"
        match = re.search(pattern, text, re.S)

        if match:
            try:
                result = ast.literal_eval(match.group())
                if isinstance(result, list):
                    return [str(item) for item in result]
            except Exception:
                pass

        # 兜底：按行拆分
        lines = [
            line.strip("-0123456789.、 ")
            for line in text.splitlines()
            if line.strip()
        ]

        return lines[:5] if lines else ["直接回答问题"]

    def run(self, input_text: str, **kwargs) -> str:
        print(f"{self.name} 正在规划任务...")

        planner_prompt = self.prompts["planner"].format(question=input_text)
        raw_plan = self._ask_llm(planner_prompt, **kwargs)

        plan = self._extract_plan(raw_plan)

        print("生成的计划：")
        for i, step in enumerate(plan, start=1):
            print(f"{i}. {step}")

        results = []

        for i, step in enumerate(plan, start=1):
            print(f"{self.name} 正在执行步骤 {i}：{step}")

            history_text = "\n".join(
                [f"步骤 {idx + 1}: {item['step']}\n结果: {item['result']}" for idx, item in enumerate(results)]
            ) or "暂无"

            executor_prompt = self.prompts["executor"].format(
                question=input_text,
                plan=plan,
                history=history_text,
                current_step=step
            )

            step_result = self._ask_llm(executor_prompt, **kwargs)

            results.append({
                "step": step,
                "result": step_result
            })

        final_history = "\n".join(
            [f"步骤 {idx + 1}: {item['step']}\n结果: {item['result']}" for idx, item in enumerate(results)]
        )

        summarizer_prompt = self.prompts["summarizer"].format(
            question=input_text,
            plan=plan,
            history=final_history
        )

        final_answer = self._ask_llm(summarizer_prompt, **kwargs)

        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=final_answer, role="assistant"))

        return final_answer