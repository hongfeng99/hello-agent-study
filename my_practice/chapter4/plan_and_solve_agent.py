import ast
import sys
from pathlib import Path
from typing import List


# 让当前文件可以导入同目录下的 llm_client.py
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from llm_client import HelloAgentsLLM


PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。
你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。

要求：
1. 每个步骤必须是一个独立、明确、可执行的子任务。
2. 步骤必须严格按照逻辑顺序排列。
3. 你不要直接回答最终答案，只需要生成计划。
4. 你的输出必须是一个 Python 列表。
5. Python 列表中的每个元素必须是字符串。
6. 必须使用 ```python 和 ``` 包裹输出。

问题：
{question}

请严格按照以下格式输出：

```python
["步骤1", "步骤2", "步骤3"]
```
"""


EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。
你的任务是严格按照给定计划，一步一步解决问题。

你会收到：
1. 原始问题
2. 完整计划
3. 历史步骤与结果
4. 当前步骤

请注意：
- 你只需要完成“当前步骤”。
- 不要跳到后面的步骤。
- 不要重复输出完整计划。
- 只输出当前步骤的结果。

# 原始问题
{question}

# 完整计划
{plan}

# 历史步骤与结果
{history}

# 当前步骤
{current_step}

请仅输出当前步骤的结果：
"""


class Planner:
    """
    规划器：
    接收用户问题，生成一个 Python 列表形式的行动计划。
    """

    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def plan(self, question: str) -> List[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]

        print("\n--- 正在生成计划 ---")
        response_text = self.llm_client.think(messages=messages) or ""

        print("\n--- Planner 原始输出 ---")
        print(response_text)

        plan = self._parse_plan(response_text)

        print("\n--- 解析后的计划 ---")
        for i, step in enumerate(plan, start=1):
            print(f"{i}. {step}")

        return plan

    def _parse_plan(self, response_text: str) -> List[str]:
        """
        从 LLM 输出中解析 Python 列表。
        文档中要求模型输出类似：
        ```python
        ["步骤1", "步骤2", "步骤3"]
        ```
        """

        try:
            if "```python" in response_text:
                plan_str = response_text.split("```python")[1].split("```")[0].strip()
            else:
                # 兼容模型没有输出 markdown 代码块的情况
                start = response_text.find("[")
                end = response_text.rfind("]")
                if start == -1 or end == -1:
                    print("没有找到列表格式的计划。")
                    return []
                plan_str = response_text[start:end + 1]

            plan = ast.literal_eval(plan_str)

            if not isinstance(plan, list):
                print("解析结果不是 Python list。")
                return []

            clean_plan = []
            for item in plan:
                if isinstance(item, str) and item.strip():
                    clean_plan.append(item.strip())

            return clean_plan

        except Exception as e:
            print(f"解析计划失败：{e}")
            print(f"原始响应：{response_text}")
            return []


class Executor:
    """
    执行器：
    严格按照 Planner 生成的计划逐步执行。
    """

    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def execute(self, question: str, plan: List[str]) -> str:
        history = ""
        final_answer = ""

        print("\n--- 正在执行计划 ---")

        for i, current_step in enumerate(plan, start=1):
            print(f"\n-> 正在执行步骤 {i}/{len(plan)}：{current_step}")

            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=current_step,
            )

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages) or ""
            step_result = response_text.strip()

            print(f"\n步骤 {i} 结果：{step_result}")

            history += f"步骤 {i}: {current_step}\n"
            history += f"结果: {step_result}\n\n"

            final_answer = step_result

        return final_answer


class PlanAndSolveAgent:
    """
    Plan-and-Solve 智能体：
    先调用 Planner 生成计划，再调用 Executor 逐步执行计划。
    """

    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client
        self.planner = Planner(llm_client)
        self.executor = Executor(llm_client)

    def run(self, question: str):
        print("\n==============================")
        print("开始处理问题")
        print("==============================")
        print(f"问题：{question}")

        # 1. 规划阶段
        plan = self.planner.plan(question)

        if not plan:
            print("\n任务终止：没有生成有效计划。")
            return ""

        # 2. 执行阶段
        final_answer = self.executor.execute(question, plan)

        print("\n==============================")
        print("任务完成")
        print("==============================")
        print(f"最终答案：{final_answer}")

        return final_answer


if __name__ == "__main__":
    llm = HelloAgentsLLM()

    agent = PlanAndSolveAgent(llm_client=llm)

    question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"

    agent.run(question)