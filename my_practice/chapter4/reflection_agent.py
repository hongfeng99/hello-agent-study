import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from llm_client import HelloAgentsLLM
from memory import Memory


INITIAL_PROMPT_TEMPLATE = """
你是一位资深的 Python 程序员。请根据以下要求，编写一个 Python 函数。

要求：
1. 代码必须包含完整的函数签名。
2. 代码必须包含文档字符串。
3. 代码必须遵循 PEP 8 编码规范。
4. 请直接输出代码，不要输出解释。

任务：
{task}
"""


REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师。

你的任务是审查下面这段 Python 代码，并重点分析它在“算法效率”方面的问题。

# 原始任务
{task}

# 待审查代码
【待审查代码开始】
{code}
【待审查代码结束】

请完成以下事情：
1. 分析当前代码的时间复杂度。
2. 判断当前算法是否存在明显的效率瓶颈。
3. 如果存在更优算法，请明确指出改进方向。
4. 如果当前代码在算法层面已经足够好，请回答“无需改进”。

请直接输出反馈，不要输出多余解释。
"""


REFINE_PROMPT_TEMPLATE = """
你是一位资深的 Python 程序员。

你需要根据代码评审专家的反馈，优化上一轮生成的代码。

# 原始任务
{task}

# 上一轮代码
【上一轮代码开始】
{last_code_attempt}
【上一轮代码结束】

# 评审员反馈
{feedback}

请根据反馈生成优化后的新版本代码。

要求：
1. 代码必须包含完整的函数签名。
2. 代码必须包含文档字符串。
3. 代码必须遵循 PEP 8 编码规范。
4. 请直接输出优化后的代码，不要输出解释。
"""


class ReflectionAgent:
    """
    Reflection Agent:
    1. 初始执行：先生成一个初版代码。
    2. 反思：让模型扮演评审员，指出代码问题。
    3. 优化：根据反馈生成新版本代码。
    4. 循环：直到无需改进，或达到最大迭代次数。
    """

    def __init__(self, llm_client: HelloAgentsLLM, max_iterations: int = 2):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str) -> str:
        print("\n==============================")
        print("开始处理任务")
        print("==============================")
        print(f"任务：{task}")

        # 1. 初始执行
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        print("\n--- 初始代码 ---")
        print(initial_code)

        # 2. 反思与优化循环
        for i in range(self.max_iterations):
            print("\n==============================")
            print(f"第 {i + 1}/{self.max_iterations} 轮迭代")
            print("==============================")

            # a. 获取上一轮代码
            last_code = self.memory.get_last_execution()

            if not last_code:
                print("没有找到上一轮代码，任务终止。")
                return ""

            # b. 反思
            print("\n-> 正在进行反思...")

            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(
                task=task,
                code=last_code,
            )

            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)

            print("\n--- 反思反馈 ---")
            print(feedback)

            # c. 判断是否停止
            if self._should_stop(feedback):
                print("\n反思认为当前代码已无需改进，停止迭代。")
                break

            # d. 根据反馈优化
            print("\n-> 正在根据反馈优化代码...")

            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback,
            )

            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)

            print("\n--- 优化后的代码 ---")
            print(refined_code)

        final_code = self.memory.get_last_execution() or ""

        print("\n==============================")
        print("任务完成")
        print("==============================")
        print("最终生成的代码：")
        print(final_code)

        return final_code

    def _get_llm_response(self, prompt: str) -> str:
        """
        调用 LLM，返回模型输出文本。
        """
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        response_text = self.llm_client.think(messages=messages) or ""
        return response_text.strip()

    def _should_stop(self, feedback: str) -> bool:
        """
        判断是否停止迭代。
        """
        stop_keywords = [
            "无需改进",
            "不需要进一步优化",
            "已经足够好",
            "已经足够高效",
            "没有明显的效率瓶颈",
        ]

        return any(keyword in feedback for keyword in stop_keywords)


if __name__ == "__main__":
    llm = HelloAgentsLLM()

    agent = ReflectionAgent(
        llm_client=llm,
        max_iterations=2,
    )

    task = "编写一个 Python 函数，找出 1 到 n 之间所有的素数 prime numbers。"

    agent.run(task)