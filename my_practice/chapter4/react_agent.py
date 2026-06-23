import re
import sys
from pathlib import Path

# 让当前文件可以正确导入同目录下的 llm_client.py 和 tools.py
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from llm_client import HelloAgentsLLM
from tools import ToolExecutor, search


REACT_PROMPT_TEMPLATE = """
你是一个可以调用外部工具的智能助手。

你需要严格按照 ReAct 范式解决问题：

Thought: 分析当前问题，说明你下一步要做什么。
Action: 执行动作，格式只能是以下两种之一：
- 工具调用：工具名[工具输入]
- 结束回答：Finish[最终答案]

可用工具如下：
{tools}

用户问题：
{question}

历史记录：
{history}

请严格输出：
Thought: ...
Action: ...
"""


class ReActAgent:
    """
    ReAct Agent 主体：
    1. 把问题、工具、历史记录组合成提示词
    2. 调用 LLM
    3. 解析 Thought 和 Action
    4. 如果 Action 是工具调用，则执行工具并得到 Observation
    5. 把 Action 和 Observation 加入历史记录
    6. 重复以上过程，直到模型输出 Finish[最终答案]
    """

    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        self.history = []

        for step in range(1, self.max_steps + 1):
            print(f"\n========== 第 {step} 步 ==========")

            tools_desc = self.tool_executor.get_available_tools()
            history_str = "\n".join(self.history)

            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str,
            )

            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("错误：LLM 没有返回有效内容。")
                return None

            print("\n--- LLM 原始输出 ---")
            print(response_text)

            thought, action = self._parse_output(response_text)

            if thought:
                print("\n--- Thought ---")
                print(thought)

            if not action:
                print("\n错误：没有解析到 Action。")
                print("建议：检查模型是否严格输出了 Action: ...")
                return None

            print("\n--- Action ---")
            print(action)

            # 如果模型认为已经完成任务
            if action.startswith("Finish"):
                final_answer = self._parse_finish(action)
                print("\n========== 最终答案 ==========")
                print(final_answer)
                return final_answer

            # 否则解析工具调用
            tool_name, tool_input = self._parse_action(action)

            if not tool_name or tool_input is None:
                observation = f"错误：Action 格式不正确。收到的 Action 是：{action}"
                print("\n--- Observation ---")
                print(observation)
                self.history.append(f"Action: {action}")
                self.history.append(f"Observation: {observation}")
                continue

            tool_function = self.tool_executor.get_tool(tool_name)

            if not tool_function:
                observation = f"错误：没有找到名为 {tool_name} 的工具。"
            else:
                observation = tool_function(tool_input)

            print("\n--- Observation ---")
            print(observation)

            # 把本轮行动和观察结果加入历史，让下一轮 LLM 能看到
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        print("\n达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        """
        从 LLM 输出中解析 Thought 和 Action。
        """
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\nAction:|$)",
            text,
            re.DOTALL,
        )

        action_match = re.search(
            r"Action:\s*(.*?)$",
            text,
            re.DOTALL,
        )

        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None

        return thought, action

    def _parse_action(self, action_text: str):
        """
        把 Search[关键词] 解析成：
        tool_name = Search
        tool_input = 关键词
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)

        if not match:
            return None, None

        tool_name = match.group(1).strip()
        tool_input = match.group(2).strip()

        return tool_name, tool_input

    def _parse_finish(self, action_text: str):
        """
        把 Finish[最终答案] 解析成最终答案。
        """
        match = re.match(r"Finish\[(.*)\]", action_text, re.DOTALL)

        if not match:
            return action_text

        return match.group(1).strip()


if __name__ == "__main__":
    # 1. 初始化 LLM
    llm = HelloAgentsLLM()

    # 2. 初始化工具执行器
    tool_executor = ToolExecutor()

    # 3. 注册 Search 工具
    tool_executor.register_tool(
        name="Search",
        description="网页搜索工具。当你需要查询实时信息、新闻、产品、事实资料时，使用这个工具。",
        func=search,
    )

    # 4. 初始化 ReAct Agent
    agent = ReActAgent(
        llm_client=llm,
        tool_executor=tool_executor,
        max_steps=5,
    )

    # 5. 运行问题
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)