# test_7_5_react_with_tools.py

from dotenv import load_dotenv

from my_llm import MyLLM
from chapter7.agents import MyReActAgent
from chapter7.tools import ToolRegistry, CalculatorTool, SearchTool


load_dotenv()


def test_react_with_calculator():
    print("\n====== 1. ReAct + Calculator 测试 ======")

    llm = MyLLM(provider="modelscope")

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    agent = MyReActAgent(
        name="ReAct工具助手",
        llm=llm,
        tool_registry=registry,
        max_steps=5,
    )

    result = agent.run("请计算 sqrt(16) + 2 * 3，并给出最终答案。")

    print("\n最终结果：")
    print(result)

    print("\n历史消息数量：")
    print(len(agent.get_history()))


def test_react_with_search():
    print("\n====== 2. ReAct + Search 测试 ======")

    llm = MyLLM(provider="modelscope")

    registry = ToolRegistry()
    registry.register_tool(SearchTool())

    agent = MyReActAgent(
        name="ReAct搜索助手",
        llm=llm,
        tool_registry=registry,
        max_steps=5,
    )

    result = agent.run("请搜索 HelloAgents 项目的主要内容，并用一句话总结。")

    print("\n最终结果：")
    print(result)

    print("\n历史消息数量：")
    print(len(agent.get_history()))


if __name__ == "__main__":
    test_react_with_calculator()

    # 如果你的 SERPAPI_API_KEY 可用，可以取消下面这一行注释：
    test_react_with_search()

    print("\n====== 7.5 ReAct 接入工具系统测试完成 ======")