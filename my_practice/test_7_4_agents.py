# test_7_4_agents.py

from dotenv import load_dotenv

from my_llm import MyLLM
from chapter7.agents import (
    MySimpleAgent,
    MyReflectionAgent,
    MyPlanAndSolveAgent,
    MyReActAgent,
)


load_dotenv()

llm = MyLLM(provider="modelscope")


def test_simple_agent():
    print("\n====== 1. 测试 MySimpleAgent ======")

    agent = MySimpleAgent(
        name="简单对话助手",
        llm=llm,
        system_prompt="你是一个适合初学者的人工智能助教，请用通俗语言回答。"
    )

    result = agent.run("请用两句话解释什么是 Agent。")
    print("\n最终结果：")
    print(result)
    print(f"历史消息数：{len(agent.get_history())}")


def test_reflection_agent():
    print("\n====== 2. 测试 MyReflectionAgent ======")

    agent = MyReflectionAgent(
        name="反思助手",
        llm=llm,
        max_reflections=1
    )

    result = agent.run("请写一段简短说明：为什么要学习智能体框架？")
    print("\n最终结果：")
    print(result)
    print(f"历史消息数：{len(agent.get_history())}")


def test_plan_solve_agent():
    print("\n====== 3. 测试 MyPlanAndSolveAgent ======")

    agent = MyPlanAndSolveAgent(
        name="规划执行助手",
        llm=llm
    )

    question = "一个水果店周一卖出15个苹果，周二卖出的是周一的两倍，周三比周二少5个。三天总共卖出多少个苹果？"

    result = agent.run(question)
    print("\n最终结果：")
    print(result)
    print(f"历史消息数：{len(agent.get_history())}")


def test_react_agent():
    print("\n====== 4. 测试 MyReActAgent ======")

    agent = MyReActAgent(
        name="ReAct计算助手",
        llm=llm,
        max_steps=5
    )

    result = agent.run("请计算 15 * 8 + 32，并给出最终答案。")
    print("\n最终结果：")
    print(result)
    print(f"历史消息数：{len(agent.get_history())}")


if __name__ == "__main__":
    test_simple_agent()
    test_reflection_agent()
    test_plan_solve_agent()
    test_react_agent()

    print("\n====== 7.4 Agent范式框架化实现测试完成 ======")