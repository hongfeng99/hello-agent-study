# test_7_5_4_advanced_tools.py

from dotenv import load_dotenv

from chapter7.tools import (
    ToolRegistry,
    CalculatorTool,
    ToolChain,
    ToolChainManager,
    AsyncToolExecutor,
)


load_dotenv()


def test_tool_chain():
    print("\n====== 1. 测试工具链 ToolChain ======")

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    chain = ToolChain(
        name="两步计算链",
        description="先计算一个表达式，再基于结果继续计算。",
    )

    chain.add_step(
        tool_name="calculator",
        input_template="{input}",
        output_key="first_result",
    )

    chain.add_step(
        tool_name="calculator",
        input_template="{first_result} * 2",
        output_key="second_result",
    )

    manager = ToolChainManager(registry)
    manager.register_chain(chain)

    result = manager.execute_chain("两步计算链", "10 + 5")

    print("工具链最终结果：")
    print(result)


def test_async_executor():
    print("\n====== 2. 测试异步工具执行 AsyncToolExecutor ======")

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    executor = AsyncToolExecutor(registry)

    tasks = [
        {
            "tool_name": "calculator",
            "parameters": "2 + 3 * 4",
        },
        {
            "tool_name": "calculator",
            "parameters": "sqrt(16) + 6",
        },
        {
            "tool_name": "calculator",
            "parameters": "100 / 4 + 8",
        },
    ]

    results = executor.run_many(tasks)

    for idx, item in enumerate(results, start=1):
        print(f"\n任务 {idx}")
        print(f"工具：{item['tool_name']}")
        print(f"输入：{item['parameters']}")
        print(f"结果：{item['result']}")


if __name__ == "__main__":
    test_tool_chain()
    test_async_executor()

    print("\n====== 7.5.4 工具系统高级特性测试完成 ======")