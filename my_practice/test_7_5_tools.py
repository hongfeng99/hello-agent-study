# test_7_5_tools.py

from dotenv import load_dotenv

from chapter7.tools import (
    ToolRegistry,
    CalculatorTool,
    SearchTool,
    ToolChain,
    ToolChainManager,
)


load_dotenv()


def test_calculator_tool():
    print("\n====== 1. 测试 CalculatorTool ======")

    calculator = CalculatorTool()

    print(calculator)
    print("参数定义：")
    print(calculator.get_parameters())

    result_1 = calculator.run({"expression": "2 + 3 * 4"})
    print("2 + 3 * 4 =", result_1)

    result_2 = calculator.run({"expression": "sqrt(16) + 2 * 3"})
    print("sqrt(16) + 2 * 3 =", result_2)

    result_3 = calculator.run({"expression": "pi * 2"})
    print("pi * 2 =", result_3)

    print("OpenAI schema：")
    print(calculator.to_openai_schema())


def test_registry():
    print("\n====== 2. 测试 ToolRegistry ======")

    registry = ToolRegistry()

    registry.register_tool(CalculatorTool())

    def echo(text: str) -> str:
        return f"Echo: {text}"

    registry.register_function(
        name="echo",
        description="回声工具，返回输入内容。",
        func=echo,
    )

    print("工具列表：")
    print(registry.list_tools())

    print("工具描述：")
    print(registry.get_tools_description())

    print("执行 calculator：")
    print(registry.execute_tool("calculator", "10 + 20 * 3"))

    print("执行 echo：")
    print(registry.execute_tool("echo", "你好，工具系统"))

    print("OpenAI tools schema：")
    print(registry.to_openai_tools())


def test_search_tool():
    print("\n====== 3. 测试 SearchTool ======")

    registry = ToolRegistry()
    registry.register_tool(SearchTool())

    print("工具描述：")
    print(registry.get_tools_description())

    print("搜索测试：")
    result = registry.execute_tool("search", "HelloAgents 工具系统")
    print(result)


def test_tool_chain():
    print("\n====== 4. 测试 ToolChain ======")

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    chain = ToolChain(
        name="计算链",
        description="先计算输入表达式，然后返回计算结果。",
    )

    chain.add_step(
        tool_name="calculator",
        input_template="{input}",
        output_key="calc_result",
    )

    manager = ToolChainManager(registry)
    manager.register_chain(chain)

    result = manager.execute_chain("计算链", "100 / 4 + 6")
    print("工具链最终结果：")
    print(result)


if __name__ == "__main__":
    test_calculator_tool()
    test_registry()
    test_search_tool()
    test_tool_chain()

    print("\n====== 7.5 工具系统测试完成 ======")