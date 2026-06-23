# chapter7/tools/chain.py

"""工具链"""

from typing import Any, Dict, List, Optional

from .registry import ToolRegistry


class ToolChain:
    """
    工具链。

    支持多个工具按顺序执行。
    后一个工具可以使用前一个工具的输出。
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []

    def add_step(
        self,
        tool_name: str,
        input_template: str,
        output_key: Optional[str] = None,
    ):
        """
        添加工具执行步骤。

        input_template 支持变量替换，例如：
        "{input}"
        "{step_0_result}"
        """
        self.steps.append(
            {
                "tool_name": tool_name,
                "input_template": input_template,
                "output_key": output_key or f"step_{len(self.steps)}_result",
            }
        )

    def execute(
        self,
        registry: ToolRegistry,
        initial_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        执行工具链。
        """
        context = context or {}
        context["input"] = initial_input

        if not self.steps:
            return "工具链为空，无法执行。"

        print(f"开始执行工具链：{self.name}")

        for index, step in enumerate(self.steps, start=1):
            tool_name = step["tool_name"]
            input_template = step["input_template"]
            output_key = step["output_key"]

            try:
                tool_input = input_template.format(**context)
            except KeyError as e:
                return f"工具链执行失败：模板变量 {e} 不存在。"

            print(f"步骤 {index}：调用工具 {tool_name}，输入：{tool_input}")

            result = registry.execute_tool(tool_name, tool_input)
            context[output_key] = result

            print(f"步骤 {index} 完成，输出保存为：{output_key}")

        final_key = self.steps[-1]["output_key"]
        return str(context[final_key])


class ToolChainManager:
    """
    工具链管理器。
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: Dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain):
        self.chains[chain.name] = chain
        print(f"工具链 {chain.name} 已注册。")

    def execute_chain(self, name: str, initial_input: str) -> str:
        chain = self.chains.get(name)

        if chain is None:
            return f"工具链不存在：{name}"

        return chain.execute(self.registry, initial_input)