# chapter7/tools/registry.py

"""工具注册表"""

from typing import Any, Callable, Dict, List, Optional

from .base import Tool, ToolParameter


class FunctionTool(Tool):
    """
    把普通 Python 函数包装成 Tool。

    适合简单工具：
    输入一个字符串，返回一个字符串。
    """

    def __init__(self, name: str, description: str, func: Callable[[str], str]):
        super().__init__(name=name, description=description)
        self.func = func

    def run(self, parameters: Dict[str, Any]) -> str:
        tool_input = parameters.get("input", "")

        try:
            return str(self.func(str(tool_input)))
        except Exception as e:
            return f"工具 {self.name} 执行失败：{e}"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="input",
                type="string",
                description="工具输入内容",
                required=True,
            )
        ]


class ToolRegistry:
    """
    工具注册表。

    负责：
    1. 注册工具；
    2. 展示工具描述；
    3. 执行工具；
    4. 转换 OpenAI function calling schema。
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """
        注册 Tool 对象。
        """
        if tool.name in self._tools:
            print(f"警告：工具 {tool.name} 已存在，将被覆盖。")

        self._tools[tool.name] = tool
        print(f"工具 {tool.name} 已注册。")

    def register_function(self, name: str, description: str, func: Callable[[str], str]):
        """
        直接注册普通函数。
        """
        function_tool = FunctionTool(
            name=name,
            description=description,
            func=func,
        )
        self.register_tool(function_tool)

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        按名称获取工具。
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """
        返回所有工具名称。
        """
        return list(self._tools.keys())

    def get_tools_description(self) -> str:
        """
        生成工具描述文本，供 ReActAgent 写入提示词。

        例如：
        - calculator: 用于数学计算
        - search: 用于网页搜索
        """
        if not self._tools:
            return "暂无可用工具"

        descriptions = []

        for tool in self._tools.values():
            params = tool.get_parameters()
            params_desc = ", ".join(
                [
                    f"{p.name}: {p.description}"
                    for p in params
                ]
            )

            descriptions.append(
                f"- {tool.name}: {tool.description}。参数：{params_desc}"
            )

        return "\n".join(descriptions)

    def execute_tool(self, name: str, parameters: Any) -> str:
        """
        执行指定工具。

        parameters 可以是：
        1. dict，例如 {"expression": "2 + 3"}
        2. str，例如 "2 + 3"

        如果是 str，会自动映射到工具的第一个参数。
        """
        tool = self.get_tool(name)

        if tool is None:
            return f"工具不存在：{name}"

        try:
            if isinstance(parameters, dict):
                tool_parameters = parameters
            else:
                tool_parameters = self._convert_string_to_parameters(tool, str(parameters))

            return tool.run(tool_parameters)

        except Exception as e:
            return f"工具 {name} 执行失败：{e}"

    def _convert_string_to_parameters(self, tool: Tool, text: str) -> Dict[str, Any]:
        """
        把字符串输入转换成工具参数字典。

        例如：
        calculator[2 + 3]
        会转成：
        {"expression": "2 + 3"}
        """
        parameters = tool.get_parameters()

        if not parameters:
            return {"input": text}

        first_param_name = parameters[0].name
        return {first_param_name: text}

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """
        转换全部工具为 OpenAI function calling 工具列表。
        """
        return [tool.to_openai_schema() for tool in self._tools.values()]