# chapter7/tools/base.py

"""工具基类与参数定义"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ToolParameter(BaseModel):
    """
    工具参数定义。

    用来描述一个工具需要哪些参数。
    例如 calculator 工具需要 expression 参数。
    """

    name: str
    type: str = "string"
    description: str
    required: bool = True
    default: Optional[Any] = None


class Tool(ABC):
    """
    工具抽象基类。

    所有工具都应该继承 Tool，并实现：
    1. run(parameters)
    2. get_parameters()
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """
        执行工具。

        parameters 是字典，例如：
        {"expression": "2 + 3 * 4"}
        """
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """
        返回工具参数定义。
        """
        pass

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        转换成 OpenAI function calling 兼容格式。

        这一部分后续 FunctionCallAgent 会用到。
        现在先实现，方便理解工具自描述机制。
        """
        properties = {}
        required = []

        for param in self.get_parameters():
            prop = {
                "type": param.type,
                "description": param.description,
            }

            if param.default is not None:
                prop["description"] = f"{param.description}，默认值：{param.default}"

            if param.type == "array":
                prop["items"] = {"type": "string"}

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def __str__(self) -> str:
        return f"Tool(name={self.name}, description={self.description})"