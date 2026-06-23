# chapter7/tools/builtin/calculator.py

"""计算器工具"""

import ast
import math
import operator
from typing import Any, Dict, List

from chapter7.tools.base import Tool, ToolParameter


class CalculatorTool(Tool):
    """
    安全计算器工具。

    支持：
    1. 加减乘除；
    2. 幂运算；
    3. 取模；
    4. sqrt、sin、cos、tan、log；
    5. pi、e 常数。
    """

    def __init__(self):
        super().__init__(
            name="calculator",
            description="数学计算工具，支持四则运算、幂运算、sqrt、sin、cos、tan、log、pi、e。",
        )

        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        self.functions = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "abs": abs,
            "round": round,
        }

        self.constants = {
            "pi": math.pi,
            "e": math.e,
        }

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="要计算的数学表达式，例如：sqrt(16) + 2 * 3",
                required=True,
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        expression = str(parameters.get("expression", "")).strip()

        if not expression:
            return "计算表达式不能为空。"

        try:
            node = ast.parse(expression, mode="eval")
            result = self._eval_node(node.body)
            return str(result)

        except ZeroDivisionError:
            return "计算失败：除数不能为 0。"
        except Exception as e:
            return f"计算失败：{e}"

    def _eval_node(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("只允许数字常量。")

        if isinstance(node, ast.Name):
            if node.id in self.constants:
                return self.constants[node.id]
            raise ValueError(f"不支持的变量：{node.id}")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            op_type = type(node.op)
            if op_type not in self.operators:
                raise ValueError(f"不支持的运算符：{op_type}")

            return self.operators[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)

            if op_type not in self.operators:
                raise ValueError(f"不支持的一元运算符：{op_type}")

            return self.operators[op_type](operand)

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("不支持复杂函数调用。")

            func_name = node.func.id

            if func_name not in self.functions:
                raise ValueError(f"不支持的函数：{func_name}")

            args = [self._eval_node(arg) for arg in node.args]
            return self.functions[func_name](*args)

        raise ValueError(f"不支持的表达式类型：{type(node).__name__}")


def create_calculator_registry():
    """
    创建只包含计算器工具的注册表。
    """
    from chapter7.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    return registry