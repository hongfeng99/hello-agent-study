# chapter7/tools/async_executor.py

"""异步工具执行器"""

import asyncio
from typing import Any, Dict, List, Optional

from .registry import ToolRegistry


class AsyncToolExecutor:
    """
    异步工具执行器。

    作用：
    1. 并发执行多个工具任务；
    2. 对同步工具进行异步包装；
    3. 避免一个耗时工具阻塞整个 Agent。
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute_tool_async(self, tool_name: str, parameters: Any) -> str:
        """
        异步执行单个工具。

        由于当前工具大多是同步 run() 方法，
        这里用 asyncio.to_thread 把同步执行放到线程里。
        """
        try:
            result = await asyncio.to_thread(
                self.registry.execute_tool,
                tool_name,
                parameters,
            )
            return result
        except Exception as e:
            return f"异步工具 {tool_name} 执行失败：{e}"

    async def execute_many(
        self,
        tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        并发执行多个工具任务。

        tasks 示例：
        [
            {"tool_name": "calculator", "parameters": "2 + 3"},
            {"tool_name": "calculator", "parameters": "10 * 5"}
        ]
        """
        coroutines = []

        for task in tasks:
            tool_name = task.get("tool_name")
            parameters = task.get("parameters", "")

            coroutines.append(
                self.execute_tool_async(tool_name, parameters)
            )

        results = await asyncio.gather(*coroutines)

        output = []

        for task, result in zip(tasks, results):
            output.append(
                {
                    "tool_name": task.get("tool_name"),
                    "parameters": task.get("parameters"),
                    "result": result,
                }
            )

        return output

    def run_many(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        同步入口。

        方便普通 Python 脚本直接调用，不需要手动写 asyncio.run。
        """
        return asyncio.run(self.execute_many(tasks))