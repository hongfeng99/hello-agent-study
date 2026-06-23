import os
from typing import Callable, Dict

from dotenv import load_dotenv
from serpapi import SerpApiClient


load_dotenv()


def search(query: str) -> str:
    """
    基于 SerpApi 的网页搜索工具。
    输入：搜索关键词
    输出：搜索结果摘要
    """
    print(f"正在执行 [Search] 网页搜索：{query}")

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "错误：SERPAPI_API_KEY 未在 .env 文件中配置。"

    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-cn",
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        # 1. 优先返回直接答案
        if "answer_box" in results:
            answer_box = results["answer_box"]
            if "answer" in answer_box:
                return answer_box["answer"]
            if "snippet" in answer_box:
                return answer_box["snippet"]

        # 2. 其次返回知识图谱描述
        if "knowledge_graph" in results:
            knowledge_graph = results["knowledge_graph"]
            if "description" in knowledge_graph:
                return knowledge_graph["description"]

        # 3. 最后返回普通搜索结果前三条
        if "organic_results" in results and results["organic_results"]:
            snippets = []
            for i, res in enumerate(results["organic_results"][:3]):
                title = res.get("title", "")
                snippet = res.get("snippet", "")
                link = res.get("link", "")
                snippets.append(f"[{i + 1}] {title}\n{snippet}\n{link}")
            return "\n\n".join(snippets)

        return "没有找到有效搜索结果。"

    except Exception as e:
        return f"搜索工具执行失败：{e}"


class ToolExecutor:
    """
    工具执行器：
    负责注册工具、展示工具、根据工具名调用工具。
    """

    def __init__(self):
        self.tools: Dict[str, Dict[str, object]] = {}

    def register_tool(self, name: str, description: str, func: Callable[[str], str]):
        self.tools[name] = {
            "description": description,
            "func": func,
        }
        print(f"工具 '{name}' 已注册。")

    def get_tool(self, name: str):
        tool_info = self.tools.get(name)
        if not tool_info:
            return None
        return tool_info["func"]

    def get_available_tools(self) -> str:
        if not self.tools:
            return "当前没有可用工具。"

        return "\n".join(
            [
                f"- {name}: {info['description']}"
                for name, info in self.tools.items()
            ]
        )


if __name__ == "__main__":
    tool_executor = ToolExecutor()

    tool_executor.register_tool(
        name="Search",
        description="网页搜索工具。当你需要查询实时信息、新闻、产品、事实资料时，使用这个工具。",
        func=search,
    )

    print("\n--- 当前可用工具 ---")
    print(tool_executor.get_available_tools())

    print("\n--- 测试 Search 工具 ---")
    tool = tool_executor.get_tool("Search")
    result = tool("英伟达最新的GPU型号是什么")
    print("\n--- Observation ---")
    print(result)