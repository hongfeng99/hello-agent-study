# chapter7/tools/builtin/search.py

"""搜索工具"""

import os
from typing import Any, Dict, List, Optional

import requests

from chapter7.tools.base import Tool, ToolParameter


class SearchTool(Tool):
    """
    搜索工具。

    当前实现：
    1. 支持 SerpApi；
    2. 如果没有 SERPAPI_API_KEY，则给出明确提示；
    3. 返回格式统一的搜索结果文本。
    """

    def __init__(self, serpapi_key: Optional[str] = None):
        super().__init__(
            name="search",
            description="网页搜索工具，用于查询最新信息、事实资料、背景知识。",
        )

        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="搜索关键词，例如：HelloAgents 工具系统",
                required=True,
            )
        ]

    def run(self, parameters: Dict[str, Any]) -> str:
        query = str(parameters.get("query", "")).strip()

        if not query:
            return "搜索关键词不能为空。"

        if not self.serpapi_key:
            return "搜索失败：未配置 SERPAPI_API_KEY。请在 .env 文件中添加 SERPAPI_API_KEY。"

        try:
            return self._search_serpapi(query)
        except Exception as e:
            return f"搜索失败：{e}"

    def _search_serpapi(self, query: str) -> str:
        url = "https://serpapi.com/search.json"

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
            "hl": "zh-cn",
            "num": 5,
        }

        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()

        data = response.json()

        organic_results = data.get("organic_results", [])

        if not organic_results:
            return "未找到相关搜索结果。"

        lines = [f"搜索关键词：{query}", ""]

        for idx, item in enumerate(organic_results[:5], start=1):
            title = item.get("title", "无标题")
            link = item.get("link", "无链接")
            snippet = item.get("snippet", "无摘要")

            lines.append(f"{idx}. {title}")
            lines.append(f"   摘要：{snippet}")
            lines.append(f"   链接：{link}")
            lines.append("")

        return "\n".join(lines)


def create_search_registry():
    """
    创建只包含搜索工具的注册表。
    """
    from chapter7.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_tool(SearchTool())

    return registry