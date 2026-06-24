from dotenv import load_dotenv
load_dotenv()

import os
from hello_agents.tools import RAGTool


def rag_call(rag_tool, action, **kwargs):
    if hasattr(rag_tool, "execute"):
        return rag_tool.execute(action, **kwargs)

    if hasattr(rag_tool, "run"):
        try:
            payload = {"action": action}
            payload.update(kwargs)
            return rag_tool.run(payload)
        except TypeError:
            return rag_tool.run(action, **kwargs)

    raise AttributeError("当前 RAGTool 既没有 execute，也没有 run。")


def print_result(title, result):
    print("\n" + "=" * 20)
    print(title)
    print("=" * 20)
    print(result)


def main():
    print("=== 8.3 文件级 RAG 测试 ===")

    os.makedirs("./my_docs", exist_ok=True)

    doc_path = "./my_docs/hello_agents_memory_rag.md"

    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(
            """# HelloAgents 第八章学习笔记

## 记忆系统

MemoryTool 用于让智能体保存和检索记忆。
记忆系统通常包括工作记忆、情景记忆、语义记忆和感知记忆。

工作记忆适合保存当前任务中的临时信息。
情景记忆适合保存具体事件和操作经历。
语义记忆适合保存概念、规则和稳定知识。

## RAG 系统

RAGTool 用于构建检索增强生成系统。
它会先把文档切分成文本块，然后生成向量，最后写入 Qdrant。
用户提问时，系统会先检索相关文本块，再把检索结果提供给大语言模型生成答案。

## Qdrant 与 Neo4j

Qdrant 主要负责向量存储和相似度检索。
Neo4j 主要负责实体关系和知识图谱存储。
"""
        )

    rag_tool = RAGTool(
        knowledge_base_path="./knowledge_base",
        collection_name="rag_8_3_file_test",
        rag_namespace="chapter8_3_file"
    )

    print("\n=== 1. 添加 Markdown 文档 ===")

    result = rag_call(
        rag_tool,
        "add_document",
        file_path=doc_path,
        chunk_size=500,
        chunk_overlap=100
    )
    print_result("添加文档结果", result)

    print("\n=== 2. 搜索文档内容 ===")

    result = rag_call(
        rag_tool,
        "search",
        query="MemoryTool 有哪些记忆类型？",
        limit=3,
        min_score=0.1
    )
    print_result("搜索：MemoryTool 有哪些记忆类型？", result)

    result = rag_call(
        rag_tool,
        "search",
        query="Qdrant 和 Neo4j 分别负责什么？",
        limit=3,
        min_score=0.1
    )
    print_result("搜索：Qdrant 和 Neo4j 分别负责什么？", result)

    print("\n=== 8.3 文件级 RAG 测试完成 ===")


if __name__ == "__main__":
    main()