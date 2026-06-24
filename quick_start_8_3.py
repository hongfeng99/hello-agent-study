from dotenv import load_dotenv
load_dotenv()

from hello_agents.tools import RAGTool


def rag_call(rag_tool, action, **kwargs):
    """
    兼容不同版本的 RAGTool 调用方式：
    1. 有些文档使用 execute(action, **kwargs)
    2. 你当前版本没有 execute，所以优先尝试 run({...})
    3. 如果 run({...}) 不匹配，再尝试 run(action, **kwargs)
    """
    if hasattr(rag_tool, "execute"):
        return rag_tool.execute(action, **kwargs)

    if hasattr(rag_tool, "run"):
        try:
            payload = {"action": action}
            payload.update(kwargs)
            return rag_tool.run(payload)
        except TypeError:
            return rag_tool.run(action, **kwargs)

    raise AttributeError("当前 RAGTool 既没有 execute 方法，也没有 run 方法。")


def print_result(title, result):
    print("\n" + "=" * 20)
    print(title)
    print("=" * 20)
    print(result)


def main():
    print("=== 8.3 RAGTool 检索增强生成快速体验 ===")

    rag_tool = RAGTool(
        knowledge_base_path="./knowledge_base",
        collection_name="rag_8_3_test",
        rag_namespace="chapter8_3"
    )

    print("\n=== 1. 添加知识 ===")

    result1 = rag_call(
        rag_tool,
        "add_text",
        text="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年首次发布。Python 的设计哲学强调代码的可读性和简洁的语法。",
        document_id="python_intro"
    )
    print_result("知识1：Python 简介", result1)

    result2 = rag_call(
        rag_tool,
        "add_text",
        text="机器学习是人工智能的一个分支，它通过算法让计算机从数据中学习模式。机器学习主要包括监督学习、无监督学习和强化学习。",
        document_id="machine_learning_intro"
    )
    print_result("知识2：机器学习简介", result2)

    result3 = rag_call(
        rag_tool,
        "add_text",
        text="RAG 是 Retrieval-Augmented Generation 的缩写，中文通常称为检索增强生成。它先从知识库中检索相关内容，再把检索结果提供给大语言模型生成答案。",
        document_id="rag_intro"
    )
    print_result("知识3：RAG 简介", result3)

    print("\n=== 2. 搜索知识 ===")

    result = rag_call(
        rag_tool,
        "search",
        query="Python 是谁发明的？",
        limit=3,
        min_score=0.1
    )
    print_result("搜索：Python 是谁发明的？", result)

    result = rag_call(
        rag_tool,
        "search",
        query="机器学习有哪些主要类型？",
        limit=3,
        min_score=0.1
    )
    print_result("搜索：机器学习有哪些主要类型？", result)

    result = rag_call(
        rag_tool,
        "search",
        query="RAG 是怎么工作的？",
        limit=3,
        min_score=0.1
    )
    print_result("搜索：RAG 是怎么工作的？", result)

    print("\n=== 3. 查看知识库统计 ===")

    result = rag_call(rag_tool, "stats")
    print_result("知识库统计", result)

    print("\n=== 8.3 RAGTool 快速体验完成 ===")


if __name__ == "__main__":
    main()