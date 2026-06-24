from dotenv import load_dotenv
load_dotenv()

from hello_agents.tools import RAGTool


def rag_call(rag_tool, action, **kwargs):
    """
    兼容不同版本的 RAGTool：
    官方文档中常见 execute(action, **kwargs)，
    你当前本地版本没有 execute，所以优先使用 run。
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

    raise AttributeError("当前 RAGTool 既没有 execute，也没有 run。")


def print_result(title, result):
    print("\n" + "=" * 20)
    print(title)
    print("=" * 20)
    print(result)


def safe_ask(rag_tool, question):
    """
    先尝试 ask。
    如果当前版本 ask 参数不兼容，则退回到 search。
    """
    print("\n问题：", question)

    try:
        result = rag_call(
            rag_tool,
            "ask",
            question=question,
            limit=3,
            enable_advanced_search=False
        )
        print_result("RAG 生成答案", result)
        return
    except Exception as e:
        print("\nask 调用失败，退回到 search。")
        print("错误信息：", repr(e))

    result = rag_call(
        rag_tool,
        "search",
        query=question,
        limit=3,
        min_score=0.1
    )
    print_result("退回检索结果", result)


def main():
    print("=== 8.3 RAG ask 问答测试 ===")

    rag_tool = RAGTool(
        knowledge_base_path="./knowledge_base",
        collection_name="rag_8_3_test",
        rag_namespace="chapter8_3"
    )

    questions = [
        "Python 是谁发明的？",
        "机器学习主要包括哪些类型？",
        "RAG 的工作流程是什么？"
    ]

    for question in questions:
        safe_ask(rag_tool, question)

    print("\n=== 8.3 RAG ask 问答测试完成 ===")


if __name__ == "__main__":
    main()