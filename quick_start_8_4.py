from dotenv import load_dotenv
load_dotenv()

import os
import json
from datetime import datetime
from typing import Any, Dict, Optional

from hello_agents.tools import RAGTool, MemoryTool


def tool_call(tool, action, **kwargs):
    """
    兼容不同版本的 Tool 调用方式。

    官方文档中常见：
        tool.execute("action", ...)

    你当前本地版本中：
        RAGTool 没有 execute
        MemoryTool 之前用 run({...}) 可以跑通

    所以这里统一做兼容。
    """
    if hasattr(tool, "execute"):
        return tool.execute(action, **kwargs)

    if hasattr(tool, "run"):
        try:
            payload = {"action": action}
            payload.update(kwargs)
            return tool.run(payload)
        except TypeError:
            return tool.run(action, **kwargs)

    raise AttributeError(f"{tool.__class__.__name__} 既没有 execute，也没有 run。")


class PDFLearningAssistant:
    """8.4 智能文档问答助手：RAG + Memory"""

    def __init__(self, user_id: str = "user_8_4"):
        self.user_id = user_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.rag_tool = RAGTool(
            knowledge_base_path="./knowledge_base_8_4",
            collection_name="rag_8_4_assistant",
            rag_namespace=f"pdf_{user_id}"
        )

        self.memory_tool = MemoryTool(
            user_id=user_id,
            memory_types=["working", "episodic", "semantic"]
        )

        self.current_document = None

        self.stats = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "notes_added": 0
        }

    def load_document(self, file_path: str) -> Dict[str, Any]:
        """加载文档到 RAG 知识库，并写入情景记忆"""

        if not os.path.exists(file_path):
            return {
                "success": False,
                "message": f"文件不存在：{file_path}"
            }

        print("\n正在加载文档到 RAG 知识库...")

        result = tool_call(
            self.rag_tool,
            "add_document",
            file_path=file_path,
            chunk_size=800,
            chunk_overlap=100
        )

        self.current_document = os.path.basename(file_path)
        self.stats["documents_loaded"] += 1

        tool_call(
            self.memory_tool,
            "add",
            content=f"用户加载了文档《{self.current_document}》用于学习。",
            memory_type="episodic",
            importance=0.9,
            event_type="document_loaded",
            session_id=self.session_id
        )

        return {
            "success": True,
            "message": f"文档已加载：{self.current_document}",
            "rag_result": result
        }

    def ask(self, question: str, use_advanced_search: bool = False) -> str:
        """对当前文档提问：优先 ask，失败则退回 search"""

        if not self.current_document:
            return "请先加载文档。"

        self.stats["questions_asked"] += 1

        # 记录当前问题到工作记忆
        tool_call(
            self.memory_tool,
            "add",
            content=f"用户提出问题：{question}",
            memory_type="working",
            importance=0.6,
            session_id=self.session_id
        )

        print("\n正在进行 RAG 问答...")

        try:
            answer = tool_call(
                self.rag_tool,
                "ask",
                question=question,
                limit=5,
                enable_advanced_search=use_advanced_search,
                enable_mqe=use_advanced_search,
                enable_hyde=use_advanced_search
            )
        except Exception as e:
            print("ask 调用失败，退回到 search。")
            print("错误信息：", repr(e))

            answer = tool_call(
                self.rag_tool,
                "search",
                query=question,
                limit=5,
                min_score=0.1
            )

        # 记录问答事件到情景记忆
        tool_call(
            self.memory_tool,
            "add",
            content=f"用户围绕《{self.current_document}》提问：{question}",
            memory_type="episodic",
            importance=0.7,
            event_type="qa_interaction",
            session_id=self.session_id
        )

        return str(answer)

    def add_note(self, content: str, concept: Optional[str] = None) -> str:
        """添加学习笔记到语义记忆"""

        self.stats["notes_added"] += 1

        result = tool_call(
            self.memory_tool,
            "add",
            content=content,
            memory_type="semantic",
            importance=0.8,
            concept=concept or "general",
            session_id=self.session_id
        )

        return str(result)

    def recall(self, query: str, limit: int = 5) -> str:
        """从记忆系统中回顾学习记录"""

        result = tool_call(
            self.memory_tool,
            "search",
            query=query,
            limit=limit
        )

        return str(result)

    def get_stats(self) -> Dict[str, Any]:
        """查看当前学习统计"""

        duration = (datetime.now() - self.stats["session_start"]).total_seconds()

        return {
            "用户ID": self.user_id,
            "会话ID": self.session_id,
            "会话时长": f"{duration:.0f} 秒",
            "当前文档": self.current_document or "未加载",
            "加载文档数": self.stats["documents_loaded"],
            "提问次数": self.stats["questions_asked"],
            "笔记数量": self.stats["notes_added"]
        }

    def generate_report(self) -> Dict[str, Any]:
        """生成学习报告 JSON"""

        memory_summary = tool_call(
            self.memory_tool,
            "summary",
            limit=10
        )

        rag_stats = tool_call(
            self.rag_tool,
            "stats"
        )

        report = {
            "session_info": {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "start_time": self.stats["session_start"].isoformat(),
                "current_document": self.current_document
            },
            "learning_stats": self.get_stats(),
            "memory_summary": str(memory_summary),
            "rag_stats": str(rag_stats)
        }

        report_file = f"learning_report_{self.session_id}.json"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        report["report_file"] = report_file

        return report


def create_test_document():
    """创建一个测试用 Markdown 文档，避免一开始就处理 PDF 带来额外依赖问题"""

    os.makedirs("./docs_8_4", exist_ok=True)

    file_path = "./docs_8_4/hello_agents_chapter8_note.md"

    content = """# HelloAgents 第八章学习笔记

## 记忆系统

HelloAgents 的记忆系统用于让智能体保存、检索和管理历史信息。
记忆系统通常包括工作记忆、情景记忆、语义记忆和感知记忆。

工作记忆适合保存当前任务中的临时上下文。
情景记忆适合保存具体事件，例如用户加载了哪个文档、提出了哪些问题。
语义记忆适合保存稳定知识、概念和学习笔记。
感知记忆适合处理图像、音频、文档等多模态信息。

## RAG 系统

RAG 是 Retrieval-Augmented Generation 的缩写，中文通常称为检索增强生成。
RAG 系统会先从知识库中检索相关内容，再把检索结果提供给大语言模型生成答案。

在 HelloAgents 中，RAGTool 可以完成文档加载、文本分块、向量化、向量存储和相似度检索。
Qdrant 主要负责向量存储和相似度检索。
Neo4j 主要负责实体关系和知识图谱存储。

## 智能文档问答助手

智能文档问答助手可以把 RAGTool 和 MemoryTool 结合起来。
RAGTool 负责回答“文档中有什么”。
MemoryTool 负责记录“用户学过什么、问过什么、记了什么笔记”。

这种组合可以用于论文阅读助手、技术文档助手、课程学习助手和企业知识库助手。
"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return file_path


def print_section(title):
    print("\n" + "=" * 20)
    print(title)
    print("=" * 20)


def main():
    print("=== 8.4 构建智能文档问答助手：命令行版 ===")

    assistant = PDFLearningAssistant(user_id="student_8_4")

    # 1. 创建并加载测试文档
    print_section("1. 创建测试文档")
    doc_path = create_test_document()
    print("测试文档路径：", doc_path)

    print_section("2. 加载文档")
    result = assistant.load_document(doc_path)
    print(result["message"])
    print(result["rag_result"])

    # 2. 文档问答
    print_section("3. 文档问答")

    questions = [
        "HelloAgents 的记忆系统包括哪些类型？",
        "RAGTool 和 MemoryTool 分别负责什么？",
        "Qdrant 和 Neo4j 在系统中分别起什么作用？"
    ]

    for q in questions:
        print("\n问题：", q)
        answer = assistant.ask(q, use_advanced_search=False)
        print("回答：")
        print(answer)

    # 3. 添加学习笔记
    print_section("4. 添加学习笔记")

    note_result = assistant.add_note(
        content="我理解了：RAGTool 负责外部文档检索，MemoryTool 负责保存用户学习过程。",
        concept="RAG与Memory组合"
    )
    print(note_result)

    # 4. 回顾学习记录
    print_section("5. 回顾学习记录")

    recall_result = assistant.recall("我学习了哪些关于 RAG 和 Memory 的内容？")
    print(recall_result)

    # 5. 查看统计
    print_section("6. 学习统计")

    stats = assistant.get_stats()
    for k, v in stats.items():
        print(f"{k}: {v}")

    # 6. 生成学习报告
    print_section("7. 生成学习报告")

    report = assistant.generate_report()
    print("报告文件：", report["report_file"])

    print("\n=== 8.4 命令行版测试完成 ===")


if __name__ == "__main__":
    main()