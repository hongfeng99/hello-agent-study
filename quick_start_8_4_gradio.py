from dotenv import load_dotenv
load_dotenv()

import os
import json
from datetime import datetime
from typing import Any, Dict, Optional

import gradio as gr

from hello_agents.tools import RAGTool, MemoryTool


def tool_call(tool, action, **kwargs):
    """
    兼容不同版本的 Tool 调用方式：
    - 有些版本使用 execute(action, **kwargs)
    - 你当前版本主要使用 run({...})
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
    """8.4 智能文档问答助手：Gradio 版"""

    def __init__(self, user_id: str = "student_8_4_gradio"):
        self.user_id = user_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.rag_tool = RAGTool(
            knowledge_base_path="./knowledge_base_8_4_gradio",
            collection_name="rag_8_4_gradio_assistant",
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

    def load_document(self, file_path: str) -> str:
        """上传并加载文档"""

        if not file_path:
            return "请先上传文档。"

        if not os.path.exists(file_path):
            return f"文件不存在：{file_path}"

        file_name = os.path.basename(file_path)

        try:
            result = tool_call(
                self.rag_tool,
                "add_document",
                file_path=file_path,
                chunk_size=800,
                chunk_overlap=100
            )

            self.current_document = file_name
            self.stats["documents_loaded"] += 1

            tool_call(
                self.memory_tool,
                "add",
                content=f"用户加载了文档《{file_name}》用于学习。",
                memory_type="episodic",
                importance=0.9,
                event_type="document_loaded",
                session_id=self.session_id
            )

            return f"文档加载成功：{file_name}\n\n{result}"

        except Exception as e:
            return f"文档加载失败：{repr(e)}"

    def ask(self, question: str, use_advanced_search: bool = False) -> str:
        """文档问答"""

        if not self.current_document:
            return "请先上传并加载文档。"

        if not question or not question.strip():
            return "请输入问题。"

        self.stats["questions_asked"] += 1

        question = question.strip()

        try:
            tool_call(
                self.memory_tool,
                "add",
                content=f"用户提出问题：{question}",
                memory_type="working",
                importance=0.6,
                session_id=self.session_id
            )

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
                answer = tool_call(
                    self.rag_tool,
                    "search",
                    query=question,
                    limit=5,
                    min_score=0.1
                )
                answer = f"ask 调用失败，已退回到 search。\n错误信息：{repr(e)}\n\n检索结果：\n{answer}"

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

        except Exception as e:
            return f"问答失败：{repr(e)}"

    def add_note(self, note: str, concept: str = "general") -> str:
        """保存学习笔记"""

        if not note or not note.strip():
            return "请输入笔记内容。"

        self.stats["notes_added"] += 1

        try:
            result = tool_call(
                self.memory_tool,
                "add",
                content=note.strip(),
                memory_type="semantic",
                importance=0.8,
                concept=concept or "general",
                session_id=self.session_id
            )

            return f"笔记已保存。\n\n{result}"

        except Exception as e:
            return f"保存笔记失败：{repr(e)}"

    def recall(self, query: str) -> str:
        """回顾学习记录"""

        if not query or not query.strip():
            query = "用户最近学习了什么？"

        try:
            result = tool_call(
                self.memory_tool,
                "search",
                query=query.strip(),
                limit=5
            )

            return str(result)

        except Exception as e:
            return f"回顾失败：{repr(e)}"

    def get_stats_text(self) -> str:
        """显示学习统计"""

        duration = (datetime.now() - self.stats["session_start"]).total_seconds()

        lines = [
            f"用户ID：{self.user_id}",
            f"会话ID：{self.session_id}",
            f"会话时长：{duration:.0f} 秒",
            f"当前文档：{self.current_document or '未加载'}",
            f"加载文档数：{self.stats['documents_loaded']}",
            f"提问次数：{self.stats['questions_asked']}",
            f"笔记数量：{self.stats['notes_added']}",
        ]

        return "\n".join(lines)

    def generate_report(self) -> str:
        """生成学习报告"""

        try:
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
                "learning_stats": self.get_stats_text(),
                "memory_summary": str(memory_summary),
                "rag_stats": str(rag_stats)
            }

            report_file = f"learning_report_{self.session_id}.json"

            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)

            return f"学习报告已生成：{report_file}"

        except Exception as e:
            return f"生成报告失败：{repr(e)}"


assistant = PDFLearningAssistant()


def upload_and_load(file_path):
    if file_path is None or not str(file_path).strip():
        return "请输入文档路径。"

    file_path = str(file_path).strip().strip('"').strip("'")

    if not os.path.exists(file_path):
        return f"文件不存在：{file_path}"

    return assistant.load_document(file_path)


def ask_question(question, use_advanced_search):
    return assistant.ask(question, use_advanced_search)


def save_note(note, concept):
    return assistant.add_note(note, concept)


def recall_memory(query):
    return assistant.recall(query)


def show_stats():
    return assistant.get_stats_text()


def make_report():
    return assistant.generate_report()


with gr.Blocks(title="HelloAgents 8.4 智能文档问答助手") as demo:
    gr.Markdown("# HelloAgents 8.4 智能文档问答助手")
    gr.Markdown("功能：上传文档、文档问答、保存学习笔记、回顾学习记录、生成学习报告。")

    with gr.Tab("1. 加载文档"):
        file_input = gr.Textbox(
            label="文档路径",
            value="./docs_8_4/hello_agents_chapter8_note.md",
            placeholder="例如：./docs_8_4/hello_agents_chapter8_note.md",
            lines=1
        )

        load_button = gr.Button("加载文档")
        load_output = gr.Textbox(label="加载结果", lines=8)

        load_button.click(
            fn=upload_and_load,
            inputs=file_input,
            outputs=load_output,
            api_name=False
        )

    with gr.Tab("2. 文档问答"):
        question_input = gr.Textbox(
            label="输入问题",
            placeholder="例如：RAGTool 和 MemoryTool 分别负责什么？",
            lines=3
        )
        advanced_checkbox = gr.Checkbox(
            label="启用高级检索",
            value=False
        )
        ask_button = gr.Button("提问")
        answer_output = gr.Textbox(label="回答", lines=18)

        ask_button.click(
            fn=ask_question,
            inputs=[question_input, advanced_checkbox],
            outputs=answer_output,
            api_name=False
        )

    with gr.Tab("3. 学习笔记"):
        note_input = gr.Textbox(
            label="笔记内容",
            placeholder="例如：我理解了 RAGTool 负责检索外部文档，MemoryTool 负责保存学习过程。",
            lines=6
        )
        concept_input = gr.Textbox(
            label="概念标签",
            value="RAG与Memory组合"
        )
        note_button = gr.Button("保存笔记")
        note_output = gr.Textbox(label="保存结果", lines=8)

        note_button.click(
            fn=save_note,
            inputs=[note_input, concept_input],
            outputs=note_output,
            api_name=False
        )

    with gr.Tab("4. 学习回顾"):
        recall_input = gr.Textbox(
            label="回顾问题",
            value="我学习了哪些关于 RAG 和 Memory 的内容？",
            lines=3
        )
        recall_button = gr.Button("回顾学习记录")
        recall_output = gr.Textbox(label="回顾结果", lines=15)

        recall_button.click(
            fn=recall_memory,
            inputs=recall_input,
            outputs=recall_output,
            api_name=False
        )

    with gr.Tab("5. 统计与报告"):
        stats_button = gr.Button("查看学习统计")
        stats_output = gr.Textbox(label="学习统计", lines=10)

        report_button = gr.Button("生成学习报告")
        report_output = gr.Textbox(label="报告结果", lines=6)

        stats_button.click(
            fn=show_stats,
            inputs=[],
            outputs=stats_output,
            api_name=False
        )

        report_button.click(
            fn=make_report,
            inputs=[],
            outputs=report_output,
            api_name=False
        )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_api=False
    )