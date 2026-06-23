from pathlib import Path
import os
from dotenv import load_dotenv

# ============================
# 1. 自动寻找 .env 文件
# ============================
def find_env_file(start_dir: Path) -> Path:
    """
    从当前脚本所在目录开始，逐级向上查找 .env 文件。
    这样无论 quick_start_8_1_4.py 放在项目根目录，
    还是放在 my_practice/chapter8 目录，都可以读取到 .env。
    """
    for folder in [start_dir, *start_dir.parents]:
        env_file = folder / ".env"
        if env_file.exists():
            return env_file
    raise FileNotFoundError("没有找到 .env 文件，请确认项目目录下存在 .env")


script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)

env_path = find_env_file(script_dir)
load_dotenv(env_path)

print("已加载 .env 文件：", env_path)
print("QDRANT_URL =", os.getenv("QDRANT_URL"))
print("NEO4J_URI =", os.getenv("NEO4J_URI"))
print("EMBED_MODEL_TYPE =", os.getenv("EMBED_MODEL_TYPE"))
print("LLM_MODEL_ID =", os.getenv("LLM_MODEL_ID"))


# ============================
# 2. 导入 HelloAgents
# ============================
from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool, RAGTool


# ============================
# 3. 创建知识库目录
# ============================
Path("./knowledge_base").mkdir(exist_ok=True)


# ============================
# 4. 创建 LLM
# ============================
llm = HelloAgentsLLM()


# ============================
# 5. 创建 Agent
# ============================
agent = SimpleAgent(
    name="智能助手",
    llm=llm,
    system_prompt="你是一个有记忆和知识检索能力的AI助手"
)


# ============================
# 6. 创建工具注册表
# ============================
tool_registry = ToolRegistry()


# ============================
# 7. 注册 MemoryTool
# ============================
memory_tool = MemoryTool(user_id="user123")
tool_registry.register_tool(memory_tool)


# ============================
# 8. 注册 RAGTool
# ============================
rag_tool = RAGTool(
    knowledge_base_path="./knowledge_base"
)
tool_registry.register_tool(rag_tool)


# ============================
# 9. 把工具注册表交给 Agent
# ============================
agent.tool_registry = tool_registry


# ============================
# 10. 运行官方 8.1.4 快速体验
# ============================
print("\n========== Agent 快速体验 ==========")
response = agent.run("你好！请记住我叫张三，我是一名Python开发者")
print(response)


# ============================
# 11. 直接测试 MemoryTool
# ============================
print("\n========== 直接测试 MemoryTool ==========")

result = memory_tool.run({
    "action": "add",
    "content": "张三是一名Python开发者，正在学习HelloAgents第八章记忆与检索。",
    "memory_type": "semantic",
    "importance": 0.8
})
print("添加记忆结果：")
print(result)

result = memory_tool.run({
    "action": "search",
    "query": "张三 Python开发者 第八章",
    "limit": 3
})
print("\n搜索记忆结果：")
print(result)


# ============================
# 12. 直接测试 RAGTool
# ============================
print("\n========== 直接测试 RAGTool ==========")

result = rag_tool.run({
    "action": "add_text",
    "text": (
        "HelloAgents第八章主要介绍记忆系统和检索增强生成RAG。"
        "MemoryTool用于让Agent保存和检索历史记忆。"
        "RAGTool用于让Agent基于外部知识库进行问答。"
    ),
    "document_id": "chapter8_intro"
})
print("添加知识结果：")
print(result)

result = rag_tool.run({
    "action": "ask",
    "question": "HelloAgents第八章主要介绍什么？",
    "limit": 3
})
print("\nRAG问答结果：")
print(result)

print("\n========== 运行完成 ==========")