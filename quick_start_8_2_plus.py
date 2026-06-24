from dotenv import load_dotenv
load_dotenv()

from hello_agents.tools import MemoryTool


def print_result(title, result):
    print("\n" + "=" * 20)
    print(title)
    print("=" * 20)
    print(result)


def main():
    print("=== 8.2 MemoryTool 进阶练习 ===")

    memory_tool = MemoryTool(
        user_id="user_8_2_plus",
        memory_types=["working", "episodic", "semantic"]
    )

    # 1. 添加多条记忆
    print("\n=== 1. 添加多条记忆 ===")

    memories = [
        {
            "content": "用户正在学习 HelloAgents 第八章记忆系统。",
            "memory_type": "working",
            "importance": 0.6,
            "task_type": "learning"
        },
        {
            "content": "用户已经成功修复 Neo4j Aura 的 database=None 配置问题。",
            "memory_type": "episodic",
            "importance": 0.9,
            "event_type": "debug_success"
        },
        {
            "content": "Qdrant 在记忆系统中主要负责向量存储和相似度检索。",
            "memory_type": "semantic",
            "importance": 0.8,
            "knowledge_type": "concept"
        },
        {
            "content": "Neo4j 在记忆系统中主要负责实体关系和知识图谱存储。",
            "memory_type": "semantic",
            "importance": 0.8,
            "knowledge_type": "concept"
        },
        {
            "content": "这是一条不太重要的临时记忆，用于测试遗忘机制。",
            "memory_type": "working",
            "importance": 0.2,
            "task_type": "temporary"
        }
    ]

    for item in memories:
        result = memory_tool.run({
            "action": "add",
            **item
        })
        print(result)

    # 2. 搜索全部记忆
    result = memory_tool.run({
        "action": "search",
        "query": "用户修复了什么问题？",
        "limit": 5
    })
    print_result("2. 搜索：用户修复了什么问题？", result)

    # 3. 只搜索 semantic 语义记忆
    result = memory_tool.run({
        "action": "search",
        "query": "Qdrant 和 Neo4j 分别负责什么？",
        "memory_type": "semantic",
        "limit": 5
    })
    print_result("3. 只搜索语义记忆", result)

    # 4. 查看摘要
    result = memory_tool.run({
        "action": "summary",
        "limit": 10
    })
    print_result("4. 查看记忆摘要", result)

    # 5. 查看统计
    result = memory_tool.run({
        "action": "stats"
    })
    print_result("5. 查看统计信息", result)

    # 6. 遗忘低价值记忆
    result = memory_tool.run({
        "action": "forget",
        "importance_threshold": 0.3
    })
    print_result("6. 遗忘低价值记忆", result)

    # 7. 再次查看摘要，确认低重要性记忆是否减少
    result = memory_tool.run({
        "action": "summary",
        "limit": 10
    })
    print_result("7. 遗忘后再次查看摘要", result)

    # 8. 整合工作记忆到情景记忆
    result = memory_tool.run({
        "action": "consolidate",
        "from_type": "working",
        "to_type": "episodic",
        "importance_threshold": 0.5
    })
    print_result("8. 记忆整合：working -> episodic", result)

    print("\n=== 8.2 进阶练习完成 ===")


if __name__ == "__main__":
    main()