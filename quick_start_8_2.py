from dotenv import load_dotenv
load_dotenv()

from hello_agents.tools import MemoryTool


def main():
    print("=== 8.2 MemoryTool 记忆系统快速体验 ===")

    # 先只启用 working / episodic / semantic，避免感知记忆的多模态依赖干扰
    memory_tool = MemoryTool(
        user_id="user123",
        memory_types=["working", "episodic", "semantic"]
    )

    print("\n=== 1. 添加记忆 ===")

    result = memory_tool.run({
        "action": "add",
        "content": "用户正在学习 HelloAgents 第八章 记忆系统",
        "memory_type": "working",
        "importance": 0.7,
        "task_type": "learning"
    })
    print("工作记忆：", result)

    result = memory_tool.run({
        "action": "add",
        "content": "用户已经成功配置了 Qdrant、Embedding 和 Neo4j",
        "memory_type": "episodic",
        "importance": 0.8,
        "event_type": "configuration_success"
    })
    print("情景记忆：", result)

    result = memory_tool.run({
        "action": "add",
        "content": "Agent 的记忆系统通常包括工作记忆、情景记忆、语义记忆和感知记忆",
        "memory_type": "semantic",
        "importance": 0.9,
        "knowledge_type": "concept"
    })
    print("语义记忆：", result)

    print("\n=== 2. 搜索记忆 ===")

    result = memory_tool.run({
        "action": "search",
        "query": "用户配置了哪些数据库？",
        "limit": 3
    })
    print(result)

    print("\n=== 3. 只搜索语义记忆 ===")

    result = memory_tool.run({
        "action": "search",
        "query": "Agent 记忆系统有哪些类型？",
        "memory_type": "semantic",
        "limit": 3
    })
    print(result)

    print("\n=== 4. 查看记忆摘要 ===")

    result = memory_tool.run({
        "action": "summary",
        "limit": 5
    })
    print(result)

    print("\n=== 5. 查看统计信息 ===")

    result = memory_tool.run({
        "action": "stats"
    })
    print(result)

    print("\n=== 6. 记忆整合：working -> episodic ===")

    result = memory_tool.run({
        "action": "consolidate",
        "from_type": "working",
        "to_type": "episodic",
        "importance_threshold": 0.6
    })
    print(result)

    print("\n=== 8.2 测试完成 ===")


if __name__ == "__main__":
    main()