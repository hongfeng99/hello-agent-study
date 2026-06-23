from typing import List, Dict, Any, Optional


class Memory:
    """
    一个简单的短期记忆模块，用于存储 Reflection Agent 的执行与反思轨迹。
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        添加一条记录。

        record_type:
        - execution: 表示模型生成的代码
        - reflection: 表示模型对代码的反思反馈
        """
        record = {
            "type": record_type,
            "content": content,
        }
        self.records.append(record)
        print(f"记忆已更新，新增一条 '{record_type}' 记录。")

    def get_trajectory(self) -> str:
        """
        将所有历史记录整理成文本，方便后续放进提示词。
        """
        trajectory_parts = []

        for record in self.records:
            if record["type"] == "execution":
                trajectory_parts.append(
                    f"--- 上一轮尝试代码 ---\n{record['content']}"
                )
            elif record["type"] == "reflection":
                trajectory_parts.append(
                    f"--- 评审员反馈 ---\n{record['content']}"
                )

        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:
        """
        获取最近一次生成的代码。
        """
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]

        return None