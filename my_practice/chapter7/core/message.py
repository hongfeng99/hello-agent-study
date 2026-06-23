# chapter7/core/message.py

"""消息系统"""

from typing import Optional, Dict, Any, Literal
from datetime import datetime

from pydantic import BaseModel, Field


# 限定消息角色只能是这四种
MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """
    消息类。

    作用：
    1. 统一框架内部的消息格式；
    2. 记录消息角色、内容、时间戳和元数据；
    3. 可以转换成 OpenAI API 兼容的字典格式。
    """

    content: str
    role: MessageRole
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, str]:
        """
        转换为 OpenAI API 兼容格式。

        OpenAI 格式只需要：
        {
            "role": "...",
            "content": "..."
        }
        """
        return {
            "role": self.role,
            "content": self.content
        }

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"