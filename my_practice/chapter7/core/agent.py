# chapter7/core/agent.py

"""Agent 抽象基类"""

from abc import ABC, abstractmethod
from typing import Optional, Any, List

from .message import Message
from .config import Config


class Agent(ABC):
    """
    Agent 抽象基类。

    作用：
    1. 规定所有 Agent 都必须有 run() 方法；
    2. 统一管理 name、llm、system_prompt、config；
    3. 提供历史消息管理能力。
    """

    def __init__(
        self,
        name: str,
        llm: Any,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config.get_instance()
        self._history: List[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """
        运行 Agent。

        所有具体 Agent 子类都必须实现这个方法。
        """
        pass

    def add_message(self, message: Message):
        """
        添加消息到历史记录。
        """
        self._history.append(message)

        # 控制历史长度，避免无限增长
        max_len = self.config.max_history_length
        if max_len is not None and max_len > 0 and len(self._history) > max_len:
            self._history = self._history[-max_len:]

    def clear_history(self):
        """
        清空历史记录。
        """
        self._history.clear()

    def get_history(self) -> List[Message]:
        """
        获取历史记录副本。
        """
        return self._history.copy()

    def get_history_dicts(self) -> List[dict]:
        """
        获取 OpenAI API 兼容格式的历史记录。
        """
        return [message.to_dict() for message in self._history]

    def __str__(self) -> str:
        provider = getattr(self.llm, "provider", "unknown")
        return f"Agent(name={self.name}, provider={provider})"