# chapter7/core/config.py

"""配置管理"""

import os
from typing import Optional, Dict, Any, ClassVar

from pydantic import BaseModel


class Config(BaseModel):
    """
    HelloAgents 配置类。

    作用：
    1. 集中管理默认模型、provider、temperature 等参数；
    2. 支持从环境变量读取配置；
    3. 提供一个简单的单例入口 get_instance()。
    """

    # LLM 配置
    default_model: str = "gpt-3.5-turbo"
    default_provider: str = "openai"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # 系统配置
    debug: bool = False
    log_level: str = "INFO"

    # 历史记录配置
    max_history_length: int = 100

    # 单例对象，不作为 Pydantic 字段
    _instance: ClassVar[Optional["Config"]] = None

    @classmethod
    def from_env(cls) -> "Config":
        """
        从环境变量创建配置。

        可以在 .env 中配置：
        DEBUG=true
        LOG_LEVEL=DEBUG
        TEMPERATURE=0.5
        MAX_TOKENS=1024
        MAX_HISTORY_LENGTH=50
        """
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
            max_history_length=int(os.getenv("MAX_HISTORY_LENGTH", "100")),
        )

    @classmethod
    def get_instance(cls) -> "Config":
        """
        获取全局唯一配置对象。

        第一次调用时从环境变量加载；
        后续调用直接复用同一个 Config。
        """
        if cls._instance is None:
            cls._instance = cls.from_env()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """
        重置配置单例。

        测试时可能会用到。
        """
        cls._instance = None

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典。

        兼容 Pydantic v1 和 v2。
        """
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()