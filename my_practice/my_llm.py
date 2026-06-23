# my_llm.py

import os
from typing import Optional, List, Dict, Any

from openai import OpenAI
from hello_agents import HelloAgentsLLM


class MyLLM(HelloAgentsLLM):
    """
    自定义 LLM 客户端。

    7.2.1 支持多提供商的核心逻辑：
    1. provider="modelscope" 时，使用我们自己写的 ModelScope 逻辑；
    2. 其他 provider 交给父类 HelloAgentsLLM 处理。
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs
    ):
        if provider == "modelscope":
            print("正在使用自定义的 ModelScope Provider")

            self.provider = "modelscope"

            self.api_key = (
                api_key
                or os.getenv("MODELSCOPE_API_KEY")
                or os.getenv("LLM_API_KEY")
            )

            self.base_url = (
                base_url
                or os.getenv("LLM_BASE_URL")
                or "https://api-inference.modelscope.cn/v1"
            )

            self.model = (
                model
                or os.getenv("LLM_MODEL_ID")
                or "Qwen/Qwen3.5-35B-A3B"
            )

            if not self.api_key:
                raise ValueError(
                    "未找到 API Key，请在 .env 中设置 MODELSCOPE_API_KEY 或 LLM_API_KEY。"
                )

            self.temperature = kwargs.get("temperature", 0.7)
            self.max_tokens = kwargs.get("max_tokens", None)
            self.timeout = kwargs.get("timeout", 60)

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )

        else:
            print(f"交给父类 HelloAgentsLLM 处理 provider={provider}")

            super().__init__(
                model=model,
                api_key=api_key,
                base_url=base_url,
                provider=provider,
                **kwargs
            )

    def _normalize_messages(self, messages: List[Any]) -> List[Dict[str, str]]:
        """
        把消息统一转换成 OpenAI 接口需要的格式：
        [
            {"role": "user", "content": "..."}
        ]
        """
        normalized = []

        for message in messages:
            if isinstance(message, dict):
                normalized.append({
                    "role": message["role"],
                    "content": message["content"]
                })
            elif hasattr(message, "role") and hasattr(message, "content"):
                normalized.append({
                    "role": message.role,
                    "content": message.content
                })
            else:
                raise TypeError(f"不支持的消息格式: {type(message)}")

        return normalized

    def think(self, messages: List[Any], **kwargs):
        """
        重写 think 方法，解决 ModelScope 流式输出最后 chunk.choices 为空导致的报错。
        """
        if self.provider != "modelscope":
            yield from super().think(messages, **kwargs)
            return

        formatted_messages = self._normalize_messages(messages)

        print(f"🧠 正在调用 {self.model} 模型...")

        request_params = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,
        }

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens

        try:
            stream = self._client.chat.completions.create(**request_params)

            print("✅ 大语言模型响应成功:")

            for chunk in stream:
                # 关键修复点：
                # ModelScope 有时会返回 choices=[] 的结束块，必须跳过。
                if not getattr(chunk, "choices", None):
                    continue

                choice = chunk.choices[0]

                if not hasattr(choice, "delta"):
                    continue

                content = getattr(choice.delta, "content", None)

                if content:
                    yield content

        except Exception as e:
            raise RuntimeError(f"ModelScope 调用失败: {e}") from e
        
    def invoke(self, messages, **kwargs) -> str:
        """
        非流式调用。7.4 Agent 统一用 invoke 获取完整回答。
        """
        if self.provider != "modelscope":
            return super().invoke(messages, **kwargs)

        formatted_messages = self._normalize_messages(messages)

        request_params = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False,
        }

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens

        try:
            response = self._client.chat.completions.create(**request_params)

            if not response.choices:
                return ""

            return response.choices[0].message.content or ""

        except Exception as e:
            raise RuntimeError(f"ModelScope 调用失败: {e}") from e