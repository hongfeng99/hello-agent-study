import os
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI


# 加载项目根目录下的 .env 文件
load_dotenv()


class HelloAgentsLLM:
    """
    Hello-Agents 项目中的基础 LLM 客户端。
    作用：统一封装大语言模型调用逻辑，后续 ReAct、Plan-and-Solve、Reflection 都可以复用它。
    """

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        timeout: int = None,
    ):
        self.model = model or os.getenv("LLM_MODEL_ID")
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not self.model:
            raise ValueError("缺少 LLM_MODEL_ID，请检查 .env 文件。")

        if not api_key:
            raise ValueError("缺少 LLM_API_KEY，请检查 .env 文件。")

        if not base_url:
            raise ValueError("缺少 LLM_BASE_URL，请检查 .env 文件。")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型，返回模型回复文本。
        """
        print(f"正在调用模型：{self.model}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            collected_content = []

            print("模型输出：")
            for chunk in response:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)

            print()
            return "".join(collected_content)

        except Exception as e:
            print(f"调用 LLM API 时发生错误：{e}")
            return ""


if __name__ == "__main__":
    llm = HelloAgentsLLM()

    messages = [
        {
            "role": "system",
            "content": "你是一个擅长解释 Python 代码的助手。",
        },
        {
            "role": "user",
            "content": "请用通俗语言解释什么是函数。",
        },
    ]

    result = llm.think(messages)

    print("\n完整返回结果：")
    print(result)