# test_7_2_3_auto_detect_safe.py

import os
from dotenv import load_dotenv
from my_llm import MyLLM

load_dotenv()


def detect_provider():
    base_url = os.getenv("LLM_BASE_URL", "")

    if "modelscope" in base_url:
        return "modelscope"

    if "localhost:11434" in base_url or ":11434" in base_url:
        return "ollama"

    if "localhost:8000" in base_url or ":8000" in base_url:
        return "vllm"

    return "auto"


provider = detect_provider()

print(f"自动检测到 provider = {provider}")

llm = MyLLM(provider=provider)

messages = [
    {
        "role": "user",
        "content": "请用一句话解释什么是 provider 自动检测。"
    }
]

print("====== 7.2.3 安全自动检测测试开始 ======")

for chunk in llm.think(messages):
    print(chunk, end="", flush=True)

print("\n====== 7.2.3 安全自动检测测试结束 ======")