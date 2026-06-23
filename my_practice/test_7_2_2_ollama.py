# test_7_2_2_ollama.py

from hello_agents import HelloAgentsLLM

llm = HelloAgentsLLM(
    provider="ollama",
    model="qwen2.5:0.5b",
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

messages = [
    {
        "role": "user",
        "content": "请用一句话解释什么是本地大语言模型。"
    }
]

print("====== 7.2.2 Ollama 本地模型测试开始 ======")

for chunk in llm.think(messages):
    print(chunk, end="", flush=True)

print("\n====== 7.2.2 Ollama 本地模型测试结束 ======")