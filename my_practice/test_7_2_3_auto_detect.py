# test_7_2_3_auto_detect.py

from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM

load_dotenv()

# 不指定 provider，让 HelloAgentsLLM 自动检测
llm = HelloAgentsLLM()

messages = [
    {
        "role": "user",
        "content": "请用一句话说明你现在是通过哪类接口被调用的。"
    }
]

print("====== 7.2.3 自动检测测试开始 ======")

for chunk in llm.think(messages):
    print(chunk, end="", flush=True)

print("\n====== 7.2.3 自动检测测试结束 ======")