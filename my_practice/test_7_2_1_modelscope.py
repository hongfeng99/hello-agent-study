# test_7_2_1_modelscope.py

from dotenv import load_dotenv
from my_llm import MyLLM

load_dotenv()

llm = MyLLM(provider="modelscope")

messages = [
    {
        "role": "user",
        "content": "你好，请用三句话介绍一下你自己。"
    }
]

print("====== 7.2.1 ModelScope 测试开始 ======")

for chunk in llm.think(messages):
    print(chunk, end="", flush=True)

print("\n====== 7.2.1 ModelScope 测试结束 ======")