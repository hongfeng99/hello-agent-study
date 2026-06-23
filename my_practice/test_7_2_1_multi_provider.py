# test_7_2_1_multi_provider.py

from dotenv import load_dotenv
from my_llm import MyLLM

load_dotenv()


def test_llm(provider: str):
    print(f"\n====== 测试 provider={provider} ======")

    llm = MyLLM(provider=provider)

    messages = [
        {
            "role": "user",
            "content": "请用一句话解释什么是大语言模型客户端。"
        }
    ]

    for chunk in llm.think(messages):
        print(chunk, end="", flush=True)

    print("\n====== 测试结束 ======")


# 你当前已经配置了 ModelScope，所以先只测试这个
test_llm("modelscope")

# 如果以后你配置了 OpenAI，可以取消下面注释：
# test_llm("openai")

# 如果以后你本地启动了 Ollama，可以取消下面注释：
# test_llm("ollama")

# 如果以后你本地启动了 VLLM，可以取消下面注释：
# test_llm("vllm")