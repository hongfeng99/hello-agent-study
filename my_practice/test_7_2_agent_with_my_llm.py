# test_7_2_agent_with_my_llm.py

from dotenv import load_dotenv
from hello_agents import SimpleAgent
from my_llm import MyLLM

load_dotenv()

llm = MyLLM(provider="modelscope")

agent = SimpleAgent(
    name="7.2测试助手",
    llm=llm,
    system_prompt="你是一个适合初学者的人工智能助教，请用清楚、通俗的方式回答问题。"
)

response = agent.run("请解释一下 HelloAgentsLLM 在 Agent 框架中的作用。")

print("====== Agent 回复 ======")
print(response)

print("\n====== 历史消息数量 ======")
print(len(agent.get_history()))