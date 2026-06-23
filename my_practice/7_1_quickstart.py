from dotenv import load_dotenv
from hello_agents import SimpleAgent, HelloAgentsLLM

# 1. 加载 .env 文件中的 API Key
load_dotenv()

# 2. 创建 LLM 实例
# 不传 provider 时，框架会根据 .env 自动检测
llm = HelloAgentsLLM()

# 3. 创建一个最简单的 Agent
agent = SimpleAgent(
    name="AI助手",
    llm=llm,
    system_prompt="你是一个有用的AI助手，请用简洁、清楚的方式回答问题。"
)

# 4. 运行一次基础对话
response = agent.run("你好，请用三句话介绍一下你自己。")

print("====== Agent 回复 ======")
print(response)

# 5. 查看历史消息数量
print("\n====== 历史消息数量 ======")
print(len(agent.get_history()))