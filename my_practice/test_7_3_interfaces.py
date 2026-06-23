# test_7_3_interfaces.py

from dotenv import load_dotenv
from pydantic import ValidationError

from chapter7.core import Message, Config, Agent


load_dotenv()


class EchoAgent(Agent):
    """
    一个最小可运行 Agent。

    作用：
    只用于测试 Agent 抽象基类是否可用。
    它不调用大模型，只返回 Echo 结果。
    """

    def run(self, input_text: str, **kwargs) -> str:
        self.add_message(Message(content=input_text, role="user"))

        response = f"Echo: {input_text}"

        self.add_message(Message(content=response, role="assistant"))

        return response


def test_message():
    print("====== 1. 测试 Message 类 ======")

    message = Message(content="你好", role="user")

    print("Message 对象：")
    print(message)

    print("Message 转 dict：")
    print(message.to_dict())

    try:
        Message(content="错误角色测试", role="human")
    except ValidationError:
        print("非法 role 已被 Pydantic 成功拦截。")


def test_config():
    print("\n====== 2. 测试 Config 类 ======")

    config = Config.from_env()

    print("Config 对象：")
    print(config)

    print("Config 转 dict：")
    print(config.to_dict())

    config_1 = Config.get_instance()
    config_2 = Config.get_instance()

    print("单例测试：")
    print(config_1 is config_2)


def test_agent():
    print("\n====== 3. 测试 Agent 抽象基类 ======")

    config = Config.from_env()

    agent = EchoAgent(
        name="测试EchoAgent",
        llm=None,
        system_prompt="你是一个测试助手。",
        config=config
    )

    result = agent.run("请测试一下 Agent 基类。")

    print("Agent 返回：")
    print(result)

    print("历史消息数量：")
    print(len(agent.get_history()))

    print("历史消息对象：")
    for message in agent.get_history():
        print(message)

    print("历史消息 dict：")
    print(agent.get_history_dicts())

    print("Agent 字符串表示：")
    print(agent)


if __name__ == "__main__":
    test_message()
    test_config()
    test_agent()

    print("\n====== 7.3 框架接口实现测试完成 ======")