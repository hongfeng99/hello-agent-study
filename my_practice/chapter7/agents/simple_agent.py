# chapter7/agents/simple_agent.py

from typing import Any, Optional, List, Dict

from chapter7.core import Agent, Message, Config


class MySimpleAgent(Agent):
    """
    最基础的对话 Agent。

    作用：
    1. 构造 system + history + user 消息；
    2. 调用 llm.invoke；
    3. 把 user 和 assistant 消息写入历史记录。
    """

    def __init__(
        self,
        name: str,
        llm: Any,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or "你是一个有用的AI助手。",
            config=config
        )

    def _build_messages(self, input_text: str) -> List[Dict[str, str]]:
        messages = []

        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        for msg in self.get_history():
            messages.append(msg.to_dict())

        messages.append({
            "role": "user",
            "content": input_text
        })

        return messages

    def run(self, input_text: str, **kwargs) -> str:
        print(f"{self.name} 正在处理：{input_text}")

        messages = self._build_messages(input_text)

        response = self.llm.invoke(messages, **kwargs)

        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=response, role="assistant"))

        return response