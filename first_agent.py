import os
import re
import requests
from tavily import TavilyClient
from openai import OpenAI

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对 Thought 和 Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action 的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对 Thought-Action
- Action 必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""


def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实天气信息。
    """
    url = f"https://wttr.in/{city}?format=j1"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        current_condition = data["current_condition"][0]

        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]

        return f"{city}当前天气：{weather_desc}，气温 {temp_c} 摄氏度"

    except requests.exceptions.RequestException as e:
        return f"错误：查询天气时遇到网络问题 - {e}"

    except (KeyError, IndexError) as e:
        return f"错误：解析天气数据失败，可能是城市名称无效 - {e}"


def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用 Tavily 搜索推荐旅游景点。
    """
    api_key = os.environ.get("TAVILY_API_KEY")

    if not api_key:
        return "错误：未配置 TAVILY_API_KEY 环境变量。"

    tavily = TavilyClient(api_key=api_key)

    query = f"{city} 在 {weather} 天气下最值得去的旅游景点推荐及理由"

    try:
        response = tavily.search(
            query=query,
            search_depth="basic",
            include_answer=True
        )

        if response.get("answer"):
            return response["answer"]

        results = response.get("results", [])

        if not results:
            return "抱歉，没有找到相关的旅游景点推荐。"

        formatted_results = []
        for result in results:
            title = result.get("title", "无标题")
            content = result.get("content", "无内容")
            formatted_results.append(f"- {title}: {content}")

        return "根据搜索，为您找到以下信息：\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误：执行 Tavily 搜索时出现问题 - {e}"


available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

class OpenAICompatibleClient:
    """
    一个用于调用兼容 OpenAI 接口的大语言模型服务的客户端。
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def generate(self, prompt: str, system_prompt: str) -> str:
        """
        调用大语言模型 API，返回模型生成的文本。
        """
        print("正在调用大语言模型...")

        try:
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )

            answer = response.choices[0].message.content

            print("大语言模型响应成功。")
            return answer

        except Exception as e:
            print(f"调用 LLM API 时发生错误: {e}")
            return "错误：调用语言模型服务时出错。"

if __name__ == "__main__":
    # =========================
    # 1. 读取环境变量
    # =========================
    API_KEY = os.environ.get("LLM_API_KEY")
    BASE_URL = os.environ.get("LLM_BASE_URL")
    MODEL_ID = os.environ.get("LLM_MODEL_ID")

    if not API_KEY:
        raise ValueError("未设置环境变量 LLM_API_KEY")

    if not BASE_URL:
        raise ValueError("未设置环境变量 LLM_BASE_URL")

    if not MODEL_ID:
        raise ValueError("未设置环境变量 LLM_MODEL_ID")

    if not os.environ.get("TAVILY_API_KEY"):
        print("警告：未检测到 TAVILY_API_KEY。后续调用 get_attraction 时可能失败。")

    # =========================
    # 2. 初始化 LLM 客户端
    # =========================
    llm = OpenAICompatibleClient(
        model=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # =========================
    # 3. 初始化用户任务和历史记录
    # =========================
    user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"

    prompt_history = [
        f"用户请求: {user_prompt}"
    ]

    print(f"用户输入: {user_prompt}")
    print("=" * 60)

    # =========================
    # 4. Agent 主循环
    # =========================
    for i in range(5):
        print(f"\n--- 循环 {i + 1} ---")

        # 4.1 构造完整 Prompt
        full_prompt = "\n".join(prompt_history)

        # 4.2 调用 LLM，让模型根据当前上下文决定下一步 Action
        llm_output = llm.generate(
            prompt=full_prompt,
            system_prompt=AGENT_SYSTEM_PROMPT
        )

        # 4.3 尝试截断多余输出，只保留一组 Thought-Action
        match = re.search(
            r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
            llm_output,
            re.DOTALL
        )

        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("已截断多余的 Thought-Action 对。")

        print("\n模型输出:")
        print(llm_output)

        # 把模型输出加入历史
        prompt_history.append(llm_output)

        # 4.4 解析 Action
        action_match = re.search(r"Action:\s*(.*)", llm_output, re.DOTALL)

        if not action_match:
            observation = (
                "错误：未能解析到 Action 字段。"
                "请确保你的回复严格遵循 Thought: ... 和 Action: ... 的格式。"
            )
            observation_str = f"Observation: {observation}"
            print("\n" + observation_str)
            print("=" * 60)
            prompt_history.append(observation_str)
            continue

        action_str = action_match.group(1).strip()

        # 4.5 如果模型决定结束任务
        if action_str.startswith("Finish"):
            finish_match = re.match(r"Finish\[(.*)\]", action_str, re.DOTALL)

            if finish_match:
                final_answer = finish_match.group(1).strip()
                print("\n任务完成，最终答案:")
                print(final_answer)
                break
            else:
                observation = "错误：Finish 格式不正确，正确格式应为 Finish[最终答案]。"
                observation_str = f"Observation: {observation}"
                print("\n" + observation_str)
                print("=" * 60)
                prompt_history.append(observation_str)
                continue

        # 4.6 解析工具名称
        tool_name_match = re.search(r"(\w+)\(", action_str)

        if not tool_name_match:
            observation = f"错误：无法从 Action 中解析工具名称：{action_str}"
            observation_str = f"Observation: {observation}"
            print("\n" + observation_str)
            print("=" * 60)
            prompt_history.append(observation_str)
            continue

        tool_name = tool_name_match.group(1)

        # 4.7 解析工具参数
        args_match = re.search(r"\((.*)\)", action_str, re.DOTALL)

        if not args_match:
            observation = f"错误：无法从 Action 中解析工具参数：{action_str}"
            observation_str = f"Observation: {observation}"
            print("\n" + observation_str)
            print("=" * 60)
            prompt_history.append(observation_str)
            continue

        args_str = args_match.group(1)

        # 只解析形如 key="value" 的参数
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

        # 4.8 执行工具
        if tool_name in available_tools:
            try:
                observation = available_tools[tool_name](**kwargs)
            except TypeError as e:
                observation = f"错误：工具参数不匹配 - {e}"
            except Exception as e:
                observation = f"错误：工具执行失败 - {e}"
        else:
            observation = f"错误：未定义的工具 '{tool_name}'"

        # 4.9 把工具结果作为 Observation 加入历史
        observation_str = f"Observation: {observation}"

        print("\n" + observation_str)
        print("=" * 60)

        prompt_history.append(observation_str)

    else:
        print("\n达到最大循环次数，任务未正常结束。")