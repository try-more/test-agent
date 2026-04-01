import os
from datetime import datetime
from typing import Optional

os.environ["OPENAI_API_KEY"] = "your-api-key-here"

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent


@tool
def get_current_time(timezone: Optional[str] = None) -> str:
    now = datetime.now()
    if timezone:
        return f"当前时间（{timezone}时区）: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


tools = [get_current_time, calculator]

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

graph = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""你是一个有用的助手，可以回答用户的问题并使用工具来获取信息。
    你可以使用的工具：
    1. get_current_time - 获取当前时间
    2. calculator - 计算数学表达式

    如果用户询问时间或日期，请使用get_current_time工具。
    如果用户需要数学计算，请使用calculator工具。

    请用中文回答用户的问题。""",
)


def run_agent_demo():
    print("=" * 50)
    print("AI Agent Demo 已启动!")
    print("可以尝试询问：")
    print("1. '现在几点了？'")
    print("2. '计算一下 25 * 4 + 18 等于多少'")
    print("3. '今天是什么日期？'")
    print("输入 '退出' 或 'quit' 结束程序")
    print("=" * 50)

    while True:
        user_input = input("\n您的问题: ").strip()

        if user_input.lower() in ["退出", "quit", "exit"]:
            print("再见！")
            break

        if not user_input:
            continue

        try:
            inputs = {"messages": [{"role": "user", "content": user_input}]}
            response = graph.invoke(inputs)
            output = response["messages"][-1].content
            print(f"\n助手：{output}")
        except Exception as e:
            print(f"\n发生错误：{e}")


if __name__ == "__main__":
    run_agent_demo()
