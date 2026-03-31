import os
from datetime import datetime
from typing import Optional

# 设置环境变量（请替换为你的实际API密钥）
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # 或通过 .env 文件设置

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents import create_tool_calling_agent, AgentExecutor


# 1. 定义一个工具 - 获取当前时间
@tool
def get_current_time(timezone: Optional[str] = None) -> str:
    """获取当前的日期和时间。可以指定时区，默认为系统本地时间。"""
    now = datetime.now()
    if timezone:
        return f"当前时间（{timezone}时区）: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}"


# 2. 再定义一个工具 - 计算器
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持加减乘除和括号。"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# 3. 设置工具列表
tools = [get_current_time, calculator]

# 4. 初始化大语言模型（使用 gpt-3.5-turbo）
llm = ChatOpenAI(
    model="gpt-3.5-turbo", temperature=0, api_key=os.environ.get("OPENAI_API_KEY")
)

# 5. 创建 Agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个有用的助手，可以回答用户的问题并使用工具来获取信息。
    你可以使用的工具：
    1. get_current_time - 获取当前时间
    2. calculator - 计算数学表达式

    如果用户询问时间或日期，请使用get_current_time工具。
    如果用户需要数学计算，请使用calculator工具。

    请用中文回答用户的问题。""",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# 6. 运行 Agent
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
            # 执行 Agent
            result = agent_executor.invoke({"input": user_input})
            print(f"\n助手：{result['output']}")
        except Exception as e:
            print(f"\n发生错误：{e}")


# 如果直接运行此文件
if __name__ == "__main__":
    run_agent_demo()
