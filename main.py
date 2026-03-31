from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()  # 加载环境变量

# 1. 初始化大模型
llm = ChatOpenAI(
    model="gpt-4",  # 或 "gpt-3.5-turbo" 控制成本
    temperature=0,  # 降低随机性，让Agent更稳定
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 2. 定义工具
search = SerpAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))
tools = [
    Tool(
        name="Search",  # 工具名称
        func=search.run,  # 工具函数
        description="在互联网上搜索当前问题的最新信息。当你需要了解实时或事实性信息时，必须使用此工具。"  # 关键：告诉Agent何时使用
    ),
    # 你可以在这里添加更多工具，如计算器、维基百科API等
]

# 3. 创建并运行Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # 最通用的Agent类型
    verbose=True,  # 打印详细思考过程，方便调试
    handle_parsing_errors=True  # 优雅处理解析错误
)

# 4. 运行测试
if __name__ == "__main__":
    query = "特斯拉2024年第四季度营收是多少？主要增长点是什么？"
    print(f"用户问题: {query}\n")
    result = agent.invoke({"input": query})
    print(f"\n=== 最终答案 ===\n{result['output']}")