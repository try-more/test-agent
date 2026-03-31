import sys

try:
    from langchain.agents import create_tool_calling_agent, AgentExecutor

    print("IMPORT_SUCCESS")
except ImportError as e:
    print(f"IMPORT_ERROR: {e}")
except Exception as e:
    print(f"ERROR: {e}")
