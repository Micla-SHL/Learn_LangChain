# demos/03_agent_llm_debug.py

"""
学习目标: 学习LangChain的 Agent调用（修复版本）
时间: 2025/10/05
说明: 使用JSON Chat Agent的正确实现示例
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import os

api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")


# 从环境变量读取 API Key，而不是硬编码
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("请设置环境变量: TAVILY_API_KEY")

# 设置到环境变量中（如果工具需要从环境变量读取）
os.environ["TAVILY_API_KEY"] = tavily_api_key



from langchain import hub
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]
prompt = hub.pull("hwchase17/react-chat-json")


print("\n模板内容预览:")
print(prompt)
print("\n" + "="*60)

llm = ChatZhipuAI(temperature=0.01, model="glm-4")

agent = create_json_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )


agent_executor.invoke({"input":"什么是Python？"})
