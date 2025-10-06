# demos/01_basic_llm_usage.py

"""
学习目标: 学习LangChain的 Agent调用
时间: 2025/10/05
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import os


from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub



api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")


chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.5,
        api_key = api_key
)


# 2. 定义工具
tools = [
    DuckDuckGoSearchRun(name="搜索引擎"),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(), name="维基百科"),
    Tool(
        name="学术论文",
        func=ArxivAPIWrapper().run,
        description="搜索 arXiv 学术论文数据库，适合查找科研文献"
    )
    ]


# 3. 获取最新的 Prompt 模板（从 LangChain Hub）
prompt = hub.pull("hwchase17/react")

# 4. 创建 Agent
agent = create_react_agent(llm, tools, prompt)

# 5. 创建 AgentExecutor（新版必须）
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5  # 防止无限循环
    )


# 6. 使用新的 invoke 方法（不再是 run）
result = agent_executor.invoke({
    "input": "帮我搜索一下深度学习的最新进展"
})

print(result["output"])




