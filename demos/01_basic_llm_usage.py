# demos/01_basic_llm_usage.py

"""
学习目标: 学习LangChain的 LLM调用
时间: 2025/10/04
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import os

api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")


chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.5,
        api_key = api_key
)

messages = [
        AIMessage(content="Hi."),
        SystemMessage(content="Your role is a poet."),
        HumanMessage(content="Write a short poem about AI in four lines."),
        ]


response = chat.invoke(messages)
print(response.content) #Displays the AI-generated poem
