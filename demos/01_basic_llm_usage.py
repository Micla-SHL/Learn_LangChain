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

# 正确的消息顺序：系统消息 -> 用户消息
messages = [
    SystemMessage(content="你是一个诗人，擅长写简短优美的诗歌。"),
    HumanMessage(content="请用中文写一首关于人工智能的四行诗。")
]

print("=== LangChain 基础LLM调用示例 ===")
print(f"用户问题: {messages[1].content}")
print()

response = chat.invoke(messages)
print(f"AI回复: {response.content}")
