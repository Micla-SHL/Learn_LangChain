#!/usr/bin/env python3
"""
测试Agent最小化版本
"""

import os
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 简单工具
@tool
def calculator(expression: str) -> str:
    """简单计算器"""
    try:
        result = eval(expression)
        return f"结果: {result}"
    except Exception as e:
        return f"错误: {e}"

# 初始化
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    print("错误: 未设置ZHIPUAI_API_KEY")
    exit(1)

print("测试1: 模型初始化...")
try:
    chat = ChatZhipuAI(model="glm-4", api_key=api_key)
    result = chat.invoke("你好")
    print("✅ 模型初始化成功")
except Exception as e:
    print(f"❌ 模型初始化失败: {e}")
    exit(1)

print("测试2: 工具调用...")
try:
    tools = [calculator]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(chat, tools, prompt)
    executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        max_iterations=3, 
        max_execution_time=30
    )

    result = executor.invoke({"input": "计算 15 * 8"})
    print("✅ 工具调用成功")
    print(f"结果: {result}")
except Exception as e:
    print(f"❌ 工具调用失败: {e}")
    exit(1)

print("🎉 所有测试通过!")