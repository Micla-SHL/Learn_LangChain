# demos/08_prompt_templates.py

"""
学习目标: LangChain 提示模板 (Prompt Templates)
时间: 2025/10/14
说明: 学习如何创建和使用提示模板来标准化LLM输入
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatZhipuAI
import os

# 初始化模型
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")

chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.7,
    api_key=api_key
)

print("=== 1. 基础提示模板 (PromptTemplate) ===")
# 创建简单的提示模板
basic_prompt = PromptTemplate.from_template(
    "你是一个{role}。请{task}。主题是：{topic}"
)

# 格式化提示
formatted_prompt = basic_prompt.format(
    role="专业的中文教师",
    task="解释这个概念的含义",
    topic="人工智能"
)
print("格式化后的提示:")
print(formatted_prompt)

# 使用模板生成回答
response = chat.invoke(formatted_prompt)
print("AI回答:", response.content)
print()

print("=== 2. 多变量提示模板 ===")
# 创建更复杂的模板
complex_prompt = PromptTemplate(
    input_variables=["product", "feature", "benefit"],
    template="作为{product}的产品经理，请介绍{feature}功能如何为用户带来{benefit}。"
)

formatted_complex = complex_prompt.format(
    product="LangChain",
    feature="链式调用",
    benefit="开发效率提升"
)
print("复杂模板示例:")
print(formatted_complex)

response = chat.invoke(formatted_complex)
print("AI回答:", response.content)
print()

print("=== 3. 聊天提示模板 (ChatPromptTemplate) ===")
# 创建聊天模板
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{expertise}，专门帮助用户解决{domain}相关问题。"),
    ("human", "我的问题是：{question}"),
    ("ai", "我来帮您分析这个问题..."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{follow_up}")
])

# 格式化聊天模板
from langchain_core.messages import AIMessage

history = [AIMessage(content="这是一个很好的问题，需要从多个角度来考虑。")]

formatted_chat = chat_prompt.format_messages(
    expertise="软件架构师",
    domain="系统设计",
    question="如何设计一个高可用的微服务架构？",
    history=history,
    follow_up="能具体介绍一下服务发现机制吗？"
)

print("聊天模板消息数量:", len(formatted_chat))
for i, msg in enumerate(formatted_chat):
    print(f"消息 {i+1} ({type(msg).__name__}): {msg.content[:50]}...")

response = chat.invoke(formatted_chat)
print("AI回答:", response.content)
print()

print("=== 4. 模板验证 ===")
# 验证模板输入变量
try:
    incomplete_prompt = basic_prompt.format(role="老师")  # 缺少变量
except Exception as e:
    print("模板验证错误:", str(e))

print("=== 5. 动态模板组合 ===")
# 组合多个模板
title_template = PromptTemplate.from_template("标题：{title}")
content_template = PromptTemplate.from_template("内容：{content}")
footer_template = PromptTemplate.from_template("作者：{author}")

# 组合使用
combined_prompt = "\n".join([
    title_template.format(title="LangChain学习笔记"),
    content_template.format(content="今天学习了提示模板的使用方法"),
    footer_template.format(author="学习者")
])

print("组合模板:")
print(combined_prompt)
print()

print("=== 6. 条件模板 ==="
)
def create_conditional_prompt(is_formal: bool, topic: str):
    if is_formal:
        return PromptTemplate.from_template(
            "尊敬的用户，关于{topic}这个话题，我将为您提供专业的分析和建议。"
        )
    else:
        return PromptTemplate.from_template(
            "嗨！让我们聊聊{topic}吧，我会用轻松的方式为你解答！"
        )

# 使用条件模板
formal_prompt = create_conditional_prompt(True, "机器学习")
informal_prompt = create_conditional_prompt(False, "机器学习")

print("正式模板:", formal_prompt.format(topic="机器学习"))
print("非正式模板:", informal_prompt.format(topic="机器学习"))

print("\n=== 提示模板学习完成 ===")