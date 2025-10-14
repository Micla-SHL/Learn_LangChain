# demos/11_memory_types.py

"""
学习目标: LangChain 记忆管理 (Memory)
时间: 2025/10/14
说明: 学习如何为对话应用添加不同类型的记忆功能
"""

import os
from typing import List, Dict, Any
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage,
    BaseMessage, get_buffer_string
)
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.memory import BaseMemory
# LLMChain 已弃用，使用 LCEL 语法替代
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import asyncio

# 初始化模型
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")

chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.7,
    api_key=api_key
)

print("=== 1. 简单缓冲记忆 ===")
# 创建内存对话历史
memory = InMemoryChatMessageHistory()

# 添加一些对话历史
memory.add_user_message("你好，我叫小明")
memory.add_ai_message("你好小明，很高兴认识你！")
memory.add_user_message("我想学习编程")
memory.add_ai_message("编程是一个很好的技能！你想从哪种语言开始？")

# 获取对话历史
messages = memory.messages
print("对话历史:")
for i, msg in enumerate(messages):
    msg_type = "用户" if isinstance(msg, HumanMessage) else "AI"
    print(f"{i+1}. {msg_type}: {msg.content}")
print()

print("=== 2. 使用提示模板的记忆链 ===")
# 创建带记忆的提示模板
memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手，能够记住之前的对话内容。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# 创建简单的记忆链
def simple_memory_chain(input_text: str, history: List[BaseMessage]) -> str:
    """简单的记忆链函数"""
    # 格式化提示
    formatted_prompt = memory_prompt.format_messages(
        chat_history=history,
        input=input_text
    )

    # 调用模型
    response = chat.invoke(formatted_prompt)

    return response.content

# 测试记忆链
print("测试记忆链:")
history = [
    HumanMessage(content="我喜欢科幻电影"),
    AIMessage(content="科幻电影确实很有趣！你最喜欢哪部科幻电影？"),
    HumanMessage(content="我喜欢《星际穿越》"),
    AIMessage(content="《星际穿越》是一部非常优秀的科幻电影！它探讨了很多深刻的科学和哲学问题。")
]

response = simple_memory_chain("你还记得我喜欢什么类型的电影吗？", history)
print("AI回答:", response)
print()

print("=== 3. 自定义记忆类 ===")
class ConversationSummaryMemory(BaseMemory):
    """对话摘要记忆"""

    summary: str = ""
    buffer: List[BaseMessage] = []
    max_buffer_size: int = 5

    def __init__(self):
        super().__init__()
        # 使用 object.__setattr__ 来设置 Pydantic 模型字段
        object.__setattr__(self, 'summary', "")
        object.__setattr__(self, 'buffer', [])
        object.__setattr__(self, 'max_buffer_size', 5)

    @property
    def memory_variables(self) -> List[str]:
        return ["history_summary"]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"history_summary": self.summary}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """保存对话上下文并更新摘要"""
        # 获取用户输入和AI输出
        user_input = inputs.get("input", "")
        ai_output = outputs.get("text", "")

        # 添加到缓冲区
        self.buffer.append(HumanMessage(content=user_input))
        self.buffer.append(AIMessage(content=ai_output))

        # 如果缓冲区满了，更新摘要
        if len(self.buffer) >= self.max_buffer_size:
            self._update_summary()
            self.buffer = []

    def _update_summary(self):
        """更新对话摘要"""
        if not self.buffer:
            return

        # 创建摘要提示
        summary_prompt = PromptTemplate.from_template(
            """以下是之前的对话摘要：
            {current_summary}

            以下是新的对话内容：
            {new_conversation}

            请生成更新后的对话摘要，保持简洁明了。"""
        )

        # 格式化新对话
        conversation_text = get_buffer_string(self.buffer)

        # 生成新摘要
        formatted_prompt = summary_prompt.format(
            current_summary=self.summary or "没有之前的对话记录",
            new_conversation=conversation_text
        )

        response = chat.invoke(formatted_prompt)
        self.summary = response.content

    def clear(self):
        """清空记忆"""
        self.summary = ""
        self.buffer = []

# 测试自定义记忆
print("测试自定义摘要记忆:")
summary_memory = ConversationSummaryMemory()

# 模拟多轮对话
conversations = [
    ("你好，我想学习Python", "Python是一个很好的编程语言选择！"),
    ("Python有什么优势？", "Python语法简洁，社区活跃，应用广泛。"),
    ("我应该从哪里开始？", "建议从基础语法开始，然后练习小项目。"),
    ("有什么好的学习资源吗？", "官方文档、在线课程和实战项目都是不错的选择。"),
    ("我打算开始学习了", "祝您学习愉快！有具体问题随时可以问我。"),
]

for user_input, ai_output in conversations:
    summary_memory.save_context(
        {"input": user_input},
        {"text": ai_output}
    )

print("生成的对话摘要:")
print(summary_memory.summary)
print()

print("=== 4. 带状态的图记忆 (LangGraph) ===")
# 使用LangGraph的MemorySaver创建持久化记忆
memory_saver = MemorySaver()

# 定义状态
class State(MessagesState):
    pass

# 创建带记忆的图
async def call_model_with_memory(state: State):
    """带记忆的模型调用函数"""
    messages = state["messages"]
    response = await chat.ainvoke(messages)
    return {"messages": response}

# 构建图
workflow = StateGraph(State)
workflow.add_node("model", call_model_with_memory)
workflow.add_edge(START, "model")

# 编译时添加记忆
app = workflow.compile(checkpointer=memory_saver)

async def test_graph_memory():
    """测试图记忆"""
    print("测试LangGraph记忆:")

    # 第一轮对话
    config = {"configurable": {"thread_id": "test-thread-1"}}

    query1 = "我的名字是张三，我是一名软件工程师"
    input_messages = [HumanMessage(query1)]

    response1 = await app.ainvoke(
        {"messages": input_messages},
        config
    )
    print(f"用户: {query1}")
    print(f"AI: {response1['messages'][-1].content}")
    print()

    # 第二轮对话（测试记忆）
    query2 = "你还记得我的名字和职业吗？"
    input_messages = [HumanMessage(query2)]

    response2 = await app.ainvoke(
        {"messages": input_messages},
        config
    )
    print(f"用户: {query2}")
    print(f"AI: {response2['messages'][-1].content}")
    print()

# 运行异步测试
asyncio.run(test_graph_memory())

print("=== 5. 长期记忆系统 ===")
class LongTermMemory:
    """模拟长期记忆系统"""

    def __init__(self):
        self.facts: Dict[str, str] = {}
        self.preferences: Dict[str, Any] = {}
        self.relationships: Dict[str, str] = {}

    def add_fact(self, key: str, value: str):
        """添加事实信息"""
        self.facts[key] = value

    def add_preference(self, key: str, value: Any):
        """添加偏好信息"""
        self.preferences[key] = value

    def add_relationship(self, person: str, relationship: str):
        """添加关系信息"""
        self.relationships[person] = relationship

    def get_memory_summary(self) -> str:
        """获取记忆摘要"""
        summary_parts = []

        if self.facts:
            facts_text = ", ".join([f"{k}: {v}" for k, v in self.facts.items()])
            summary_parts.append(f"已知信息: {facts_text}")

        if self.preferences:
            pref_text = ", ".join([f"{k}: {v}" for k, v in self.preferences.items()])
            summary_parts.append(f"偏好: {pref_text}")

        if self.relationships:
            rel_text = ", ".join([f"{k}: {v}" for k, v in self.relationships.items()])
            summary_parts.append(f"关系: {rel_text}")

        return "; ".join(summary_parts) if summary_parts else "没有存储的记忆信息"

# 测试长期记忆
print("测试长期记忆系统:")
long_memory = LongTermMemory()

# 添加记忆
long_memory.add_fact("职业", "数据科学家")
long_memory.add_fact("城市", "北京")
long_memory.add_preference("编程语言", "Python")
long_memory.add_preference("爱好", "阅读")
long_memory.add_relationship("朋友", "李四")

print("长期记忆摘要:")
print(long_memory.get_memory_summary())
print()

# 创建带长期记忆的提示
long_term_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能助手，具有长期记忆能力。

    关于用户的信息：
    {user_memory}

    请根据这些信息提供个性化的回答。"""),
    ("human", "{question}")
])

formatted_prompt = long_term_prompt.format_messages(
    user_memory=long_memory.get_memory_summary(),
    question="根据我的背景，推荐一些学习资源"
)

response = chat.invoke(formatted_prompt)
print("带长期记忆的回答:")
print(response.content)
print()

print("=== 6. 记忆类型总结 ===")
memory_types = {
    "缓冲记忆": "简单存储最近的对话历史",
    "摘要记忆": "将对话内容压缩成摘要",
    "图记忆": "使用LangGraph的持久化记忆",
    "长期记忆": "结构化存储用户信息和偏好",
    "实体记忆": "记住对话中提到的实体和关系",
    "时间窗口记忆": "只在特定时间范围内记住内容"
}

print("不同类型的记忆:")
for memory_type, description in memory_types.items():
    print(f"• {memory_type}: {description}")

print("\n=== 记忆管理学习完成 ===")
print("\n选择记忆类型的考虑因素：")
print("1. 应用场景：简单对话vs复杂应用")
print("2. 存储需求：临时vs持久化")
print("3. 性能要求：响应速度vs记忆完整性")
print("4. 隐私考虑：数据敏感性和用户隐私")
print("5. 成本因素：存储和计算成本")