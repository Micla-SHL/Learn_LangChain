
# demos/06_build_chatbot.py
"""
学习目标: 学习LangChain的 教程: Build a ChatBot 第二部分
时间: 2025/10/05
"""
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

import os
import asyncio  # ✅ 导入 asyncio

api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")

chat = ChatZhipuAI(
    model="glm-4.5-flash",
    temperature=0.7,
    top_p=0.9,
    api_key=api_key
)


trimmer = trim_messages(
    max_tokens=20,
    strategy="last",
    token_counter=count_tokens_approximately,
    include_system=True,
    allow_partial=False,
    start_on="human",
    )

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个友好的助手，用{language}语言尽可能好地回答用户的问题。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
    )

# 定义新的图

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


workflow = StateGraph(state_schema=State)


# Define the function that calls the model
async def call_model(state: State):    
    print(f"Messages before trimming: {len(state['messages'])}")
    trimmed_messages = await trimmer.ainvoke(state["messages"])
    print(f"Messages after trimming: {len(trimmed_messages)}")
    print("Remaining messages:")
    for msg in trimmed_messages:
        print(f"  {type(msg).__name__}: {msg.content}")
    
    prompt = await prompt_template.ainvoke(
        {"messages": trimmed_messages, "language": state["language"]}
        )
    response = await chat.ainvoke(prompt)
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
# ✅ 将所有异步调用包装在 async 函数中
async def main():
    config = {"configurable": {"thread_id": "abc123"}}
    
    # 第一轮对话
    query = "你好，我是杨顶天。请写一篇300字的关于春天的文章"
    language = "中文"
    input_messages = [HumanMessage(query)]

    
    async for chunk, metadata in app.astream(
        {"messages": input_messages, "language": language},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):  # Filter to just model responses
            print(chunk.content, end="|")

    # output = await app.ainvoke({"messages": input_messages, "language": language}, config)
    #output["messages"][-1].pretty_print()
    
    # 第二轮对话（同一个 thread）
    query = "我叫什么名字？"
    input_messages = [HumanMessage(query)]
    output = await app.ainvoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    """
    # 第三轮对话（新的 thread，应该不记得名字）
    config = {"configurable": {"thread_id": "abc234"}}
    input_messages = [HumanMessage(query)]
    output = await app.ainvoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    
    # 第四轮对话（回到原来的 thread，应该记得名字）
    config = {"configurable": {"thread_id": "abc123"}}
    input_messages = [HumanMessage(query)]
    output = await app.ainvoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    """
# ✅ 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())
