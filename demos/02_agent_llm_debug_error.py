# demos/02_agent_llm_debug_error.py

"""
学习目标: 学习LangChain的 Agent调用（调试版本）
时间: 2025/10/05
说明: 包含错误的Agent实现示例，用于调试和学习错误处理
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import os


from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
import time
import random



api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")


chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.5,
        api_key = api_key
)


# 2. 定义简单的工具（避免外部依赖）
@tool
def get_current_time() -> str:
    """获取当前时间"""
    return time.strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate_sum(a: str, b: str) -> str:
    """计算两个数的和"""
    try:
        num_a = float(a)
        num_b = float(b)
        return str(num_a + num_b)
    except ValueError:
        return "输入必须是数字"

@tool
def get_random_number() -> str:
    """生成一个随机数"""
    return str(random.randint(1, 100))

# 3. 定义工具列表
tools = [get_current_time, calculate_sum, get_random_number]

# 4. 创建 ReAct 提示模板
prompt = PromptTemplate.from_template("""
你是一个智能助手，可以使用以下工具来帮助用户：

可用工具：
{tools}

工具名称：{tool_names}

请使用以下格式回答：

Question: 用户的问题
Thought: 我需要思考如何回答这个问题
Action: 选择一个工具
Action Input: 工具的输入参数
Observation: 工具的输出结果
... (可以重复 Thought/Action/Action Input/Observation)
Thought: 我现在知道最终答案了
Final Answer: 对用户问题的最终回答

开始！

Question: {input}
Thought: {agent_scratchpad}
""")

# 5. 创建 Agent
agent = create_react_agent(chat, tools, prompt)

# 6. 创建 AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

print("=== Agent ReAct 模式演示 ===")

# 7. 测试不同的查询
test_queries = [
    "现在几点了？",
    "帮我计算 25 + 37 的和",
    "给我一个随机数"
]

for query in test_queries:
    print(f"\n查询: {query}")
    try:
        result = agent_executor.invoke({"input": query})
        print(f"回答: {result['output']}")
    except Exception as e:
        print(f"❌ 处理查询时出错: {e}")
    print("-" * 50)




