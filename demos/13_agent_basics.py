# demos/13_agent_basics.py

"""
学习目标: LangChain Agent 基础和 ReAct 模式
时间: 2025/10/14
说明: 学习如何创建和使用基础的 ReAct Agent，包括自定义工具开发
"""

import os
import json
import math
from typing import Dict, Any, List
from datetime import datetime

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# 初始化模型
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")

chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.1,
    api_key=api_key
)

print("=== Agent 系统进阶学习 ===\n")

print("=== 1. 理解 ReAct 模式 ===")
print("ReAct = Reasoning + Acting")
print("Agent 通过思考-行动-观察的循环来解决问题")
print()

print("=== 2. 创建自定义工具 ===")

# 定义计算器工具
@tool
def calculator(expression: str) -> str:
    """计算数学表达式，支持加减乘除、幂运算、三角函数等

    Args:
        expression: 数学表达式，如 "2 + 3 * 4" 或 "sin(0.5)"
    """
    try:
        # 安全的数学表达式求值
        allowed_names = {
            k: v for k, v in math.__dict__.items()
            if not k.startswith("_")
        }

        # 创建安全环境
        safe_dict = {
            "__builtins__": {},
            "pi": math.pi,
            "e": math.e,
        }
        safe_dict.update(allowed_names)

        result = eval(expression, safe_dict)
        return f"计算结果: {result}"

    except Exception as e:
        return f"计算错误: {str(e)}"

# 定义当前时间工具
@tool
def get_current_time(format_type: str = "standard") -> str:
    """获取当前时间和日期

    Args:
        format_type: 时间格式类型，可选 "standard", "iso", "timestamp", "chinese"
    """
    now = datetime.now()

    if format_type == "standard":
        return now.strftime("%Y-%m-%d %H:%M:%S")
    elif format_type == "iso":
        return now.isoformat()
    elif format_type == "timestamp":
        return str(int(now.timestamp()))
    elif format_type == "chinese":
        return now.strftime("%Y年%m月%d日 %H时%M分%S秒")
    else:
        return f"不支持的格式类型: {format_type}"

# 定义文本分析工具
@tool
def analyze_text(text: str, analysis_type: str = "basic") -> str:
    """分析文本的基本信息

    Args:
        text: 要分析的文本内容
        analysis_type: 分析类型，可选 "basic", "detailed", "sentiment"
    """
    if analysis_type == "basic":
        char_count = len(text)
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?') + text.count('。') + text.count('！') + text.count('？')

        return f"""文本基本信息:
- 字符数: {char_count}
- 单词数: {word_count}
- 句子数: {sentence_count}"""

    elif analysis_type == "detailed":
        # 更详细的分析
        char_count = len(text)
        word_count = len(text.split())
        char_no_space = len(text.replace(' ', '').replace('\t', '').replace('\n', ''))
        avg_word_length = sum(len(word) for word in text.split()) / max(word_count, 1)

        return f"""详细文本分析:
- 总字符数: {char_count}
- 不含空格字符数: {char_no_space}
- 单词数: {word_count}
- 平均单词长度: {avg_word_length:.2f}
- 段落数: {text.count('\n\n') + 1}"""

    elif analysis_type == "sentiment":
        # 简单的情感分析
        positive_words = ["好", "棒", "优秀", "喜欢", "美好", "amazing", "great", "excellent", "good", "love"]
        negative_words = ["坏", "差", "糟糕", "讨厌", "terrible", "bad", "awful", "hate", "horrible"]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            sentiment = "积极"
        elif negative_count > positive_count:
            sentiment = "消极"
        else:
            sentiment = "中性"

        return f"""情感分析:
- 积极词汇数量: {positive_count}
- 消极词汇数量: {negative_count}
- 整体情感倾向: {sentiment}"""

    else:
        return f"不支持的分析类型: {analysis_type}"

# 定义单位转换工具
@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """单位转换工具

    Args:
        value: 要转换的数值
        from_unit: 原始单位
        to_unit: 目标单位
    """
    # 长度单位转换
    length_units = {
        "meter": 1.0,
        "kilometer": 0.001,
        "centimeter": 100.0,
        "millimeter": 1000.0,
        "mile": 0.000621371,
        "yard": 1.09361,
        "foot": 3.28084,
        "inch": 39.3701
    }

    # 重量单位转换
    weight_units = {
        "kilogram": 1.0,
        "gram": 1000.0,
        "milligram": 1000000.0,
        "pound": 2.20462,
        "ounce": 35.274
    }

    # 温度单位转换
    if from_unit.lower() in ["celsius", "c"] and to_unit.lower() in ["fahrenheit", "f"]:
        result = (value * 9/5) + 32
        return f"{value}°C = {result:.2f}°F"
    elif from_unit.lower() in ["fahrenheit", "f"] and to_unit.lower() in ["celsius", "c"]:
        result = (value - 32) * 5/9
        return f"{value}°F = {result:.2f}°C"

    # 长度转换
    elif from_unit.lower() in length_units and to_unit.lower() in length_units:
        base_value = value * length_units[from_unit.lower()]
        result = base_value / length_units[to_unit.lower()]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"

    # 重量转换
    elif from_unit.lower() in weight_units and to_unit.lower() in weight_units:
        base_value = value * weight_units[from_unit.lower()]
        result = base_value / weight_units[to_unit.lower()]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"

    else:
        return f"不支持的单位转换: {from_unit} 到 {to_unit}"

# 创建工具列表
tools = [
    calculator,
    get_current_time,
    analyze_text,
    unit_converter
]

print("已创建的工具:")
for tool in tools:
    print(f"- {tool.name}: {tool.description[:50]}...")
print()

print("=== 3. 创建 ReAct Agent ===")
# 使用 LangChain Hub 的 ReAct 提示模板
# 注意：这个可能需要网络连接，我们提供一个本地模板作为备选

try:
    prompt = hub.pull("hwchase17/react")
    print("使用 LangChain Hub 的 ReAct 提示模板")
except Exception as e:
    print(f"无法从 Hub 加载提示模板: {e}")
    print("使用本地 ReAct 提示模板")

    # 本地 ReAct 提示模板
    prompt = ChatPromptTemplate.from_template("""回答以下问题，你可以使用这些工具：

{tools}

使用以下格式：

Question: 你需要回答的问题
Thought: 你应该思考要做什么
Action: 选择要使用的工具名称
Action Input: 工具的输入参数
Observation: 工具执行的结果
... (这个 Thought/Action/Action Input/Observation 可以重复)
Thought: 我现在知道最终答案了
Final Answer: 最终的答案

开始！

Question: {input}
Thought: {agent_scratchpad}""")

# 创建 ReAct Agent
agent = create_react_agent(chat, tools, prompt)

# 创建 Agent 执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    return_intermediate_steps=True
)

print("ReAct Agent 创建成功!")
print()

print("=== 4. 测试基础 ReAct Agent ===")

# 测试用例1: 数学计算
print("测试用例1: 数学计算")
try:
    result1 = agent_executor.invoke({
        "input": "帮我计算 (15 * 8 + 32) / 4 - 6 的结果"
    })
    print(f"结果: {result1['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

# 测试用例2: 时间查询
print("测试用例2: 时间查询")
try:
    result2 = agent_executor.invoke({
        "input": "现在是什么时间？请用中文格式显示"
    })
    print(f"结果: {result2['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

# 测试用例3: 文本分析
print("测试用例3: 文本分析")
sample_text = "人工智能正在改变我们的世界。这项技术非常棒，让生活变得更美好。虽然有挑战，但未来充满希望！"
try:
    result3 = agent_executor.invoke({
        "input": f"请分析这段文本的情感倾向：'{sample_text}'"
    })
    print(f"结果: {result3['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

# 测试用例4: 单位转换
print("测试用例4: 单位转换")
try:
    result4 = agent_executor.invoke({
        "input": "将 25 摄氏度转换为华氏度"
    })
    print(f"结果: {result4['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

# 测试用例5: 复杂任务（多步推理）
print("测试用例5: 复杂任务")
try:
    result5 = agent_executor.invoke({
        "input": "如果现在是 2024 年，那么我出生年份是 1990 年，我现在多大？还有距离 2030 年还有多少天？"
    })
    print(f"结果: {result5['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

print("=== 5. Agent 工作原理分析 ===")
print("ReAct Agent 的核心思想：")
print("1. Thought: 分析问题，决定下一步行动")
print("2. Action: 选择合适的工具和参数")
print("3. Observation: 观察工具执行结果")
print("4. 循环直到得出最终答案")
print()

print("=== 6. 常见问题和解决方案 ===")
print("问题1: Agent 无限循环")
print("解决: 设置 max_iterations 参数")
print()

print("问题2: 工具调用失败")
print("解决: 在工具函数中添加异常处理")
print()

print("问题3: Agent 不使用工具")
print("解决: 改进提示模板，明确工具用途")
print()

print("问题4: 解析错误")
print("解决: 设置 handle_parsing_errors=True")
print()

print("=== 基础 ReAct Agent 学习完成 ===")
print("\n关键要点:")
print("✅ 理解 ReAct (Reasoning + Acting) 模式")
print("✅ 学会创建自定义工具")
print("✅ 掌握 Agent 配置和错误处理")
print("✅ 了解多步推理的工作流程")
print("\n下一步: 学习更高级的工具调用和规划能力")
