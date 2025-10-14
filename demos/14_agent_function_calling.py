# demos/14_agent_function_calling.py

"""
学习目标: LangChain Agent 工具调用 (Function Calling)
时间: 2025/10/14
说明: 学习使用现代化的 function calling 方式创建更强大的 Agent
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import RunnablePassthrough

# 初始化模型
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")

chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.1,
    api_key=api_key
)

print("=== Agent 工具调用进阶学习 ===\n")

print("=== 1. 现代化工具调用优势 ===")
print("• 更自然的对话流程")
print("• 自动参数验证和类型检查")
print("• 更好的错误处理")
print("• 支持多工具并行调用")
print()

print("=== 2. 创建高级工具集 ===")

# 天气查询工具
@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """获取指定城市的天气信息

    Args:
        city: 城市名称，如 "北京"、"上海"、"New York"
        unit: 温度单位，可选 "celsius" 或 "fahrenheit"
    """
    # 模拟天气数据（实际应用中可以调用真实的天气API）
    weather_data = {
        "北京": {"temp": 22, "condition": "晴朗", "humidity": 45},
        "上海": {"temp": 25, "condition": "多云", "humidity": 60},
        "广州": {"temp": 28, "condition": "小雨", "humidity": 75},
        "深圳": {"temp": 27, "condition": "晴朗", "humidity": 65},
        "New York": {"temp": 18, "condition": "晴朗", "humidity": 50},
        "London": {"temp": 15, "condition": "阴天", "humidity": 70},
        "Tokyo": {"temp": 20, "condition": "晴朗", "humidity": 55}
    }

    if city not in weather_data:
        return f"抱歉，无法获取 {city} 的天气信息。支持的城市：{', '.join(weather_data.keys())}"

    data = weather_data[city]
    temp = data["temp"]

    if unit == "fahrenheit":
        temp = (temp * 9/5) + 32
        temp_unit = "°F"
    else:
        temp_unit = "°C"

    return f"""{city} 当前天气：
温度：{temp}{temp_unit}
天气状况：{data['condition']}
湿度：{data['humidity']}%
更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""

# 新闻摘要工具
@tool
def get_news_summary(category: str, limit: int = 3) -> str:
    """获取指定类别的新闻摘要

    Args:
        category: 新闻类别，可选 "科技"、"财经"、"体育"、"娱乐"
        limit: 返回新闻条数，默认3条
    """
    # 模拟新闻数据
    news_data = {
        "科技": [
            {"title": "人工智能技术取得重大突破", "summary": "研究人员在自然语言处理领域取得显著进展", "time": "2小时前"},
            {"title": "量子计算机实现新里程碑", "summary": "新型量子处理器在特定任务上超越传统计算机", "time": "5小时前"},
            {"title": "5G网络覆盖率持续提升", "summary": "全国5G基站建设加速，网络质量显著改善", "time": "1天前"}
        ],
        "财经": [
            {"title": "股市收盘上涨", "summary": "主要指数全线上扬，科技股领涨", "time": "3小时前"},
            {"title": "央行发布最新政策", "summary": "货币政策保持稳健，支持实体经济发展", "time": "6小时前"},
            {"title": "全球经济复苏势头良好", "summary": "多国经济数据显示积极信号", "time": "1天前"}
        ],
        "体育": [
            {"title": "国足获得重要胜利", "summary": "在友谊赛中以2-0战胜对手", "time": "4小时前"},
            {"title": "篮球联赛精彩对决", "summary": "卫冕冠军主场险胜", "time": "8小时前"},
            {"title": "网球大师赛开幕", "summary": "世界顶级选手齐聚", "time": "2天前"}
        ],
        "娱乐": [
            {"title": "新电影票房突破纪录", "summary": "国产大片首周票房表现优异", "time": "1小时前"},
            {"title": "音乐节阵容公布", "summary": "多位知名艺术家将参加", "time": "7小时前"},
            {"title": "电视剧收视率创新高", "summary": "热播剧引发观众热议", "time": "1天前"}
        ]
    }

    if category not in news_data:
        return f"抱歉，没有找到 {category} 类别的新闻。支持的类别：{', '.join(news_data.keys())}"

    news_list = news_data[category][:limit]
    result = f"{category}新闻摘要（最新{len(news_list)}条）：\n\n"

    for i, news in enumerate(news_list, 1):
        result += f"{i}. {news['title']}\n"
        result += f"   {news['summary']}\n"
        result += f"   {news['time']}\n\n"

    return result

# 邮件发送工具
@tool
def send_email(to: str, subject: str, body: str, priority: str = "normal") -> str:
    """发送邮件（模拟功能）

    Args:
        to: 收件人邮箱地址
        subject: 邮件主题
        body: 邮件正文内容
        priority: 优先级，可选 "low"、"normal"、"high"
    """
    # 验证邮箱格式
    if "@" not in to or "." not in to.split("@")[-1]:
        return "错误：邮箱地址格式不正确"

    priority_levels = {"low": "低", "normal": "普通", "high": "高"}
    if priority not in priority_levels:
        priority = "normal"

    # 模拟邮件发送
    email_id = f"MSG_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return f"""邮件发送成功！
邮件ID: {email_id}
收件人: {to}
主题: {subject}
优先级: {priority_levels[priority]}
发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
状态: 已发送

注：这是模拟邮件发送功能"""

# 日程管理工具
@tool
def schedule_event(title: str, date: str, time: str, duration: int = 60, reminder: bool = True) -> str:
    """创建日程安排

    Args:
        title: 事件标题
        date: 日期，格式 YYYY-MM-DD
        time: 时间，格式 HH:MM
        duration: 持续时间（分钟），默认60分钟
        reminder: 是否设置提醒，默认True
    """
    try:
        # 验证日期格式
        event_date = datetime.strptime(date, "%Y-%m-%d")
        event_time = datetime.strptime(time, "%H:%M")

        # 创建事件时间
        event_datetime = datetime.combine(event_date.date(), event_time.time())
        end_datetime = event_datetime + timedelta(minutes=duration)

        # 生成事件ID
        event_id = f"EVENT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        reminder_text = "已设置提醒" if reminder else "未设置提醒"

        return f"""日程创建成功！
事件ID: {event_id}
标题: {title}
日期: {event_date.strftime('%Y年%m月%d日')}
时间: {time} - {end_datetime.strftime('%H:%M')}
持续时间: {duration} 分钟
提醒: {reminder_text}
创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

状态: 已添加到日程表"""

    except ValueError as e:
        return f"错误：日期或时间格式不正确。请使用 YYYY-MM-DD 格式的日期和 HH:MM 格式的时间。"

# 文件操作工具
@tool
def file_operations(operation: str, filename: str, content: Optional[str] = None) -> str:
    """文件操作工具（模拟）

    Args:
        operation: 操作类型，可选 "read"、"write"、"delete"、"list"
        filename: 文件名
        content: 文件内容（写入操作时需要）
    """
    # 模拟文件操作
    if operation == "read":
        return f"读取文件 '{filename}' 的内容：这是模拟的文件内容。"
    elif operation == "write":
        if content is None:
            return "错误：写入操作需要提供文件内容"
        return f"成功写入文件 '{filename}'，内容长度：{len(content)} 字符"
    elif operation == "delete":
        return f"文件 '{filename}' 已删除"
    elif operation == "list":
        return "当前目录文件列表：\n- document.txt\n- data.json\n- report.pdf"
    else:
        return f"错误：不支持的操作类型 '{operation}'。支持的操作：read, write, delete, list"

# 汇率转换工具
@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """货币汇率转换

    Args:
        amount: 金额
        from_currency: 原始货币代码，如 "USD", "CNY", "EUR"
        to_currency: 目标货币代码，如 "USD", "CNY", "EUR"
    """
    # 模拟汇率数据（实际应用中应该调用实时汇率API）
    exchange_rates = {
        "USD": {"CNY": 7.2, "EUR": 0.92, "JPY": 149.5},
        "CNY": {"USD": 0.139, "EUR": 0.128, "JPY": 20.8},
        "EUR": {"USD": 1.09, "CNY": 7.8, "JPY": 162.3},
        "JPY": {"USD": 0.0067, "CNY": 0.048, "EUR": 0.0062}
    }

    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    if from_currency == to_currency:
        return f"{amount} {from_currency} = {amount} {to_currency}"

    if from_currency in exchange_rates and to_currency in exchange_rates[from_currency]:
        rate = exchange_rates[from_currency][to_currency]
        result = amount * rate
        return f"{amount} {from_currency} = {result:.2f} {to_currency}\n汇率：1 {from_currency} = {rate} {to_currency}"
    else:
        return f"错误：不支持的货币转换。支持的货币：{', '.join(exchange_rates.keys())}"

# 创建工具列表
tools = [
    get_weather,
    get_news_summary,
    send_email,
    schedule_event,
    file_operations,
    currency_converter
]

print("已创建的高级工具:")
for tool in tools:
    print(f"- {tool.name}: {tool.description}")
print()

print("=== 3. 创建工具调用 Agent ===")

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能助手，可以使用多种工具来帮助用户完成任务。

你可以使用天气查询、新闻摘要、邮件发送、日程管理、文件操作和货币转换等工具。

使用以下格式：
- 如果需要使用工具，调用相应的工具函数
- 如果不需要工具，直接回答用户问题
- 可以同时使用多个工具来完成复杂任务
- 请确保提供准确和有用的信息

请根据用户的需求选择合适的工具并提供帮助。"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 创建工具调用 Agent
agent = create_tool_calling_agent(chat, tools, prompt)

# 创建 Agent 执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

print("工具调用 Agent 创建成功!")
print()

print("=== 4. 测试工具调用 Agent ===")

# 测试用例1: 天气查询
print("测试用例1: 天气查询")
try:
    result1 = agent_executor.invoke({
        "input": "帮我查询北京今天的天气情况"
    })
    print(f"结果: {result1['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

# 测试用例2: 复合任务（天气 + 邮件）
print("测试用例2: 复合任务")
try:
    result2 = agent_executor.invoke({
        "input": "查询上海的天气，然后发送一封邮件给 friend@example.com，主题是'今日天气报告'，包含天气信息"
    })
    print(f"结果: {result2['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

# 测试用例3: 日程管理
print("测试用例3: 日程管理")
try:
    result3 = agent_executor.invoke({
        "input": "帮我安排明天下午2点的会议，主题是'项目讨论'，持续90分钟，需要提醒"
    })
    print(f"结果: {result3['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

# 测试用例4: 货币转换
print("测试用例4: 货币转换")
try:
    result4 = agent_executor.invoke({
        "input": "将100美元转换成人民币，当前的汇率是多少？"
    })
    print(f"结果: {result4['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

# 测试用例5: 新闻查询
print("测试用例5: 新闻查询")
try:
    result5 = agent_executor.invoke({
        "input": "给我看看今天的科技新闻摘要"
    })
    print(f"结果: {result5['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

print("=== 5. 高级功能特性 ===")

# 测试多工具协作
print("测试多工具协作:")
try:
    result6 = agent_executor.invoke({
        "input": "查询广州天气，获取科技新闻，然后创建一个明天的日程提醒我关注天气变化和科技动态"
    })
    print(f"结果: {result6['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

print("=== 6. 工具调用最佳实践 ===")
print("1. 工具设计原则:")
print("   • 单一职责：每个工具专注一个功能")
print("   • 参数明确：提供清晰的参数说明和类型定义")
print("   • 错误处理：包含完善的异常处理机制")
print("   • 文档完整：提供详细的工具描述")
print()

print("2. Agent 配置优化:")
print("   • 合理设置 max_iterations 防止无限循环")
print("   • 启用 handle_parsing_errors 处理解析错误")
print("   • 根据任务复杂度调整 temperature 参数")
print()

print("3. 提示工程:")
print("   • 清晰说明工具用途和使用方法")
print("   • 提供使用示例和格式要求")
print("   • 引导合理使用工具组合")
print()

print("=== 工具调用 Agent 学习完成 ===")
print("\n核心技能:")
print("✅ 掌握现代化 function calling 技术")
print("✅ 学会设计和实现高级工具")
print("✅ 理解多工具协作机制")
print("✅ 掌握错误处理和调试技巧")
print("\n下一步: 学习多步骤规划和复杂任务分解")