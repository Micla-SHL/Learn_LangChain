# demos/15_agent_planning.py

"""
学习目标: LangChain Agent 多步骤规划和复杂任务分解
时间: 2025/10/14
说明: 学习如何创建能够进行任务分解、规划和执行的智能 Agent
"""

import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
import asyncio

# 初始化模型
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")

llm = ChatZhipuAI(
    model="glm-4",
    temperature=0.2,
    api_key=api_key
)

print("=== Agent 多步骤规划进阶学习 ===\n")

print("=== 1. 规划 Agent 核心概念 ===")
print("• 任务分解：将复杂任务拆分为可执行的子任务")
print("• 状态管理：跟踪任务执行进度和中间结果")
print("• 动态规划：根据执行结果调整后续计划")
print("• 错误恢复：处理执行失败和异常情况")
print()

print("=== 2. 任务规划数据结构 ===")

@dataclass
class Task:
    """任务数据结构"""
    id: str
    description: str
    status: str  # "pending", "in_progress", "completed", "failed"
    priority: int  # 1-5, 5为最高优先级
    dependencies: List[str]  # 依赖的任务ID
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class Plan:
    """执行计划数据结构"""
    id: str
    goal: str
    tasks: List[Task]
    current_task_id: Optional[str] = None
    status: str = "created"  # "created", "in_progress", "completed", "failed"
    created_at: datetime = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

# 任务管理器
class TaskManager:
    """任务管理器，负责任务的创建、执行和状态跟踪"""

    def __init__(self):
        self.plans: Dict[str, Plan] = {}
        self.current_plan_id: Optional[str] = None

    def create_plan(self, goal: str, tasks_description: List[str]) -> str:
        """创建执行计划"""
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tasks = []

        for i, desc in enumerate(tasks_description):
            task = Task(
                id=f"task_{i+1}_{plan_id}",
                description=desc,
                status="pending",
                priority=3,
                dependencies=[]
            )
            tasks.append(task)

        plan = Plan(
            id=plan_id,
            goal=goal,
            tasks=tasks
        )
        self.plans[plan_id] = plan
        self.current_plan_id = plan_id
        return plan_id

    def get_ready_tasks(self, plan_id: str) -> List[Task]:
        """获取可以执行的任务（没有未完成的依赖）"""
        if plan_id not in self.plans:
            return []

        plan = self.plans[plan_id]
        ready_tasks = []

        for task in plan.tasks:
            if task.status == "pending":
                # 检查依赖是否都已完成
                deps_completed = all(
                    self.get_task_by_id(dep_id).status == "completed"
                    for dep_id in task.dependencies
                    if self.get_task_by_id(dep_id) is not None
                )
                if deps_completed:
                    ready_tasks.append(task)

        # 按优先级排序
        ready_tasks.sort(key=lambda x: x.priority, reverse=True)
        return ready_tasks

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """根据ID获取任务"""
        for plan in self.plans.values():
            for task in plan.tasks:
                if task.id == task_id:
                    return task
        return None

    def update_task_status(self, task_id: str, status: str, result: Optional[str] = None, error: Optional[str] = None):
        """更新任务状态"""
        task = self.get_task_by_id(task_id)
        if task:
            task.status = status
            if result:
                task.result = result
            if error:
                task.error = error
            if status == "completed":
                task.completed_at = datetime.now()

    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """获取计划状态"""
        if plan_id not in self.plans:
            return {"error": "Plan not found"}

        plan = self.plans[plan_id]
        completed_tasks = [t for t in plan.tasks if t.status == "completed"]
        failed_tasks = [t for t in plan.tasks if t.status == "failed"]
        pending_tasks = [t for t in plan.tasks if t.status == "pending"]
        in_progress_tasks = [t for t in plan.tasks if t.status == "in_progress"]

        return {
            "plan_id": plan_id,
            "goal": plan.goal,
            "status": plan.status,
            "total_tasks": len(plan.tasks),
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "pending_tasks": len(pending_tasks),
            "in_progress_tasks": len(in_progress_tasks),
            "progress": len(completed_tasks) / len(plan.tasks) * 100 if plan.tasks else 0
        }

# 全局任务管理器
task_manager = TaskManager()

print("=== 3. 规划专用工具集 ===")

@tool
def create_execution_plan(goal: str, task_breakdown: str) -> str:
    """创建执行计划，将目标分解为具体任务

    Args:
        goal: 最终目标描述
        task_breakdown: 任务分解描述，应该包含具体的执行步骤
    """
    # 解析任务分解（简化版，实际应用中可以使用更复杂的NLP解析）
    tasks = []
    lines = task_breakdown.strip().split('\n')

    for line in lines:
        line = line.strip()
        if line and (line.startswith('-') or line.startswith('*') or line[0].isdigit()):
            # 清理格式
            task_desc = re.sub(r'^[-*\d.\s]+', '', line).strip()
            if task_desc:
                tasks.append(task_desc)

    if not tasks:
        # 如果没有解析出任务，创建默认分解
        tasks = [
            "分析目标和需求",
            "制定详细执行方案",
            "收集必要的信息和资源",
            "执行主要任务",
            "验证和优化结果"
        ]

    plan_id = task_manager.create_plan(goal, tasks)

    return f"""执行计划创建成功！

计划ID: {plan_id}
目标: {goal}

任务列表:
"""
    for i, task in enumerate(tasks, 1):
        print(f"{i}. {task}")

    return f"计划创建完成，共 {len(tasks)} 个任务待执行。"

@tool
def get_next_task(plan_id: str) -> str:
    """获取下一个可执行的任务

    Args:
        plan_id: 计划ID
    """
    ready_tasks = task_manager.get_ready_tasks(plan_id)
    if not ready_tasks:
        return "没有可执行的任务。可能所有任务都已完成，或有任务失败。"

    next_task = ready_tasks[0]
    task_manager.plans[plan_id].current_task_id = next_task.id
    task_manager.update_task_status(next_task.id, "in_progress")

    return f"""下一个任务：
任务ID: {next_task.id}
描述: {next_task.description}
优先级: {next_task.priority}
创建时间: {next_task.created_at.strftime('%Y-%m-%d %H:%M:%S')}

请开始执行此任务。"""

@tool
def complete_task(task_id: str, result: str) -> str:
    """标记任务为完成状态

    Args:
        task_id: 任务ID
        result: 任务执行结果
    """
    task_manager.update_task_status(task_id, "completed", result=result)

    # 检查是否所有任务都已完成
    for plan in task_manager.plans.values():
        if any(t.id == task_id for t in plan.tasks):
            all_completed = all(t.status in ["completed", "failed"] for t in plan.tasks)
            if all_completed:
                plan.status = "completed"
                plan.completed_at = datetime.now()
                return f"""任务 {task_id} 已完成！

执行结果: {result}

🎉 恭喜！计划 {plan.id} 已全部完成！
目标: {plan.goal}
完成时间: {plan.completed_at.strftime('%Y-%m-%d %H:%M:%S')}"""
            break

    return f"""任务 {task_id} 已完成！

执行结果: {result}

可以继续执行下一个任务。"""

@tool
def fail_task(task_id: str, error_message: str) -> str:
    """标记任务为失败状态

    Args:
        task_id: 任务ID
        error_message: 失败原因
    """
    task_manager.update_task_status(task_id, "failed", error=error_message)
    return f"任务 {task_id} 标记为失败：{error_message}"

@tool
def get_plan_progress(plan_id: str) -> str:
    """获取计划执行进度

    Args:
        plan_id: 计划ID
    """
    status = task_manager.get_plan_status(plan_id)
    if "error" in status:
        return status["error"]

    return f"""计划进度报告:
计划ID: {status['plan_id']}
目标: {status['goal']}
状态: {status['status']}

任务统计:
• 总任务数: {status['total_tasks']}
• 已完成: {status['completed_tasks']}
• 进行中: {status['in_progress_tasks']}
• 待执行: {status['pending_tasks']}
• 失败: {status['failed_tasks']}

完成进度: {status['progress']:.1f}%

当前任务: {task_manager.plans[plan_id].current_task_id or '无'}"""

@tool
def search_information(query: str) -> str:
    """搜索信息（模拟）

    Args:
        query: 搜索查询
    """
    # 模拟搜索结果
    search_results = {
        "人工智能": "人工智能(AI)是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "机器学习": "机器学习是人工智能的子领域，专注于算法和统计模型，使计算机系统能够从数据中学习并改进。",
        "深度学习": "深度学习是机器学习的子领域，使用多层神经网络来学习数据的复杂模式。",
        "自然语言处理": "NLP是人工智能的一个分支，专注于计算机与人类语言之间的交互。",
        "计算机视觉": "计算机视觉是人工智能的一个分支，专注于使计算机能够从数字图像或视频中获取高级理解。"
    }

    for key, value in search_results.items():
        if key in query:
            return f"搜索结果：{value}"

    return f"关于 '{query}' 的搜索结果：这是一个模拟的搜索结果。在实际应用中，这里会调用真实的搜索API。"

@tool
def analyze_data(data_description: str) -> str:
    """数据分析工具（模拟）

    Args:
        data_description: 数据描述
    """
    return f"""数据分析报告：
数据类型: {data_description}
分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

主要发现:
• 数据质量良好，无明显异常值
• 存在明显的趋势模式
• 建议进行进一步的时间序列分析
• 可以考虑建立预测模型

建议:
1. 收集更多历史数据以提高分析准确性
2. 考虑外部因素的影响
3. 定期更新分析结果
4. 建立自动监控系统"""

@tool
def generate_report(topic: str, content: str, format_type: str = "markdown") -> str:
    """生成报告

    Args:
        topic: 报告主题
        content: 报告内容
        format_type: 报告格式，可选 "markdown", "html", "plain"
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if format_type == "markdown":
        report = f"""# {topic}

生成时间: {timestamp}

## 内容概述
{content}

## 总结
本报告由AI助手自动生成，基于提供的分析内容。

---
*报告结束*"""
    elif format_type == "html":
        report = f"""<html>
<head><title>{topic}</title></head>
<body>
<h1>{topic}</h1>
<p><em>生成时间: {timestamp}</em></p>
<h2>内容概述</h2>
<p>{content}</p>
<h2>总结</h2>
<p>本报告由AI助手自动生成，基于提供的分析内容。</p>
</body>
</html>"""
    else:
        report = f"""{topic}
生成时间: {timestamp}

内容概述:
{content}

总结:
本报告由AI助手自动生成，基于提供的分析内容。"""

    return f"报告已生成（{format_type}格式）：\n\n{report}"

# 创建规划工具列表
planning_tools = [
    create_execution_plan,
    get_next_task,
    complete_task,
    fail_task,
    get_plan_progress,
    search_information,
    analyze_data,
    generate_report
]

print("已创建的规划工具:")
for tool in planning_tools:
    print(f"- {tool.name}: {tool.description}")
print()

print("=== 4. 创建规划 Agent ===")

# 规划 Agent 提示模板
planning_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的任务规划和管理AI助手。你的主要职责是：

1. **任务分解**：将用户的目标分解为具体、可执行的子任务
2. **计划制定**：创建详细的执行计划，包含任务优先级和依赖关系
3. **进度跟踪**：监控任务执行进度，及时调整计划
4. **结果验证**：确保任务完成质量达到预期

工作流程：
1. 首先使用 create_execution_plan 工具创建执行计划
2. 使用 get_next_task 获取下一个任务
3. 执行任务（可能需要使用其他工具）
4. 使用 complete_task 标记任务完成
5. 重复步骤2-4直到所有任务完成
6. 使用 get_plan_progress 检查整体进度

注意事项：
- 每次只执行一个任务
- 确保任务完成质量
- 遇到问题时使用 fail_task 标记失败
- 保持与用户沟通进度"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 创建规划 Agent
planning_agent = create_tool_calling_agent(llm, planning_tools, planning_prompt)

# 创建规划 Agent 执行器
planning_executor = AgentExecutor(
    agent=planning_agent,
    tools=planning_tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=15
)

print("规划 Agent 创建成功!")
print()

print("=== 5. 测试规划 Agent ===")

# 测试用例1: 简单任务规划
print("测试用例1: 简单任务规划")
try:
    result1 = planning_executor.invoke({
        "input": "帮我制定一个学习人工智能的计划，包括理论基础学习和实践项目"
    })
    print(f"结果: {result1['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

# 测试用例2: 研究项目规划
print("测试用例2: 研究项目规划")
try:
    result2 = planning_executor.invoke({
        "input": "我需要完成一个关于机器学习在医疗领域应用的研究报告，请帮我制定详细的执行计划"
    })
    print(f"结果: {result2['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

# 测试用例3: 商业分析项目
print("测试用例3: 商业分析项目")
try:
    result3 = planning_executor.invoke({
        "input": "分析竞品并制定市场策略，包括数据收集、分析和报告生成"
    })
    print(f"结果: {result3['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

print("=== 6. 高级规划功能演示 ===")

print("当前任务管理器状态:")
for plan_id, plan in task_manager.plans.items():
    status = task_manager.get_plan_status(plan_id)
    print(f"计划 {plan_id}: {status['goal']} (进度: {status['progress']:.1f}%)")
print()

print("=== 7. 规划 Agent 最佳实践 ===")
print("1. 任务设计原则:")
print("   • 明确具体：每个任务都应该有明确的描述和预期结果")
print("   • 可执行性：任务应该可以在合理时间内完成")
print("   • 独立性：减少任务间的复杂依赖关系")
print("   • 可测量：能够判断任务是否成功完成")
print()

print("2. 规划策略:")
print("   • 分层分解：从高层目标分解到具体执行步骤")
print("   • 优先级排序：先执行重要且紧急的任务")
print("   • 风险控制：识别关键路径和潜在风险")
print("   • 迭代优化：根据执行结果动态调整计划")
print()

print("3. 状态管理:")
print("   • 实时跟踪：监控任务执行状态和进度")
print("   • 异常处理：及时处理失败任务和异常情况")
print("   • 资源协调：合理安排和调度可用资源")
print("   • 结果验证：确保每个任务的输出质量")
print()

print("=== 多步骤规划 Agent 学习完成 ===")
print("\n核心技能:")
print("✅ 掌握任务分解和规划技术")
print("✅ 学会复杂任务的状态管理")
print("✅ 理解动态规划和调整机制")
print("✅ 掌握多步骤执行和错误恢复")
print("\n🎯 Agent 系统进阶学习总结:")
print("1. 基础 ReAct Agent - 理解思考和行动循环")
print("2. 工具调用 Agent - 掌握现代化 function calling")
print("3. 多步骤规划 Agent - 学会复杂任务分解和管理")
print("\n你已经掌握了 LangChain Agent 系统的核心技术！")