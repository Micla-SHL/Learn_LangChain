# demos/17_multi_agent_system.py

"""
学习目标: Multi-Agent 系统
时间: 2025/10/14
说明: 学习如何构建多智能体协作系统，包括通信、任务分配、协调和冲突解决
"""

import os
import json
import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, PriorityQueue
import uuid

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
import time

# 初始化模型
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")

chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.1,
    api_key=api_key
)

print("=== Multi-Agent 系统学习 ===\n")

print("=== 1. Multi-Agent 核心概念 ===")
print("Multi-Agent 系统是由多个协作的智能体组成的系统，具有以下特点：")
print("• 分布式智能: 每个 Agent 都有自己的智能和决策能力")
print("• 协作与协调: Agent 之间可以协作完成任务")
print("• 通信机制: Agent 之间通过消息传递进行交流")
print("• 任务分配: 根据能力动态分配任务")
print("• 自适应性: 系统能够适应环境变化和 Agent 增减")
print()

# 定义消息类型
class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    COORDINATION_REQUEST = "coordination_request"
    COORDINATION_RESPONSE = "coordination_response"
    STATUS_UPDATE = "status_update"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"

# 定义 Agent 状态
class AgentState(Enum):
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"

# 定义任务优先级
class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class Message:
    """Agent 之间的通信消息"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    receiver: str = ""
    message_type: MessageType = MessageType.TASK_REQUEST
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: TaskPriority = TaskPriority.NORMAL

@dataclass
class Task:
    """任务定义"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    requirements: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    status: str = "pending"  # pending, assigned, in_progress, completed, failed
    assigned_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class AgentCapability:
    """Agent 能力定义"""
    name: str
    description: str
    can_handle: List[str]  # 可以处理的任务类型
    max_concurrent_tasks: int = 1
    avg_task_time: float = 60.0  # 平均任务处理时间（秒）

print("=== 2. Agent 基础类定义 ===")

class BaseAgent:
    """Agent 基类"""

    def __init__(self, agent_id: str, name: str, capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.state = AgentState.IDLE
        self.current_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.message_queue = PriorityQueue()
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.system = None  # 引用多智能体系统

        # 注册默认消息处理器
        self.register_message_handler(MessageType.TASK_REQUEST, self.handle_task_request)
        self.register_message_handler(MessageType.STATUS_UPDATE, self.handle_status_update)

        # 启动消息处理线程
        self.running = True
        self.message_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.message_thread.start()

    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """注册消息处理器"""
        self.message_handlers[message_type] = handler

    def send_message(self, receiver: str, message_type: MessageType, content: Dict[str, Any],
                    priority: TaskPriority = TaskPriority.NORMAL):
        """发送消息给其他 Agent"""
        if self.system:
            message = Message(
                sender=self.agent_id,
                receiver=receiver,
                message_type=message_type,
                content=content,
                priority=priority
            )
            self.system.route_message(message)

    def receive_message(self, message: Message):
        """接收消息"""
        self.message_queue.put((-message.priority.value, message))

    def _process_messages(self):
        """处理消息队列"""
        while self.running:
            try:
                if not self.message_queue.empty():
                    _, message = self.message_queue.get(timeout=1)
                    if message.message_type in self.message_handlers:
                        self.message_handlers[message.message_type](message)
                time.sleep(0.1)
            except:
                continue

    def handle_task_request(self, message: Message):
        """处理任务请求"""
        task_data = message.content
        task = Task(
            id=task_data.get('id', str(uuid.uuid4())),
            title=task_data.get('title', ''),
            description=task_data.get('description', ''),
            requirements=task_data.get('requirements', []),
            priority=task_data.get('priority', TaskPriority.NORMAL)
        )

        # 检查是否能处理该任务
        if self.can_handle_task(task):
            self.accept_task(task)
            # 回复任务接受消息
            self.send_message(
                receiver=message.sender,
                message_type=MessageType.TASK_RESPONSE,
                content={
                    'task_id': task.id,
                    'status': 'accepted',
                    'agent_id': self.agent_id
                }
            )
        else:
            # 回复任务拒绝消息
            self.send_message(
                receiver=message.sender,
                message_type=MessageType.TASK_RESPONSE,
                content={
                    'task_id': task.id,
                    'status': 'rejected',
                    'reason': 'Agent 无法处理此任务类型',
                    'agent_id': self.agent_id
                }
            )

    def handle_status_update(self, message: Message):
        """处理状态更新消息"""
        # 可以根据状态更新调整自己的行为
        pass

    def can_handle_task(self, task: Task) -> bool:
        """检查是否能处理指定任务"""
        if self.state != AgentState.IDLE:
            return False

        if len(self.current_tasks) >= sum(cap.max_concurrent_tasks for cap in self.capabilities):
            return False

        # 检查是否有匹配的能力
        for capability in self.capabilities:
            for requirement in task.requirements:
                if requirement in capability.can_handle:
                    return True

        return False

    def accept_task(self, task: Task):
        """接受任务"""
        task.status = "assigned"
        task.assigned_agent = self.agent_id
        task.started_at = datetime.now()
        self.current_tasks.append(task)
        self.state = AgentState.BUSY

        # 异步执行任务
        threading.Thread(target=self.execute_task, args=(task,), daemon=True).start()

    def execute_task(self, task: Task):
        """执行任务（子类需要重写）"""
        task.status = "in_progress"

        try:
            # 模拟任务执行
            time.sleep(2)

            # 这里应该有实际的任务处理逻辑
            result = {
                'task_id': task.id,
                'agent_id': self.agent_id,
                'result': f'任务 {task.title} 已由 {self.name} 完成',
                'completion_time': datetime.now().isoformat()
            }

            task.result = result
            task.status = "completed"
            task.completed_at = datetime.now()

            # 通知系统任务完成
            self.notify_task_completion(task)

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            self.notify_task_failure(task)

        finally:
            self.current_tasks.remove(task)
            self.completed_tasks.append(task)
            if len(self.current_tasks) == 0:
                self.state = AgentState.IDLE

    def notify_task_completion(self, task: Task):
        """通知任务完成"""
        if self.system:
            self.system.on_task_completed(task)

    def notify_task_failure(self, task: Task):
        """通知任务失败"""
        if self.system:
            self.system.on_task_failed(task)

    def get_status(self) -> Dict[str, Any]:
        """获取 Agent 状态"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'state': self.state.value,
            'current_tasks': len(self.current_tasks),
            'completed_tasks': len(self.completed_tasks),
            'capabilities': [cap.name for cap in self.capabilities]
        }

    def shutdown(self):
        """关闭 Agent"""
        self.running = False
        self.state = AgentState.OFFLINE

print("=== 3. 具体类型 Agent 实现 ===")

class ResearchAgent(BaseAgent):
    """研究型 Agent - 专注于信息收集和分析"""

    def __init__(self, agent_id: str, name: str):
        capabilities = [
            AgentCapability(
                name="信息检索",
                description="搜索和收集信息",
                can_handle=["搜索", "信息收集", "文献调研", "数据分析"],
                max_concurrent_tasks=3,
                avg_task_time=30.0
            ),
            AgentCapability(
                name="报告撰写",
                description="撰写研究报告",
                can_handle=["报告", "文档", "总结", "分析"],
                max_concurrent_tasks=2,
                avg_task_time=120.0
            )
        ]
        super().__init__(agent_id, name, capabilities)

    def execute_task(self, task: Task):
        """执行研究任务"""
        task.status = "in_progress"

        try:
            if "搜索" in task.description or "信息收集" in task.description:
                # 模拟信息检索
                time.sleep(1)
                result = {
                    'task_id': task.id,
                    'agent_id': self.agent_id,
                    'type': '信息检索',
                    'result': f'已收集关于 "{task.title}" 的相关信息，找到 15 个相关资料',
                    'sources': ['资料1', '资料2', '资料3'],
                    'completion_time': datetime.now().isoformat()
                }
            else:
                # 模拟报告撰写
                time.sleep(3)
                result = {
                    'task_id': task.id,
                    'agent_id': self.agent_id,
                    'type': '报告撰写',
                    'result': f'已完成 "{task.title}" 的研究报告，共 5 页',
                    'report_summary': '报告主要包含了...',
                    'completion_time': datetime.now().isoformat()
                }

            task.result = result
            task.status = "completed"
            task.completed_at = datetime.now()
            self.notify_task_completion(task)

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            self.notify_task_failure(task)

        finally:
            self.current_tasks.remove(task)
            self.completed_tasks.append(task)
            if len(self.current_tasks) == 0:
                self.state = AgentState.IDLE

class AnalysisAgent(BaseAgent):
    """分析型 Agent - 专注于数据分析和决策"""

    def __init__(self, agent_id: str, name: str):
        capabilities = [
            AgentCapability(
                name="数据分析",
                description="分析和处理数据",
                can_handle=["数据分析", "统计", "计算", "建模"],
                max_concurrent_tasks=2,
                avg_task_time=60.0
            ),
            AgentCapability(
                name="决策支持",
                description="提供决策建议",
                can_handle=["决策", "建议", "评估", "优化"],
                max_concurrent_tasks=1,
                avg_task_time=90.0
            )
        ]
        super().__init__(agent_id, name, capabilities)

    def execute_task(self, task: Task):
        """执行分析任务"""
        task.status = "in_progress"

        try:
            if "分析" in task.description or "统计" in task.description:
                # 模拟数据分析
                time.sleep(2.5)
                result = {
                    'task_id': task.id,
                    'agent_id': self.agent_id,
                    'type': '数据分析',
                    'result': f'已完成 "{task.title}" 的数据分析',
                    'insights': ['洞察1', '洞察2', '洞察3'],
                    'metrics': {'准确率': 0.95, '置信度': 0.98},
                    'completion_time': datetime.now().isoformat()
                }
            else:
                # 模拟决策支持
                time.sleep(1.5)
                result = {
                    'task_id': task.id,
                    'agent_id': self.agent_id,
                    'type': '决策支持',
                    'result': f'为 "{task.title}" 提供决策建议',
                    'recommendation': '建议采取方案A，因为...',
                    'confidence': 0.87,
                    'completion_time': datetime.now().isoformat()
                }

            task.result = result
            task.status = "completed"
            task.completed_at = datetime.now()
            self.notify_task_completion(task)

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            self.notify_task_failure(task)

        finally:
            self.current_tasks.remove(task)
            self.completed_tasks.append(task)
            if len(self.current_tasks) == 0:
                self.state = AgentState.IDLE

class CoordinationAgent(BaseAgent):
    """协调型 Agent - 专注于任务协调和资源管理"""

    def __init__(self, agent_id: str, name: str):
        capabilities = [
            AgentCapability(
                name="任务协调",
                description="协调多个Agent的任务",
                can_handle=["协调", "分配", "调度", "管理"],
                max_concurrent_tasks=5,
                avg_task_time=20.0
            ),
            AgentCapability(
                name="冲突解决",
                description="解决Agent间的冲突",
                can_handle=["冲突", "协商", "调解", "仲裁"],
                max_concurrent_tasks=3,
                avg_task_time=45.0
            )
        ]
        super().__init__(agent_id, name, capabilities)

    def execute_task(self, task: Task):
        """执行协调任务"""
        task.status = "in_progress"

        try:
            if "协调" in task.description or "分配" in task.description:
                # 模拟任务协调
                time.sleep(1)
                result = {
                    'task_id': task.id,
                    'agent_id': self.agent_id,
                    'type': '任务协调',
                    'result': f'已完成 "{task.title}" 的任务协调',
                    'assigned_agents': ['agent1', 'agent2'],
                    'coordination_plan': '协调计划详情...',
                    'completion_time': datetime.now().isoformat()
                }
            else:
                # 模拟冲突解决
                time.sleep(2)
                result = {
                    'task_id': task.id,
                    'agent_id': self.agent_id,
                    'type': '冲突解决',
                    'result': f'已解决 "{task.title}" 相关的冲突',
                    'resolution': '解决方案...',
                    'affected_agents': ['agent1', 'agent2'],
                    'completion_time': datetime.now().isoformat()
                }

            task.result = result
            task.status = "completed"
            task.completed_at = datetime.now()
            self.notify_task_completion(task)

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            self.notify_task_failure(task)

        finally:
            self.current_tasks.remove(task)
            self.completed_tasks.append(task)
            if len(self.current_tasks) == 0:
                self.state = AgentState.IDLE

print("=== 4. Multi-Agent 系统管理器 ===")

class MultiAgentSystem:
    """多智能体系统管理器"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue = PriorityQueue()
        self.running = True
        self.task_assignment_strategy = "capability_based"  # capability_based, load_balance, priority

        # 启动任务分配线程
        self.assignment_thread = threading.Thread(target=self._task_assignment_loop, daemon=True)
        self.assignment_thread.start()

    def register_agent(self, agent: BaseAgent):
        """注册 Agent"""
        self.agents[agent.agent_id] = agent
        agent.system = self
        print(f"Agent 已注册: {agent.name} ({agent.agent_id})")

    def unregister_agent(self, agent_id: str):
        """注销 Agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.shutdown()
            del self.agents[agent_id]
            print(f"Agent 已注销: {agent.name} ({agent_id})")

    def route_message(self, message: Message):
        """路由消息到目标 Agent"""
        if message.receiver in self.agents:
            self.agents[message.receiver].receive_message(message)
        else:
            print(f"错误：找不到接收者 Agent: {message.receiver}")

    def submit_task(self, title: str, description: str, requirements: List[str],
                   priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """提交任务到系统"""
        task = Task(
            title=title,
            description=description,
            requirements=requirements,
            priority=priority
        )

        self.tasks[task.id] = task
        self.task_queue.put((-priority.value, task))

        print(f"任务已提交: {title} (ID: {task.id})")
        return task.id

    def _task_assignment_loop(self):
        """任务分配循环"""
        while self.running:
            try:
                if not self.task_queue.empty():
                    _, task = self.task_queue.get(timeout=1)
                    self.assign_task(task)
                time.sleep(0.5)
            except:
                continue

    def assign_task(self, task: Task):
        """分配任务给合适的 Agent"""
        suitable_agents = []

        # 找到能够处理该任务的 Agent
        for agent in self.agents.values():
            if agent.can_handle_task(task):
                # 计算匹配度
                match_score = self._calculate_match_score(agent, task)
                suitable_agents.append((agent, match_score))

        if suitable_agents:
            # 根据策略选择 Agent
            selected_agent = self._select_agent(suitable_agents)

            # 发送任务给选中的 Agent
            task_data = {
                'id': task.id,
                'title': task.title,
                'description': task.description,
                'requirements': task.requirements,
                'priority': task.priority
            }

            selected_agent.send_message(
                receiver=selected_agent.agent_id,
                message_type=MessageType.TASK_REQUEST,
                content=task_data,
                priority=task.priority
            )

            print(f"任务已分配: {task.title} -> {selected_agent.name}")
        else:
            print(f"没有合适的 Agent 可以处理任务: {task.title}")
            task.status = "failed"
            task.error = "没有可用的 Agent"

    def _calculate_match_score(self, agent: BaseAgent, task: Task) -> float:
        """计算 Agent 与任务的匹配度"""
        score = 0.0

        # 能力匹配
        capability_match = 0
        for capability in agent.capabilities:
            for requirement in task.requirements:
                if requirement in capability.can_handle:
                    capability_match += 1

        score += capability_match * 0.6

        # 负载考虑
        load_factor = 1.0 - (len(agent.current_tasks) /
                           sum(cap.max_concurrent_tasks for cap in agent.capabilities))
        score += load_factor * 0.3

        # 状态考虑
        if agent.state == AgentState.IDLE:
            score += 0.1
        elif agent.state == AgentState.BUSY:
            score -= 0.05

        return score

    def _select_agent(self, suitable_agents: List[tuple]) -> BaseAgent:
        """根据策略选择最合适的 Agent"""
        if self.task_assignment_strategy == "capability_based":
            # 选择匹配度最高的
            suitable_agents.sort(key=lambda x: x[1], reverse=True)
            return suitable_agents[0][0]
        elif self.task_assignment_strategy == "load_balance":
            # 选择负载最轻的
            suitable_agents.sort(key=lambda x: len(x[0].current_tasks))
            return suitable_agents[0][0]
        else:
            # 默认选择第一个
            return suitable_agents[0][0]

    def on_task_completed(self, task: Task):
        """任务完成回调"""
        print(f"任务完成: {task.title} by {task.assigned_agent}")

        # 可以在这里添加后续处理逻辑
        # 比如触发相关任务或通知其他 Agent

    def on_task_failed(self, task: Task):
        """任务失败回调"""
        print(f"任务失败: {task.title} - {task.error}")

        # 可以在这里添加重试逻辑或错误处理

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        agent_status = {}
        for agent_id, agent in self.agents.items():
            agent_status[agent_id] = agent.get_status()

        task_status = {
            'total_tasks': len(self.tasks),
            'pending_tasks': len([t for t in self.tasks.values() if t.status == 'pending']),
            'assigned_tasks': len([t for t in self.tasks.values() if t.status == 'assigned']),
            'in_progress_tasks': len([t for t in self.tasks.values() if t.status == 'in_progress']),
            'completed_tasks': len([t for t in self.tasks.values() if t.status == 'completed']),
            'failed_tasks': len([t for t in self.tasks.values() if t.status == 'failed'])
        }

        return {
            'agents': agent_status,
            'tasks': task_status,
            'system_running': self.running
        }

    def shutdown(self):
        """关闭系统"""
        self.running = False
        for agent in self.agents.values():
            agent.shutdown()
        print("Multi-Agent 系统已关闭")

print("=== 5. 演示 Multi-Agent 系统 ===")

# 创建多智能体系统
multi_agent_system = MultiAgentSystem()

# 创建并注册不同类型的 Agent
research_agent1 = ResearchAgent("research_001", "研究助手1")
research_agent2 = ResearchAgent("research_002", "研究助手2")
analysis_agent1 = AnalysisAgent("analysis_001", "分析师1")
coordination_agent = CoordinationAgent("coordination_001", "协调员")

# 注册所有 Agent
multi_agent_system.register_agent(research_agent1)
multi_agent_system.register_agent(research_agent2)
multi_agent_system.register_agent(analysis_agent1)
multi_agent_system.register_agent(coordination_agent)

print(f"\n已注册 {len(multi_agent_system.agents)} 个 Agent")
print()

# 提交各种类型的任务
print("=== 6. 提交测试任务 ===")

# 任务1: 信息检索
task1_id = multi_agent_system.submit_task(
    title="人工智能市场调研",
    description="搜索和收集人工智能市场相关信息",
    requirements=["搜索", "信息收集"],
    priority=TaskPriority.HIGH
)

# 任务2: 数据分析
task2_id = multi_agent_system.submit_task(
    title="用户行为数据分析",
    description="分析用户行为数据并提供洞察",
    requirements=["数据分析", "统计"],
    priority=TaskPriority.NORMAL
)

# 任务3: 报告撰写
task3_id = multi_agent_system.submit_task(
    title="技术报告撰写",
    description="撰写技术发展报告",
    requirements=["报告", "文档"],
    priority=TaskPriority.NORMAL
)

# 任务4: 决策支持
task4_id = multi_agent_system.submit_task(
    title="产品决策分析",
    description="为新产品发布提供决策建议",
    requirements=["决策", "建议"],
    priority=TaskPriority.URGENT
)

# 任务5: 任务协调
task5_id = multi_agent_system.submit_task(
    title="项目协调",
    description="协调多个研究项目",
    requirements=["协调", "分配"],
    priority=TaskPriority.NORMAL
)

print(f"\n已提交 5 个任务")
print()

# 等待任务执行
print("=== 7. 等待任务执行 ===")
time.sleep(8)  # 等待任务执行

# 显示系统状态
print("=== 8. 系统状态报告 ===")
status = multi_agent_system.get_system_status()

print("Agent 状态:")
for agent_id, agent_info in status['agents'].items():
    print(f"  {agent_info['name']} ({agent_id[:8]}...): "
          f"{agent_info['state']} - 当前任务: {agent_info['current_tasks']}, "
          f"已完成: {agent_info['completed_tasks']}")

print(f"\n任务统计:")
for key, value in status['tasks'].items():
    print(f"  {key}: {value}")

print()

# 显示完成的任务详情
print("=== 9. 已完成任务详情 ===")
completed_tasks = [task for task in multi_agent_system.tasks.values() if task.status == "completed"]
for task in completed_tasks:
    print(f"任务: {task.title}")
    print(f"  执行者: {multi_agent_system.agents[task.assigned_agent].name}")
    print(f"  结果: {task.result['result'] if task.result else '无结果'}")
    print(f"  完成时间: {task.completed_at.strftime('%H:%M:%S')}")
    print()

print("=== 10. Multi-Agent 系统特性总结 ===")
print("✅ 分布式架构: 每个 Agent 独立运行和管理")
print("✅ 智能任务分配: 基于能力和负载自动分配任务")
print("✅ 异步消息传递: Agent 之间通过消息进行通信")
print("✅ 动态负载均衡: 根据实时负载情况调整任务分配")
print("✅ 容错机制: 单个 Agent 失败不影响整个系统")
print("✅ 可扩展性: 可以动态添加和移除 Agent")
print()

print("=== 11. Multi-Agent 最佳实践 ===")
print("1. Agent 设计:")
print("   • 单一职责: 每个 Agent 专注特定领域")
print("   • 能力明确: 清确定义 Agent 的能力范围")
print("   • 状态管理: 完善的状态转换机制")
print()

print("2. 通信机制:")
print("   • 消息标准化: 统一的消息格式和协议")
print("   • 异步处理: 避免阻塞和死锁")
print("   • 错误处理: 完善的消息处理异常机制")
print()

print("3. 任务分配:")
print("   • 能力匹配: 根据 Agent 能力分配任务")
print("   • 负载均衡: 考虑 Agent 当前负载")
print("   • 优先级处理: 高优先级任务优先分配")
print()

print("4. 系统监控:")
print("   • 实时监控: 监控 Agent 状态和任务进度")
print("   • 性能分析: 分析系统性能和瓶颈")
print("   • 日志记录: 完整的操作日志和审计")
print()

print("=== Multi-Agent 系统学习完成 ===")
print("\n核心成就:")
print("✅ 构建了完整的多智能体协作系统")
print("✅ 实现了智能任务分配和协调机制")
print("✅ 掌握了 Agent 间通信和消息传递")
print("✅ 学会了分布式系统设计和实现")
print("\n下一步: 生产环境部署和优化")

# 关闭系统
multi_agent_system.shutdown()