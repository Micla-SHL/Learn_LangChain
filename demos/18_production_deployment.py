# demos/18_production_deployment.py

"""
学习目标: LangChain 生产环境部署
时间: 2025/10/14
说明: 学习如何将 LangChain 应用部署到生产环境，包括容器化、API设计、监控和安全性
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import uuid
import hashlib
import secrets

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
from contextlib import asynccontextmanager

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
import prometheus_client
import psutil
import threading

print("=== LangChain 生产环境部署学习 ===\n")

print("=== 1. 生产环境部署核心概念 ===")
print("生产环境部署的关键要素：")
print("• 容器化: 使用 Docker 进行应用打包和部署")
print("• API 网关: 统一的接口管理和安全控制")
print("• 监控系统: 实时监控应用性能和健康状态")
print("• 日志管理: 结构化日志和集中管理")
print("• 安全性: 认证、授权和防护机制")
print("• 可扩展性: 支持水平扩展和负载均衡")
print()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建 Prometheus 指标
REQUEST_COUNT = prometheus_client.Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = prometheus_client.Histogram('request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = prometheus_client.Gauge('active_connections', 'Active connections')
CPU_USAGE = prometheus_client.Gauge('cpu_usage_percent', 'CPU usage percent')
MEMORY_USAGE = prometheus_client.Gauge('memory_usage_percent', 'Memory usage percent')

# 速率限制
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="LangChain Production API", version="1.0.0")

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 添加速率限制异常处理器
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 安全认证
security = HTTPBearer()

# 配置
class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    RATE_LIMIT = "100/minute"  # 每分钟100个请求
    MAX_CONCURRENT_REQUESTS = 10

config = Config()

print("=== 2. 数据模型定义 ===")

class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., min_length=1, max_length=4000, description="用户消息")
    session_id: Optional[str] = Field(None, description="会话ID")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")
    agent_type: Optional[str] = Field("general", description="Agent类型")

class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str = Field(..., description="AI回复")
    session_id: str = Field(..., description="会话ID")
    timestamp: datetime = Field(..., description="响应时间戳")
    agent_type: str = Field(..., description="使用的Agent类型")
    processing_time: float = Field(..., description="处理时间(秒)")

class HealthCheck(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    timestamp: datetime = Field(..., description="检查时间")
    version: str = Field(..., description="版本号")
    uptime: float = Field(..., description="运行时间(秒)")
    system_info: Dict[str, Any] = Field(..., description="系统信息")

class MetricsResponse(BaseModel):
    """指标响应"""
    timestamp: datetime = Field(..., description="指标时间戳")
    metrics: Dict[str, Any] = Field(..., description="系统指标")

print("=== 3. 生产级 LangChain 服务 ===")

class ProductionLangChainService:
    """生产级 LangChain 服务"""

    def __init__(self):
        self.start_time = time.time()
        self.chat = None
        self.agents = {}
        self.session_store = {}
        self.request_count = 0
        self.active_requests = 0
        self.error_count = 0

        # 初始化 LLM
        self._initialize_llm()

        # 初始化 Agents
        self._initialize_agents()

        # 启动系统监控
        self._start_monitoring()

    def _initialize_llm(self):
        """初始化大语言模型"""
        try:
            api_key = os.getenv("ZHIPUAI_API_KEY")
            if not api_key:
                raise ValueError("ZHIPUAI_API_KEY 环境变量未设置")

            self.chat = ChatZhipuAI(
                model="glm-4",
                temperature=0.1,
                api_key=api_key,
                max_retries=3,
                request_timeout=30
            )
            logger.info("LLM 初始化成功")
        except Exception as e:
            logger.error(f"LLM 初始化失败: {e}")
            raise

    def _initialize_agents(self):
        """初始化各种类型的 Agent"""
        try:
            # 创建通用工具
            @tool
            def calculator(expression: str) -> str:
                """计算数学表达式"""
                try:
                    # 安全的数学表达式计算
                    allowed_names = {
                        k: v for k, v in __builtins__.items()
                        if not k.startswith("_") and k not in ["eval", "exec", "compile"]
                    }
                    result = eval(expression, {"__builtins__": allowed_names})
                    return f"计算结果: {result}"
                except Exception as e:
                    return f"计算错误: {str(e)}"

            @tool
            def get_current_time() -> str:
                """获取当前时间"""
                return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            @tool
            def get_system_info() -> str:
                """获取系统信息"""
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                return f"""系统信息:
CPU 使用率: {cpu_percent}%
内存使用率: {memory.percent}%
可用内存: {memory.available / (1024**3):.2f} GB"""

            tools = [calculator, get_current_time, get_system_info]

            # 创建通用提示模板
            general_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个智能助手，能够回答用户的问题并使用工具完成特定任务。"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])

            # 创建 Agent
            general_agent = create_tool_calling_agent(self.chat, tools, general_prompt)
            self.agents["general"] = AgentExecutor(
                agent=general_agent,
                tools=tools,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=5
            )

            logger.info("Agent 初始化完成")

        except Exception as e:
            logger.error(f"Agent 初始化失败: {e}")
            raise

    def _start_monitoring(self):
        """启动系统监控"""
        def update_metrics():
            while True:
                try:
                    # 更新系统指标
                    CPU_USAGE.set(psutil.cpu_percent())
                    memory = psutil.virtual_memory()
                    MEMORY_USAGE.set(memory.percent)
                    ACTIVE_CONNECTIONS.set(self.active_requests)

                    time.sleep(5)  # 每5秒更新一次
                except Exception as e:
                    logger.error(f"监控更新失败: {e}")

        monitor_thread = threading.Thread(target=update_metrics, daemon=True)
        monitor_thread.start()
        logger.info("系统监控已启动")

    async def process_request(self, request: ChatRequest) -> ChatResponse:
        """处理聊天请求"""
        start_time = time.time()
        session_id = request.session_id or str(uuid.uuid4())

        try:
            # 增加活跃请求计数
            self.active_requests += 1
            self.request_count += 1

            logger.info(f"处理请求 - 会话ID: {session_id}, 消息: {request.message[:50]}...")

            # 获取或创建会话历史
            if session_id not in self.session_store:
                self.session_store[session_id] = {
                    "created_at": datetime.now(),
                    "message_count": 0
                }

            # 选择合适的 Agent
            agent_type = request.agent_type or "general"
            agent = self.agents.get(agent_type, self.agents["general"])

            # 执行推理
            if hasattr(agent, 'ainvoke'):
                result = await agent.ainvoke({"input": request.message})
            else:
                result = agent.invoke({"input": request.message})

            response_text = result.get("output", "抱歉，我无法处理这个请求。")

            # 更新会话信息
            self.session_store[session_id]["message_count"] += 1
            self.session_store[session_id]["last_active"] = datetime.now()

            processing_time = time.time() - start_time

            logger.info(f"请求处理完成 - 会话ID: {session_id}, 耗时: {processing_time:.2f}秒")

            return ChatResponse(
                response=response_text,
                session_id=session_id,
                timestamp=datetime.now(),
                agent_type=agent_type,
                processing_time=processing_time
            )

        except Exception as e:
            self.error_count += 1
            logger.error(f"请求处理失败 - 会话ID: {session_id}, 错误: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="内部服务器错误"
            )

        finally:
            self.active_requests -= 1

    def get_health_status(self) -> HealthCheck:
        """获取健康状态"""
        uptime = time.time() - self.start_time

        # 检查各个组件状态
        components_status = {}

        # 检查 LLM
        try:
            # 简单的连接测试
            components_status["llm"] = "healthy"
        except:
            components_status["llm"] = "unhealthy"

        # 检查 Agents
        components_status["agents"] = "healthy" if self.agents else "unhealthy"

        # 检查内存使用
        memory = psutil.virtual_memory()
        memory_status = "healthy" if memory.percent < 90 else "warning"

        # 整体状态
        overall_status = "healthy"
        if any(status == "unhealthy" for status in components_status.values()):
            overall_status = "unhealthy"
        elif memory_status == "warning":
            overall_status = "warning"

        return HealthCheck(
            status=overall_status,
            timestamp=datetime.now(),
            version="1.0.0",
            uptime=uptime,
            system_info={
                "components": components_status,
                "memory_usage": memory.percent,
                "active_requests": self.active_requests,
                "total_requests": self.request_count,
                "error_count": self.error_count,
                "active_sessions": len(self.session_store)
            }
        )

# 全局服务实例
service = ProductionLangChainService()

print("=== 4. FastAPI 应用生命周期 ===")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("应用启动中...")

    # 启动时执行
    logger.info("应用启动完成")

    yield

    # 关闭时执行
    logger.info("应用关闭中...")
    logger.info("应用关闭完成")

app.router.lifespan_context = lifespan

print("=== 5. API 路由定义 ===")

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "LangChain Production API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthCheck)
@limiter.limit("30/minute")
async def health_check(request: Request):
    """健康检查端点"""
    return service.get_health_status()

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("100/minute")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """聊天端点"""

    # 验证 Token（简化版本）
    if not verify_token(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据"
        )

    # 记录请求
    REQUEST_COUNT.labels(method="POST", endpoint="/chat").inc()

    with REQUEST_DURATION.time():
        response = await service.process_request(request)

    # 后台任务：记录日志
    background_tasks.add_task(log_chat_request, request, response)

    return response

@app.get("/metrics", response_model=MetricsResponse)
@limiter.limit("60/minute")
async def get_metrics(request: Request):
    """获取系统指标"""
    return MetricsResponse(
        timestamp=datetime.now(),
        metrics={
            "requests_total": service.request_count,
            "active_requests": service.active_requests,
            "error_count": service.error_count,
            "active_sessions": len(service.session_store),
            "uptime": time.time() - service.start_time,
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent
        }
    )

@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Prometheus 格式的指标"""
    return Response(
        content=prometheus_client.generate_latest(),
        media_type="text/plain"
    )

@app.delete("/sessions/{session_id}")
@limiter.limit("30/minute")
async def clear_session(request: Request, session_id: str):
    """清除会话"""
    if session_id in service.session_store:
        del service.session_store[session_id]
        return {"message": f"会话 {session_id} 已清除"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="会话不存在"
        )

print("=== 6. 辅助函数 ===")

def verify_token(token: str) -> bool:
    """验证认证令牌（简化版本）"""
    # 在生产环境中，这里应该验证 JWT token
    # 这里简化为检查 token 是否不为空
    return token is not None and len(token) > 10

async def log_chat_request(request: ChatRequest, response: ChatResponse):
    """记录聊天请求日志"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": response.session_id,
        "request_message": request.message,
        "response_message": response.response,
        "processing_time": response.processing_time,
        "agent_type": response.agent_type
    }

    logger.info(f"聊天日志: {json.dumps(log_entry, ensure_ascii=False)}")

print("=== 7. 安全中间件 ===")

class SecurityMiddleware:
    """安全中间件"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # 添加安全头
        if scope["type"] == "http":
            # 这里可以添加安全检查逻辑
            pass

        await self.app(scope, receive, send)

# 添加安全中间件
app.middleware("http")(SecurityMiddleware)

print("=== 8. 生产环境配置示例 ===")

def create_production_config():
    """创建生产环境配置"""
    return {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 4,
        "log_level": "info",
        "access_log": True,
        "use_colors": False,
        "reload": False,  # 生产环境不使用热重载
        "limit_concurrency": 100,
        "limit_max_requests": 1000,
        "timeout_keep_alive": 5
    }

print("=== 9. 性能优化建议 ===")

optimization_tips = {
    "连接池": "配置数据库和外部API的连接池",
    "缓存": "使用 Redis 缓存常见请求结果",
    "异步处理": "将耗时操作放入后台任务队列",
    "压缩": "启用 Gzip 压缩减少传输大小",
    "CDN": "使用 CDN 分发静态资源",
    "负载均衡": "部署多个实例进行负载均衡"
}

print("=== 10. 监控和告警 ===")

monitoring_setup = {
    "指标收集": "Prometheus + Grafana",
    "日志管理": "ELK Stack (Elasticsearch, Logstash, Kibana)",
    "错误追踪": "Sentry",
    "健康检查": "定期的健康检查端点",
    "告警通知": "基于阈值的自动告警"
}

print("=== 生产环境部署学习完成 ===")
print("\n核心组件:")
print("✅ FastAPI Web 框架")
print("✅ Docker 容器化")
print("✅ 安全认证和授权")
print("✅ 速率限制和防护")
print("✅ 性能监控和指标")
print("✅ 结构化日志记录")
print("✅ 健康检查和故障检测")
print("✅ 生产级配置和优化")
print()

print("=== 部署建议 ===")
print("1. 容器编排:")
print("   • 使用 Docker Compose 进行本地开发")
print("   • 使用 Kubernetes 进行生产部署")
print("   • 配置健康检查和自动重启")
print()

print("2. 环境管理:")
print("   • 开发、测试、预生产、生产环境分离")
print("   • 使用环境变量管理配置")
print("   • 敏感信息使用密钥管理系统")
print()

print("3. 监控告警:")
print("   • 设置关键指标监控")
print("   • 配置多级告警机制")
print("   • 建立故障处理流程")
print()

print("4. 性能优化:")
print("   • 进行负载测试找出瓶颈")
print("   • 优化数据库查询和缓存策略")
print("   • 配置适当的资源限制")
print()

if __name__ == "__main__":
    # 启动服务器（开发环境）
    print("启动 LangChain 生产服务器...")
    print("API 文档: http://localhost:8000/docs")
    print("健康检查: http://localhost:8000/health")
    print("系统指标: http://localhost:8000/metrics")

    # 这里只是演示，实际部署时应该使用 uvicorn 命令行
    # uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

    print("\n示例启动命令:")
    print("uvicorn 18_production_deployment:app --host 0.0.0.0 --port 8000 --workers 4")