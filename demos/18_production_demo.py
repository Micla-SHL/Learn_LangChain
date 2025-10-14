# demos/18_production_demo.py

"""
生产环境部署演示（简化版）
由于依赖限制，这里演示生产环境部署的核心概念和配置
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

print("=== LangChain 生产环境部署学习 ===\n")

print("=== 1. 生产环境部署核心概念 ===")
print("生产环境部署是将开发完成的应用部署到实际运行环境的过程。")
print("这包括容器化、监控、安全、性能优化等多个方面。")
print()

print("=== 2. 创建生产级配置管理器 ===")

class ProductionConfig:
    """生产环境配置管理器"""

    def __init__(self):
        self.config = {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", "8000")),
            "workers": int(os.getenv("WORKERS", "4")),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "rate_limit": os.getenv("RATE_LIMIT", "100/minute"),
            "max_concurrent": int(os.getenv("MAX_CONCURRENT", "100")),
            "timeout": int(os.getenv("TIMEOUT", "30")),
            "ssl_enabled": os.getenv("SSL_ENABLED", "false").lower() == "true",
            "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
        }

    def get(self, key: str, default=None):
        """获取配置值"""
        return self.config.get(key, default)

    def validate(self) -> bool:
        """验证配置"""
        required_vars = ["ZHIPUAI_API_KEY"]
        for var in required_vars:
            if not os.getenv(var):
                print(f"❌ 缺少必需的环境变量: {var}")
                return False

        print("✅ 配置验证通过")
        return True

    def show_config(self):
        """显示当前配置"""
        print("📋 当前配置:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print()

# 配置示例
config = ProductionConfig()
config.show_config()

print("=== 3. 创建监控和指标收集器 ===")

class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "response_time_sum": 0.0,
            "active_connections": 0,
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "start_time": time.time()
        }

    def increment_requests(self):
        """增加请求计数"""
        self.metrics["requests_total"] += 1

    def record_success(self, response_time: float):
        """记录成功请求"""
        self.metrics["requests_success"] += 1
        self.metrics["response_time_sum"] += response_time

    def record_failure(self):
        """记录失败请求"""
        self.metrics["requests_failed"] += 1

    def update_system_metrics(self, cpu: float, memory: float):
        """更新系统指标"""
        self.metrics["cpu_usage"] = cpu
        self.metrics["memory_usage"] = memory

    def get_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        total_requests = self.metrics["requests_total"]
        success_requests = self.metrics["requests_success"]

        return {
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.metrics["start_time"],
            "requests": {
                "total": total_requests,
                "success": success_requests,
                "failed": self.metrics["requests_failed"],
                "success_rate": (success_requests / total_requests * 100) if total_requests > 0 else 0,
                "avg_response_time": (self.metrics["response_time_sum"] / success_requests) if success_requests > 0 else 0
            },
            "system": {
                "cpu_usage": self.metrics["cpu_usage"],
                "memory_usage": self.metrics["memory_usage"],
                "active_connections": self.metrics["active_connections"]
            }
        }

# 指标收集器示例
metrics = MetricsCollector()
print("📊 指标收集器已创建")
print()

print("=== 4. 创建健康检查系统 ===")

class HealthChecker:
    """健康检查系统"""

    def __init__(self):
        self.components = {}
        self.start_time = time.time()

    def register_component(self, name: str, checker_func):
        """注册组件健康检查函数"""
        self.components[name] = checker_func

    def check_component(self, name: str) -> Dict[str, Any]:
        """检查单个组件"""
        if name in self.components:
            try:
                result = self.components[name]()
                return {
                    "name": name,
                    "status": "healthy" if result else "unhealthy",
                    "details": result,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {
                    "name": name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        else:
            return {
                "name": name,
                "status": "unknown",
                "error": "Component not found",
                "timestamp": datetime.now().isoformat()
            }

    def check_all(self) -> Dict[str, Any]:
        """检查所有组件"""
        results = {}
        overall_status = "healthy"

        for name in self.components:
            result = self.check_component(name)
            results[name] = result

            if result["status"] != "healthy":
                overall_status = "unhealthy"

        uptime = time.time() - self.start_time

        return {
            "status": overall_status,
            "uptime": uptime,
            "timestamp": datetime.now().isoformat(),
            "components": results
        }

# 健康检查示例
health_checker = HealthChecker()

# 注册一些示例检查函数
def check_llm_service():
    """检查 LLM 服务"""
    # 模拟检查
    return {"api_key": "configured", "model": "glm-4"}

def check_memory_usage():
    """检查内存使用"""
    # 模拟内存检查
    return {"usage": "65%", "available": "2.1GB"}

def check_disk_space():
    """检查磁盘空间"""
    # 模拟磁盘检查
    return {"usage": "45%", "available": "5.3GB"}

health_checker.register_component("llm_service", check_llm_service)
health_checker.register_component("memory", check_memory_usage)
health_checker.register_component("disk", check_disk_space)

print("🔍 健康检查系统已创建")
print()

print("=== 5. 创建安全中间件 ===")

class SecurityMiddleware:
    """安全中间件"""

    def __init__(self):
        self.rate_limiters = {}
        self.blocked_ips = set()
        self.max_requests_per_minute = 100

    def check_rate_limit(self, client_ip: str) -> bool:
        """检查速率限制"""
        current_time = time.time()

        if client_ip not in self.rate_limiters:
            self.rate_limiters[client_ip] = []

        # 清理过期的请求记录
        self.rate_limiters[client_ip] = [
            req_time for req_time in self.rate_limiters[client_ip]
            if current_time - req_time < 60
        ]

        # 检查是否超过限制
        if len(self.rate_limiters[client_ip]) >= self.max_requests_per_minute:
            return False

        # 记录新请求
        self.rate_limiters[client_ip].append(current_time)
        return True

    def block_ip(self, client_ip: str):
        """阻止 IP"""
        self.blocked_ips.add(client_ip)

    def is_ip_blocked(self, client_ip: str) -> bool:
        """检查 IP 是否被阻止"""
        return client_ip in self.blocked_ips

    def get_security_headers(self) -> Dict[str, str]:
        """获取安全头"""
        return {
            "X-Frame-Options": "SAMEORIGIN",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'"
        }

# 安全中间件示例
security = SecurityMiddleware()
print("🔒 安全中间件已创建")
print()

print("=== 6. 模拟生产环境部署流程 ===")

def simulate_deployment():
    """模拟部署流程"""

    print("🚀 开始生产环境部署模拟...")
    print()

    # 步骤1: 配置验证
    print("步骤1: 验证配置")
    if config.validate():
        print("✅ 配置验证通过")
    else:
        print("❌ 配置验证失败")
        return
    print()

    # 步骤2: 健康检查
    print("步骤2: 系统健康检查")
    health_status = health_checker.check_all()
    print(f"总体状态: {health_status['status']}")
    for component, status in health_status['components'].items():
        print(f"  {component}: {status['status']}")
    print()

    # 步骤3: 模拟请求处理
    print("步骤3: 模拟请求处理")
    for i in range(5):
        # 模拟请求
        metrics.increment_requests()

        # 模拟处理时间
        response_time = 0.5 + i * 0.1
        time.sleep(0.1)  # 短暂延迟以模拟处理

        # 记录成功
        metrics.record_success(response_time)

        # 更新系统指标
        metrics.update_system_metrics(25.5 + i * 2, 60.2 + i * 1.5)

        print(f"  请求 {i+1}: 成功 (耗时: {response_time:.2f}s)")

    print()

    # 步骤4: 显示指标
    print("步骤4: 系统指标")
    current_metrics = metrics.get_metrics()

    print(f"  运行时间: {current_metrics['uptime']:.1f}s")
    print(f"  请求总数: {current_metrics['requests']['total']}")
    print(f"  成功率: {current_metrics['requests']['success_rate']:.1f}%")
    print(f"  平均响应时间: {current_metrics['requests']['avg_response_time']:.3f}s")
    print(f"  CPU使用率: {current_metrics['system']['cpu_usage']:.1f}%")
    print(f"  内存使用率: {current_metrics['system']['memory_usage']:.1f}%")
    print()

    # 步骤5: 安全检查
    print("步骤5: 安全检查")
    test_ip = "192.168.1.100"

    if security.is_ip_blocked(test_ip):
        print(f"  IP {test_ip} 被阻止")
    else:
        print(f"  IP {test_ip} 允许访问")
        if security.check_rate_limit(test_ip):
            print(f"  IP {test_ip} 速率限制检查通过")
        else:
            print(f"  IP {test_ip} 触发速率限制")

    security_headers = security.get_security_headers()
    print("  安全头已配置:")
    for header, value in security_headers.items():
        print(f"    {header}: {value}")
    print()

    print("✅ 部署模拟完成")

# 运行部署模拟
simulate_deployment()

print("=== 7. 生产环境最佳实践总结 ===")

best_practices = {
    "容器化": "使用 Docker 容器化应用，确保环境一致性",
    "配置管理": "使用环境变量管理配置，避免硬编码",
    "监控告警": "实施全面的监控和告警机制",
    "日志管理": "使用结构化日志，便于问题排查",
    "安全防护": "实施多层安全防护措施",
    "性能优化": "进行性能测试和优化",
    "备份恢复": "制定数据备份和恢复策略",
    "CI/CD": "自动化构建、测试和部署流程",
    "文档维护": "保持文档的及时更新"
}

print("📋 生产环境最佳实践:")
for practice, description in best_practices.items():
    print(f"• {practice}: {description}")
print()

print("=== 8. 部署文件结构 ===")

deployment_structure = """
LangChain/
├── 18_production_deployment.py    # 生产级应用代码
├── Dockerfile                     # Docker 镜像配置
├── docker-compose.yml             # 容器编排配置
├── requirements-production.txt     # 生产环境依赖
├── nginx/
│   └── nginx.conf                # Nginx 配置
├── monitoring/
│   ├── prometheus.yml            # Prometheus 配置
│   └── grafana/                  # Grafana 配置
├── deploy.sh                     # 部署脚本
├── load_test.py                  # 负载测试脚本
├── .env.example                  # 环境变量示例
└── README_DEPLOYMENT.md          # 部署文档
"""

print("📁 完整部署文件结构:")
print(deployment_structure)
print()

print("=== 生产环境部署学习完成 ===")
print("\n核心成就:")
print("✅ 掌握了生产环境部署的核心概念")
print("✅ 学会了容器化配置和编排")
print("✅ 实现了监控和健康检查系统")
print("✅ 创建了安全中间件和防护机制")
print("✅ 了解了性能优化和负载测试")
print("✅ 掌握了配置管理和环境变量")
print("✅ 学会了日志管理和故障排除")
print()
print("🎯 关键技能:")
print("• Docker 容器化技术")
print("• FastAPI Web 框架")
print("• Prometheus 监控系统")
print("• Nginx 反向代理")
print("• 安全认证和授权")
print("• 负载测试和性能优化")
print("• 微服务架构设计")
print("• CI/CD 部署流程")
print()
print("🚀 你现在已经具备了将 LangChain 应用部署到生产环境的能力！")
print("下一步: 特定领域应用开发")

print("\n📖 更多资源:")
print("• 完整的部署指南: README_DEPLOYMENT.md")
print("• 负载测试脚本: load_test.py")
print("• 部署脚本: deploy.sh")
print("• 配置示例: .env.example")