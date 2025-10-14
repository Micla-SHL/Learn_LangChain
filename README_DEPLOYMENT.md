# LangChain 生产环境部署指南

本文档详细说明了如何将 LangChain 应用部署到生产环境。

## 📋 目录结构

```
LangChain/
├── 18_production_deployment.py    # 生产级应用代码
├── Dockerfile                     # Docker 镜像配置
├── docker-compose.yml             # 容器编排配置
├── requirements-production.txt     # 生产环境依赖
├── nginx/
│   └── nginx.conf                # Nginx 配置
├── monitoring/
│   ├── prometheus.yml            # Prometheus 配置
│   └── grafana/
│       └── provisioning/         # Grafana 配置
├── deploy.sh                     # 部署脚本
├── load_test.py                  # 负载测试脚本
├── .env.example                  # 环境变量示例
└── README_DEPLOYMENT.md          # 本文档
```

## 🚀 快速开始

### 1. 环境准备

#### 系统要求
- Docker 20.10+
- Docker Compose 2.0+
- 至少 2GB RAM
- 至少 2GB 可用磁盘空间

#### 安装依赖
```bash
# 安装 Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量
nano .env
```

必须配置的环境变量：
- `ZHIPUAI_API_KEY`: 智谱AI API密钥
- `SECRET_KEY`: 应用密钥（用于JWT签名）

### 3. 部署应用

```bash
# 使用部署脚本（推荐）
./deploy.sh

# 或手动部署
docker-compose up -d
```

### 4. 验证部署

访问以下地址验证部署：

- **API 服务**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)

## 🔧 详细配置

### Docker 配置

#### Dockerfile 特性
- 使用 Python 3.11 slim 基础镜像
- 多阶段构建优化镜像大小
- 非 root 用户运行
- 健康检查配置
- 优化的依赖安装

#### docker-compose.yml 服务
- **langchain-app**: 主应用服务
- **redis**: 缓存服务
- **prometheus**: 监控服务
- **grafana**: 可视化服务
- **nginx**: 反向代理服务

### Nginx 配置

#### 主要功能
- 反向代理和负载均衡
- Gzip 压缩
- 速率限制
- 安全头设置
- 静态文件服务

#### 配置文件位置
```
nginx/
├── nginx.conf          # 主配置文件
└── ssl/               # SSL 证书目录
```

### 监控配置

#### Prometheus 指标
- 应用性能指标
- 系统资源指标
- 自定义业务指标

#### Grafana 仪表板
- 系统概览
- 性能监控
- 错误统计

## 🔒 安全配置

### 1. API 安全
- JWT 认证
- 速率限制
- CORS 配置
- 请求大小限制

### 2. 网络安全
- 防火墙配置
- SSL/TLS 加密
- 安全头设置

### 3. 应用安全
- 非 root 用户运行
- 密钥管理
- 输入验证

## 📊 监控和日志

### 1. 应用监控
```bash
# 查看应用日志
docker-compose logs -f langchain-app

# 查看系统指标
curl http://localhost:8000/metrics
```

### 2. 系统监控
- CPU、内存、磁盘使用率
- 网络流量
- 容器状态

### 3. 日志管理
- 结构化日志
- 日志轮转
- 集中收集

## ⚡ 性能优化

### 1. 应用优化
- 异步处理
- 连接池
- 缓存策略

### 2. 基础设施优化
- 资源限制
- 自动扩缩容
- 负载均衡

### 3. 数据库优化
- 索引优化
- 查询优化
- 连接池配置

## 🧪 测试

### 1. 负载测试
```bash
# 运行负载测试
python load_test.py --requests 100 --concurrency 10

# 高并发测试
python load_test.py --requests 1000 --concurrency 50
```

### 2. 压力测试
```bash
# 持续压力测试
python load_test.py --requests 5000 --concurrency 100
```

### 3. 故障测试
- 模拟服务故障
- 网络分区测试
- 资源耗尽测试

## 🚨 故障排除

### 常见问题

#### 1. 容器启动失败
```bash
# 查看容器日志
docker-compose logs langchain-app

# 检查容器状态
docker-compose ps
```

#### 2. API 调用失败
```bash
# 检查健康状态
curl http://localhost:8000/health

# 检查网络连接
docker network ls
```

#### 3. 性能问题
```bash
# 检查资源使用
docker stats

# 查看系统负载
htop
```

### 紧急恢复
```bash
# 重启服务
docker-compose restart

# 完全重建
docker-compose down
docker-compose up -d
```

## 📈 扩展指南

### 1. 水平扩展
```yaml
# docker-compose.yml
services:
  langchain-app:
    deploy:
      replicas: 3
```

### 2. 微服务拆分
- 拆分不同的 Agent 类型
- 独立部署服务
- 服务间通信

### 3. 云原生部署
- Kubernetes 配置
- Helm Charts
- CI/CD 流水线

## 🔄 备份和恢复

### 1. 数据备份
```bash
# 备份应用数据
docker run --rm -v langchain_data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup-$(date +%Y%m%d).tar.gz -C /data .

# 备份配置文件
tar czf config-backup-$(date +%Y%m%d).tar.gz .env nginx/ monitoring/
```

### 2. 恢复数据
```bash
# 恢复应用数据
docker run --rm -v langchain_data:/data -v $(pwd):/backup alpine tar xzf /backup/data-backup-20241014.tar.gz -C /data
```

## 📞 支持和维护

### 1. 监控告警
- 设置关键指标告警
- 邮件/短信通知
- 自动故障恢复

### 2. 定期维护
- 依赖更新
- 安全补丁
- 性能调优

### 3. 版本升级
- 蓝绿部署
- 金丝雀发布
- 回滚策略

---

## 🎯 部署检查清单

- [ ] 环境变量配置正确
- [ ] Docker 和 Docker Compose 安装
- [ ] 网络端口可用
- [ ] SSL 证书配置（生产环境）
- [ ] 监控系统配置
- [ ] 日志系统配置
- [ ] 备份策略制定
- [ ] 安全配置验证
- [ ] 性能基准测试
- [ ] 故障恢复测试

部署完成后，请确保所有检查项都已完成。

## 📞 技术支持

如果遇到问题，请：

1. 查看本文档的故障排除部分
2. 检查容器日志
3. 运行健康检查
4. 联系技术支持团队