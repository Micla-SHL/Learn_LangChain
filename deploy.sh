#!/bin/bash

# LangChain 生产环境部署脚本

set -e

echo "=== LangChain 生产环境部署 ==="

# 检查环境变量
check_env() {
    echo "检查环境变量..."

    if [ -z "$ZHIPUAI_API_KEY" ]; then
        echo "错误: 请设置 ZHIPUAI_API_KEY 环境变量"
        exit 1
    fi

    if [ -z "$SECRET_KEY" ]; then
        echo "警告: SECRET_KEY 未设置，将使用随机生成的密钥"
        export SECRET_KEY=$(openssl rand -hex 32)
    fi

    echo "环境变量检查完成"
}

# 检查 Docker
check_docker() {
    echo "检查 Docker 环境..."

    if ! command -v docker &> /dev/null; then
        echo "错误: Docker 未安装"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        echo "错误: Docker Compose 未安装"
        exit 1
    fi

    echo "Docker 环境检查完成"
}

# 创建必要的目录
create_directories() {
    echo "创建必要的目录..."

    mkdir -p logs
    mkdir -p data
    mkdir -p nginx/ssl
    mkdir -p monitoring/grafana/provisioning

    echo "目录创建完成"
}

# 构建和启动服务
deploy_services() {
    echo "构建和启动服务..."

    # 停止现有服务
    docker-compose down

    # 构建镜像
    docker-compose build

    # 启动服务
    docker-compose up -d

    echo "服务部署完成"
}

# 等待服务启动
wait_for_services() {
    echo "等待服务启动..."

    # 等待应用启动
    echo "等待 LangChain 应用启动..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            echo "LangChain 应用启动成功"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done

    if [ $timeout -le 0 ]; then
        echo "警告: LangChain 应用启动超时"
    fi

    # 等待 Prometheus 启动
    echo "等待 Prometheus 启动..."
    timeout=30
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:9090 &> /dev/null; then
            echo "Prometheus 启动成功"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done

    if [ $timeout -le 0 ]; then
        echo "警告: Prometheus 启动超时"
    fi

    # 等待 Grafana 启动
    echo "等待 Grafana 启动..."
    timeout=30
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:3000 &> /dev/null; then
            echo "Grafana 启动成功"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done

    if [ $timeout -le 0 ]; then
        echo "警告: Grafana 启动超时"
    fi
}

# 显示服务状态
show_status() {
    echo "=== 服务状态 ==="
    docker-compose ps

    echo ""
    echo "=== 访问地址 ==="
    echo "LangChain API: http://localhost:8000"
    echo "API 文档: http://localhost:8000/docs"
    echo "健康检查: http://localhost:8000/health"
    echo "Prometheus: http://localhost:9090"
    echo "Grafana: http://localhost:3000 (admin/admin123)"
    echo "Nginx 状态: http://localhost/nginx_status"

    echo ""
    echo "=== 日志查看 ==="
    echo "查看应用日志: docker-compose logs -f langchain-app"
    echo "查看所有日志: docker-compose logs -f"
}

# 主函数
main() {
    echo "开始部署..."

    check_env
    check_docker
    create_directories
    deploy_services
    wait_for_services
    show_status

    echo ""
    echo "=== 部署完成 ==="
    echo "如需停止服务，请运行: docker-compose down"
    echo "如需查看日志，请运行: docker-compose logs -f"
}

# 执行主函数
main "$@"