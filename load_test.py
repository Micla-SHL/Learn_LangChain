#!/usr/bin/env python3
"""
负载测试脚本
测试 LangChain API 的性能和稳定性
"""

import asyncio
import aiohttp
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any
import argparse
import sys

class LoadTester:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "test-key"):
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
        self.results = []

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def send_request(self, message: str) -> Dict[str, Any]:
        """发送单个请求"""
        start_time = time.time()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "message": message,
                "session_id": f"test_session_{int(time.time())}",
                "agent_type": "general"
            }

            async with self.session.post(
                f"{self.base_url}/chat",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()

                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "status_code": response.status,
                        "response_time": end_time - start_time,
                        "response_length": len(result.get("response", "")),
                        "error": None
                    }
                else:
                    return {
                        "success": False,
                        "status_code": response.status,
                        "response_time": end_time - start_time,
                        "response_length": 0,
                        "error": f"HTTP {response.status}"
                    }

        except asyncio.TimeoutError:
            end_time = time.time()
            return {
                "success": False,
                "status_code": 0,
                "response_time": end_time - start_time,
                "response_length": 0,
                "error": "Timeout"
            }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "status_code": 0,
                "response_time": end_time - start_time,
                "response_length": 0,
                "error": str(e)
            }

    async def run_concurrent_test(self, num_requests: int, concurrency: int, messages: List[str]) -> List[Dict[str, Any]]:
        """运行并发测试"""
        print(f"开始负载测试: {num_requests} 个请求，并发数: {concurrency}")

        semaphore = asyncio.Semaphore(concurrency)

        async def limited_request(message):
            async with semaphore:
                return await self.send_request(message)

        # 创建任务列表
        tasks = []
        for i in range(num_requests):
            message = messages[i % len(messages)]
            task = limited_request(message)
            tasks.append(task)

        # 执行所有任务
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # 过滤异常结果
        valid_results = []
        for result in results:
            if isinstance(result, dict):
                valid_results.append(result)
            else:
                valid_results.append({
                    "success": False,
                    "status_code": 0,
                    "response_time": 0,
                    "response_length": 0,
                    "error": str(result)
                })

        total_time = end_time - start_time

        print(f"测试完成，总耗时: {total_time:.2f} 秒")

        return valid_results

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析测试结果"""
        if not results:
            return {"error": "没有测试结果"}

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        success_rate = len(successful) / len(results) * 100

        if successful:
            response_times = [r["response_time"] for r in successful]
            response_lengths = [r["response_length"] for r in successful]

            analysis = {
                "total_requests": len(results),
                "successful_requests": len(successful),
                "failed_requests": len(failed),
                "success_rate": success_rate,
                "response_time": {
                    "min": min(response_times),
                    "max": max(response_times),
                    "mean": statistics.mean(response_times),
                    "median": statistics.median(response_times),
                    "stdev": statistics.stdev(response_times) if len(response_times) > 1 else 0
                },
                "response_length": {
                    "min": min(response_lengths),
                    "max": max(response_lengths),
                    "mean": statistics.mean(response_lengths),
                    "median": statistics.median(response_lengths)
                }
            }
        else:
            analysis = {
                "total_requests": len(results),
                "successful_requests": 0,
                "failed_requests": len(failed),
                "success_rate": 0,
                "errors": [r["error"] for r in failed[:10]]  # 只显示前10个错误
            }

        return analysis

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with self.session.get(f"{self.base_url}/health", timeout=5) as response:
                return response.status == 200
        except:
            return False

async def main():
    parser = argparse.ArgumentParser(description="LangChain API 负载测试")
    parser.add_argument("--url", default="http://localhost:8000", help="API 基础 URL")
    parser.add_argument("--requests", type=int, default=100, help="总请求数")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数")
    parser.add_argument("--api-key", default="test-key", help="API 密钥")

    args = parser.parse_args()

    # 测试消息
    test_messages = [
        "你好，请介绍一下自己",
        "什么是人工智能？",
        "帮我计算 2+2*3",
        "现在几点了？",
        "LangChain 是什么框架？",
        "请推荐一些学习资源",
        "如何优化代码性能？",
        "解释机器学习的基本概念",
        "Python 有哪些优势？",
        "请写一个简单的函数"
    ]

    print("=== LangChain API 负载测试 ===")
    print(f"目标 URL: {args.url}")
    print(f"总请求数: {args.requests}")
    print(f"并发数: {args.concurrency}")
    print()

    async with LoadTester(args.url, args.api_key) as tester:
        # 健康检查
        print("进行健康检查...")
        is_healthy = await tester.health_check()
        if not is_healthy:
            print("❌ 服务健康检查失败，请确保服务正在运行")
            sys.exit(1)
        print("✅ 服务健康检查通过")
        print()

        # 执行负载测试
        results = await tester.run_concurrent_test(
            args.requests,
            args.concurrency,
            test_messages
        )

        # 分析结果
        analysis = tester.analyze_results(results)

        # 显示结果
        print("=== 测试结果 ===")
        print(f"总请求数: {analysis['total_requests']}")
        print(f"成功请求数: {analysis['successful_requests']}")
        print(f"失败请求数: {analysis['failed_requests']}")
        print(f"成功率: {analysis['success_rate']:.2f}%")
        print()

        if "response_time" in analysis:
            rt = analysis["response_time"]
            print("响应时间统计:")
            print(f"  最小值: {rt['min']:.3f}s")
            print(f"  最大值: {rt['max']:.3f}s")
            print(f"  平均值: {rt['mean']:.3f}s")
            print(f"  中位数: {rt['median']:.3f}s")
            print(f"  标准差: {rt['stdev']:.3f}s")
            print()

        if "response_length" in analysis:
            rl = analysis["response_length"]
            print("响应长度统计:")
            print(f"  最小值: {rl['min']} 字符")
            print(f"  最大值: {rl['max']} 字符")
            print(f"  平均值: {rl['mean']:.1f} 字符")
            print(f"  中位数: {rl['median']} 字符")
            print()

        if "errors" in analysis and analysis["errors"]:
            print("错误信息:")
            for error in analysis["errors"]:
                print(f"  - {error}")
            print()

        # 性能评估
        if analysis["success_rate"] >= 95:
            print("🟢 性能评估: 优秀")
        elif analysis["success_rate"] >= 90:
            print("🟡 性能评估: 良好")
        else:
            print("🔴 性能评估: 需要改进")

        if "response_time" in analysis:
            avg_rt = analysis["response_time"]["mean"]
            if avg_rt <= 2.0:
                print("🟢 响应时间: 优秀")
            elif avg_rt <= 5.0:
                print("🟡 响应时间: 良好")
            else:
                print("🔴 响应时间: 需要优化")

        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"load_test_result_{timestamp}.json"

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "config": {
                    "url": args.url,
                    "requests": args.requests,
                    "concurrency": args.concurrency
                },
                "analysis": analysis,
                "detailed_results": results
            }, f, indent=2, ensure_ascii=False)

        print(f"详细结果已保存到: {result_file}")

if __name__ == "__main__":
    asyncio.run(main())