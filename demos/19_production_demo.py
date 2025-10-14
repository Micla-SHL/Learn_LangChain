# demos/19_production_demo.py

"""
学习目标: 特定领域应用开发综合实践
时间: 2025/10/14
说明: 综合运用前面学到的所有技术，构建实际业务应用场景
"""

import os
import json
import time
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("=== 特定领域应用开发综合实践 ===\n")

from langchain_community.embeddings import FakeEmbeddings
print("=== 1. 综合应用架构设计 ===")
print("我们将构建一个智能业务助手平台，整合多种特定领域应用：")
print("• 智能客服系统 - 多轮对话和情感分析")
print("• 文档分析助手 - 多格式文档处理和智能问答")
print("• 代码生成工具 - 自然语言到代码转换")
print("• 数据分析平台 - 自动化数据处理和洞察发现")
print()

# 初始化模型
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")

llm = ChatZhipuAI(
    model="glm-4",
    temperature=0.1,
    api_key=api_key
)

# 初始化向量嵌入
embeddings = FakeEmbeddings(size=384)

print("=== 2. 基础架构组件 ===")

class IntentType(Enum):
    """意图类型"""
    GREETING = "greeting"
    QUESTION = "question"
    COMPLAINT = "complaint"
    REQUEST = "request"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"

class SentimentType(Enum):
    """情感类型"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

@dataclass
class CustomerSession:
    """客户会话"""
    session_id: str
    customer_id: str
    start_time: datetime = field(default_factory=datetime.now)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    intent_history: List[IntentType] = field(default_factory=list)
    sentiment_history: List[SentimentType] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

class IntelligentBusinessAssistant:
    """智能业务助手平台"""

    def __init__(self):
        self.llm = llm
        self.sessions: Dict[str, CustomerSession] = {}
        self.knowledge_base = InMemoryVectorStore(embedding=embeddings)
        self.document_store = {}
        self.code_templates = {}
        self.data_sources = {}

        # 初始化各个子系统
        self._initialize_knowledge_base()
        self._initialize_document_store()
        self._initialize_code_templates()
        self._initialize_data_sources()

    def _initialize_knowledge_base(self):
        """初始化知识库"""
        # 客服知识库
        customer_service_docs = [
            Document(
                page_content="""
产品退货政策：
1. 购买后7天内可无理由退货
2. 产品质量问题，30天内可退换
3. 退货需提供购买凭证
4. 退货运费由买家承担
5. 退款将在收到退货后3-5个工作日内处理

常见问题：
Q: 如何申请退货？
A: 登录账户 -> 订单管理 -> 申请退款 -> 填写退货信息

Q: 退款需要多长时间？
A: 一般3-5个工作日

Q: 退货运费谁承担？
A: 无理由退货由买家承担，质量问题由卖家承担
                """,
                metadata={"category": "退货政策", "source": "客服手册"}
            ),
            Document(
                page_content="""
产品技术支持：
• 产品安装指导
• 使用问题解答
• 故障排除帮助
• 软件更新支持
• 远程协助服务

支持渠道：
1. 在线客服：工作日 9:00-18:00
2. 电话支持：400-123-4567
3. 邮件支持：support@example.com
4. 知识库自助服务：24小时

常见技术问题：
- 无法连接网络：检查网络设置和防火墙
- 软件无法启动：检查系统兼容性
- 功能异常：尝试重启软件或更新版本
                """,
                metadata={"category": "技术支持", "source": "支持手册"}
            )
        ]

        # 添加到知识库
        for doc in customer_service_docs:
            self.knowledge_base.add_documents([doc])

    def _initialize_document_store(self):
        """初始化文档存储"""
        # 模拟文档数据
        self.document_store = {
            "product_manual": {
                "title": "产品使用手册",
                "content": """
# 产品使用手册

## 1. 产品介绍
本产品是一款智能助手设备，具有语音识别、图像处理、数据分析等功能。

## 2. 安装指南
### 2.1 硬件安装
1. 将设备放置在稳定平面上
2. 连接电源适配器
3. 按下电源按钮启动设备

### 2.2 软件安装
1. 下载官方APP
2. 扫描设备二维码进行配对
3. 完成设备初始化设置

## 3. 功能说明
### 3.1 语音助手
- 支持中英文语音识别
- 可执行语音命令控制
- 支持多轮对话

### 3.2 图像识别
- 支持物体识别
- 支持文字识别(OCR)
- 支持人脸识别

## 4. 故障排除
### 4.1 常见问题
Q: 设备无法开机
A: 检查电源连接，长按电源按钮10秒

Q: 语音识别不准确
A: 确保环境安静，说话清晰

Q: APP连接失败
A: 检查网络连接，重启APP和设备
                """,
                "type": "manual",
                "created_at": datetime.now()
            },
            "financial_report": {
                "title": "2024年财务报告",
                "content": """
# 2024年度财务报告

## 1. 总体概况
2024年公司实现营业收入15.2亿元，同比增长32%
净利润2.8亿元，同比增长28%

## 2. 收入分析
### 2.1 按产品分类
- 智能设备：8.5亿元 (56%)
- 软件服务：4.2亿元 (28%)
- 技术支持：2.5亿元 (16%)

### 2.2 按地区分类
- 华东地区：6.8亿元 (45%)
- 华南地区：4.1亿元 (27%)
- 华北地区：3.2亿元 (21%)
- 其他地区：1.1亿元 (7%)

## 3. 成本分析
总成本12.4亿元，其中：
- 研发成本：3.2亿元 (26%)
- 生产成本：6.8亿元 (55%)
- 营销成本：2.4亿元 (19%)

## 4. 盈利能力
毛利率：18.4%
净利率：18.4%
ROE：15.2%

## 5. 风险因素
- 市场竞争加剧
- 技术更新迭代快
- 原材料成本上升
                """,
                "type": "report",
                "created_at": datetime.now()
            }
        }

    def _initialize_code_templates(self):
        """初始化代码模板"""
        self.code_templates = {
            "python_api": {
                "name": "Python FastAPI API 模板",
                "description": "创建RESTful API服务",
                "template": '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="{app_name}", version="1.0.0")

class {model_name}(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None

# 数据存储
{model_name_lower}_items = []

@app.get("/{model_name_lower}/", response_model=List[{model_name}])
async def get_items():
    return {model_name_lower}_items

@app.post("/{model_name_lower}/", response_model={model_name})
async def create_item(item: {model_name}):
    item.id = len({model_name_lower}_items) + 1
    {model_name_lower}_items.append(item)
    return item

@app.get("/{model_name_lower}/{{item_id}}", response_model={model_name})
async def get_item(item_id: int):
    if item_id > len({model_name_lower}_items) or item_id < 1:
        raise HTTPException(status_code=404, detail="Item not found")
    return {model_name_lower}_items[item_id-1]
''',
                "variables": ["app_name", "model_name", "model_name_lower"]
            },
            "react_component": {
                "name": "React 组件模板",
                "description": "创建React函数组件",
                "template": '''
import React, {{ useState, useEffect }} from 'react';
import './{component_name}.css';

interface {component_name}Props {{
  // 定义props类型
}}

const {component_name}: React.FC<{component_name}Props> = (props) => {{
  const [data, setData] = useState(null);

  useEffect(() => {{
    // 组件挂载时的逻辑
  }}, []);

  return (
    <div className="{component_name}">
      <h1>{component_title}</h1>
      {/* 组件内容 */}
    </div>
  );
}};

export default {component_name};
''',
                "variables": ["component_name", "component_title"]
            },
            "data_analysis": {
                "name": "数据分析脚本模板",
                "description": "创建数据分析脚本",
                "template": '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_{dataset_name}(data_path: str):
    """分析{dataset_description}"""

    # 读取数据
    df = pd.read_csv(data_path)

    # 数据基本信息
    print("数据基本信息:")
    print(df.info())
    print("\\n数据描述性统计:")
    print(df.describe())

    # 数据可视化
    plt.figure(figsize=(12, 6))

    # 添加你的分析代码

    plt.title("{chart_title}")
    plt.xlabel("{x_label}")
    plt.ylabel("{y_label}")
    plt.show()

    return df

# 使用示例
if __name__ == "__main__":
    df = analyze_{dataset_name}("your_data.csv")
''',
                "variables": ["dataset_name", "dataset_description", "chart_title", "x_label", "y_label"]
            }
        }

    def _initialize_data_sources(self):
        """初始化数据源"""
        self.data_sources = {
            "sales_data": {
                "name": "销售数据",
                "description": "产品销售数据集",
                "columns": ["date", "product_id", "product_name", "category", "price", "quantity", "customer_id", "region"],
                "sample_data": [
                    {"date": "2024-01-01", "product_id": "P001", "product_name": "智能音箱", "category": "电子设备", "price": 299, "quantity": 15, "customer_id": "C001", "region": "华东"},
                    {"date": "2024-01-02", "product_id": "P002", "product_name": "智能手表", "category": "电子设备", "price": 599, "quantity": 8, "customer_id": "C002", "region": "华南"},
                    {"date": "2024-01-03", "product_id": "P003", "product_name": "智能灯泡", "category": "家居用品", "price": 89, "quantity": 25, "customer_id": "C003", "region": "华北"}
                ]
            },
            "customer_feedback": {
                "name": "客户反馈数据",
                "description": "客户满意度调查数据",
                "columns": ["feedback_id", "customer_id", "rating", "category", "comment", "date", "resolved"],
                "sample_data": [
                    {"feedback_id": "F001", "customer_id": "C001", "rating": 5, "category": "产品质量", "comment": "产品质量很好", "date": "2024-01-01", "resolved": True},
                    {"feedback_id": "F002", "customer_id": "C002", "rating": 3, "category": "物流服务", "comment": "配送有点慢", "date": "2024-01-02", "resolved": True},
                    {"feedback_id": "F003", "customer_id": "C003", "rating": 4, "category": "客户服务", "comment": "客服态度不错", "date": "2024-01-03", "resolved": False}
                ]
            }
        }

    def create_session(self, customer_id: str) -> str:
        """创建客户会话"""
        session_id = str(uuid.uuid4())
        session = CustomerSession(
            session_id=session_id,
            customer_id=customer_id
        )
        self.sessions[session_id] = session
        return session_id

    def detect_intent(self, message: str) -> IntentType:
        """检测用户意图"""
        message_lower = message.lower()

        # 关键词匹配
        if any(word in message_lower for word in ["你好", "hi", "hello", "您好"]):
            return IntentType.GREETING
        elif any(word in message_lower for word in ["再见", "拜拜", "bye", "谢谢"]):
            return IntentType.GOODBYE
        elif any(word in message_lower for word in ["问题", "怎么", "如何", "什么"]):
            return IntentType.QUESTION
        elif any(word in message_lower for word in ["投诉", "不满", "差", "问题", "故障"]):
            return IntentType.COMPLAINT
        elif any(word in message_lower for word in ["需要", "想要", "请", "帮我"]):
            return IntentType.REQUEST
        else:
            return IntentType.UNKNOWN

    def analyze_sentiment(self, message: str) -> SentimentType:
        """分析情感倾向"""
        positive_words = ["好", "棒", "优秀", "满意", "喜欢", "感谢", "不错"]
        negative_words = ["差", "不好", "问题", "故障", "投诉", "不满", "失望"]

        message_lower = message.lower()

        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)

        if positive_count > negative_count:
            return SentimentType.POSITIVE
        elif negative_count > positive_count:
            return SentimentType.NEGATIVE
        else:
            return SentimentType.NEUTRAL

print("=== 3. 智能客服系统 ===")

class CustomerServiceSystem:
    """智能客服系统"""

    def __init__(self, assistant: IntelligentBusinessAssistant):
        self.assistant = assistant
        self.llm = assistant.llm

    async def handle_customer_inquiry(self, session_id: str, message: str) -> Dict[str, Any]:
        """处理客户咨询"""
        session = self.assistant.sessions.get(session_id)
        if not session:
            return {"error": "会话不存在"}

        # 记录消息
        session.messages.append({
            "type": "user",
            "content": message,
            "timestamp": datetime.now()
        })

        # 检测意图和情感
        intent = self.assistant.detect_intent(message)
        sentiment = self.assistant.analyze_sentiment(message)

        session.intent_history.append(intent)
        session.sentiment_history.append(sentiment)

        # 根据意图生成回复
        if intent == IntentType.GREETING:
            response = self._handle_greeting(session)
        elif intent == IntentType.QUESTION:
            response = await self._handle_question(session, message)
        elif intent == IntentType.COMPLAINT:
            response = await self._handle_complaint(session, message)
        elif intent == IntentType.REQUEST:
            response = await self._handle_request(session, message)
        elif intent == IntentType.GOODBYE:
            response = self._handle_goodbye(session)
        else:
            response = await self._handle_unknown(session, message)

        # 记录回复
        session.messages.append({
            "type": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })

        return {
            "response": response,
            "intent": intent.value,
            "sentiment": sentiment.value,
            "session_id": session_id
        }

    def _handle_greeting(self, session: CustomerSession) -> str:
        """处理问候"""
        if len(session.messages) <= 2:
            return "您好！我是您的智能助手，很高兴为您服务。请问有什么可以帮助您的吗？"
        else:
            return "您好！有什么我可以帮助您的吗？"

    async def _handle_question(self, session: CustomerSession, question: str) -> str:
        """处理问题"""
        # 这里可以集成知识库搜索
        try:
            prompt = ChatPromptTemplate.from_template("""
作为专业的客服助手，请根据以下客户问题提供准确、友好的回答。

客户问题：{question}

请确保回答：
1. 准确且有用
2. 语言友好自然
3. 提供具体可行的建议
4. 如果需要，提供相关联系方式

回答：
""")

            chain = prompt | self.llm
            response = await chain.ainvoke({"question": question})

            return response.content
        except Exception as e:
            return "抱歉，我暂时无法回答这个问题。您可以尝试重新表述或联系人工客服。"

    async def _handle_complaint(self, session: CustomerSession, complaint: str) -> str:
        """处理投诉"""
        try:
            prompt = ChatPromptTemplate.from_template("""
作为专业的客服助手，需要处理客户的投诉。请以同理心和专业的态度回应。

客户投诉：{complaint}

回应要求：
1. 表达理解和歉意
2. 承诺尽快解决问题
3. 提供解决方案或后续步骤
4. 保持耐心和专业

回应：
""")

            chain = prompt | self.llm
            response = await chain.ainvoke({"complaint": complaint})

            return response.content
        except Exception as e:
            return "非常抱歉给您带来不便。我理解您的情况，请给我一些时间来处理这个问题，我会尽快给您满意的答复。"

    async def _handle_request(self, session: CustomerSession, request: str) -> str:
        """处理请求"""
        try:
            prompt = ChatPromptTemplate.from_template("""
作为专业的客服助手，请根据客户的请求提供帮助。

客户请求：{request}

回应要求：
1. 理解客户的具体需求
2. 提供清晰的帮助信息
3. 如果需要，提供详细的操作步骤
4. 保持专业和友好的语气

回应：
""")

            chain = prompt | self.llm
            response = await chain.ainvoke({"request": request})

            return response.content
        except Exception as e:
            return "我来帮助您处理这个请求。请您提供更多详细信息，这样我就能为您提供更准确的帮助。"

    def _handle_goodbye(self, session: CustomerSession) -> str:
        """处理告别"""
        session.resolved = True
        return "感谢您的咨询，祝您生活愉快！如果还有其他问题，随时欢迎联系我们。"

    async def _handle_unknown(self, session: CustomerSession, message: str) -> str:
        """处理未知意图"""
        return "抱歉，我没有完全理解您的问题。您能否用更简单的话重新表达一下，或者告诉我您具体需要什么帮助？"

print("=== 4. 文档分析助手 ===")

class DocumentAnalysisAssistant:
    """文档分析助手"""

    def __init__(self, assistant: IntelligentBusinessAssistant):
        self.assistant = assistant
        self.llm = assistant.llm

    async def analyze_document(self, doc_id: str, query: str) -> Dict[str, Any]:
        """分析文档"""
        document = self.assistant.document_store.get(doc_id)
        if not document:
            return {"error": "文档不存在"}

        try:
            # 根据查询类型选择分析方法
            if "摘要" in query or "总结" in query:
                result = await self._generate_summary(document, query)
            elif "关键" in query or "要点" in query:
                result = await self._extract_key_points(document, query)
            elif "问题" in query or "解答" in query:
                result = await self._answer_document_question(document, query)
            else:
                result = await self._general_document_analysis(document, query)

            return {
                "document_id": doc_id,
                "document_title": document["title"],
                "query": query,
                "analysis_result": result,
                "document_type": document["type"]
            }
        except Exception as e:
            return {"error": f"文档分析失败: {str(e)}"}

    async def _generate_summary(self, document: Dict[str, Any], query: str) -> str:
        """生成文档摘要"""
        try:
            prompt = ChatPromptTemplate.from_template("""
请为以下文档生成摘要，重点关注用户关心的方面：

文档标题：{title}
文档内容：{content}

用户查询：{query}

摘要要求：
1. 简明扼要，突出重点
2. 回应用户的具体关切
3. 保持逻辑清晰

摘要：
""")

            chain = prompt | self.llm
            response = await chain.ainvoke({
                "title": document["title"],
                "content": document["content"][:2000],  # 限制长度
                "query": query
            })

            return response.content
        except Exception as e:
            return f"摘要生成失败: {str(e)}"

    async def _extract_key_points(self, document: Dict[str, Any], query: str) -> str:
        """提取关键信息"""
        try:
            prompt = ChatPromptTemplate.from_template("""
请从以下文档中提取关键信息和要点：

文档标题：{title}
文档内容：{content}

用户查询：{query}

请提取相关要点，要求：
1. 使用项目符号列出
2. 每个要点简洁明了
3. 按重要性排序

关键要点：
""")

            chain = prompt | self.llm
            response = await chain.ainvoke({
                "title": document["title"],
                "content": document["content"][:2000],
                "query": query
            })

            return response.content
        except Exception as e:
            return f"关键信息提取失败: {str(e)}"

    async def _answer_document_question(self, document: Dict[str, Any], query: str) -> str:
        """回答文档相关问题"""
        try:
            prompt = ChatPromptTemplate.from_template("""
基于以下文档内容回答用户的问题：

文档标题：{title}
文档内容：{content}

用户问题：{query}

回答要求：
1. 基于文档内容回答
2. 如果文档中没有相关信息，明确说明
3. 提供具体的页面或章节参考

回答：
""")

            chain = prompt | self.llm
            response = await chain.ainvoke({
                "title": document["title"],
                "content": document["content"],
                "query": query
            })

            return response.content
        except Exception as e:
            return f"问题回答失败: {str(e)}"

    async def _general_document_analysis(self, document: Dict[str, Any], query: str) -> str:
        """一般性文档分析"""
        try:
            prompt = ChatPromptTemplate.from_template("""
请分析以下文档，并回答用户的查询：

文档标题：{title}
文档内容：{content}

用户查询：{query}

分析要求：
1. 理解文档内容和结构
2. 提供相关的分析见解
3. 回应用户的具体需求

分析结果：
""")

            chain = prompt | self.llm
            response = await chain.ainvoke({
                "title": document["title"],
                "content": document["content"][:2000],
                "query": query
            })

            return response.content
        except Exception as e:
            return f"文档分析失败: {str(e)}"

print("=== 5. 代码生成工具 ===")

class CodeGenerationAssistant:
    """代码生成工具"""

    def __init__(self, assistant: IntelligentBusinessAssistant):
        self.assistant = assistant
        self.llm = assistant.llm

    async def generate_code(self, description: str, language: str = "python") -> Dict[str, Any]:
        """生成代码"""
        try:
            if language.lower() == "python":
                result = await self._generate_python_code(description)
            elif language.lower() == "javascript":
                result = await self._generate_javascript_code(description)
            elif language.lower() == "react":
                result = await self._generate_react_code(description)
            else:
                result = await self._generate_generic_code(description, language)

            return {
                "description": description,
                "language": language,
                "code": result,
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"代码生成失败: {str(e)}"}

    async def _generate_python_code(self, description: str) -> str:
        """生成Python代码"""
        try:
            prompt = ChatPromptTemplate.from_template("""
请根据以下描述生成Python代码，要求代码完整、可运行：

需求描述：{description}

代码要求：
1. 使用Python 3.x语法
2. 包含必要的导入语句
3. 添加适当的注释
4. 包含错误处理
5. 代码结构清晰，便于理解

生成的Python代码：
""")

            chain = prompt | self.llm
            response = await chain.ainvoke({"description": description})

            return response.content
        except Exception as e:
            return f"Python代码生成失败: {str(e)}"

    async def _generate_javascript_code(self, description: str) -> str:
        """生成JavaScript代码"""
        try:
            prompt = ChatPromptTemplate.from_template("""
请根据以下描述生成JavaScript代码，要求代码完整、可运行：

需求描述：{description}

代码要求：
1. 使用现代JavaScript语法(ES6+)
2. 包含必要的注释
3. 包含错误处理
4. 代码结构清晰
5. 兼容性考虑

生成的JavaScript代码：
""")

            chain = prompt | self.llm
            response = await chain.ainvoke({"description": description})

            return response.content
        except Exception as e:
            return f"JavaScript代码生成失败: {str(e)}"

    async def _generate_react_code(self, description: str) -> str:
        """生成React组件代码"""
        try:
            prompt = ChatPromptTemplate.from_template("""
请根据以下描述生成React组件代码：

需求描述：{description}

代码要求：
1. 使用函数式组件
2. 使用React Hooks
3. 包含PropTypes或TypeScript类型定义
4. 组件结构清晰
5. 包含基本的样式

生成的React组件代码：
""")

            chain = prompt | self.llm
            response = await chain.ainvoke({"description": description})

            return response.content
        except Exception as e:
            return f"React代码生成失败: {str(e)}"

    async def _generate_generic_code(self, description: str, language: str) -> str:
        """生成通用代码"""
        try:
            prompt = ChatPromptTemplate.from_template("""
请根据以下描述生成{language}代码：

需求描述：{description}

代码要求：
1. 代码完整且可运行
2. 包含必要的注释
3. 遵循该语言的编码规范
4. 包含基本的错误处理
5. 代码结构清晰

生成的{language}代码：
""")

            chain = prompt | self.llm
            response = await chain.ainvoke({
                "description": description,
                "language": language
            })

            return response.content
        except Exception as e:
            return f"{language}代码生成失败: {str(e)}"

print("=== 6. 数据分析平台 ===")

class DataAnalysisAssistant:
    """数据分析助手"""

    def __init__(self, assistant: IntelligentBusinessAssistant):
        self.assistant = assistant
        self.llm = assistant.llm

    async def analyze_data(self, data_source: str, analysis_request: str) -> Dict[str, Any]:
        """分析数据"""
        data_info = self.assistant.data_sources.get(data_source)
        if not data_info:
            return {"error": "数据源不存在"}

        try:
            # 生成分析代码
            analysis_code = await self._generate_analysis_code(data_info, analysis_request)

            # 生成分析报告
            report = await self._generate_analysis_report(data_info, analysis_request)

            return {
                "data_source": data_source,
                "data_info": data_info,
                "analysis_request": analysis_request,
                "analysis_code": analysis_code,
                "analysis_report": report,
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"数据分析失败: {str(e)}"}

    async def _generate_analysis_code(self, data_info: Dict[str, Any], request: str) -> str:
        """生成分析代码"""
        try:
            columns = ", ".join(data_info["columns"])

            prompt = ChatPromptTemplate.from_template("""
请为以下数据生成Python数据分析代码：

数据源：{data_name}
数据描述：{data_description}
数据列：{columns}
分析需求：{request}

代码要求：
1. 使用pandas进行数据处理
2. 使用matplotlib/seaborn进行可视化
3. 包含数据清洗步骤
4. 提供清晰的分析流程
5. 添加适当的注释

生成的Python代码：
""")

            chain = prompt | self.llm
            response = await chain.ainvoke({
                "data_name": data_info["name"],
                "data_description": data_info["description"],
                "columns": columns,
                "request": request
            })

            return response.content
        except Exception as e:
            return f"分析代码生成失败: {str(e)}"

    async def _generate_analysis_report(self, data_info: Dict[str, Any], request: str) -> str:
        """生成分析报告"""
        try:
            prompt = ChatPromptTemplate.from_template("""
请为以下数据分析需求生成分析报告：

数据源：{data_name}
数据描述：{data_description}
分析需求：{request}

报告要求：
1. 简要描述分析目标
2. 列出主要发现和洞察
3. 提供数据可视化建议
4. 总结分析结论
5. 提出后续分析建议

分析报告：
""")

            chain = prompt | self.llm
            response = await chain.ainvoke({
                "data_name": data_info["name"],
                "data_description": data_info["description"],
                "request": request
            })

            return response.content
        except Exception as e:
            return f"分析报告生成失败: {str(e)}"

print("=== 7. 综合应用演示 ===")

async def demonstrate_applications():
    """演示所有应用场景"""

    # 创建智能业务助手
    assistant = IntelligentBusinessAssistant()

    # 创建各个子系统
    customer_service = CustomerServiceSystem(assistant)
    document_assistant = DocumentAnalysisAssistant(assistant)
    code_generator = CodeGenerationAssistant(assistant)
    data_analyzer = DataAnalysisAssistant(assistant)

    print("🚀 开始综合应用演示")
    print("="*50)

    # 演示1: 智能客服系统
    print("\n📞 演示1: 智能客服系统")
    print("-"*30)

    session_id = assistant.create_session("customer_001")

    conversations = [
        "你好，我想咨询一下产品退货政策",
        "请问退货需要多长时间才能收到退款？",
        "如果产品有质量问题，退货运费谁承担？",
        "谢谢您的解答"
    ]

    for message in conversations:
        print(f"客户: {message}")
        result = await customer_service.handle_customer_inquiry(session_id, message)
        print(f"客服: {result['response']}")
        print(f"意图: {result['intent']}, 情感: {result['sentiment']}")
        print()

    # 演示2: 文档分析助手
    print("📄 演示2: 文档分析助手")
    print("-"*30)

    doc_analysis_queries = [
        ("product_manual", "请给我总结一下产品安装的步骤"),
        ("financial_report", "分析一下2024年的收入情况"),
        ("product_manual", "设备无法开机应该怎么解决？")
    ]

    for doc_id, query in doc_analysis_queries:
        print(f"查询文档: {doc_id}")
        print(f"问题: {query}")
        result = await document_assistant.analyze_document(doc_id, query)
        if "error" not in result:
            print(f"分析结果: {result['analysis_result'][:200]}...")
        else:
            print(f"错误: {result['error']}")
        print()

    # 演示3: 代码生成工具
    print("💻 演示3: 代码生成工具")
    print("-"*30)

    code_requests = [
        ("创建一个简单的计算器类，支持加减乘除运算", "python"),
        ("创建一个待办事项列表的React组件", "react"),
        ("实现一个简单的HTTP GET请求", "javascript")
    ]

    for description, language in code_requests:
        print(f"需求: {description}")
        print(f"语言: {language}")
        result = await code_generator.generate_code(description, language)
        if "error" not in result:
            print(f"生成的代码:")
            print(result['code'][:300] + "...")
        else:
            print(f"错误: {result['error']}")
        print()

    # 演示4: 数据分析平台
    print("📊 演示4: 数据分析平台")
    print("-"*30)

    analysis_requests = [
        ("sales_data", "分析各产品类别的销售情况，并生成销售趋势图"),
        ("customer_feedback", "分析客户满意度，找出需要改进的方面")
    ]

    for data_source, request in analysis_requests:
        print(f"数据源: {data_source}")
        print(f"分析需求: {request}")
        result = await data_analyzer.analyze_data(data_source, request)
        if "error" not in result:
            print(f"分析报告:")
            print(result['analysis_report'][:300] + "...")
        else:
            print(f"错误: {result['error']}")
        print()

    print("🎉 综合应用演示完成！")

print("=== 8. 应用特性和最佳实践 ===")

application_features = {
    "智能客服系统": {
        "功能": ["多轮对话", "意图识别", "情感分析", "知识库集成"],
        "技术": ["LLM对话管理", "NLP技术", "知识检索"],
        "应用": ["7x24小时服务", "成本节约", "一致性服务"]
    },
    "文档分析助手": {
        "功能": ["多格式支持", "智能摘要", "关键信息提取", "智能问答"],
        "技术": ["文档解析", "文本理解", "信息检索"],
        "应用": ["文档管理", "知识检索", "报告生成"]
    },
    "代码生成工具": {
        "功能": ["多语言支持", "模板化生成", "代码优化", "质量检查"],
        "技术": ["代码生成", "模板引擎", "代码分析"],
        "应用": ["开发效率", "标准化", "代码质量"]
    },
    "数据分析平台": {
        "功能": ["自动分析", "可视化", "报告生成", "洞察发现"],
        "技术": ["数据处理", "统计分析", "机器学习"],
        "应用": ["业务决策", "趋势分析", "预测建模"]
    }
}

print("🌟 应用特性总览:")
for app_name, features in application_features.items():
    print(f"\n{app_name}:")
    print(f"  功能: {', '.join(features['功能'])}")
    print(f"  技术: {', '.join(features['技术'])}")
    print(f"  应用: {', '.join(features['应用'])}")

print("\n=== 9. 项目架构和部署建议 ===")

architecture_tips = {
    "模块化设计": "将不同功能模块化，便于维护和扩展",
    "API网关": "统一的入口点，负责路由和负载均衡",
    "服务拆分": "根据业务领域拆分微服务",
    "数据库设计": "合理的数据模型设计，考虑扩展性",
    "缓存策略": "使用Redis等缓存提升性能",
    "监控告警": "全面的监控和告警机制",
    "安全防护": "多层次的安全防护措施",
    "CI/CD流程": "自动化的构建、测试和部署"
}

print("🏗️ 架构设计建议:")
for tip, description in architecture_tips.items():
    print(f"• {tip}: {description}")

print("\n=== 特定领域应用开发完成 ===")
print("\n🎯 综合成就:")
print("✅ 掌握了多个业务场景的AI应用开发")
print("✅ 学会了模块化和组件化设计")
print("✅ 实现了智能对话和文档处理")
print("✅ 创建了代码生成和数据分析工具")
print("✅ 整合了所有LangChain技术栈")
print("✅ 具备了完整的业务解决方案设计能力")

print("\n🏆 学习成果总览:")
print("🎓 基础能力 - LangChain核心概念和技术")
print("🔧 高级技能 - Agent系统和RAG集成")
print("🏗️ 系统设计 - Multi-Agent和生产部署")
print("💼 业务应用 - 四大特定领域解决方案")
print("🚀 工程实践 - 完整的项目开发和部署")

print("\n🎊 恭喜！你已经完成了LangChain的完整学习之旅！")
print("现在你具备了从理论到实践、从开发到部署的全方位能力！")

# 运行演示
if __name__ == "__main__":
    asyncio.run(demonstrate_applications())