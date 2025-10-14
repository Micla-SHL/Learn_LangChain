# demos/16_rag_agent_integration.py

"""
学习目标: RAG + Agent 集成系统
时间: 2025/10/14
说明: 学习如何将检索增强生成(RAG)与Agent系统结合，创建基于知识库的智能对话系统
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import requests
from bs4 import BeautifulSoup

# 初始化模型
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")

chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.1,
    api_key=api_key
)

embeddings = ZhipuAIEmbeddings(model="embedding-3")

print("=== RAG + Agent 集成系统学习 ===\n")

print("=== 1. 理解 RAG + Agent 集成的核心思想 ===")
print("RAG + Agent 集成结合了两种强大的技术：")
print("• RAG (检索增强生成): 基于知识库的准确信息检索")
print("• Agent 系统: 智能决策和工具调用能力")
print()
print("集成优势：")
print("✅ 知识准确性: 基于检索到的文档回答，减少幻觉")
print("✅ 动态推理: Agent 可以分析查询并选择最佳检索策略")
print("✅ 多源信息: 可以整合多个知识源的信息")
print("✅ 交互式问答: 支持追问和深度对话")
print()

print("=== 2. 创建知识库系统 ===")

class KnowledgeBase:
    """知识库管理类"""

    def __init__(self):
        self.vector_store = InMemoryVectorStore(embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.documents = []
        self.metadata = {}

    def add_documents(self, documents: List[Document]) -> None:
        """添加文档到知识库"""
        self.documents.extend(documents)
        # 分割文档
        splits = self.text_splitter.split_documents(documents)
        # 添加到向量存储
        self.vector_store.add_documents(splits)
        print(f"已添加 {len(documents)} 个文档，分割为 {len(splits)} 个块")

    def add_text(self, text: str, source: str = "manual") -> None:
        """添加文本到知识库"""
        doc = Document(page_content=text, metadata={"source": source})
        self.add_documents([doc])

    def search(self, query: str, k: int = 4) -> List[Document]:
        """搜索相关文档"""
        return self.vector_store.similarity_search(query, k=k)

    def get_document_count(self) -> int:
        """获取文档数量"""
        return len(self.documents)

# 创建全局知识库
knowledge_base = KnowledgeBase()

print("=== 3. 创建示例知识库 ===")

# 添加 AI 技术相关文档
ai_documents = [
    Document(
        page_content="""
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。

机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。通过训练数据，机器学习模型可以识别模式并做出预测或决策。

深度学习是机器学习的一个进一步子领域，使用神经网络来模拟人脑的工作方式。深度学习在图像识别、自然语言处理和语音识别等领域取得了显著成果。

自然语言处理（NLP）是人工智能的另一个重要分支，专注于计算机与人类语言之间的交互。NLP技术使计算机能够理解、解释和生成人类语言。
        """,
        metadata={"source": "ai_textbook", "category": "基础概念", "priority": 1}
    ),
    Document(
        page_content="""
LangChain 是一个用于构建基于大语言模型（LLM）应用程序的框架。它提供了一套工具和组件，帮助开发者快速构建复杂的AI应用。

LangChain 的核心组件包括：
1. Models: 模型接口，支持各种LLM提供商
2. Prompts: 提示模板系统，用于构建和管理提示
3. Chains: 链式调用，将多个组件连接成处理流水线
4. Memory: 记忆管理，为对话应用添加记忆功能
5. Agents: 智能代理，使用工具执行复杂任务
6. Indexes: 索引系统，用于文档检索和知识库管理

LangChain 支持多种用例，包括问答系统、文档分析、聊天机器人、代码生成等。它简化了LLM应用的开发过程，让开发者能够专注于业务逻辑而不是底层技术实现。
        """,
        metadata={"source": "langchain_docs", "category": "框架介绍", "priority": 1}
    ),
    Document(
        page_content="""
检索增强生成（RAG，Retrieval-Augmented Generation）是一种结合了信息检索和文本生成的技术架构。

RAG 的工作流程：
1. 文档处理：将文档分割成小块并转换为向量表示
2. 向量存储：将向量存储在专门的向量数据库中
3. 检索：根据用户查询检索最相关的文档片段
4. 增强：将检索到的文档片段作为上下文提供给生成模型
5. 生成：基于检索到的上下文生成准确、相关的回答

RAG 的优势：
- 减少模型幻觉：基于真实文档回答问题
- 知识更新：可以通过更新文档来更新知识
- 可解释性：可以追溯到回答的来源文档
- 领域专业化：可以为特定领域构建专门的知识库

RAG 广泛应用于问答系统、文档搜索、知识管理、客户服务等场景。
        """,
        metadata={"source": "rag_tutorial", "category": "RAG技术", "priority": 1}
    ),
    Document(
        page_content="""
Agent 系统是人工智能中的一个重要概念，指的是能够感知环境、做出决策并执行行动的智能实体。

LangChain 中的 Agent 具有以下特点：
1. 推理能力：能够分析问题并制定解决方案
2. 工具使用：可以调用各种工具来完成任务
3. 规划能力：能够将复杂任务分解为多个步骤
4. 学习能力：可以从执行结果中学习和改进

常见的 Agent 类型：
- ReAct Agent: 使用思考-行动-观察的循环模式
- Tool Calling Agent: 使用现代化的函数调用机制
- Planning Agent: 专注于任务分解和执行规划
- Multi-Agent System: 多个协作的智能体

Agent 的应用场景包括自动化助手、数据分析、研究助理、客户服务等。随着大语言模型的发展，Agent 系统变得越来越智能和实用。
        """,
        metadata={"source": "agent_guide", "category": "Agent技术", "priority": 1}
    ),
    Document(
        page_content="""
构建生产级的 AI 系统需要考虑多个方面的因素：

1. 性能优化：
   - 向量检索的效率优化
   - 模型推理的速度优化
   - 缓存机制的设计
   - 负载均衡和扩缩容

2. 可靠性保障：
   - 错误处理和重试机制
   - 监控和日志系统
   - 降级策略和备选方案
   - 数据备份和恢复

3. 安全性考虑：
   - API 密钥和访问控制
   - 数据隐私保护
   - 输入验证和内容过滤
   - 模型输出的安全检查

4. 成本控制：
   - API 调用的成本监控
   - 资源使用的优化
   - 缓存策略以减少重复计算
   - 模型选择的成本效益分析

5. 用户体验：
   - 响应时间优化
   - 交互界面设计
   - 个性化推荐
   - 多语言支持

遵循这些最佳实践可以构建出高质量、可靠的AI应用系统。
        """,
        metadata={"source": "production_guide", "category": "生产实践", "priority": 2}
    )
]

# 添加文档到知识库
knowledge_base.add_documents(ai_documents)

print(f"知识库创建完成！共添加 {knowledge_base.get_document_count()} 个文档")
print()

print("=== 4. 创建 RAG 专用工具 ===")

@tool
def search_knowledge_base(query: str, category: Optional[str] = None, max_results: int = 4) -> str:
    """在知识库中搜索相关信息

    Args:
        query: 搜索查询
        category: 可选的文档类别过滤
        max_results: 最大返回结果数量
    """
    try:
        # 搜索文档
        docs = knowledge_base.search(query, k=max_results)

        if not docs:
            return f"在知识库中没有找到关于 '{query}' 的相关信息。"

        # 如果指定了类别，进行过滤
        if category:
            filtered_docs = [doc for doc in docs
                           if doc.metadata.get('category', '').lower() == category.lower()]
            if filtered_docs:
                docs = filtered_docs
            else:
                return f"在类别 '{category}' 中没有找到关于 '{query}' 的相关信息。"

        # 格式化搜索结果
        result = f"找到 {len(docs)} 个相关文档：\n\n"

        for i, doc in enumerate(docs, 1):
            result += f"文档 {i}:\n"
            result += f"来源: {doc.metadata.get('source', '未知')}\n"
            result += f"类别: {doc.metadata.get('category', '未分类')}\n"
            result += f"内容: {doc.page_content[:500]}...\n\n"

        return result

    except Exception as e:
        return f"搜索过程中出现错误: {str(e)}"

@tool
def get_knowledge_categories() -> str:
    """获取知识库中的所有文档类别"""
    try:
        categories = set()
        for doc in knowledge_base.documents:
            category = doc.metadata.get('category', '未分类')
            categories.add(category)

        if not categories:
            return "知识库中暂无文档类别信息。"

        result = "知识库中的文档类别：\n"
        for category in sorted(categories):
            count = sum(1 for doc in knowledge_base.documents
                       if doc.metadata.get('category') == category)
            result += f"• {category}: {count} 个文档\n"

        return result

    except Exception as e:
        return f"获取类别信息时出现错误: {str(e)}"

@tool
def add_document_to_knowledge_base(content: str, source: str, category: str) -> str:
    """向知识库添加新文档

    Args:
        content: 文档内容
        source: 文档来源
        category: 文档类别
    """
    try:
        doc = Document(
            page_content=content,
            metadata={
                "source": source,
                "category": category,
                "added_at": datetime.now().isoformat()
            }
        )

        knowledge_base.add_documents([doc])

        return f"文档已成功添加到知识库！\n来源: {source}\n类别: {category}\n内容长度: {len(content)} 字符"

    except Exception as e:
        return f"添加文档时出现错误: {str(e)}"

@tool
def get_knowledge_base_stats() -> str:
    """获取知识库统计信息"""
    try:
        total_docs = knowledge_base.get_document_count()

        # 按类别统计
        category_stats = {}
        for doc in knowledge_base.documents:
            category = doc.metadata.get('category', '未分类')
            category_stats[category] = category_stats.get(category, 0) + 1

        result = f"""知识库统计信息：
总文档数: {total_docs}

文档类别分布：
"""
        for category, count in sorted(category_stats.items()):
            result += f"• {category}: {count} 个文档\n"

        return result

    except Exception as e:
        return f"获取统计信息时出现错误: {str(e)}"

@tool
def web_search_and_add(query: str, source_name: str, category: str) -> str:
    """搜索网络信息并添加到知识库（模拟功能）

    Args:
        query: 搜索查询
        source_name: 来源名称
        category: 文档类别
    """
    # 模拟网络搜索结果
    search_results = {
        "人工智能最新发展": "最新的人工智能发展包括大语言模型的进步、多模态AI的应用、以及AI在各个行业的深度集成。GPT-4、Claude等模型展示了强大的推理和生成能力。",
        "机器学习算法": "机器学习算法不断演进，包括深度学习、强化学习、联邦学习等新技术。这些算法在图像识别、自然语言处理、推荐系统等领域取得了突破性进展。",
        "AI应用场景": "AI技术在医疗诊断、自动驾驶、金融风控、教育个性化、智能制造等领域广泛应用，提高了效率和准确性。"
    }

    # 查找匹配的搜索结果
    content = search_results.get(query, f"关于 '{query}' 的搜索结果：这是一个模拟的搜索内容，包含了相关的信息和见解。")

    # 添加到知识库
    doc = Document(
        page_content=content,
        metadata={
            "source": f"网络搜索 - {source_name}",
            "category": category,
            "search_query": query,
            "added_at": datetime.now().isoformat()
        }
    )

    knowledge_base.add_documents([doc])

    return f"网络搜索完成并已添加到知识库！\n搜索查询: {query}\n来源: {source_name}\n类别: {category}\n内容长度: {len(content)} 字符"

# 创建 RAG 工具列表
rag_tools = [
    search_knowledge_base,
    get_knowledge_categories,
    add_document_to_knowledge_base,
    get_knowledge_base_stats,
    web_search_and_add
]

print("已创建的 RAG 工具:")
for tool in rag_tools:
    print(f"- {tool.name}: {tool.description}")
print()

print("=== 5. 创建 RAG Agent ===")

# RAG Agent 提示模板
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个基于知识库的智能问答助手，结合了检索增强生成(RAG)和Agent技术。

你的核心能力：
1. **知识检索**: 可以从知识库中搜索相关信息来回答问题
2. **智能推理**: 基于检索到的信息进行逻辑推理和分析
3. **知识管理**: 可以添加、更新和管理知识库内容
4. **多源整合**: 能够整合来自不同来源的信息

工作原则：
- 优先使用知识库中的信息回答问题
- 如果知识库中没有相关信息，明确说明
- 可以建议添加相关文档到知识库
- 保持回答的准确性和客观性
- 提供信息来源和引用

可用的工具：
- search_knowledge_base: 搜索知识库
- get_knowledge_categories: 查看知识库类别
- add_document_to_knowledge_base: 添加新文档
- get_knowledge_base_stats: 获取统计信息
- web_search_and_add: 搜索网络并添加到知识库

请根据用户的问题选择合适的工具来提供准确、有用的回答。"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# 创建 RAG Agent
rag_agent = create_tool_calling_agent(llm, rag_tools, rag_prompt)

# 创建 RAG Agent 执行器
rag_executor = AgentExecutor(
    agent=rag_agent,
    tools=rag_tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

print("RAG Agent 创建成功!")
print()

print("=== 6. 测试 RAG Agent 系统 ===")

# 测试用例1: 基础知识问答
print("测试用例1: 基础知识问答")
try:
    result1 = rag_executor.invoke({
        "input": "什么是人工智能？请详细介绍一下。"
    })
    print(f"回答: {result1['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

# 测试用例2: LangChain 相关问题
print("测试用例2: LangChain 相关问题")
try:
    result2 = rag_executor.invoke({
        "input": "LangChain 框架有哪些核心组件？"
    })
    print(f"回答: {result2['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

# 测试用例3: RAG 技术查询
print("测试用例3: RAG 技术查询")
try:
    result3 = rag_executor.invoke({
        "input": "RAG 技术的优势是什么？它的工作流程是怎样的？"
    })
    print(f"回答: {result3['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

# 测试用例4: 知识库管理
print("测试用例4: 知识库管理")
try:
    result4 = rag_executor.invoke({
        "input": "请查看当前知识库的统计信息和文档类别。"
    })
    print(f"回答: {result4['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

# 测试用例5: 动态知识添加
print("测试用例5: 动态知识添加")
try:
    result5 = rag_executor.invoke({
        "input": "请搜索关于'机器学习算法'的最新信息，并添加到知识库中。"
    })
    print(f"回答: {result5['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

# 测试用例6: 复杂查询
print("测试用例6: 复杂查询")
try:
    result6 = rag_executor.invoke({
        "input": "我想了解如何构建一个生产级的AI系统，需要注意哪些方面？"
    })
    print(f"回答: {result6['output']}")
except Exception as e:
    print(f"测试失败: {e}")
print()

print("=== 7. RAG Agent 系统特点总结 ===")
print("✅ 知识准确性: 基于检索到的文档回答问题")
print("✅ 动态知识管理: 可以实时添加和更新知识")
print("✅ 智能推理: Agent 能够分析查询并选择最佳策略")
print("✅ 多源整合: 整合不同来源的信息")
print("✅ 可追溯性: 能够追溯到回答的来源")
print("✅ 交互性: 支持追问和深度对话")
print()

print("=== 8. RAG + Agent 最佳实践 ===")
print("1. 知识库设计:")
print("   • 高质量文档: 确保知识库内容的准确性和时效性")
print("   • 合理分割: 选择合适的文档分割策略")
print("   • 元数据管理: 完善的文档分类和标签系统")
print()

print("2. 检索策略:")
print("   • 查询优化: 改进用户查询的表述方式")
print("   • 混合检索: 结合向量检索和关键词检索")
print("   • 结果排序: 基于相关性和质量的智能排序")
print()

print("3. Agent 设计:")
print("   • 工具选择: 合理设计检索和推理工具")
print("   • 提示工程: 优化系统提示以指导Agent行为")
print("   • 错误处理: 优雅处理检索失败和异常情况")
print()

print("4. 用户体验:")
print("   • 响应速度: 优化检索和生成的性能")
print("   • 回答质量: 确保回答的准确性和有用性")
print("   • 交互流畅: 支持自然的对话体验")
print()

print("=== RAG + Agent 集成学习完成 ===")
print("\n核心成就:")
print("✅ 成功集成了 RAG 和 Agent 技术")
print("✅ 构建了智能知识问答系统")
print("✅ 实现了动态知识管理功能")
print("✅ 掌握了多源信息整合技术")
print("\n下一步: Multi-Agent 系统开发")