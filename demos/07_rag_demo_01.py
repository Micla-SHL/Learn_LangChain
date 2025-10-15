# demos/07_rag_demo_01.py

"""
学习目标: 学习LangChain的 RAG（检索增强生成）调用
时间: 2025/10/06
说明: 完整的RAG系统实现，包含文档加载、分割、向量化和检索
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.embeddings import ZhipuAIEmbeddings
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.prompts import PromptTemplate

import os

api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")


chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.5,
        api_key = api_key
)


# 创建嵌入模型
embeddings = ZhipuAIEmbeddings(
    model="embedding-3"
)

# 测试嵌入模型
try:
    text = "这是一段测试文本"
    vector = embeddings.embed_query(text)
    print(f"嵌入模型测试成功，向量维度: {len(vector)}")
except Exception as e:
    print(f"嵌入模型测试失败: {e}")
    print("请检查智谱AI API密钥是否正确设置")

##在本指南中，我们将构建一个应用程序来回答有关网站内容的问题。
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)



# Load and chunk contents of the blog
# 使用本地示例文档作为备用，避免网络依赖
sample_docs = [
    Document(
        page_content="""
        LangChain Agent 系统架构

        Agent 是LangChain框架中的一个核心概念，它能够自主地执行任务。
        Agent的核心思想是通过推理（Reasoning）和行动（Acting）来解决问题。

        主要组成部分：
        1. 推理引擎：负责分析问题并制定行动计划
        2. 工具集：提供各种功能工具，如搜索、计算、查询等
        3. 记忆系统：保存对话历史和上下文信息
        4. 执行器：协调各个组件的工作流程

        Agent的优势：
        - 能够处理复杂的、多步骤的任务
        - 可以根据中间结果调整策略
        - 支持与外部系统的交互
        - 具备一定程度的自主决策能力
        """,
        metadata={"source": "本地示例文档", "type": "Agent架构说明"}
    ),
    Document(
        page_content="""
        RAG（检索增强生成）技术详解

        RAG是一种结合了信息检索和文本生成的技术，能够提高AI回答的准确性。

        工作流程：
        1. 文档处理：将各种类型的文档转换为统一的文本格式
        2. 文本分割：将长文档分割成适合处理的片段
        3. 向量化：将文本片段转换为数学向量表示
        4. 存储索引：将向量和原文一起存储到向量数据库
        5. 相似性搜索：根据问题找到最相关的文档片段
        6. 上下文增强：将检索到的内容添加到提示中
        7. 生成回答：基于增强的上下文生成准确的回答

        应用场景：
        - 知识库问答系统
        - 文档分析助手
        - 智能客服系统
        - 教育辅导工具
        """,
        metadata={"source": "本地示例文档", "type": "RAG技术说明"}
    )
]

# 尝试从网络加载文档，失败时使用本地示例
try:
    print("尝试从网络加载文档...")
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    print(f"✅ 网络文档加载成功，共 {len(docs)} 个文档")
except Exception as e:
    print(f"⚠️ 网络文档加载失败: {e}")
    print("使用本地示例文档...")
    docs = sample_docs
    print(f"✅ 本地文档加载成功，共 {len(docs)} 个文档")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# 定义问答提示模板

# 自定义中文 Prompt
custom_prompt = PromptTemplate.from_template("""
你是一个专业的问答助手，擅长从给定的文档中提取信息回答问题。

请根据以下文档内容回答用户的问题：

【文档内容】
{context}

【用户问题】
{question}

【回答要求】
1. 如果文档中有相关信息，请详细准确地回答
2. 如果文档中没有相关信息，请明确说"根据提供的文档无法回答这个问题"
3. 不要编造或推测文档中没有的信息
4. 回答要清晰、有条理

【你的答案】
""")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = custom_prompt.invoke({"question": state["question"], "context": docs_content})
    response = chat.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


print("\n=== RAG 系统测试 ===")
print("可用问题示例：")
print("1. 'Agent的主要组成部分是什么？'")
print("2. 'RAG技术的工作流程是什么？'")
print("3. 'Agent有什么优势？'")
print("4. 'RAG的应用场景有哪些？'")
print()

# 测试多个问题
test_questions = [
    "Agent的主要组成部分是什么？",
    "RAG技术如何提高AI回答的准确性？",
    "请介绍RAG的应用场景"
]

for i, question in enumerate(test_questions, 1):
    print(f"问题 {i}: {question}")
    try:
        response = graph.invoke({"question": question})
        print(f"回答: {response['answer']}")
    except Exception as e:
        print(f"❌ 处理问题时出错: {e}")
    print("-" * 50)
