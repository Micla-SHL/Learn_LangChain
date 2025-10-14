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


"""
# 测试
text = "这是一段测试文本"
vector = embeddings.embed_query(text)
print(f"向量维度: {len(vector)}")
"""

##在本指南中，我们将构建一个应用程序来回答有关网站内容的问题。
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)



# Load and chunk contents of the blog

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
#prompt = hub.pull("rlm/rag-prompt")

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


response = graph.invoke({"question": "这篇文章讲什么？"})
print(response["answer"])
