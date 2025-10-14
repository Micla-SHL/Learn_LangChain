# demos/10_data_connection.py

"""
学习目标: LangChain 数据连接 (Data Connection)
时间: 2025/10/14
说明: 学习如何加载、处理和存储各种类型的数据
"""

import os
from typing import List
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader
)
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
import tempfile
import csv

# 初始化模型
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")

chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.7,
    api_key=api_key
)

embeddings = ZhipuAIEmbeddings(model="embedding-3")

print("=== 1. 文本文档加载 ===")
# 创建临时文本文件
sample_text = """
LangChain学习指南

第一章：基础概念
LangChain是一个用于构建基于LLM应用程序的框架。
它提供了丰富的工具和组件，帮助开发者快速构建智能应用。

第二章：核心组件
- Models: 模型接口
- Prompts: 提示模板
- Chains: 链式调用
- Memory: 记忆管理
- Indexes: 索引和检索

第三章：实践应用
通过实际案例学习如何使用LangChain构建各种AI应用。
"""

# 写入临时文件
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
    f.write(sample_text)
    temp_file_path = f.name

try:
    # 加载文本文件
    text_loader = TextLoader(temp_file_path, encoding='utf-8')
    documents = text_loader.load()

    print("加载的文档数量:", len(documents))
    print("第一页内容预览:")
    print(documents[0].page_content[:200] + "...")
    print("元数据:", documents[0].metadata)
    print()

finally:
    # 清理临时文件
    os.unlink(temp_file_path)

print("=== 2. CSV数据加载 ===")
# 创建临时CSV文件
csv_data = [
    ["产品", "价格", "描述"],
    ["iPhone", "999", "苹果公司的智能手机"],
    ["MacBook", "1999", "苹果公司的笔记本电脑"],
    ["iPad", "599", "苹果公司的平板电脑"]
]

with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)
    csv_file_path = f.name

try:
    # 加载CSV文件
    csv_loader = CSVLoader(csv_file_path, encoding='utf-8')
    csv_docs = csv_loader.load()

    print("CSV文档数量:", len(csv_docs))
    print("CSV内容示例:")
    for doc in csv_docs[:2]:
        print(f"内容: {doc.page_content}")
        print(f"元数据: {doc.metadata}")
    print()

finally:
    os.unlink(csv_file_path)

print("=== 3. 网页内容加载 ===")
try:
    # 加载网页内容（使用一个简单的方法）
    web_loader = WebBaseLoader("https://zh.wikipedia.org/wiki/人工智能")
    web_docs = web_loader.load()

    print("网页文档数量:", len(web_docs))
    print("网页内容预览:")
    print(web_docs[0].page_content[:300] + "...")
    print("元数据:", web_docs[0].metadata)
except Exception as e:
    print(f"网页加载失败（可能网络问题）: {e}")
    print("这是一个示例，实际使用时需要网络连接")
print()

print("=== 4. 目录批量加载 ===")
# 创建临时目录和多个文件
temp_dir = tempfile.mkdtemp()
try:
    # 创建多个文件
    files_content = {
        "doc1.txt": "这是第一个文档，包含基础概念介绍。",
        "doc2.txt": "这是第二个文档，详细介绍高级特性。",
        "doc3.txt": "这是第三个文档，提供实际应用案例。"
    }

    for filename, content in files_content.items():
        with open(os.path.join(temp_dir, filename), 'w', encoding='utf-8') as f:
            f.write(content)

    # 批量加载目录中的文本文件
    dir_loader = DirectoryLoader(
        temp_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )

    dir_docs = dir_loader.load()
    print("目录加载的文档数量:", len(dir_docs))
    for i, doc in enumerate(dir_docs):
        print(f"文档{i+1}: {doc.metadata['source'][:50]}...")

finally:
    # 清理临时目录
    import shutil
    shutil.rmtree(temp_dir)
print()

print("=== 5. 文本分割策略 ===")
# 创建长文档用于分割演示
long_text = """
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
这些任务包括学习、推理、问题解决、感知和语言理解。

机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。
通过训练数据，机器学习模型可以识别模式并做出预测或决策。

深度学习是机器学习的一个进一步子领域，使用神经网络来模拟人脑的工作方式。
深度学习在图像识别、自然语言处理和语音识别等领域取得了显著成果。

自然语言处理（NLP）是人工智能的另一个重要分支，专注于计算机与人类语言之间的交互。
NLP技术使计算机能够理解、解释和生成人类语言。
""" * 3  # 重复3次以创建更长的文本

# 创建文档对象
long_doc = Document(page_content=long_text)

print("原始文档长度:", len(long_doc.page_content))
print()

# 1. 字符分割器
char_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separator="\n"
)
char_chunks = char_splitter.split_documents([long_doc])
print("字符分割结果:")
print(f"分割块数: {len(char_chunks)}")
print(f"第一块长度: {len(char_chunks[0].page_content)}")
print(f"第一块内容: {char_chunks[0].page_content[:100]}...")
print()

# 2. 递归字符分割器
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " "]
)
recursive_chunks = recursive_splitter.split_documents([long_doc])
print("递归分割结果:")
print(f"分割块数: {len(recursive_chunks)}")
print(f"第一块长度: {len(recursive_chunks[0].page_content)}")
print()

# 3. Token分割器（需要tiktoken）
try:
    token_splitter = TokenTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    token_chunks = token_splitter.split_documents([long_doc])
    print("Token分割结果:")
    print(f"分割块数: {len(token_chunks)}")
except Exception as e:
    print(f"Token分割失败（可能缺少tiktoken）: {e}")
print()

print("=== 6. 向量存储 ===")
# 使用分割后的文档创建向量存储
try:
    # 使用FAISS（内存向量存储）
    faiss_store = FAISS.from_documents(recursive_chunks[:3], embeddings)

    print("FAISS向量存储创建成功")
    print("存储的文档数量:", len(recursive_chunks[:3]))

    # 相似性搜索
    query = "什么是机器学习？"
    results = faiss_store.similarity_search(query, k=2)

    print(f"\n查询: {query}")
    print("相似性搜索结果:")
    for i, doc in enumerate(results):
        print(f"结果 {i+1}: {doc.page_content[:100]}...")
    print()

    # 保存和加载向量存储
    save_path = tempfile.mkdtemp()
    faiss_save_path = os.path.join(save_path, "faiss_index")
    faiss_store.save_local(faiss_save_path)
    print(f"FAISS索引已保存到: {faiss_save_path}")

    # 加载索引
    loaded_store = FAISS.load_local(faiss_save_path, embeddings)
    print("FAISS索引加载成功")

    # 清理
    import shutil
    shutil.rmtree(save_path)

except Exception as e:
    print(f"向量存储创建失败: {e}")
    print("可能需要安装额外的依赖包")

print()
print("=== 7. 自定义文档加载器 ===")
class CustomDocumentLoader:
    """自定义文档加载器示例"""

    def __init__(self, data_source: str):
        self.data_source = data_source

    def load(self) -> List[Document]:
        """模拟从自定义数据源加载文档"""
        # 这里可以是数据库、API等
        mock_data = [
            {"title": "产品A", "content": "这是产品A的详细描述", "category": "电子"},
            {"title": "产品B", "content": "这是产品B的详细描述", "category": "家居"},
            {"title": "产品C", "content": "这是产品C的详细描述", "category": "服装"}
        ]

        documents = []
        for item in mock_data:
            doc = Document(
                page_content=item["content"],
                metadata={
                    "title": item["title"],
                    "category": item["category"],
                    "source": self.data_source
                }
            )
            documents.append(doc)

        return documents

# 使用自定义加载器
custom_loader = CustomDocumentLoader("产品数据库")
custom_docs = custom_loader.load()

print("自定义加载器结果:")
for doc in custom_docs:
    print(f"标题: {doc.metadata['title']}")
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
    print()

print("=== 数据连接学习完成 ===")
print("\n关键概念总结：")
print("1. Document Loaders: 从各种数据源加载文档")
print("2. Text Splitters: 将长文档分割成合适的块")
print("3. Vector Stores: 存储和检索文档向量")
print("4. Embeddings: 将文本转换为向量表示")
print("5. 自定义加载器: 处理特殊数据源")