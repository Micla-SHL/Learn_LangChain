# demos/09_chains_basics.py

"""
学习目标: LangChain 链式调用 (Chains) 基础
时间: 2025/10/14
说明: 学习如何将多个组件链接起来形成处理流水线
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_community.chat_models import ChatZhipuAI
import os
import json

# 初始化模型
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")

chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.7,
    api_key=api_key
)

print("=== 1. 最简单的链 (LLMChain) ===")
# 创建提示模板
story_prompt = PromptTemplate.from_template(
    "请写一个关于{topic}的{genre}故事，大约{length}字。"
)

# 创建链
story_chain = story_prompt | chat | StrOutputParser()

# 执行链
result = story_chain.invoke({
    "topic": "人工智能",
    "genre": "科幻",
    "length": "200"
})

print("故事生成结果:")
print(result)
print()

print("=== 2. 使用 LCEL 创建链 ===")
# 使用 LCEL 方式创建链（推荐方式）
lcel_chain = story_prompt | chat | StrOutputParser()

result2 = lcel_chain.invoke({
    "topic": "时间旅行",
    "genre": "悬疑",
    "length": "150"
})

print("LCEL 链结果:")
print(result2)
print()

print("=== 3. 顺序链 (使用 LCEL) ===")
# 创建第一个链：生成大纲
outline_prompt = PromptTemplate.from_template(
    "为一篇关于{topic}的文章生成一个大纲，包含3-4个要点。"
)
outline_chain = outline_prompt | chat | StrOutputParser()

# 创建第二个链：根据大纲写内容
content_prompt = PromptTemplate.from_template(
    "根据以下大纲写一篇文章：\n{outline}\n\n请详细展开每个要点。"
)
content_chain = content_prompt | chat | StrOutputParser()

# 使用 LCEL 创建顺序链
def create_sequential_chain(outline_chain, content_chain):
    def process_topic(topic):
        # 第一步：生成大纲
        outline = outline_chain.invoke({"topic": topic})
        # 第二步：根据大纲写内容
        content = content_chain.invoke({"outline": outline})
        return content
    return RunnableLambda(process_topic)

# 创建顺序链
sequential_chain = create_sequential_chain(outline_chain, content_chain)

print("顺序链执行:")
result3 = sequential_chain.invoke("可持续发展")
print("\n最终文章:")
print(result3)
print()

print("=== 4. 并行链 (RunnableParallel) ===")
# 创建多个并行任务
translation_prompt = PromptTemplate.from_template(
    "将以下中文翻译成英文：{text}"
)
summary_prompt = PromptTemplate.from_template(
    "将以下内容总结成一句话：{text}"
)
keyword_prompt = PromptTemplate.from_template(
    "从以下文本中提取3个关键词：{text}"
)

# 创建并行链
parallel_chain = RunnableParallel(
    translation=translation_prompt | chat | StrOutputParser(),
    summary=summary_prompt | chat | StrOutputParser(),
    keywords=keyword_prompt | chat | StrOutputParser()
)

sample_text = "人工智能正在改变我们的生活方式，从智能家居到自动驾驶，AI技术无处不在。"

print("并行链执行:")
parallel_result = parallel_chain.invoke({"text": sample_text})
print("翻译:", parallel_result["translation"])
print("总结:", parallel_result["summary"])
print("关键词:", parallel_result["keywords"])
print()

print("=== 5. 带条件判断的链 ===")
def conditional_chain(topic: str):
    """根据主题选择不同的处理链"""
    if "技术" in topic:
        # 技术主题的链
        tech_prompt = PromptTemplate.from_template(
            "详细解释{topic}的技术原理和应用场景。"
        )
        return tech_prompt | chat | StrOutputParser()
    else:
        # 非技术主题的链
        general_prompt = PromptTemplate.from_template(
            "通俗易懂地介绍{topic}，让普通读者也能理解。"
        )
        return general_prompt | chat | StrOutputParser()

print("条件链测试:")
tech_result = conditional_chain("区块链技术").invoke({"topic": "区块链技术"})
print("技术主题结果:", tech_result[:100] + "...")

general_result = conditional_chain("音乐欣赏").invoke({"topic": "音乐欣赏"})
print("一般主题结果:", general_result[:100] + "...")
print()

print("=== 6. 带数据转换的链 ===")
# 数据预处理链
preprocess_prompt = PromptTemplate.from_template(
    "清理和整理以下文本，使其更清晰：{raw_text}"
)

# 数据后处理链
format_prompt = PromptTemplate.from_template(
    "将以下内容格式化为Markdown格式：\n{content}"
)

# 完整处理链
processing_chain = (
    preprocess_prompt | chat | StrOutputParser() |
    format_prompt | chat | StrOutputParser()
)

messy_text = "这个很重要因为我们需要知道怎么弄好它所以应该学习"
print("数据处理链测试:")
processed_result = processing_chain.invoke({"raw_text": messy_text})
print("处理结果:")
print(processed_result)
print()

print("=== 7. JSON输出链 ===")
# 创建JSON输出解析器
json_parser = JsonOutputParser()

# JSON格式化提示模板
json_prompt = PromptTemplate.from_template(
    """请以JSON格式回答以下问题。返回格式要求：
    {{
        "answer": "主要答案",
        "confidence": "置信度(高/中/低)",
        "reasoning": "推理过程"
    }}

    问题：{question}"""
)

json_chain = json_prompt | chat | json_parser

print("JSON输出链测试:")
json_result = json_chain.invoke({"question": "什么是机器学习？"})
print("JSON结果:")
print(json.dumps(json_result, ensure_ascii=False, indent=2))
print()

print("=== 链式调用学习完成 ===")
print("\n关键概念总结 (LangChain v0.3)：")
print("1. LCEL 链: 使用管道操作符创建链 (推荐方式)")
print("2. 顺序链: 使用 RunnableLambda 创建顺序执行逻辑")
print("3. RunnableParallel: 并行执行的链")
print("4. 条件链: 根据条件选择不同的处理路径")
print("5. 数据转换链: 包含数据预处理的复杂链")
print("6. JSON输出链: 结构化数据输出")
print("\n注意：LLMChain 和 SimpleSequentialChain 已弃用，建议使用 LCEL 语法")