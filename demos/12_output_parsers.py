# demos/12_output_parsers.py

"""
学习目标: LangChain 输出解析器 (Output Parsers)
时间: 2025/10/14
说明: 学习如何解析和结构化LLM的输出结果
"""

import os
import json
import re
from typing import List, Dict, Any
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser
)
# EnumOutputParser 和 DatetimeOutputParser 在 LangChain v0.3 中已移除，我们将实现自定义解析器
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

# 初始化模型
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")

chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.1,  # 使用较低温度以获得更稳定的输出
    api_key=api_key
)

print("=== 1. 字符串输出解析器 ===")
# 最简单的输出解析器
str_parser = StrOutputParser()

prompt = PromptTemplate.from_template("请简单回答：{question}")

chain = prompt | chat | str_parser

result = chain.invoke({"question": "什么是人工智能？"})
print("字符串解析结果:")
print(result)
print("类型:", type(result))
print()

print("=== 2. JSON输出解析器 ===")
# JSON输出解析器
json_parser = JsonOutputParser()

json_prompt = PromptTemplate.from_template(
    """请以JSON格式回答以下问题。
    返回格式示例：
    {{
        "answer": "主要答案",
        "confidence": "高/中/低",
        "details": ["要点1", "要点2", "要点3"]
    }}

    问题：{question}"""
)

json_chain = json_prompt | chat | json_parser

json_result = json_chain.invoke({"question": "什么是机器学习的主要类型？"})
print("JSON解析结果:")
print(json.dumps(json_result, ensure_ascii=False, indent=2))
print("类型:", type(json_result))
print()

print("=== 3. Pydantic输出解析器 ===")
# 定义Pydantic模型
class MovieReview(BaseModel):
    """电影评论模型"""
    title: str = Field(description="电影标题")
    rating: float = Field(description="评分，0-10分", ge=0, le=10)
    summary: str = Field(description="评论摘要")
    pros: List[str] = Field(description="优点列表")
    cons: List[str] = Field(description="缺点列表")
    recommended: bool = Field(description="是否推荐")

# 创建Pydantic解析器
pydantic_parser = PydanticOutputParser(pydantic_object=MovieReview)

# 获取格式说明
format_instructions = pydantic_parser.get_format_instructions()

pydantic_prompt = PromptTemplate.from_template(
    """请分析以下电影并提供结构化评论。

    电影：《阿凡达：水之道》

    {format_instructions}

    请提供详细的分析。"""
).partial(format_instructions=format_instructions)

pydantic_chain = pydantic_prompt | chat | pydantic_parser

try:
    pydantic_result = pydantic_chain.invoke({})
    print("Pydantic解析结果:")
    print(f"标题: {pydantic_result.title}")
    print(f"评分: {pydantic_result.rating}/10")
    print(f"摘要: {pydantic_result.summary}")
    print(f"优点: {', '.join(pydantic_result.pros)}")
    print(f"缺点: {', '.join(pydantic_result.cons)}")
    print(f"推荐: {'是' if pydantic_result.recommended else '否'}")
    print("类型:", type(pydantic_result))
except Exception as e:
    print(f"Pydantic解析失败: {e}")
print()

print("=== 4. 逗号分隔列表解析器 ===")
# 列表解析器
list_parser = CommaSeparatedListOutputParser()

list_format_instructions = list_parser.get_format_instructions()

list_prompt = PromptTemplate.from_template(
    """请列出Python编程的5个主要应用领域。

    {format_instructions}"""
).partial(format_instructions=list_format_instructions)

list_chain = list_prompt | chat | list_parser

try:
    list_result = list_chain.invoke({})
    print("列表解析结果:")
    print(list_result)
    print("类型:", type(list_result))
    for i, item in enumerate(list_result, 1):
        print(f"{i}. {item}")
except Exception as e:
    print(f"列表解析失败: {e}")
print()

print("=== 5. 枚举输出解析器 (自定义) ===")
# 定义枚举
class Sentiment(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class Priority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# 自定义枚举解析器
class CustomEnumParser:
    """自定义枚举解析器"""

    def __init__(self, enum_class):
        self.enum_class = enum_class

    def get_format_instructions(self) -> str:
        """返回格式说明"""
        options = [e.value for e in self.enum_class]
        return f"请从以下选项中选择一个: {', '.join(options)}"

    def parse(self, text: str) -> Enum:
        """解析枚举值"""
        text_lower = text.lower().strip()

        # 查找匹配的枚举值
        for enum_item in self.enum_class:
            if enum_item.value.lower() == text_lower:
                return enum_item

        # 如果直接匹配失败，尝试模糊匹配
        for enum_item in self.enum_class:
            if enum_item.value.lower() in text_lower or text_lower in enum_item.value.lower():
                return enum_item

        raise OutputParserException(f"无法解析枚举值: {text}")

# 创建枚举解析器
sentiment_parser = CustomEnumParser(Sentiment)

# 情感分析提示
sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个情感分析专家。请分析以下文本的情感倾向。"),
    ("human", "文本：{text}\n\n{format_instructions}")
]).partial(format_instructions=sentiment_parser.get_format_instructions())

sentiment_chain = sentiment_prompt | chat | StrOutputParser()

try:
    # 获取原始响应
    raw_response = sentiment_chain.invoke({"text": "今天天气真好，心情很愉快！"})
    print("原始响应:", raw_response)

    # 解析枚举
    sentiment_result = sentiment_parser.parse(raw_response)
    print("情感分析结果:", sentiment_result.value)
    print("类型:", type(sentiment_result))
except Exception as e:
    print(f"枚举解析失败: {e}")
print()

print("=== 6. 日期时间解析器 (自定义) ===")
# 自定义日期解析器，替代已移除的 DatetimeOutputParser
class CustomDatetimeParser:
    """自定义日期时间解析器"""

    def get_format_instructions(self) -> str:
        return "请以 YYYY-MM-DD 格式返回日期，例如：2024-01-15"

    def parse(self, text: str) -> datetime:
        """解析日期字符串"""
        try:
            # 尝试提取日期
            import re
            date_match = re.search(r'\d{4}-\d{2}-\d{2}', text)
            if date_match:
                return datetime.strptime(date_match.group(), '%Y-%m-%d')
            else:
                raise ValueError("无法找到有效的日期格式")
        except Exception as e:
            raise OutputParserException(f"日期解析失败: {e}")

# 创建自定义日期解析器
datetime_parser = CustomDatetimeParser()

datetime_format_instructions = datetime_parser.get_format_instructions()

datetime_prompt = PromptTemplate.from_template(
    """将以下自然语言描述转换为标准日期格式。

    描述：{date_description}

    {format_instructions}"""
).partial(format_instructions=datetime_format_instructions)

# 创建解析链
datetime_chain = datetime_prompt | chat | StrOutputParser()

try:
    # 获取原始响应
    raw_response = datetime_chain.invoke({"date_description": "2024年1月15日"})
    print("原始响应:", raw_response)

    # 解析日期
    datetime_result = datetime_parser.parse(raw_response)
    print("解析后的日期:", datetime_result.strftime('%Y-%m-%d'))
    print("类型:", type(datetime_result))
except Exception as e:
    print(f"日期解析失败: {e}")
print()

print("=== 7. 自定义解析器 ===")
class CustomMarkdownParser:
    """自定义Markdown解析器"""

    def __init__(self):
        self.pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def parse(self, text: str) -> Dict[str, Any]:
        """解析Markdown标题结构"""
        headers = {}
        current_section = None

        lines = text.split('\n')
        for line in lines:
            match = self.pattern.match(line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                key = f"h{level}_{title}"
                headers[key] = {
                    'level': level,
                    'title': title,
                    'content': ''
                }
                current_section = key
            elif current_section and line.strip():
                headers[current_section]['content'] += line.strip() + ' '

        return headers

    def get_format_instructions(self) -> str:
        """返回格式说明"""
        return """请以Markdown格式回答，使用标题(## ###)来组织内容结构。
        例如：
        ## 主要观点
        这里是主要观点的详细说明...

        ## 具体例子
        1. 例子一
        2. 例子二
        """

# 使用自定义解析器
custom_parser = CustomMarkdownParser()

markdown_prompt = PromptTemplate.from_template(
    """请用Markdown格式分析人工智能的发展历程。

    {format_instructions}"""
).partial(format_instructions=custom_parser.get_format_instructions())

custom_chain = markdown_prompt | chat | StrOutputParser()

try:
    raw_response = custom_chain.invoke({})
    print("原始Markdown响应:")
    print(raw_response[:300] + "...")
    print()

    parsed_result = custom_parser.parse(raw_response)
    print("自定义解析结果:")
    for key, value in parsed_result.items():
        print(f"{key}: {value['title']} (Level {value['level']})")
        if value['content']:
            print(f"  内容预览: {value['content'][:50]}...")
except Exception as e:
    print(f"自定义解析失败: {e}")
print()

print("=== 8. 组合解析器 ===")
class ComplexAnalysis(BaseModel):
    """复杂分析模型"""
    main_topic: str = Field(description="主要主题")
    key_points: List[str] = Field(description="关键要点")
    sentiment: str = Field(description="情感倾向")
    confidence_score: float = Field(description="置信度", ge=0, le=1)
    tags: List[str] = Field(description="标签")

# 复合解析器
complex_parser = PydanticOutputParser(pydantic_object=ComplexAnalysis)

def complex_parse_chain(text: str) -> Dict[str, Any]:
    """复合解析链"""
    # 第一步：获取原始响应
    analysis_prompt = PromptTemplate.from_template(
        """请全面分析以下文本：

        文本：{input_text}

        {format_instructions}
        """
    ).partial(format_instructions=complex_parser.get_format_instructions())

    chain = analysis_prompt | chat | complex_parser
    result = chain.invoke({"input_text": text})

    # 第二步：后处理
    processed_result = {
        "analysis": result.model_dump(),
        "word_count": len(text.split()),
        "character_count": len(text),
        "has_questions": "?" in text or "？" in text,
        "complexity_score": min(len(result.key_points) / 5.0, 1.0)
    }

    return processed_result

sample_text = "人工智能技术正在快速发展，它改变了我们的生活方式。从智能家居到自动驾驶，AI的应用越来越广泛。虽然存在一些挑战，但我对未来充满期待！"

try:
    complex_result = complex_parse_chain(sample_text)
    print("复合解析结果:")
    print(json.dumps(complex_result, ensure_ascii=False, indent=2))
except Exception as e:
    print(f"复合解析失败: {e}")
print()

print("=== 9. 错误处理和重试 ===")
class RobustJsonParser:
    """健壮的JSON解析器，带错误处理"""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.base_parser = JsonOutputParser()

    def parse(self, text: str) -> Dict[str, Any]:
        """带重试的JSON解析"""
        for attempt in range(self.max_retries):
            try:
                return self.base_parser.parse(text)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # 最后一次尝试，尝试修复常见的JSON错误
                    return self._fix_and_parse(text)
                print(f"解析失败，重试 {attempt + 1}/{self.max_retries}: {e}")
                continue
        return {"error": "解析失败", "original_text": text}

    def _fix_and_parse(self, text: str) -> Dict[str, Any]:
        """尝试修复常见的JSON格式错误"""
        # 移除可能的markdown代码块标记
        cleaned = re.sub(r'```json\s*', '', text)
        cleaned = re.sub(r'```\s*$', '', cleaned)

        # 尝试提取JSON部分
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group()

        try:
            return json.loads(cleaned)
        except:
            return {"error": "无法解析", "cleaned_text": cleaned}

# 测试健壮解析器
robust_parser = RobustJsonParser()

test_cases = [
    '{"answer": "test", "confidence": "high"}',  # 正常JSON
    '```json\n{"answer": "test"}\n```',  # 带markdown标记
    '这是一个回答：{"answer": "test"}结束',  # 带额外文本
    '不是JSON格式的内容'  # 无效内容
]

print("健壮解析器测试:")
for i, test_case in enumerate(test_cases, 1):
    print(f"测试 {i}: {test_case[:50]}...")
    result = robust_parser.parse(test_case)
    print(f"结果: {result}")
    print()

print("=== 输出解析器学习完成 ===")
print("\n解析器选择指南：")
print("• StrOutputParser: 简单文本输出")
print("• JsonOutputParser: 结构化数据")
print("• PydanticOutputParser: 严格类型定义")
print("• ListOutputParser: 列表数据")
print("• EnumOutputParser: 限定选项")
print("• DatetimeOutputParser: 时间日期")
print("• 自定义解析器: 特殊格式需求")
print("• 健壮解析器: 生产环境容错")