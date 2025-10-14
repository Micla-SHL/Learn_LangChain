# demos/04_tutorials_001.py

"""
学习目标: 教程 - 构建简单的LLM应用程序
时间: 2025/10/05
说明: 基于LangChain官方教程，展示如何构建一个简单的LLM应用
包含输入验证、错误处理和响应格式化
"""

from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
import os
import re

api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("请设置环境变量: ZHIPUAI_API_KEY")

# 初始化模型
chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.7,
    api_key=api_key
)

print("=== 构建简单的LLM应用程序教程 ===")

# 1. 基础应用：文本生成
def basic_text_generation():
    """基础文本生成功能"""
    print("\n1. 基础文本生成")
    print("-" * 30)

    prompt = "请用中文写一个关于人工智能的简短介绍，大约100字。"
    response = chat.invoke(prompt)
    print(f"AI回复: {response.content}")
    return response.content

# 2. 结构化应用：使用模板
def structured_application():
    """使用提示模板的结构化应用"""
    print("\n2. 结构化应用（使用模板）")
    print("-" * 30)

    # 创建提示模板
    template = PromptTemplate.from_template(
        "作为一个{role}，请就{topic}这个话题，用{tone}的语气写一段话，字数控制在{word_count}字以内。"
    )

    # 使用模板
    formatted_prompt = template.format(
        role="技术专家",
        topic="区块链技术的应用前景",
        tone="专业且易懂",
        word_count="150"
    )

    print(f"格式化提示: {formatted_prompt}")
    response = chat.invoke(formatted_prompt)
    print(f"AI回复: {response.content}")
    return response.content

# 3. 交互式应用：聊天机器人
def interactive_chatbot():
    """交互式聊天机器人应用"""
    print("\n3. 交互式聊天机器人")
    print("-" * 30)

    # 创建聊天提示模板
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的AI助手，专门回答关于{domain}的问题。请用简洁明了的方式回答。"),
        ("human", "{user_input}")
    ])

    # 模拟多轮对话
    conversations = [
        {"domain": "编程", "user_input": "Python有什么优点？"},
        {"domain": "编程", "user_input": "如何开始学习Python？"},
        {"domain": "编程", "user_input": "推荐一些Python学习资源"}
    ]

    for i, conv in enumerate(conversations, 1):
        print(f"\n第{i}轮对话:")
        print(f"用户: {conv['user_input']}")

        # 格式化提示
        messages = chat_template.format_messages(**conv)
        response = chat.invoke(messages)
        print(f"AI助手: {response.content}")

# 4. 实用工具应用：文本处理
def text_processing_tool():
    """文本处理工具应用"""
    print("\n4. 文本处理工具")
    print("-" * 30)

    # 创建文本处理提示模板
    processing_template = PromptTemplate.from_template(
        """
请对以下文本进行{operation}：

原文：{text}

要求：
- 保持原文的核心意思
- 输出结果要{requirements}
- 不要添加额外的解释
        """
    )

    sample_text = "机器学习是人工智能的一个重要分支，它让计算机能够从数据中学习并做出预测。"

    operations = [
        {
            "operation": "简化总结",
            "requirements": "简洁明了，不超过50字"
        },
        {
            "operation": "扩写说明",
            "requirements": "详细解释，包含具体例子"
        }
    ]

    for op in operations:
        print(f"\n操作: {op['operation']}")
        prompt = processing_template.format(
            text=sample_text,
            operation=op['operation'],
            requirements=op['requirements']
        )
        response = chat.invoke(prompt)
        print(f"处理结果: {response.content}")

# 5. 错误处理和验证
def robust_application():
    """包含错误处理的健壮应用"""
    print("\n5. 健壮的应用（包含错误处理）")
    print("-" * 30)

    def validate_input(user_input: str) -> bool:
        """验证用户输入"""
        if not user_input or not user_input.strip():
            print("错误：输入不能为空")
            return False
        if len(user_input) > 1000:
            print("错误：输入过长，请控制在1000字符以内")
            return False
        return True

    def generate_response(topic: str, style: str = "专业") -> str:
        """生成响应的函数"""
        try:
            template = PromptTemplate.from_template(
                "请用{style}的风格介绍{topic}这个主题，内容要准确且有用。"
            )

            prompt = template.format(topic=topic, style=style)
            response = chat.invoke(prompt)
            return response.content

        except Exception as e:
            return f"生成回复时出错：{str(e)}"

    # 测试不同的输入
    test_cases = [
        ("量子计算", "通俗"),
        ("", "专业"),  # 空输入
        ("机器学习", "专业"),
        ("人工智能", "幽默")
    ]

    for topic, style in test_cases:
        print(f"\n测试 - 主题: '{topic}', 风格: '{style}'")

        if validate_input(topic):
            response = generate_response(topic, style)
            print(f"生成的内容: {response[:100]}...")
        else:
            print("输入验证失败，跳过生成")

# 6. 链式应用示例
def chained_application():
    """展示链式处理的应用"""
    print("\n6. 链式处理应用")
    print("-" * 30)

    # 创建处理链
    template1 = PromptTemplate.from_template("将以下内容翻译成英文：{content}")
    template2 = PromptTemplate.from_template("将以下英文内容总结成一个要点：{content}")

    # 第一级处理：翻译
    translation_response = chat.invoke(template1.format(content="人工智能正在改变世界"))
    translation = translation_response.content
    print(f"翻译结果: {translation}")

    # 第二级处理：总结
    summary_response = chat.invoke(template2.format(content=translation))
    summary = summary_response.content
    print(f"总结结果: {summary}")

# 主函数：运行所有示例
def main():
    """主函数，按顺序运行所有示例"""
    print("LangChain 简单应用程序构建教程")
    print("=" * 50)

    try:
        # 依次运行各个示例
        basic_text_generation()
        structured_application()
        interactive_chatbot()
        text_processing_tool()
        robust_application()
        chained_application()

        print("\n" + "=" * 50)
        print("✅ 所有示例运行完成！")
        print("\n学习要点总结：")
        print("1. 基础文本生成：直接调用LLM")
        print("2. 结构化应用：使用提示模板")
        print("3. 交互式应用：构建聊天机器人")
        print("4. 实用工具：专门的文本处理")
        print("5. 健壮性：错误处理和输入验证")
        print("6. 链式处理：多步骤处理流程")

    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")

if __name__ == "__main__":
    main()