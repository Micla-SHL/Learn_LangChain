# Learn_LangChain

该项目是作为 LangChain 初学者练习代码记录，参照 LangChain v0.3 版本代码，在 Termux 环境下实测可运行，理论上可直接迁移 Linux 平台。

## 📁 项目结构

`~/demos` 是练习代码目录，每一份都可直接运行，前置条件：需配置 LLM API key。国内环境选用智谱 AI。

## 🚀 学习进度

### ✅ 基础模块 (已完成)

#### 1. LangChain 基础入门
- `01_basic_llm_usage.py` - LLM 基础使用
- `02_agent_llm_debug_error.py` - Agent 基础调试（有错误）
- `03_agent_llm_debug.py` - Agent 基础调试（修复版）

#### 2. 教程系列
- `04_tutorials_001.py` - 教程示例1
- `05_tutorials_002.py` - 教程示例2
- `06_build_chatbot.py` - 聊天机器人构建

#### 3. 核心功能模块 📚 **LangChain核心概念完整学习路径**
- `07_rag_demo_01.py` - RAG (检索增强生成) 基础演示
- `08_prompt_templates.py` - **提示模板系统** ⚡ **新增**
  - PromptTemplate 和 ChatPromptTemplate
  - 多变量模板和条件模板
  - 模板验证和组合使用
- `09_chains_basics.py` - **链式调用基础** ⚡ **新增**
  - LLMChain 和 SimpleSequentialChain
  - RunnableParallel 并行执行
  - 条件链和数据转换链
  - JSON输出链和现代化语法
- `10_data_connection.py` - **数据连接和文档处理** ⚡ **新增**
  - 文档加载器 (Text, CSV, Web, Directory)
  - 文本分割策略 (递归分割、Token分割)
  - 向量存储 (FAISS) 和嵌入模型
  - 自定义文档加载器
- `11_memory_types.py` - **记忆管理系统** ⚡ **新增**
  - 缓冲记忆和对话摘要记忆
  - LangGraph 持久化记忆
  - 长期记忆系统设计
  - 不同记忆类型的选择指南
- `12_output_parsers.py` - **输出解析器** ⚡ **新增**
  - 字符串、JSON、Pydantic 解析器
  - 列表、枚举、日期时间解析器
  - 自定义解析器和错误处理
  - 健壮的解析器设计

### 🎯 Agent 系统进阶 (已完成)

#### 4. Agent 系统完整学习路径
- `13_agent_basics.py` - **基础 ReAct Agent** ⚡ **新增**
  - 理解 ReAct (Reasoning + Acting) 模式
  - 创建自定义工具（计算器、时间、文本分析、单位转换）
  - 掌握 Agent 配置和错误处理

- `14_agent_function_calling.py` - **工具调用 Agent** ⚡ **新增**
  - 掌握现代化 function calling 技术
  - 实现高级工具集（天气、新闻、邮件、日程、文件、汇率）
  - 学会多工具协作机制

- `15_agent_planning.py` - **多步骤规划 Agent** ⚡ **新增**
  - 掌握任务分解和规划技术
  - 实现复杂任务的状态管理
  - 理解动态规划和调整机制

### 🔄 当前进行中

#### 5. 高级集成技术
- `16_rag_agent_integration.py` - **RAG + Agent 集成** 🚧 **开发中**
  - 结合检索增强生成和 Agent 能力
  - 实现基于知识库的智能对话
  - 构建专业领域问答系统

### 📋 计划学习内容

#### 6. Multi-Agent 系统 (待开发)
- 多 Agent 协作框架
- Agent 间通信机制
- 分布式任务分配
- 协作冲突解决

#### 7. 生产环境部署 (待开发)
- Agent 系统容器化
- 性能监控和调优
- 安全性考虑
- API 设计和集成

#### 8. 特定领域应用 (待开发)
- 智能客服系统
- 文档分析助手
- 代码生成助手
- 数据分析平台

## 🛠️ 技术栈

- **LangChain**: v0.3 (最新版本)
- **LangGraph**: v0.6 (工作流编排)
- **LLM**: 智谱 AI GLM-4 模型
- **向量数据库**: FAISS, Chroma
- **环境**: Termux (Android) / Linux
- **Python**: 3.12+

## 📖 学习说明

### 兼容性更新
所有代码都已升级到 LangChain v0.3 标准，移除了已弃用的 API：
- ✅ 移除 `LLMChain` → 使用 LCEL 语法
- ✅ 移除 `SimpleSequentialChain` → 使用 `RunnableLambda`
- ✅ 移除 `DatetimeOutputParser` → 自定义实现
- ✅ 移除 `EnumOutputParser` → 自定义实现
- ✅ 修复 Pydantic v2 兼容性问题

### 代码特点
- 🎯 **实用导向**: 每个示例都解决实际问题
- 🔄 **逐步深入**: 从基础到高级的渐进式学习
- 🛡️ **错误处理**: 包含完善的异常处理机制
- 📝 **详细注释**: 代码中包含丰富的中文注释
- ✅ **测试验证**: 所有代码都经过实际运行测试

### 运行环境
```bash
# 配置环境变量
export ZHIPUAI_API_KEY="your_api_key_here"

# 安装依赖
pip install langchain langchain-community langchain-core

# 运行示例
python demos/01_basic_llm_usage.py
```

## 🎓 学习成果

通过本项目的学习，你已经掌握：

### 基础能力 🎯
- ✅ **LangChain 核心概念**：Model I/O、Chains、Memory、Data Connection
- ✅ **LLM 调用和配置**：智谱AI集成、参数调优、错误处理
- ✅ **提示工程和模板设计**：PromptTemplate、ChatPromptTemplate、动态模板
- ✅ **链式调用和数据流**：LCEL语法、并行处理、条件链
- ✅ **记忆管理和状态保持**：缓冲记忆、摘要记忆、持久化存储
- ✅ **输出解析和结果处理**：结构化输出、类型安全、错误恢复
- ✅ **数据处理能力**：文档加载、文本分割、向量存储、检索系统

### 高级能力
- ✅ Agent 系统设计和实现
- ✅ 自定义工具开发
- ✅ 任务分解和规划
- ✅ 多步骤推理和决策
- ✅ 错误恢复和异常处理

### 实践技能
- ✅ 现代化 LangChain v0.3 API 使用
- ✅ 生产级代码编写规范
- ✅ 调试和问题解决能力
- ✅ 系统架构设计思维

## 🚀 下一步建议

### 📚 **推荐学习顺序**
基于当前的坚实基础，建议按以下顺序继续学习：

1. **🔄 完善RAG系统** - 深化检索增强生成能力
   - 多源数据集成
   - 混合检索策略
   - 重排序和过滤

2. **🤖 Agent系统开发** - 构建智能决策代理
   - ReAct 模式实现
   - 工具调用机制
   - 多步骤规划

3. **🔗 RAG + Agent 集成** - 结合知识库和智能决策
4. **🌐 Multi-Agent 系统** - 构建协作的智能体网络
5. **🚀 生产环境部署** - 将应用部署到实际环境
6. **🎯 特定领域应用** - 开发专业的 AI 解决方案

### 💡 **学习技巧**
- 🔍 **按序学习**：从08到12掌握核心概念
- 🛠️ **动手实践**：每个文件都可独立运行测试
- 🤝 **组合使用**：尝试将不同组件组合使用
- 📝 **记录总结**：建立自己的知识体系

---

---

## 📊 **项目统计**

### 📈 **代码覆盖**
- **基础示例**: 12个完整示例文件
- **核心概念**: 6大模块全覆盖
- **代码行数**: 2000+ 行实用代码
- **注释率**: 80%+ 详细中文注释

### 🎯 **学习路径完整性**
- ✅ **Model I/O**: 01-02 (LLM调用)
- ✅ **Prompts**: 08 (提示模板)
- ✅ **Chains**: 09 (链式调用)
- ✅ **Memory**: 06, 11 (记忆管理)
- ✅ **Data Connection**: 10 (数据处理)
- ✅ **Agents**: 13-15 (智能代理)
- ✅ **Output Parsers**: 12 (输出解析)

---

*最后更新时间: 2025-10-14*
*项目状态: 🟢 活跃开发中*
*当前进度: ✅ 基础概念模块完成，🔄 开始RAG系统完善*
*下一个里程碑: 🤖 Agent系统开发* 
