# Enterprise Knowledge Agent

企业制度知识库智能助手，基于 LangChain Agent、Chroma、Streamlit 和通义千问构建。项目从原有 AIRAGAgent 改造而来，用模拟企业制度文档验证企业内部知识库问答、流程指引、申请草稿和咨询月报生成链路。

## 核心能力

- 制度问答：围绕报销、请假、采购、入职、信息安全、客户工单 SOP 等制度进行 RAG 问答。
- 知识库管理：支持将 `data/enterprise` 下的 TXT/PDF 文档切分、向量化、增量写入 Chroma，并通过 MD5 去重。
- Agent 工具调用：支持获取员工 ID、部门、当前月份、咨询统计记录，并生成申请草稿。
- 动态提示词：普通问答使用制度助手 prompt，咨询月报场景通过中间件切换到报告 prompt。
- 多轮会话：Streamlit 页面保存历史消息，并传入 Agent 维持上下文。

## 目录说明

- `app.py`：Streamlit 入口。
- `agent/`：React Agent、工具和中间件。
- `rag/`：向量库加载和 RAG 总结服务。
- `model/`：通义千问聊天模型和 embedding 模型工厂。
- `config/`：模型、Chroma、prompt 和外部数据配置。
- `data/enterprise/`：模拟企业制度知识库资料。
- `data/external/consultation_records.csv`：模拟员工制度咨询统计数据。
- `prompts/`：主 prompt、RAG 总结 prompt、报告 prompt。

## 启动方式

1. 安装依赖：

```powershell
pip install -r AIRAGAgent/requirements.txt
```

2. 配置通义千问 API Key：

```powershell
$env:DASHSCOPE_API_KEY="你的API Key"
```

3. 启动应用：

```powershell
streamlit run AIRAGAgent/app.py
```

4. 首次进入页面后，点击侧边栏“加载/更新制度知识库”，将制度文档写入 Chroma。

## 可演示问题

- 出差住宿报销标准是多少？
- 病假超过一天需要什么证明？
- 采购一台 8000 元的电脑需要哪些审批？
- 生产数据库权限可以直接申请长期权限吗？
- 帮我生成一个云资源采购申请草稿。
- 给我生成本月制度咨询报告。

## 简历描述参考

基于 LangChain、Chroma、Streamlit 构建企业制度知识库 RAG Agent，支持制度文档增量入库、语义检索、工具调用、动态 Prompt 切换、申请草稿生成和咨询月报分析。通过模拟企业制度文档和咨询统计数据，验证企业内部知识助手在制度问答、流程指引和可追溯回答场景中的完整技术链路。
