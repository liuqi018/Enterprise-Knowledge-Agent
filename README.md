# 制造企业制度知识库与流程合规 Agent

面向中小型制造企业内部制度和流程管理场景的 RAG + Agent 应用。系统基于企业制度文档构建知识库，支持制度问答、流程查询、审批路径判断、申请草稿生成和合规风险提示，覆盖采购、生产、质量、财务、人事、研发、IT 服务、信息安全等业务域。

当前项目定位不是泛泛的“企业问答机器人”，而是一个更具体的制造企业内部流程助手：员工可以查询制度依据，管理人员可以快速判断流程是否合规，系统可以在有依据时给出处置建议，在依据不足时明确提示风险。

## 项目场景

假设企业是一家中小型智能制造或设备制造公司，主营工业设备、电子设备或自动化设备的研发、生产和销售。公司内部包含研发部、生产部、质量部、采购部、仓储部、财务部、人力资源部、IT 服务部等组织。

典型业务问题包括：

- 采购金额超过 5 万元需要经过哪些审批？
- 生产异常发生后应该如何上报和处理？
- 设备维修、物料领用、产品入库分别需要走什么流程？
- 质量事故发生后如何处理和追责？
- 财务付款、费用报销、借款冲账需要准备哪些材料？
- 研发项目立项、结项、配置管理需要遵守哪些要求？
- IT 变更发布、访问权限申请、信息备份应该如何管理？

## 核心能力

### 1. 制度问答

系统对用户问题进行意图识别和业务域路由，从知识库中检索相关制度片段，并生成带依据的回答。回答会尽量引用来源文档、章节信息和制度域，避免脱离知识库编造。

### 2. 混合检索 RAG

项目支持向量检索 + BM25 关键词检索，并使用 RRF 融合召回结果。对于采购、入职、请假、报销等问题，会结合业务域元数据进行过滤和排序，降低模板、合同、表单类文档对制度问答的干扰。

### 3. Agent 流程能力

除了普通问答，系统还支持流程类任务：

- 查询审批路径
- 识别金额并判断审批风险
- 汇总制度依据
- 生成申请草稿
- 给出合规风险提示

例如用户输入“我想采购一台 5 万元的办公电脑，帮我生成申请草稿”，系统会识别采购场景和金额，查询相关制度，生成申请内容并提示审批风险。

### 4. 知识库治理

项目包含文档清洗、分类和补充导入脚本，用于降低企业文档中的噪声：

- 自动清洗页眉、页脚、重复行、低价值表格片段
- 自动识别文档类型，如制度、合同模板、表单模板、岗位职责、培训材料
- 按业务域组织知识库目录，如财务、生产、质量、研发、IT 服务等
- 从大规模原始文档中筛选高价值制度文件，排除行业案例、台账、表单和临时文件
- 支持坏文件隔离和老格式 `.doc` 转 `.docx` 的治理流程

## 知识库结构

当前推荐的知识库目录位于 `data/enterprise`。该目录不提交到 Git，避免上传原始文档和本地数据。

主要结构如下：

```text
data/enterprise/
  policies/
    procurement/          采购制度
    production/           生产管理
    quality/              质量管理
    warehouse/            仓储库存
    finance/              财务制度
    reimbursement/        报销差旅
    hr/                   人事管理
    onboarding/           入职转正离职
    leave_attendance/     请假考勤
    salary_performance/   薪酬绩效
    research/             研发管理
    it_service/           IT 服务管理
    security/             信息安全
    administration/       行政办公
    sales/                销售客户
    general/              综合制度
  templates/
    contracts/            合同协议模板
    forms/                表单申请模板
  references/
    job_descriptions/     岗位职责
    training_materials/   培训手册
    org_structure/        组织架构
    meeting_reports/      汇报总结
    culture_team/         企业文化团建
    general/              其他参考资料
```

## 技术架构

- 后端：FastAPI
- 前端：原生 HTML/CSS/JavaScript，支持流式对话
- 大模型：AutoDL OpenAI-compatible API 或 DashScope
- Embedding：DashScope Embeddings
- 向量库：Milvus 或 Chroma
- 关键词检索：BM25 本地语料缓存
- 编排框架：LangChain / LangGraph
- 数据库：MySQL，用于用户、会话和消息记录
- 异步任务：支持 Celery + Redis 或进程内任务
- 文档解析：docx、doc、pdf、txt

## 目录说明

```text
agent/       Agent 工作流和工具调用
api/         FastAPI 路由
config/      全局配置
frontend/    前端页面
knowledge/   文档加载、清洗、切分、入库服务
model/       LLM 和 Embedding 工厂
prompts/     RAG、路由和回答 Prompt
rag/         RAG 检索与回答服务
scripts/     文档分类、补充导入、评估和修复脚本
services/    聊天、鉴权、会话和路由服务
utils/       日志、路径、金额解析等工具
```

## 环境配置

复制 `.env.example` 为 `.env`，然后按需修改。

```powershell
copy .env.example .env
```

如果使用 AutoDL 的 OpenAI-compatible API：

```env
LLM_PROVIDER=autodl
AUTODL_API_KEY=your_autodl_api_key
AUTODL_MODEL=gpt-5.5
AUTODL_BASE_URL=https://www.autodl.art/api/v1
AUTODL_TIKTOKEN_MODEL=gpt-4o
```

Embedding 和 Rerank 默认使用 DashScope，需要配置：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
```

向量库可选择 Milvus 或 Chroma：

```env
VECTOR_BACKEND=milvus
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=enterprise_policy
```

或者：

```env
VECTOR_BACKEND=chroma
CHROMA_PERSIST_DIR=chroma_db/enterprise
```

## 启动方式

在项目父目录执行依赖安装：

```powershell
cd D:\PyCharm\pythonProject\PythonProject
pip install -r AIRAGAgent\requirements.txt
```

启动后端：

```powershell
cd D:\PyCharm\pythonProject\PythonProject\AIRAGAgent
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

浏览器访问：

```text
http://127.0.0.1:8000
```

默认管理员账号由 `.env` 控制：

```env
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123456
```


## Docker Compose

Docker Compose is for local deployment and interview reproduction. It is independent from the evaluation dataset.
It starts the FastAPI app and infrastructure services together:

- FastAPI app
- MySQL
- Redis
- Milvus standalone
- Etcd and MinIO, required by Milvus

Create a Docker environment file first:

```powershell
copy .env.docker.example .env.docker
```

Then edit `.env.docker` and fill in your API keys, especially:

```env
AUTODL_API_KEY=your_autodl_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key
JWT_SECRET_KEY=replace_with_a_long_random_secret
```

Start all services:

```powershell
docker compose --env-file .env.docker up -d --build
```

Open the frontend:

```text
http://127.0.0.1:8000
```

View logs:

```powershell
docker compose --env-file .env.docker logs -f app
```

Stop services:

```powershell
docker compose --env-file .env.docker down
```

Remove service data volumes when you need a clean environment:

```powershell
docker compose --env-file .env.docker down -v
```

In Docker Compose, service addresses are different from local Windows addresses:

```env
MYSQL_URL=mysql+pymysql://root:password@mysql:3306/airag_agent?charset=utf8mb4
REDIS_URL=redis://redis:6379/0
MILVUS_URI=http://milvus:19530
VECTOR_BACKEND=milvus
```

## 构建知识库

首次运行或文档变更后，需要强制重建索引。

推荐命令行方式：

```powershell
cd D:\PyCharm\pythonProject\PythonProject
python -c "from AIRAGAgent.knowledge.service import KnowledgeBaseService; print(KnowledgeBaseService().ingest(force=True))"
```

也可以在前端使用管理员账号触发“强制重建索引”。如果文档较多，命令行方式更便于观察错误信息。

## 文档治理脚本

自动分类 `data/enterprise` 根目录文档：

```powershell
python scripts\classify_enterprise_docs.py --data-dir data\enterprise --apply --report
```

从 `data/01-10` 原始资料中精选补充高价值制度文件：

```powershell
python scripts\import_enterprise_supplements.py --data-root data --enterprise-root data\enterprise --apply --report
```

如果存在旧版 `.doc` 文件，建议先用 LibreOffice 批量转换为 `.docx`，再重建索引。

```powershell
Get-ChildItem -Path data\enterprise -Recurse -Filter *.doc | ForEach-Object {
    & "D:\LenovoSoftstore\LibreOffice\program\soffice.com" --headless --convert-to docx --outdir $_.DirectoryName $_.FullName
}
```

确认转换成功后再删除原 `.doc`：

```powershell
Get-ChildItem -Path data\enterprise -Recurse -Filter *.doc | ForEach-Object {
    $docx = [System.IO.Path]::ChangeExtension($_.FullName, ".docx")
    if (Test-Path $docx) {
        Remove-Item $_.FullName
    } else {
        Write-Host "未转换成功，保留原文件：" $_.FullName
    }
}
```

## API 示例

健康检查：

```http
GET /api/health
```

登录：

```http
POST /api/auth/login
```

知识库统计：

```http
GET /api/knowledge/stats
```

强制重建知识库：

```http
POST /api/knowledge/ingest
Content-Type: application/json

{"force": true}
```

流式对话：

```http
POST /api/chat/stream
Content-Type: application/json

{
  "query": "生产异常发生后应该如何处理？",
  "session_id": "demo-session",
  "use_agent": true,
  "history": []
}
```

## Demo 问题

推荐用下面几类问题测试系统效果：

```text
生产异常发生后应该如何上报和处理？
设备维修需要走什么流程？
原材料入库和领料需要遵守哪些要求？
质量事故发生后应该怎么处理？
采购金额超过 5 万元需要哪些审批？
财务付款审批一般需要经过哪些流程？
员工费用报销需要准备哪些材料？
研发项目立项需要走哪些流程？
IT 变更发布需要经过哪些步骤？
信息系统用户访问权限如何申请和管理？
```


## 权限控制

项目提供轻量级、配置化的敏感制度域访问控制，配置文件位于：

```text
config/access_control.yml
```

核心策略：
- `allow`：允许访问该制度域的所有问题。
- `general_only`：允许查询泛化制度、流程和要求，但限制金额、明细、名单、账号、密码等敏感细节。
- `deny`：限制访问该制度域。

默认配置中，`admin` 可访问全部内容；普通 `user` 对财务、薪酬绩效、信息安全域采用 `general_only` 策略。可以通过 `ACCESS_CONTROL_CONFIG_PATH` 指定自定义配置文件。

快速验证：

```powershell
python scripts\check_access_control.py --role user
python scripts\check_access_control.py --role admin
python scripts\check_access_control.py --role finance --query "财务报销金额标准是多少？"
```

## 效果评测

项目提供一套 160 条制造企业制度问答评测集，覆盖生产、质量、财务、报销、采购、仓储、人事、考勤、薪酬绩效、研发、IT 服务和信息安全等业务域。

评测集位置：

```text
eval/manufacturing_rag_eval_160.jsonl
```

重新生成评测集：

```powershell
python scripts\generate_manufacturing_eval_160.py
```

只评测检索效果：

```powershell
python scripts\evaluate_manufacturing_rag.py --skip-answer --max-k 5
```

小样本快速验证：

```powershell
python scripts\evaluate_manufacturing_rag.py --skip-answer --limit 20 --max-k 5
```

评测脚本会输出 JSON 和 Markdown 报告到：

```text
eval/results/
```

最新 160 条全量评测结果（Milvus，`max-k=5`，启用 LLM-as-Judge 严格回答评测）：

| 指标 | 结果 |
| --- | ---: |
| Recall@3 | 99.38% |
| Recall@5 | 99.38% |
| MRR | 98.44% |
| NDCG@5 | 98.34% |
| Source Hit@3 | 98.12% |
| Domain Hit@3 | 91.87% |
| LLM Judge Pass Rate | 91.25% |
| Judge Accuracy | 4.59 / 5 |
| Judge Faithfulness | 4.69 / 5 |
| Judge Groundedness | 4.70 / 5 |

严格回答准确率评测：

```powershell
python scripts\evaluate_manufacturing_rag.py --max-k 5 --judge-answer
```

主要指标包括：

- `recall@1 / recall@3 / recall@5`：Top-K 召回是否命中期望业务域或来源关键词，等价于报告中的 `hit@1 / hit@3 / hit@5`
- `domain_hit@3`：Top-3 中是否命中期望业务域
- `source_hit@3`：Top-3 中是否命中期望来源关键词
- `mrr`：首个相关结果排名质量
- `ndcg@5`：Top-5 排序质量
- `answer_keyword_coverage`：回答关键词覆盖率，非 `--skip-answer` 模式下统计
- `judge_pass_rate`：LLM-as-Judge 严格回答通过率，可作为最终回答准确率参考
- `failure_analysis`：失败案例自动归因，包括检索失败、关键词误判、回答不完整、领域标注偏差和 Judge 输出异常等类别

## 项目亮点

- 将项目场景收敛到制造企业，而不是泛化问答，业务边界更清晰
- 支持 RAG 检索、业务域路由、Agent 工具调用和流式回答
- 通过文档类型和业务域元数据降低合同、表单、模板对制度问答的干扰
- 对采购、财务、生产、质量、研发、IT 服务等企业流程有较完整覆盖
- 支持知识库增量更新、强制重建、BM25 缓存和 Milvus/Chroma 切换
- 前端支持多轮会话、历史记录和回答流式输出

## 简历描述参考

制造企业制度知识库与流程合规 Agent：面向中小型制造企业内部制度与流程管理场景，基于 FastAPI、LangChain、Milvus、BM25 和 OpenAI-compatible 大模型 API 构建 RAG + Agent 应用，支持制度问答、流程查询、审批路径判断、申请草稿生成和合规风险提示。项目通过文档清洗、业务域分类、混合检索、RRF 融合和元数据过滤提升企业制度检索质量，覆盖采购、生产、质量、财务、人事、研发、IT 服务和信息安全等业务域。

量化效果：基于 160 条制造业企业制度问答评测集进行离线评估，检索 Recall@3 达 99.38%，MRR 达 98.44%，NDCG@5 达 98.34%；引入 LLM-as-Judge 严格评测最终回答质量，回答通过率达 91.25%，平均准确性 4.59/5，平均忠实性 4.69/5，平均证据支撑 4.70/5。

## 注意事项

- `.env`、`data/`、向量库缓存、BM25 缓存和日志文件不应提交到 Git。
- 原始企业文档可能包含损坏文件、旧版 `.doc`、重复制度和低价值表单，重建索引前建议先做文档治理。
- 如果强制重建时长时间无进展，优先检查坏文件和 Word/LibreOffice 转换结果，不要直接删除 Milvus。
