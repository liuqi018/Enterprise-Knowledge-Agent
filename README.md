# 制造企业制度知识库与流程合规 Agent

面向制造企业内部制度查询、流程审批与合规检查场景的 RAG + Agent 应用。系统基于企业制度文档构建知识库，支持制度问答、流程查询、审批路径判断、申请草稿生成、敏感制度权限控制、多轮会话记忆和离线效果评测。

项目不是泛化闲聊机器人，而是聚焦制造企业内部制度和流程辅助：用户可以查询制度依据，系统在资料明确时给出可追溯回答，在资料不足时明确说明证据边界，避免无依据补全。

## 业务场景

假设企业是一家中小型智能制造或设备制造公司，包含研发、生产、质量、采购、仓储、财务、人事、IT 服务、信息安全等部门。

典型问题包括：

- 采购金额超过 5 万元需要经过哪些审批？
- 生产异常发生后应该如何上报和处理？
- 出库发料审批不完整时，仓库能不能先发料再补签？
- 没有质量检验记录，产品能否先出厂再补检？
- 员工借款后如何进行报销冲账？
- IT 变更没有评审，能否先上线再补流程？
- 普通员工能否查询其他人的工资明细？

## 核心能力

### 制度文档治理

项目提供文档解析、清洗、分类和审计隔离流程，用于降低企业原始文档中的噪声。

- 支持 `.docx`、`.doc`、`.pdf`、`.txt` 等格式解析。
- 清洗页眉页脚、重复行、低价值表格片段等内容。
- 按业务域组织文档，例如采购、生产、质量、财务、报销、人事、研发、IT 服务、信息安全等。
- 审计并隔离低价值或业务场景不匹配的文档，例如模板、总结、案例、门店类文档等。
- 支持强制重建索引，自动清理旧分片并重新写入向量库和 BM25 语料。

### 混合检索 RAG

系统使用 Milvus 向量检索 + BM25 关键词检索构建混合召回链路，并通过 RRF 融合多路结果。

- 向量检索用于语义召回。
- BM25 用于制度条款、金额、流程节点、岗位名称等关键词召回。
- RRF 融合向量与关键词结果，降低单一路径漏召回风险。
- Metadata 过滤用于按业务域收敛检索范围。
- 对采购、入职、请假、报销、信息安全等问题做业务域路由，降低模板、合同、表单对制度问答的干扰。

### 流程 Agent

除普通制度问答外，项目基于 LangGraph 编排流程类任务，节点包括意图识别、任务规划、工具执行、风险复核和最终回答。

已封装的工具能力包括：

- 制度依据汇总
- 制度条款定位
- 审批路径查询
- 审批风险判断
- 申请草稿生成

适合处理单轮 RAG 难以覆盖的流程类问题，例如“采购电脑需要生成申请草稿时应包含哪些信息？”、“出库审批不完整是否可以发料？”、“研发代码没有评审记录能否直接发布？”。

### 可信回答与兜底

项目重点控制三类风险：无依据补全、结构化输出不稳定、工具调用失败。

- 制度问答 Prompt 要求回答基于检索资料，并区分“资料明确”和“资料未明确”。
- 对“先执行再补流程”“是否允许”“是否完整覆盖”等问题设置更严格的证据边界。
- JSON Guard 支持从 Markdown、解释文本和嵌套对象中提取合法 JSON。
- 工具调用前进行参数校验、非法工具过滤和类型修正。
- 工具调用失败时返回保守提示或最小可用草稿，避免请求直接中断。
- 大模型调用失败时保存消息状态，支持后续重试。

### 多轮会话与权限控制

- 支持上下文问题改写，处理“那这个呢？”、“谁负责？”这类追问。
- 支持会话摘要持久化，避免摘要只存在内存中。
- 按 `session_id` 隔离不同会话，避免新会话误用旧上下文。
- 基于角色和制度域实现轻量权限控制。
- 普通用户对财务、薪酬绩效、信息安全等敏感制度域默认只能访问泛化规则，敏感明细返回友好无权限提示。

### 可观测性

系统增加 `trace_id` 链路日志，记录关键链路事件：

- API 请求与响应
- 上下文解析
- 路由决策
- 权限判断
- 向量召回
- BM25 召回
- RRF 融合
- 回答生成
- 失败重试

这些日志用于定位“为什么检索错了、为什么拒答、为什么触发权限限制”等问题。

## 技术栈

- 后端：FastAPI
- 前端：HTML / CSS / JavaScript
- 编排：LangChain / LangGraph
- 向量库：Milvus，兼容 Chroma
- 关键词检索：BM25
- 数据库：MySQL
- 缓存与任务：Redis / Celery 可选
- 大模型：OpenAI-compatible API，例如 AutoDL API
- Embedding：DashScope Embedding
- 部署：Docker Compose

## 目录结构

```text
agent/       Agent 工作流与工具调用
api/         FastAPI 路由
config/      配置文件
frontend/    前端页面
knowledge/   文档加载、清洗、切分、入库
model/       LLM 与 Embedding 工厂
rag/         RAG 检索与回答服务
scripts/     文档治理、索引构建、评测脚本
services/    聊天、会话、权限、上下文服务
utils/       日志、路径、JSON 解析等工具
eval/        离线评测集与评测结果
```

## 环境配置

复制环境变量模板：

```powershell
copy .env.example .env
```

如果使用 AutoDL OpenAI-compatible API：

```env
LLM_PROVIDER=autodl
AUTODL_API_KEY=your_autodl_api_key
AUTODL_MODEL=gpt-5.5
AUTODL_BASE_URL=https://www.autodl.art/api/v1
AUTODL_TIKTOKEN_MODEL=gpt-4o
```

Embedding 默认使用 DashScope：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
```

Milvus 配置示例：

```env
VECTOR_BACKEND=milvus
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=enterprise_policy
```

## 本地启动

安装依赖：

```powershell
cd D:\PyCharm\pythonProject\PythonProject
pip install -r AIRAGAgent\requirements.txt
```

启动 FastAPI：

```powershell
cd D:\PyCharm\pythonProject\PythonProject\AIRAGAgent
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

访问前端：

```text
http://127.0.0.1:8000
```

## Docker Compose

Docker Compose 用于本地复现和部署，包含：

- FastAPI app
- MySQL
- Redis
- Milvus standalone
- Etcd
- MinIO

创建 Docker 环境变量文件：

```powershell
copy .env.docker.example .env.docker
```

填写必要 API Key：

```env
AUTODL_API_KEY=your_autodl_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key
JWT_SECRET_KEY=replace_with_a_long_random_secret
```

启动服务：

```powershell
docker compose --env-file .env.docker up -d --build
```

查看日志：

```powershell
docker compose --env-file .env.docker logs -f app
```

停止服务：

```powershell
docker compose --env-file .env.docker down
```

## 知识库构建

强制重建知识库：

```powershell
cd D:\PyCharm\pythonProject\PythonProject
python -c "from AIRAGAgent.knowledge.service import KnowledgeBaseService; print(KnowledgeBaseService().ingest(force=True))"
```

也可以在前端使用管理员账号触发“强制重建索引”。强制重建会删除旧分片并重新入库，不需要手动清空 Milvus。

## 文档治理脚本

自动分类 `data/enterprise` 下的文档：

```powershell
python scripts\classify_enterprise_docs.py --data-dir data\enterprise --apply --report
```

从原始资料中筛选补充制度文档：

```powershell
python scripts\import_enterprise_supplements.py --data-root data --enterprise-root data\enterprise --apply --report
```

审计并隔离低价值或场景不匹配文档：

```powershell
python scripts\audit_enterprise_docs.py --data-dir data\enterprise --apply
```

如果存在旧版 `.doc` 文件，建议先用 LibreOffice 批量转换为 `.docx`：

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

```text
生产异常发生后应该如何上报和处理？
出库发料审批不完整时，仓库能不能先发料再补签？
没有质量检验记录，产品能否先出厂再补检？
采购金额超过 5 万元需要哪些审批？
员工借款后如何进行冲账？
研发代码没有评审记录，能否直接发布？
IT 变更没有评审，能否先上线再补流程？
普通员工能否查询所有人的工资明细和奖金计算方式？
```

## 权限控制

权限配置文件：

```text
config/access_control.yml
```

核心策略：

- `allow`：允许访问该制度域的全部内容。
- `general_only`：允许查询泛化制度、流程和要求，但限制金额、明细、名单、账号、密码等敏感细节。
- `deny`：限制访问该制度域。

默认情况下：

- `admin` 可访问全部内容。
- 普通 `user` 对财务、薪酬绩效、信息安全等制度域采用 `general_only` 策略。
- 命中敏感明细限制时，返回友好无权限提示。

快速验证：

```powershell
python scripts\check_access_control.py --role user
python scripts\check_access_control.py --role admin
python scripts\check_access_control.py --role finance --query "财务报销金额标准是多少？"
```

## 效果评测

项目提供两套离线评测集：

- 基础集：`eval/manufacturing_rag_eval_160.jsonl`，共 160 条，覆盖常规制度问答、流程查询、风险判断和 Agent 工具类问题。
- 困难集：`eval/manufacturing_rag_hard_eval_100.jsonl`，共 100 条，覆盖跨域干扰、部分证据、无依据拒答、多跳流程、模糊问题、否定约束、敏感权限和角色冲突。

### 评测指标说明

当前评测中的检索指标基于预期业务域、预期来源关键词和预期回答关键词统计，属于关键词弱监督评测，不等同于人工标注 golden chunk 的严格 Recall。

主要指标：

- `hit@k / recall@k`：脚本内部字段，表示 Top-K 检索结果是否命中预期证据关键词。README 中统一称为 Top-K 预期证据命中率。
- `source_hit@3`：Top-3 结果是否命中预期来源关键词。
- `domain_hit@3`：Top-3 结果是否命中预期业务域。
- `mrr`：首个命中预期证据的排名质量。
- `ndcg@5`：Top-5 排序质量。
- `answer_keyword_coverage`：回答中覆盖预期关键词的比例，仅作为辅助诊断。
- `judge_pass_rate`：LLM-as-Judge 对最终回答的通过率，综合评估准确性、忠实性、完整性、证据支撑和可用性。
- `no_answer_pass_rate`：无明确依据问题是否正确说明证据不足，而不是编造制度。
- `failure_analysis`：失败案例自动归因，包括检索失败、关键词误判、回答不完整、领域标注偏差、无依据拒答失败和 Judge 失败等。

### 最新评测结果

基础集结果，`160` 条，Milvus，`max-k=5`，启用 LLM-as-Judge：

| 指标 | 结果 |
| --- | ---: |
| Top-3 预期证据命中率 | 99.38% |
| Source Hit@3 | 98.12% |
| Domain Hit@3 | 91.87% |
| MRR | 98.44% |
| NDCG@5 | 98.34% |
| LLM-as-Judge 通过率 | 91.87% |
| Judge Accuracy | 4.58 / 5 |
| Judge Faithfulness | 4.68 / 5 |
| Judge Groundedness | 4.70 / 5 |

困难集结果，`100` 条，Milvus，`max-k=5`，启用 LLM-as-Judge：

| 指标 | 结果 |
| --- | ---: |
| Retrieval Cases | 88 |
| No-Answer Cases | 12 |
| Top-3 预期证据命中率 | 100.00% |
| Source Hit@3 | 100.00% |
| Domain Hit@3 | 85.23% |
| No-Answer Pass Rate | 100.00% |
| LLM-as-Judge 通过率 | 87.00% |
| Judge Accuracy | 4.51 / 5 |
| Judge Faithfulness | 4.65 / 5 |
| Judge Groundedness | 4.64 / 5 |

### 运行评测

基础集全量评测：

```powershell
D:\Anaconda\envs\AIRAGAgent\python.exe scripts\evaluate_manufacturing_rag.py --dataset eval/manufacturing_rag_eval_160.jsonl --max-k 5 --judge-answer
```

困难集全量评测：

```powershell
D:\Anaconda\envs\AIRAGAgent\python.exe scripts\evaluate_manufacturing_rag.py --dataset eval/manufacturing_rag_hard_eval_100.jsonl --max-k 5 --judge-answer
```

只评测检索，不生成答案：

```powershell
python scripts\evaluate_manufacturing_rag.py --dataset eval/manufacturing_rag_eval_160.jsonl --max-k 5 --skip-answer
```

小样本抽检：

```powershell
python scripts\evaluate_manufacturing_rag.py --dataset eval/manufacturing_rag_hard_eval_100.jsonl --max-k 5 --judge-answer --limit 10
```

评测脚本会输出 JSON 和 Markdown 报告到：

```text
eval/results/
```

## 项目亮点

- 制度文档治理与混合召回：通过文档清洗、业务域分类、低价值文档隔离、Milvus + BM25 + RRF 融合，提升复杂制度问题下的证据召回稳定性。
- 流程任务编排与工具化执行：基于 LangGraph 将制度检索、审批路径、风险判断和草稿生成串联为流程辅助能力。
- 可信回答与异常兜底：通过 Prompt 约束、JSON Guard、工具参数校验、失败降级和消息重试，降低无依据回答和请求中断风险。
- 多轮会话与评测体系：支持上下文改写、会话摘要持久化、角色权限控制，并构建基础集 + 困难集评测体系。

## 注意事项

- `data/enterprise` 中的原始企业文档不建议提交到 Git，避免上传本地资料。
- 强制重建索引前建议先做文档治理，尤其是坏文件、旧版 `.doc` 和低价值模板。
- 如果重建过程中出现 DashScope SSL 或网络错误，通常是外部服务或网络波动，建议停止任务后重试。
- 如果检索指标很高但 Judge 通过率不高，优先查看失败案例诊断，区分是回答不完整、过度拒答、领域标签偏差还是证据不足。
