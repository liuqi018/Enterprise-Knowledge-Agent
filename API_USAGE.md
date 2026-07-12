# Enterprise Knowledge Agent API

## Endpoints

- `GET /`: web chat UI.
- `GET /api/health`: health check.
- `GET /api/knowledge/stats`: vector backend, document count, data path, and manifest path.
- `POST /api/knowledge/ingest`: build or incrementally update the knowledge base.
- `POST /api/knowledge/ingest/async`: submit an async ingestion task.
- `GET /api/knowledge/tasks/{task_id}`: query async ingestion task status.
- `POST /api/rag/ask`: RAG-only question answering with sources.
- `POST /api/chat`: Agent question answering with session context.
- `POST /api/chat/stream`: streaming Agent/RAG question answering, returns NDJSON chunks.
- `DELETE /api/sessions/{session_id}`: clear a session.

## Examples

```powershell
curl -X POST http://127.0.0.1:8000/api/knowledge/ingest -H "Content-Type: application/json" -d "{\"force\":false}"
curl -X POST http://127.0.0.1:8000/api/knowledge/ingest/async -H "Content-Type: application/json" -d "{\"force\":false}"
curl -X POST http://127.0.0.1:8000/api/rag/ask -H "Content-Type: application/json" -d "{\"query\":\"病假超过一天需要什么证明？\"}"
curl -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" -d "{\"query\":\"帮我生成一个采购申请草稿\",\"use_agent\":true}"
```

## Knowledge Ingestion

The ingestion pipeline uses:

- Fast file filtering by `size + mtime`; MD5 is calculated only for suspected changed files.
- File-level MD5 manifest compatibility with older manifests.
- Chunk-level `chunk_hash` and stable `chunk_id`.
- Chunk-level delete/add for changed files, avoiding re-inserting unchanged chunks.
- Batch vector writes via `INGEST_BATCH_SIZE`.
- Concurrent file parsing via `INGEST_WORKERS`.
- Delete synchronization for files removed from `data/enterprise`.
- Async task submission for long-running ingestion jobs.
- Optional production task queue with Celery + Redis.

Useful tuning variables:

```powershell
$env:INGEST_BATCH_SIZE="64"
$env:INGEST_WORKERS="4"
```

## RAG Evaluation

The project includes an evaluation dataset and script:

- Dataset: `eval/retrieval_eval.csv`
- Script: `scripts/evaluate_rag.py`
- Outputs: `eval/results/rag_eval_details.csv` and `eval/results/rag_eval_summary.json`

After adding or changing documents, rebuild the knowledge base first, then run:

```powershell
python scripts/evaluate_rag.py
```

For retrieval-only evaluation without LLM answer generation:

```powershell
python scripts/evaluate_rag.py --skip-answer
```

Metrics include `Recall@1`, `Recall@3`, `Recall@5`, `MRR`, `nDCG@5`, keyword coverage, and latency.

## Production Task Queue

You can put environment variables in `.env`. Copy `.env.example` to `.env` and replace `DASHSCOPE_API_KEY`.

The default queue backend is in-process for local demos:

```powershell
$env:TASK_QUEUE_BACKEND="in_process"
```

For production-style deployment, use Celery with Redis as broker and result backend:

```powershell
$env:TASK_QUEUE_BACKEND="celery"
$env:REDIS_URL="redis://localhost:6379/0"
$env:CELERY_BROKER_URL="redis://localhost:6379/0"
$env:CELERY_RESULT_BACKEND="redis://localhost:6379/0"
```

Start Redis, then start a Celery worker from the parent directory of `AIRAGAgent`:

```powershell
celery -A AIRAGAgent.knowledge.tasks worker --pool=solo -l info
```

Then start FastAPI as usual:

```powershell
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## MySQL Auth And Tenant Isolation

The API now uses MySQL-backed users and JWT authentication. Configure MySQL in `.env`:

```powershell
MYSQL_URL=mysql+pymysql://root:password@localhost:3306/airag_agent?charset=utf8mb4
JWT_SECRET_KEY=replace_with_a_long_random_secret
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123456
ADMIN_TENANT_ID=default
```

Create the database before starting FastAPI:

```sql
CREATE DATABASE airag_agent CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

On startup, the service creates tables and seeds the admin user from `.env`.

Authenticated APIs require:

```text
Authorization: Bearer <access_token>
```

Tenant isolation is enforced by:

- User records carrying `tenant_id` for organization context.
- Conversation and message records carrying `user_id`, so users only see their own chat history.
- Knowledge task records carrying `user_id`, so users only query their own submitted tasks.
- Knowledge ingestion APIs requiring `admin` role.
- Enterprise policy knowledge base remaining shared at company level by default.

This matches the current product model: company policies are shared knowledge, while chat history is private per user.

## Vector Backend

The default backend is Chroma for local development. Use Milvus by setting:

```powershell
$env:VECTOR_BACKEND="milvus"
$env:MILVUS_URI="http://localhost:19530"
$env:MILVUS_COLLECTION="enterprise_policy"
```

## Retrieval Pipeline

The RAG pipeline uses intent-aware dynamic retrieval:

- Intent classification: `simple`, `professional_policy`, or `complex_process`.
- Hybrid Query Rewrite: simple questions keep the original query, professional policy questions use rule-based rewrite, and complex process questions use LLM multi-query rewrite.
- Dynamic recall: simple questions use `query -> Vector + BM25`; professional policy questions use `Query Rewrite -> Metadata Filter -> Vector + BM25`; complex process questions use `Multi Query -> Vector + BM25`.
- Metadata Filter: policy-domain metadata is inferred during ingestion and used before retrieval for professional policy queries.
- Weighted RRF: vector and BM25 results are fused with configurable route weights.
- Light filter: low-signal RRF candidates are trimmed before reranking.
- Rerank: candidates are sent to DashScope Rerank or a Cross Encoder; if the external reranker is unavailable, the system keeps weighted-RRF order instead of applying rule rerank.
- Context Compression: final top documents are compressed sentence-level before being passed to the LLM.

Useful tuning variables:

```powershell
$env:VECTOR_RECALL_TOP_K="5"
$env:BM25_RECALL_TOP_K="5"
$env:VECTOR_RRF_WEIGHT="1.0"
$env:BM25_RRF_WEIGHT="0.85"
$env:LIGHT_FILTER_TOP_N="12"
$env:RERANK_TOP_K="3"
```

Enable DashScope rerank:

```powershell
$env:RERANK_PROVIDER="dashscope"
$env:DASHSCOPE_RERANK_MODEL="gte-rerank"
```

Use Cross Encoder rerank:

```powershell
$env:RERANK_PROVIDER="cross_encoder"
$env:CROSS_ENCODER_MODEL="BAAI/bge-reranker-base"
```

After this metadata upgrade, run a force ingestion once so existing chunks include `policy_domain`:

```powershell
curl -X POST http://127.0.0.1:8000/api/knowledge/ingest -H "Content-Type: application/json" -d "{\"force\":true}"
```
