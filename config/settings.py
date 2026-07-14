"""
Global settings for the enterprise knowledge agent platform.
"""
import os

from dotenv import load_dotenv

load_dotenv(override=True)

dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
if dashscope_api_key:
    os.environ["DASHSCOPE_API_KEY"] = dashscope_api_key.strip().strip('"').strip("'")

class Settings:
    PROJECT_NAME = "Enterprise AI Agent Platform"
    VERSION = "1.0.0"

    VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma").lower()  # chroma | milvus
    MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
    MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "enterprise_policy")
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "enterprise_policy")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db/enterprise")
    KNOWLEDGE_DATA_PATH = os.getenv("KNOWLEDGE_DATA_PATH", "data/enterprise")
    KNOWLEDGE_MANIFEST_PATH = os.getenv("KNOWLEDGE_MANIFEST_PATH", "knowledge_manifest.json")
    BM25_CORPUS_PATH = os.getenv("BM25_CORPUS_PATH", "bm25_corpus.jsonl")
    CHUNKING_VERSION = os.getenv("CHUNKING_VERSION", "section_v1")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
    INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "64"))
    INGEST_WORKERS = int(os.getenv("INGEST_WORKERS", "4"))
    TASK_QUEUE_BACKEND = os.getenv("TASK_QUEUE_BACKEND", "in_process").lower()  # in_process | celery
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
    MYSQL_URL = os.getenv(
        "MYSQL_URL",
        "mysql+pymysql://root:password@localhost:3306/airag_agent?charset=utf8mb4",
    )
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "720"))
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123456")
    ADMIN_TENANT_ID = os.getenv("ADMIN_TENANT_ID", "default")

    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    VECTOR_RECALL_TOP_K = int(os.getenv("VECTOR_RECALL_TOP_K", "5"))
    BM25_RECALL_TOP_K = int(os.getenv("BM25_RECALL_TOP_K", "5"))
    RRF_K = int(os.getenv("RRF_K", "60"))
    VECTOR_RRF_WEIGHT = float(os.getenv("VECTOR_RRF_WEIGHT", "1.0"))
    BM25_RRF_WEIGHT = float(os.getenv("BM25_RRF_WEIGHT", "0.85"))
    LIGHT_FILTER_TOP_N = int(os.getenv("LIGHT_FILTER_TOP_N", "12"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
    RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "dashscope").lower()  # dashscope | cross_encoder | none
    ENABLE_COMPLEX_RERANK = os.getenv("ENABLE_COMPLEX_RERANK", "false").lower() == "true"
    SEMANTIC_ROUTER_ENABLED = os.getenv("SEMANTIC_ROUTER_ENABLED", "true").lower() == "true"
    SEMANTIC_ROUTER_THRESHOLD = float(os.getenv("SEMANTIC_ROUTER_THRESHOLD", "0.72"))
    CLARIFY_ROUTE_ENABLED = os.getenv("CLARIFY_ROUTE_ENABLED", "true").lower() == "true"
    CLARIFY_ROUTE_THRESHOLD = float(os.getenv("CLARIFY_ROUTE_THRESHOLD", "0.70"))
    ROUTER_LLM_FALLBACK_ENABLED = os.getenv("ROUTER_LLM_FALLBACK_ENABLED", "true").lower() == "true"
    AGENT_LLM_PLANNER_ENABLED = os.getenv("AGENT_LLM_PLANNER_ENABLED", "false").lower() == "true"
    DASHSCOPE_RERANK_MODEL = os.getenv("DASHSCOPE_RERANK_MODEL", "gte-rerank")
    CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base")
    COMPRESSED_CONTEXT_CHARS_PER_DOC = int(os.getenv("COMPRESSED_CONTEXT_CHARS_PER_DOC", "900"))
    MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
    MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
    CONTEXT_REWRITE_ENABLED = os.getenv("CONTEXT_REWRITE_ENABLED", "true").lower() == "true"
    CONTEXT_SUMMARY_ENABLED = os.getenv("CONTEXT_SUMMARY_ENABLED", "true").lower() == "true"
    CONTEXT_SUMMARY_TRIGGER_MESSAGES = int(os.getenv("CONTEXT_SUMMARY_TRIGGER_MESSAGES", "6"))
    CONTEXT_RECENT_MESSAGES = int(os.getenv("CONTEXT_RECENT_MESSAGES", "6"))
    ACCESS_CONTROL_CONFIG_PATH = os.getenv("ACCESS_CONTROL_CONFIG_PATH", "config/access_control.yml")

    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

settings = Settings()
