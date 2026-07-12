from celery import Celery

from AIRAGAgent.config.settings import settings

celery_app = Celery(
    "airag_agent",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=False,
    task_track_started=True,
)


@celery_app.task(name="knowledge.ingest", bind=True)
def ingest_knowledge_task(self, force: bool = False, tenant_id: str = "default", user_id: int = 0):
    from AIRAGAgent.knowledge.service import KnowledgeBaseService

    result = KnowledgeBaseService().ingest(force=force, tenant_id=tenant_id)
    return result.model_dump() if hasattr(result, "model_dump") else result.dict()
