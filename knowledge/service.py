import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.documents import Document

from AIRAGAgent.config.settings import settings
from AIRAGAgent.db.models import KnowledgeTaskRecord
from AIRAGAgent.db.session import SessionLocal
from AIRAGAgent.knowledge.loader import SourceFile, iter_source_files, load_documents
from AIRAGAgent.knowledge.splitter import split_documents
from AIRAGAgent.knowledge.vector_store import EnterpriseVectorStore
from AIRAGAgent.schemas import KnowledgeIngestResponse, KnowledgeStatsResponse, KnowledgeTaskResponse
from AIRAGAgent.utils.file_handler import get_file_md5_hex
from AIRAGAgent.utils.logger_handler import logger
from AIRAGAgent.utils.path_tool import get_abs_path


class KnowledgeBaseService:
    def __init__(self):
        self.data_path = get_abs_path(settings.KNOWLEDGE_DATA_PATH)
        self.manifest_path = get_abs_path(settings.KNOWLEDGE_MANIFEST_PATH)
        self.bm25_corpus_path = get_abs_path(settings.BM25_CORPUS_PATH)
        self.vector_store = EnterpriseVectorStore()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def _load_manifest(self) -> Dict[str, Any]:
        if not os.path.exists(self.manifest_path):
            return {"version": 2, "files": {}}
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if "tenants" in raw:
            return {
                "version": raw.get("version", 3),
                "tenants": raw.get("tenants") or {},
            }
        if "files" in raw:
            return {"version": 3, "tenants": {"default": raw["files"]}}
        return {
            "version": 3,
            "tenants": {
                "default": {
                path: {
                    "md5": md5,
                    "size": None,
                    "mtime": None,
                    "chunks": [],
                }
                for path, md5 in raw.items()
                }
            },
        }

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def ingest(self, force: bool = False, tenant_id: str = "global") -> KnowledgeIngestResponse:
        manifest = self._load_manifest()
        files = iter_source_files(self.data_path, ("txt", "pdf", "docx", "doc"), calculate_md5=False)
        current_paths = {source.path for source in files}
        existing_files = manifest.setdefault("tenants", {}).setdefault(tenant_id, {})
        indexed_files = 0
        skipped_files = 0
        indexed_chunks = 0
        deleted_files = 0
        deleted_chunks = 0
        errors: List[str] = []
        bm25_records = {} if force else self._load_bm25_records()

        for deleted_path in sorted(set(existing_files) - current_paths):
            deleted_chunks += len(existing_files[deleted_path].get("chunks", []))
            deleted_ids = {chunk["chunk_id"] for chunk in existing_files[deleted_path].get("chunks", [])}
            for chunk_id in deleted_ids:
                bm25_records.pop(chunk_id, None)
            self.vector_store.delete_by_source(deleted_path, tenant_id=tenant_id)
            del existing_files[deleted_path]
            deleted_files += 1

        changed_files = []
        for source_file in files:
            stat = os.stat(source_file.path)
            old_entry = existing_files.get(source_file.path)
            if not force and old_entry and self._fast_unchanged(old_entry, stat):
                skipped_files += 1
                continue

            md5 = get_file_md5_hex(source_file.path)
            if not force and old_entry and old_entry.get("md5") == md5:
                old_entry["size"] = stat.st_size
                old_entry["mtime"] = stat.st_mtime
                skipped_files += 1
                continue

            changed_files.append(SourceFile(path=source_file.path, md5=md5 or ""))

        with ThreadPoolExecutor(max_workers=settings.INGEST_WORKERS) as pool:
            futures = {pool.submit(self._prepare_file_chunks, source, tenant_id): source for source in changed_files}
            for future in as_completed(futures):
                source = futures[future]
                try:
                    prepared = future.result()
                    old_entry = existing_files.get(source.path, {})
                    old_chunk_ids = {chunk["chunk_id"] for chunk in old_entry.get("chunks", [])}
                    new_chunk_ids = set(prepared["chunk_ids"])
                    removed_chunk_ids = sorted(old_chunk_ids - new_chunk_ids)
                    chunks_to_add = [
                        chunk
                        for chunk in prepared["chunks"]
                        if chunk.metadata["chunk_id"] not in old_chunk_ids
                    ]
                    ids_to_add = [chunk.metadata["chunk_id"] for chunk in chunks_to_add]

                    if force:
                        self.vector_store.delete_by_source(source.path, tenant_id=tenant_id)
                        for chunk_id, record in list(bm25_records.items()):
                            if record.get("source") == source.path and record.get("tenant_id") == tenant_id:
                                bm25_records.pop(chunk_id, None)
                        chunks_to_add = prepared["chunks"]
                        ids_to_add = prepared["chunk_ids"]
                        removed_chunk_ids = list(old_chunk_ids)
                    else:
                        self.vector_store.delete_ids(removed_chunk_ids)
                        for chunk_id in removed_chunk_ids:
                            bm25_records.pop(chunk_id, None)

                    if chunks_to_add:
                        self.vector_store.add_documents_in_batches(
                            chunks_to_add,
                            ids=ids_to_add,
                            batch_size=settings.INGEST_BATCH_SIZE,
                        )
                        for chunk in chunks_to_add:
                            bm25_records[chunk.metadata["chunk_id"]] = self._bm25_record(chunk)
                    existing_files[source.path] = prepared["manifest_entry"]
                    indexed_files += 1
                    indexed_chunks += len(chunks_to_add)
                    deleted_chunks += len(removed_chunk_ids)
                except Exception as exc:
                    message = f"{source.path}: {exc}"
                    logger.error("[knowledge ingest] %s", message, exc_info=True)
                    errors.append(message)

        self._save_manifest(manifest)
        self._save_bm25_records(bm25_records)
        return KnowledgeIngestResponse(
            backend=self.vector_store.backend,
            scanned_files=len(files),
            indexed_files=indexed_files,
            skipped_files=skipped_files,
            indexed_chunks=indexed_chunks,
            deleted_files=deleted_files,
            deleted_chunks=deleted_chunks,
            errors=errors,
        )

    def start_ingest_task(
        self,
        force: bool = False,
        user_id: int = 0,
        tenant_id: str = "global",
    ) -> KnowledgeTaskResponse:
        if settings.TASK_QUEUE_BACKEND == "celery":
            from celery.result import AsyncResult

            from AIRAGAgent.knowledge.tasks import celery_app, ingest_knowledge_task

            async_result = ingest_knowledge_task.delay(force, tenant_id, user_id)
            # Touch AsyncResult here so misconfigured broker/backend fails early.
            AsyncResult(async_result.id, app=celery_app)
            self._save_task_record(async_result.id, user_id, tenant_id, "queued")
            return KnowledgeTaskResponse(task_id=async_result.id, status="queued")

        task_id = str(uuid4())
        self._save_task_record(task_id, user_id, tenant_id, "running")
        self._tasks[task_id] = {"status": "running", "result": None, "error": None}
        future = self._executor.submit(self.ingest, force, tenant_id)
        future.add_done_callback(lambda completed: self._complete_task(task_id, completed, user_id, tenant_id))
        return KnowledgeTaskResponse(task_id=task_id, status="running")

    def get_task(
        self,
        task_id: str,
        user_id: int = 0,
        tenant_id: str = "global",
    ) -> Optional[KnowledgeTaskResponse]:
        record = self._get_task_record(task_id)
        if not record or record.user_id != user_id or record.tenant_id != tenant_id:
            return None
        if settings.TASK_QUEUE_BACKEND == "celery":
            from celery.result import AsyncResult

            from AIRAGAgent.knowledge.tasks import celery_app

            result = AsyncResult(task_id, app=celery_app)
            if result.state == "PENDING":
                return KnowledgeTaskResponse(task_id=task_id, status="queued")
            if result.state in {"STARTED", "RETRY"}:
                return KnowledgeTaskResponse(task_id=task_id, status="running")
            if result.state == "SUCCESS":
                payload = result.result
                self._update_task_record(task_id, "completed", None)
                if isinstance(payload, dict):
                    return KnowledgeTaskResponse(
                        task_id=task_id,
                        status="completed",
                        result=KnowledgeIngestResponse(**payload),
                    )
                return KnowledgeTaskResponse(task_id=task_id, status="completed")
            if result.state == "FAILURE":
                self._update_task_record(task_id, "failed", str(result.result))
                return KnowledgeTaskResponse(task_id=task_id, status="failed", error=str(result.result))
            return KnowledgeTaskResponse(task_id=task_id, status=result.state.lower())

        task = self._tasks.get(task_id)
        if not task:
            return None
        return KnowledgeTaskResponse(task_id=task_id, **task)

    def _complete_task(self, task_id: str, future, user_id: int, tenant_id: str) -> None:
        try:
            result = future.result()
            result.task_id = task_id
            self._update_task_record(task_id, "completed", None)
            self._tasks[task_id] = {"status": "completed", "result": result, "error": None}
        except Exception as exc:
            logger.error("[knowledge ingest task] failed: %s", exc, exc_info=True)
            self._update_task_record(task_id, "failed", str(exc))
            self._tasks[task_id] = {"status": "failed", "result": None, "error": str(exc)}

    def _fast_unchanged(self, old_entry: Dict[str, Any], stat: os.stat_result) -> bool:
        return (
            old_entry.get("size") == stat.st_size
            and old_entry.get("mtime") == stat.st_mtime
            and old_entry.get("md5")
            and old_entry.get("chunking_version") == settings.CHUNKING_VERSION
        )

    def _prepare_file_chunks(self, source_file: SourceFile, tenant_id: str) -> Dict[str, Any]:
        documents = load_documents(source_file)
        chunks = split_documents(documents)
        stat = os.stat(source_file.path)
        chunk_entries = []
        chunk_ids = []

        for index, chunk in enumerate(chunks):
            chunk_hash = self._chunk_hash(chunk)
            chunk_id = self._chunk_id(source_file.path, index, chunk_hash)
            chunk.metadata = {
                **chunk.metadata,
                "source": chunk.metadata.get("source") or source_file.path,
                "file_name": chunk.metadata.get("file_name") or os.path.basename(source_file.path),
                "file_md5": chunk.metadata.get("file_md5") or source_file.md5,
                "document_type": chunk.metadata.get("document_type") or "unknown",
                "policy_domain": chunk.metadata.get("policy_domain") or "general",
                "chunk_id": chunk_id,
                "chunk_hash": chunk_hash,
                "tenant_id": tenant_id,
            }
            chunk_ids.append(chunk_id)
            chunk_entries.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                    "chunk_hash": chunk_hash,
                    "document_type": chunk.metadata.get("document_type"),
                    "policy_domain": chunk.metadata.get("policy_domain"),
                    "section_title": chunk.metadata.get("section_title"),
                    "section_index": chunk.metadata.get("section_index"),
                }
            )

        return {
            "chunks": chunks,
            "chunk_ids": chunk_ids,
            "manifest_entry": {
                "md5": source_file.md5,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "chunking_version": settings.CHUNKING_VERSION,
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "chunks": chunk_entries,
            },
        }

    def _chunk_hash(self, chunk: Document) -> str:
        return hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()

    def _chunk_id(self, source: str, index: int, chunk_hash: str) -> str:
        raw = f"{source}:{index}:{chunk_hash}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _load_bm25_records(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(self.bm25_corpus_path):
            return {}
        records = {}
        with open(self.bm25_corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                chunk_id = record.get("chunk_id")
                if chunk_id:
                    records[chunk_id] = record
        return records

    def _save_bm25_records(self, records: Dict[str, Dict[str, Any]]) -> None:
        with open(self.bm25_corpus_path, "w", encoding="utf-8") as f:
            for record in records.values():
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _bm25_record(self, chunk: Document) -> Dict[str, Any]:
        return {
            "chunk_id": chunk.metadata.get("chunk_id"),
            "tenant_id": chunk.metadata.get("tenant_id"),
            "source": chunk.metadata.get("source"),
            "file_name": chunk.metadata.get("file_name"),
            "file_md5": chunk.metadata.get("file_md5"),
            "document_type": chunk.metadata.get("document_type"),
            "policy_domain": chunk.metadata.get("policy_domain"),
            "chunk_index": chunk.metadata.get("chunk_index"),
            "chunk_hash": chunk.metadata.get("chunk_hash"),
            "section_title": chunk.metadata.get("section_title"),
            "section_index": chunk.metadata.get("section_index"),
            "section_chunk_index": chunk.metadata.get("section_chunk_index"),
            "page_content": chunk.page_content,
        }

    def stats(self) -> KnowledgeStatsResponse:
        return KnowledgeStatsResponse(
            backend=self.vector_store.backend,
            document_count=self.vector_store.count(),
            data_path=self.data_path,
            manifest_path=self.manifest_path,
        )

    def _save_task_record(self, task_id: str, user_id: int, tenant_id: str, status: str) -> None:
        db = SessionLocal()
        try:
            db.add(KnowledgeTaskRecord(task_id=task_id, user_id=user_id, tenant_id=tenant_id, status=status))
            db.commit()
        finally:
            db.close()

    def _get_task_record(self, task_id: str) -> Optional[KnowledgeTaskRecord]:
        db = SessionLocal()
        try:
            return db.query(KnowledgeTaskRecord).filter(KnowledgeTaskRecord.task_id == task_id).first()
        finally:
            db.close()

    def _update_task_record(self, task_id: str, status: str, error: Optional[str]) -> None:
        db = SessionLocal()
        try:
            record = db.query(KnowledgeTaskRecord).filter(KnowledgeTaskRecord.task_id == task_id).first()
            if record:
                record.status = status
                record.error = error
                db.commit()
        finally:
            db.close()
