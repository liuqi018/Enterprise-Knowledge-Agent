import inspect
import warnings
from typing import Dict, List, Optional

from langchain_core.documents import Document

from AIRAGAgent.config.settings import settings
from AIRAGAgent.model.factory import embed_model
from AIRAGAgent.utils.logger_handler import logger
from AIRAGAgent.utils.path_tool import get_abs_path


class EnterpriseVectorStore:
    """Vector store facade. Milvus is preferred when configured; Chroma remains a local fallback."""

    def __init__(self, backend: Optional[str] = None):
        self.backend = (backend or settings.VECTOR_BACKEND).lower()
        self.store = self._create_store()

    def _create_store(self):
        if self.backend == "milvus":
            try:
                from langchain_milvus import Milvus
                from pymilvus import MilvusClient, connections

                self._connect_milvus("default")
                client = MilvusClient(uri=settings.MILVUS_URI)
                client_alias = getattr(client, "_using", None)
                if client_alias:
                    self._connect_milvus(client_alias)

                kwargs = {
                    "embedding_function": embed_model,
                    "collection_name": settings.MILVUS_COLLECTION,
                    "connection_args": {"uri": settings.MILVUS_URI, "alias": "default"},
                    "search_params": {"metric_type": "L2", "params": {}},
                    "auto_id": False,
                }
                parameters = inspect.signature(Milvus.__init__).parameters
                if "enable_dynamic_field" in parameters:
                    kwargs["enable_dynamic_field"] = True
                elif "metadata_field" in parameters:
                    kwargs["metadata_field"] = "metadata"
                return Milvus(**kwargs)
            except Exception as exc:
                logger.error("[vector store] failed to initialize Milvus: %s", exc, exc_info=True)
                if settings.VECTOR_BACKEND == "milvus":
                    raise RuntimeError(f"Milvus 初始化失败，已配置 VECTOR_BACKEND=milvus，不能静默回退到 Chroma: {exc}") from exc
                self.backend = "chroma"

        from langchain_chroma import Chroma

        return Chroma(
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=embed_model,
            persist_directory=get_abs_path(settings.CHROMA_PERSIST_DIR),
        )

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> None:
        if documents:
            self._ensure_milvus_connection()
            if ids:
                self.store.add_documents(documents, ids=ids)
            else:
                self.store.add_documents(documents)

    def add_documents_in_batches(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        batch_size: int = 64,
    ) -> None:
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            batch_ids = ids[start:end] if ids else None
            self.add_documents(documents[start:end], ids=batch_ids)

    def delete_by_source(self, source: str, tenant_id: Optional[str] = None) -> None:
        if not source:
            return
        where = {"source": source}
        if tenant_id:
            where["tenant_id"] = tenant_id
        chroma_where = self._to_chroma_where(where)

        if self.backend != "milvus":
            try:
                self.store.delete(where=chroma_where)
                return
            except Exception:
                pass

        try:
            self._ensure_milvus_connection()
            self._load_milvus_collection()
            expr = f'source == "{self._escape_milvus_string(source)}"'
            if tenant_id:
                expr += f' && tenant_id == "{self._escape_milvus_string(tenant_id)}"'
            self.store.delete(expr=expr)
            return
        except Exception:
            pass

        collection = getattr(self.store, "_collection", None)
        if collection is not None:
            collection.delete(where=chroma_where)

    def delete_ids(self, ids: List[str]) -> None:
        if not ids:
            return
        try:
            self._ensure_milvus_connection()
            self._load_milvus_collection()
            self.store.delete(ids=ids)
            return
        except Exception:
            pass

        collection = getattr(self.store, "_collection", None)
        if collection is not None:
            collection.delete(ids=ids)

    def similarity_search(
        self,
        query: str,
        k: int,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        self._ensure_milvus_connection()
        if not metadata_filter:
            return self.store.similarity_search(query, k=k)

        try:
            vector_filter = self._to_vector_filter(metadata_filter)
            return self.store.similarity_search(query, k=k, filter=vector_filter)
        except Exception:
            docs = self.store.similarity_search(query, k=max(k * 4, 20))
            return [doc for doc in docs if self._metadata_match(doc, metadata_filter)][:k]

    def _metadata_match(self, doc: Document, metadata_filter: Dict[str, str]) -> bool:
        return all(doc.metadata.get(key) == value for key, value in metadata_filter.items())

    def _to_vector_filter(self, metadata_filter: Dict[str, str]):
        if self.backend == "chroma":
            return self._to_chroma_where(metadata_filter)
        return metadata_filter

    def _to_chroma_where(self, metadata_filter: Dict[str, str]):
        if len(metadata_filter) <= 1:
            return metadata_filter
        return {"$and": [{key: value} for key, value in metadata_filter.items()]}

    def _escape_milvus_string(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _load_milvus_collection(self) -> None:
        if self.backend != "milvus":
            return
        collection = getattr(self.store, "col", None) or getattr(self.store, "_collection", None)
        if collection is not None and hasattr(collection, "load"):
            collection.load()

    def _ensure_milvus_connection(self) -> None:
        if self.backend != "milvus":
            return
        from pymilvus import connections

        self._connect_milvus("default")
        alias = getattr(self.store, "alias", None)
        if alias and alias != "default":
            self._connect_milvus(alias)

    def _connect_milvus(self, alias: str) -> None:
        from pymilvus import connections

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`connections.connect` is an ORM-style.*")
            connections.connect(alias=alias, uri=settings.MILVUS_URI)

    def as_retriever(self, k: int):
        return self.store.as_retriever(search_kwargs={"k": k})

    def count(self) -> int:
        self._ensure_milvus_connection()
        if self.backend == "milvus":
            try:
                from pymilvus import Collection, utility

                if not utility.has_collection(settings.MILVUS_COLLECTION):
                    return 0
                collection = Collection(settings.MILVUS_COLLECTION)
                return collection.num_entities
            except Exception:
                return 0
        collection = getattr(self.store, "_collection", None)
        if collection is not None:
            return collection.count()
        try:
            return len(self.store.similarity_search("", k=10000))
        except Exception:
            return 0
