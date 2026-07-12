import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

from langchain_core.documents import Document
from pymilvus import MilvusClient

from AIRAGAgent.config.settings import settings
from AIRAGAgent.knowledge.vector_store import EnterpriseVectorStore
from AIRAGAgent.utils.path_tool import get_abs_path


def load_bm25_records() -> Dict[str, dict]:
    path = Path(get_abs_path(settings.BM25_CORPUS_PATH))
    if not path.exists():
        raise FileNotFoundError(f"BM25 corpus not found: {path}")

    records: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            chunk_id = record.get("chunk_id")
            if chunk_id:
                records[chunk_id] = record
    return records


def load_milvus_ids() -> Set[str]:
    client = MilvusClient(uri=settings.MILVUS_URI)
    if not client.has_collection(settings.MILVUS_COLLECTION):
        return set()

    rows = client.query(
        collection_name=settings.MILVUS_COLLECTION,
        filter='pk != ""',
        output_fields=["pk"],
        limit=16000,
    )
    return {row["pk"] for row in rows if row.get("pk")}


def records_to_documents(records: List[dict]) -> List[Document]:
    documents = []
    for record in records:
        metadata = dict(record)
        page_content = metadata.pop("page_content", "")
        if page_content:
            documents.append(Document(page_content=page_content, metadata=metadata))
    return documents


def repair_missing(records: Dict[str, dict], missing_ids: List[str], batch_size: int) -> None:
    if not missing_ids:
        return

    store = EnterpriseVectorStore()
    for start in range(0, len(missing_ids), batch_size):
        batch_ids = missing_ids[start : start + batch_size]
        docs = records_to_documents([records[chunk_id] for chunk_id in batch_ids])
        ids = [doc.metadata["chunk_id"] for doc in docs]
        store.add_documents_in_batches(docs, ids=ids, batch_size=batch_size)
        print(f"repaired {start + len(batch_ids)}/{len(missing_ids)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check or repair Milvus vectors against bm25_corpus.jsonl.")
    parser.add_argument("--repair", action="store_true", help="Only embed and insert missing chunks.")
    parser.add_argument("--batch-size", type=int, default=settings.INGEST_BATCH_SIZE)
    args = parser.parse_args()

    records = load_bm25_records()
    bm25_ids = set(records)
    milvus_ids = load_milvus_ids()
    missing_ids = sorted(bm25_ids - milvus_ids)
    extra_ids = sorted(milvus_ids - bm25_ids)

    print(f"bm25_chunks={len(bm25_ids)}")
    print(f"milvus_vectors={len(milvus_ids)}")
    print(f"missing_in_milvus={len(missing_ids)}")
    print(f"extra_in_milvus={len(extra_ids)}")

    if missing_ids[:5]:
        print("missing_samples=", missing_ids[:5])
    if extra_ids[:5]:
        print("extra_samples=", extra_ids[:5])

    if args.repair:
        repair_missing(records, missing_ids, args.batch_size)
        repaired_ids = load_milvus_ids()
        print(f"milvus_vectors_after_repair={len(repaired_ids)}")
        print(f"missing_after_repair={len(bm25_ids - repaired_ids)}")


if __name__ == "__main__":
    main()
