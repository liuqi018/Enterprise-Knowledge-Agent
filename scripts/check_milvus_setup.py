from AIRAGAgent.config.settings import settings
from AIRAGAgent.knowledge.vector_store import EnterpriseVectorStore
from pymilvus import Collection, connections, utility


def main() -> None:
    print("VECTOR_BACKEND=", settings.VECTOR_BACKEND)
    print("MILVUS_URI=", settings.MILVUS_URI)
    print("MILVUS_COLLECTION=", settings.MILVUS_COLLECTION)
    connections.connect(alias="default", uri=settings.MILVUS_URI)
    print("connected_default=", connections.has_connection("default"))
    exists = utility.has_collection(settings.MILVUS_COLLECTION)
    print("collection_exists=", exists)
    if exists:
        collection = Collection(settings.MILVUS_COLLECTION)
        print("num_entities=", collection.num_entities)
        print("schema_fields=", [(field.name, str(field.dtype), field.is_primary) for field in collection.schema.fields])
    store = EnterpriseVectorStore()
    print("vector_store_backend=", store.backend)


if __name__ == "__main__":
    main()
