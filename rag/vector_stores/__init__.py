from ..base_vector_store import BaseVectorStore
from .chroma_store import ChromaVectorStore
from .faiss_store import FAISSVectorStore
from .pinecone_store import PineconeVectorStore

def get_vector_store(config: dict) -> BaseVectorStore:
    """Factory function to get vector store based on config"""
    vector_db_type = config.get("vector_db", "chroma")

    if vector_db_type == "chroma":
        return ChromaVectorStore(config)
    elif vector_db_type == "faiss":
        return FAISSVectorStore(config)
    elif vector_db_type == "pinecone":
        return PineconeVectorStore(config)
    else:
        raise ValueError(f"Unsupported vector database: {vector_db_type}")

__all__ = ["BaseVectorStore", "ChromaVectorStore", "FAISSVectorStore", "PineconeVectorStore", "get_vector_store"]