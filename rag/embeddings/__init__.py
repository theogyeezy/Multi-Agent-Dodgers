from .base_embeddings import BaseEmbeddings
from .sentence_transformer_embeddings import SentenceTransformerEmbeddings
from .openai_embeddings import OpenAIEmbeddings

def get_embeddings(config: dict) -> BaseEmbeddings:
    """Factory function to get embedding model based on config"""
    embedding_type = config.get("embedding_model", "sentence-transformers")

    if embedding_type == "sentence-transformers":
        return SentenceTransformerEmbeddings(config)
    elif embedding_type == "openai":
        return OpenAIEmbeddings(config)
    else:
        raise ValueError(f"Unsupported embedding model: {embedding_type}")

__all__ = ["BaseEmbeddings", "SentenceTransformerEmbeddings", "OpenAIEmbeddings", "get_embeddings"]