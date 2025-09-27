from typing import List
from .base_embeddings import BaseEmbeddings

class SentenceTransformerEmbeddings(BaseEmbeddings):
    """Sentence Transformers embedding implementation"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = config.get("embedding_model_name", "all-MiniLM-L6-v2")
        self._model = None

    def _load_model(self):
        """Lazy load the model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        self._load_model()
        embeddings = self._model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        self._load_model()
        embedding = self._model.encode([text])
        return embedding[0].tolist()