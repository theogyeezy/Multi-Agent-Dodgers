from typing import List
import os
from .base_embeddings import BaseEmbeddings

class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embedding implementation"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = config.get("embedding_model_name", "text-embedding-ada-002")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self._client = None

    def _get_client(self):
        """Lazy load OpenAI client"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai not installed. Run: pip install openai")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        self._get_client()
        response = self._client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        return [embedding.embedding for embedding in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        self._get_client()
        response = self._client.embeddings.create(
            input=[text],
            model=self.model_name
        )
        return response.data[0].embedding