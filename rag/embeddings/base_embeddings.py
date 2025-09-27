from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddings(ABC):
    """Abstract base class for embedding models"""

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        pass