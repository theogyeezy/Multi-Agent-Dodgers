from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseVectorStore(ABC):
    """Abstract base class for vector stores - implement for different vector databases"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.collection_name = config.get("collection_name", "default_collection")

    @abstractmethod
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents to the vector store"""
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents"""
        pass

    @abstractmethod
    def delete_collection(self):
        """Delete the entire collection"""
        pass

    @abstractmethod
    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        pass