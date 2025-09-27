from typing import List, Dict, Any
from ..base_vector_store import BaseVectorStore

class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
        self._collection = None
        self.persist_directory = config.get("persist_directory", "./chroma_db")

    def _get_client(self):
        """Lazy load ChromaDB client"""
        if self._client is None:
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=self.persist_directory)
            except ImportError:
                raise ImportError("chromadb not installed. Run: pip install chromadb")

    def _get_collection(self):
        """Get or create collection"""
        if self._collection is None:
            self._get_client()
            try:
                self._collection = self._client.get_collection(name=self.collection_name)
            except:
                self._collection = self._client.create_collection(name=self.collection_name)
        return self._collection

    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents to the vector store"""
        collection = self._get_collection()

        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        if metadatas is None:
            metadatas = [{"source": f"document_{i}"} for i in range(len(documents))]

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents"""
        collection = self._get_collection()

        results = collection.query(
            query_texts=[query],
            n_results=k
        )

        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'id': results['ids'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })

        return formatted_results

    def delete_collection(self):
        """Delete the entire collection"""
        self._get_client()
        try:
            self._client.delete_collection(name=self.collection_name)
            self._collection = None
        except:
            pass

    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        collection = self._get_collection()
        count = collection.count()
        return {
            "name": self.collection_name,
            "count": count,
            "type": "ChromaDB"
        }