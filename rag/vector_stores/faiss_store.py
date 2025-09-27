from typing import List, Dict, Any
import numpy as np
import os
import pickle
from ..base_vector_store import BaseVectorStore

class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dimension = config.get("dimension", 384)  # Default for all-MiniLM-L6-v2
        self.persist_directory = config.get("persist_directory", "./faiss_db")
        self._index = None
        self._documents = []
        self._metadatas = []
        self._ids = []
        self._embeddings_model = None

    def _get_embeddings_model(self):
        """Get embeddings model from config"""
        if self._embeddings_model is None:
            from ..embeddings import get_embeddings
            self._embeddings_model = get_embeddings(self.config)
        return self._embeddings_model

    def _get_index(self):
        """Get or create FAISS index"""
        if self._index is None:
            try:
                import faiss
            except ImportError:
                raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

            # Try to load existing index
            index_path = os.path.join(self.persist_directory, f"{self.collection_name}.index")
            metadata_path = os.path.join(self.persist_directory, f"{self.collection_name}_metadata.pkl")

            if os.path.exists(index_path) and os.path.exists(metadata_path):
                self._index = faiss.read_index(index_path)
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self._documents = data['documents']
                    self._metadatas = data['metadatas']
                    self._ids = data['ids']
            else:
                # Create new index
                self._index = faiss.IndexFlatIP(self.dimension)  # Inner product similarity
                os.makedirs(self.persist_directory, exist_ok=True)

        return self._index

    def _save_index(self):
        """Save index and metadata to disk"""
        import faiss
        os.makedirs(self.persist_directory, exist_ok=True)

        index_path = os.path.join(self.persist_directory, f"{self.collection_name}.index")
        metadata_path = os.path.join(self.persist_directory, f"{self.collection_name}_metadata.pkl")

        faiss.write_index(self._index, index_path)

        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self._documents,
                'metadatas': self._metadatas,
                'ids': self._ids
            }, f)

    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents to the vector store"""
        index = self._get_index()
        embeddings_model = self._get_embeddings_model()

        if ids is None:
            start_id = len(self._documents)
            ids = [f"doc_{start_id + i}" for i in range(len(documents))]

        if metadatas is None:
            metadatas = [{"source": f"document_{i}"} for i in range(len(documents))]

        # Generate embeddings
        embeddings = embeddings_model.embed_documents(documents)
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)

        # Add to index
        index.add(embeddings_array)

        # Store documents and metadata
        self._documents.extend(documents)
        self._metadatas.extend(metadatas)
        self._ids.extend(ids)

        # Save to disk
        self._save_index()

    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents"""
        index = self._get_index()
        embeddings_model = self._get_embeddings_model()

        if index.ntotal == 0:
            return []

        # Generate query embedding
        query_embedding = embeddings_model.embed_query(query)
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)

        # Search
        scores, indices = index.search(query_array, min(k, index.ntotal))

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:  # Valid index
                results.append({
                    'content': self._documents[idx],
                    'metadata': self._metadatas[idx],
                    'id': self._ids[idx],
                    'score': float(score)
                })

        return results

    def delete_collection(self):
        """Delete the entire collection"""
        import faiss
        self._index = faiss.IndexFlatIP(self.dimension)
        self._documents = []
        self._metadatas = []
        self._ids = []

        # Remove files
        index_path = os.path.join(self.persist_directory, f"{self.collection_name}.index")
        metadata_path = os.path.join(self.persist_directory, f"{self.collection_name}_metadata.pkl")

        for path in [index_path, metadata_path]:
            if os.path.exists(path):
                os.remove(path)

    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        index = self._get_index()
        return {
            "name": self.collection_name,
            "count": index.ntotal,
            "dimension": self.dimension,
            "type": "FAISS"
        }