from typing import List, Dict, Any
import os
from ..base_vector_store import BaseVectorStore

class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = config.get("pinecone_environment", "us-west1-gcp")
        self.dimension = config.get("dimension", 384)
        self._client = None
        self._index = None
        self._embeddings_model = None

        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

    def _get_embeddings_model(self):
        """Get embeddings model from config"""
        if self._embeddings_model is None:
            from ..embeddings import get_embeddings
            self._embeddings_model = get_embeddings(self.config)
        return self._embeddings_model

    def _get_client(self):
        """Lazy load Pinecone client"""
        if self._client is None:
            try:
                import pinecone
                pinecone.init(api_key=self.api_key, environment=self.environment)
                self._client = pinecone
            except ImportError:
                raise ImportError("pinecone-client not installed. Run: pip install pinecone-client")

    def _get_index(self):
        """Get or create Pinecone index"""
        if self._index is None:
            self._get_client()

            # Check if index exists
            if self.collection_name not in self._client.list_indexes():
                # Create index
                self._client.create_index(
                    name=self.collection_name,
                    dimension=self.dimension,
                    metric="cosine"
                )

            self._index = self._client.Index(self.collection_name)

        return self._index

    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents to the vector store"""
        index = self._get_index()
        embeddings_model = self._get_embeddings_model()

        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        if metadatas is None:
            metadatas = [{"source": f"document_{i}"} for i in range(len(documents))]

        # Generate embeddings
        embeddings = embeddings_model.embed_documents(documents)

        # Prepare vectors for upsert
        vectors_to_upsert = []
        for i, (doc_id, embedding, document, metadata) in enumerate(zip(ids, embeddings, documents, metadatas)):
            # Add document content to metadata
            metadata_with_content = metadata.copy()
            metadata_with_content['content'] = document

            vectors_to_upsert.append({
                'id': doc_id,
                'values': embedding,
                'metadata': metadata_with_content
            })

        # Upsert vectors in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)

    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents"""
        index = self._get_index()
        embeddings_model = self._get_embeddings_model()

        # Generate query embedding
        query_embedding = embeddings_model.embed_query(query)

        # Search
        results = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )

        formatted_results = []
        for match in results['matches']:
            formatted_results.append({
                'content': match['metadata'].get('content', ''),
                'metadata': {k: v for k, v in match['metadata'].items() if k != 'content'},
                'id': match['id'],
                'score': match['score']
            })

        return formatted_results

    def delete_collection(self):
        """Delete the entire collection"""
        self._get_client()
        if self.collection_name in self._client.list_indexes():
            self._client.delete_index(self.collection_name)
        self._index = None

    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        index = self._get_index()
        stats = index.describe_index_stats()
        return {
            "name": self.collection_name,
            "count": stats.get('total_vector_count', 0),
            "dimension": self.dimension,
            "type": "Pinecone"
        }