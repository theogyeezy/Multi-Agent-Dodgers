import os
from typing import List, Dict, Any
from .vector_stores import get_vector_store
from .document_processors import DocumentProcessorFactory

class KnowledgeBaseManager:
    """Manager for RAG knowledge base operations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vector_store = get_vector_store(config)

    def add_documents_from_path(self, documents_path: str) -> Dict[str, Any]:
        """Add all supported documents from a directory path"""
        if not os.path.exists(documents_path):
            return {"status": "error", "message": f"Path does not exist: {documents_path}"}

        # Find all supported files
        file_paths = []
        supported_extensions = DocumentProcessorFactory.get_supported_extensions()

        if os.path.isfile(documents_path):
            # Single file
            if any(documents_path.lower().endswith(ext) for ext in supported_extensions):
                file_paths.append(documents_path)
        else:
            # Directory
            for root, dirs, files in os.walk(documents_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if any(file.lower().endswith(ext) for ext in supported_extensions):
                        file_paths.append(file_path)

        if not file_paths:
            return {"status": "error", "message": "No supported files found"}

        # Process documents
        chunks = DocumentProcessorFactory.process_documents(file_paths, self.config)

        if not chunks:
            return {"status": "error", "message": "No content extracted from documents"}

        # Add to vector store
        documents = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]

        self.vector_store.add_documents(documents, metadatas, ids)

        return {
            "status": "success",
            "message": f"Added {len(chunks)} chunks from {len(file_paths)} files",
            "files_processed": len(file_paths),
            "chunks_added": len(chunks)
        }

    def search_knowledge_base(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Search the knowledge base"""
        if k is None:
            k = self.config.get("retrieval_k", 3)

        return self.vector_store.similarity_search(query, k=k)

    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get information about the knowledge base"""
        return self.vector_store.get_collection_info()

    def clear_knowledge_base(self) -> Dict[str, Any]:
        """Clear all documents from the knowledge base"""
        try:
            self.vector_store.delete_collection()
            return {"status": "success", "message": "Knowledge base cleared"}
        except Exception as e:
            return {"status": "error", "message": f"Error clearing knowledge base: {str(e)}"}

    def add_single_document(self, file_path: str) -> Dict[str, Any]:
        """Add a single document to the knowledge base"""
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File does not exist: {file_path}"}

        try:
            processor = DocumentProcessorFactory.get_processor(file_path, self.config)
            chunks = processor.process_document(file_path)

            if not chunks:
                return {"status": "error", "message": "No content extracted from document"}

            documents = [chunk['content'] for chunk in chunks]
            metadatas = [chunk['metadata'] for chunk in chunks]
            ids = [f"chunk_{i}_{os.path.basename(file_path)}" for i in range(len(chunks))]

            self.vector_store.add_documents(documents, metadatas, ids)

            return {
                "status": "success",
                "message": f"Added {len(chunks)} chunks from {file_path}",
                "chunks_added": len(chunks)
            }

        except Exception as e:
            return {"status": "error", "message": f"Error processing document: {str(e)}"}