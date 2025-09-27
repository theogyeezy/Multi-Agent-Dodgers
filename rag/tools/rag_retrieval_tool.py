from typing import Type, Dict, Any, List
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

class RAGRetrievalInput(BaseModel):
    """Input schema for RAG retrieval tool"""
    query: str = Field(..., description="The search query to find relevant documents")

class RAGRetrievalTool(BaseTool):
    """CrewAI tool for RAG-based document retrieval"""

    name: str = "rag_retrieval"
    description: str = "Search through the knowledge base to find relevant documents for the given query"
    args_schema: Type[BaseModel] = RAGRetrievalInput

    def __init__(self, vector_store, config: Dict[str, Any]):
        super().__init__()
        self.vector_store = vector_store
        self.config = config
        self.retrieval_k = config.get("retrieval_k", 3)

    def _run(self, query: str) -> str:
        """Execute the RAG retrieval"""
        try:
            # Search for relevant documents
            results = self.vector_store.similarity_search(query, k=self.retrieval_k)

            if not results:
                return "No relevant documents found in the knowledge base for this query."

            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                source = metadata.get('source', 'Unknown source')

                formatted_result = f"**Document {i}:**\n"
                formatted_result += f"Source: {source}\n"
                formatted_result += f"Content: {content}\n"

                if 'chunk_index' in metadata:
                    formatted_result += f"Chunk: {metadata['chunk_index'] + 1}/{metadata.get('total_chunks', '?')}\n"

                formatted_results.append(formatted_result)

            return "\n".join(formatted_results)

        except Exception as e:
            return f"Error during retrieval: {str(e)}"

class RAGRetrievalToolFactory:
    """Factory for creating RAG retrieval tools"""

    @staticmethod
    def create_tool(config: Dict[str, Any]) -> RAGRetrievalTool:
        """Create a RAG retrieval tool based on configuration"""
        from ..vector_stores import get_vector_store

        # Initialize vector store
        vector_store = get_vector_store(config)

        # Create tool
        return RAGRetrievalTool(vector_store=vector_store, config=config)

    @staticmethod
    def create_tool_with_documents(config: Dict[str, Any], documents_path: str = None) -> RAGRetrievalTool:
        """Create a RAG retrieval tool and populate it with documents"""
        import os
        from ..document_processors import DocumentProcessorFactory

        # Create tool
        tool = RAGRetrievalToolFactory.create_tool(config)

        # Load documents if path provided
        if documents_path and os.path.exists(documents_path):
            file_paths = []
            for root, dirs, files in os.walk(documents_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if any(file.lower().endswith(ext) for ext in DocumentProcessorFactory.get_supported_extensions()):
                        file_paths.append(file_path)

            if file_paths:
                # Process documents
                chunks = DocumentProcessorFactory.process_documents(file_paths, config)

                if chunks:
                    # Extract data for vector store
                    documents = [chunk['content'] for chunk in chunks]
                    metadatas = [chunk['metadata'] for chunk in chunks]
                    ids = [f"chunk_{i}" for i in range(len(chunks))]

                    # Add to vector store
                    tool.vector_store.add_documents(documents, metadatas, ids)

        return tool