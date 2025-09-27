# API Reference Guide

Complete API documentation for the RAG Multi-Agent Template components.

## 📋 Overview

This reference covers all major classes, methods, and configuration options for building and customizing your RAG multi-agent system.

## 🏗️ Core Architecture

### Base Classes

```python
# rag/base_vector_store.py
class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize vector store with configuration

        Args:
            config: Configuration dictionary with vector store settings
        """

    @abstractmethod
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents to the vector store

        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs
        """

    @abstractmethod
    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents

        Args:
            query: Search query string
            k: Number of results to return

        Returns:
            List of documents with similarity scores
        """

    @abstractmethod
    def delete_collection(self):
        """Delete the entire collection"""

    @abstractmethod
    def get_collection_info(self) -> Dict:
        """Get information about the collection

        Returns:
            Dictionary with collection statistics
        """
```

## 🗄️ Vector Store Implementations

### ChromaVectorStore

```python
# rag/vector_stores/chroma_store.py
class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize ChromaDB vector store

        Args:
            config: Configuration dictionary
                - persist_directory: Directory for persistent storage
                - collection_name: Name of the collection
        """

    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents to ChromaDB collection

        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs

        Raises:
            ImportError: If chromadb is not installed
        """

    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Search ChromaDB collection for similar documents

        Args:
            query: Search query
            k: Number of results (default: 3)

        Returns:
            List[Dict]: Documents with keys:
                - content: Document text
                - metadata: Document metadata
                - id: Document ID
                - distance: Similarity distance
        """

    def get_collection_info(self) -> Dict:
        """Get ChromaDB collection information

        Returns:
            Dict: Collection info with keys:
                - name: Collection name
                - count: Number of documents
                - type: "ChromaDB"
        """
```

### FAISSVectorStore

```python
# rag/vector_stores/faiss_store.py
class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize FAISS vector store

        Args:
            config: Configuration dictionary
                - dimension: Embedding dimension (default: 384)
                - persist_directory: Storage directory
                - collection_name: Collection name
        """

    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents to FAISS index

        Args:
            documents: Document texts
            metadatas: Document metadata
            ids: Document IDs

        Note:
            Automatically generates embeddings and normalizes for cosine similarity
        """

    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Search FAISS index

        Args:
            query: Search query
            k: Number of results

        Returns:
            List[Dict]: Results with similarity scores
        """

    def get_collection_info(self) -> Dict:
        """Get FAISS index information

        Returns:
            Dict: Index info including:
                - name: Collection name
                - count: Number of vectors
                - dimension: Vector dimension
                - type: "FAISS"
        """
```

### PineconeVectorStore

```python
# rag/vector_stores/pinecone_store.py
class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store implementation"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Pinecone vector store

        Args:
            config: Configuration dictionary
                - dimension: Vector dimension
                - pinecone_environment: Pinecone environment

        Environment Variables:
            PINECONE_API_KEY: Required Pinecone API key
        """

    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents to Pinecone index

        Args:
            documents: Document texts
            metadatas: Document metadata
            ids: Document IDs

        Note:
            Upserts vectors in batches of 100 for efficiency
        """

    def bulk_upsert(self, vectors: List[Dict], batch_size: int = 100):
        """Efficiently upsert large amounts of data

        Args:
            vectors: List of vector dictionaries
            batch_size: Batch size for upsert operations
        """
```

## 🧠 Embedding Models

### Base Embeddings

```python
# rag/embeddings/base_embeddings.py
class BaseEmbeddings(ABC):
    """Abstract base class for embedding models"""

    def __init__(self, config: dict):
        """Initialize embedding model

        Args:
            config: Configuration dictionary
        """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents

        Args:
            texts: List of text documents

        Returns:
            List of embedding vectors
        """

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
```

### Sentence Transformer Embeddings

```python
# rag/embeddings/sentence_transformer_embeddings.py
class SentenceTransformerEmbeddings(BaseEmbeddings):
    """Sentence Transformers embedding implementation"""

    def __init__(self, config: dict):
        """Initialize Sentence Transformer model

        Args:
            config: Configuration with keys:
                - embedding_model_name: Model name (default: "all-MiniLM-L6-v2")
        """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using Sentence Transformers

        Args:
            texts: List of documents

        Returns:
            List[List[float]]: Document embeddings

        Note:
            Lazy loads model on first use
        """

    def embed_query(self, text: str) -> List[float]:
        """Embed single query

        Args:
            text: Query text

        Returns:
            List[float]: Query embedding
        """
```

### OpenAI Embeddings

```python
# rag/embeddings/openai_embeddings.py
class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embedding implementation"""

    def __init__(self, config: dict):
        """Initialize OpenAI embeddings

        Args:
            config: Configuration with keys:
                - embedding_model_name: Model name (default: "text-embedding-ada-002")

        Environment Variables:
            OPENAI_API_KEY: Required OpenAI API key
        """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using OpenAI API

        Args:
            texts: Document texts

        Returns:
            List[List[float]]: Document embeddings

        Raises:
            ValueError: If OPENAI_API_KEY not set
        """
```

## 📄 Document Processing

### Base Document Processor

```python
# rag/document_processors/base_processor.py
class BaseDocumentProcessor(ABC):
    """Abstract base class for document processors"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize document processor

        Args:
            config: Configuration with keys:
                - chunk_size: Text chunk size (default: 500)
                - chunk_overlap: Overlap between chunks (default: 50)
        """

    @abstractmethod
    def load_document(self, file_path: str) -> str:
        """Load document from file path

        Args:
            file_path: Path to document file

        Returns:
            Document text content
        """

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap

        Args:
            text: Input text

        Returns:
            List[str]: Text chunks
        """

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process document and return chunks with metadata

        Args:
            file_path: Path to document

        Returns:
            List[Dict]: Processed chunks with metadata:
                - content: Chunk text
                - metadata: Chunk metadata including source and index
        """
```

### Document Processor Factory

```python
# rag/document_processors/__init__.py
class DocumentProcessorFactory:
    """Factory for creating document processors"""

    @staticmethod
    def get_processor(file_path: str, config: Dict[str, Any]) -> BaseDocumentProcessor:
        """Get appropriate processor for file type

        Args:
            file_path: Path to document file
            config: Processor configuration

        Returns:
            BaseDocumentProcessor: Appropriate processor instance

        Raises:
            ValueError: If file type not supported
        """

    @staticmethod
    def process_documents(file_paths: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process multiple documents

        Args:
            file_paths: List of file paths
            config: Processing configuration

        Returns:
            List[Dict]: All processed chunks with metadata
        """

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Get list of supported file extensions

        Returns:
            List[str]: Supported extensions ['.txt', '.md', '.pdf', '.docx', '.doc']
        """
```

## 🛠️ RAG Tools

### RAG Retrieval Tool

```python
# rag/tools/rag_retrieval_tool.py
class RAGRetrievalTool(BaseTool):
    """CrewAI tool for RAG-based document retrieval"""

    name: str = "rag_retrieval"
    description: str = "Search through the knowledge base to find relevant documents"
    args_schema: Type[BaseModel] = RAGRetrievalInput

    def __init__(self, vector_store, config: Dict[str, Any]):
        """Initialize RAG retrieval tool

        Args:
            vector_store: Vector store instance
            config: Tool configuration
        """

    def _run(self, query: str) -> str:
        """Execute RAG retrieval

        Args:
            query: Search query

        Returns:
            str: Formatted search results
        """

class RAGRetrievalToolFactory:
    """Factory for creating RAG retrieval tools"""

    @staticmethod
    def create_tool(config: Dict[str, Any]) -> RAGRetrievalTool:
        """Create RAG retrieval tool

        Args:
            config: Tool configuration

        Returns:
            RAGRetrievalTool: Configured tool instance
        """

    @staticmethod
    def create_tool_with_documents(config: Dict[str, Any], documents_path: str = None) -> RAGRetrievalTool:
        """Create tool and populate with documents

        Args:
            config: Tool configuration
            documents_path: Path to documents directory

        Returns:
            RAGRetrievalTool: Tool with populated knowledge base
        """
```

## 🧪 Knowledge Base Management

### Knowledge Base Manager

```python
# rag/knowledge_base_manager.py
class KnowledgeBaseManager:
    """Manager for RAG knowledge base operations"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize knowledge base manager

        Args:
            config: Configuration dictionary
        """

    def add_documents_from_path(self, documents_path: str) -> Dict[str, Any]:
        """Add all supported documents from directory

        Args:
            documents_path: Path to documents directory

        Returns:
            Dict: Operation result with keys:
                - status: "success" or "error"
                - message: Description of result
                - files_processed: Number of files processed
                - chunks_added: Number of chunks added
        """

    def search_knowledge_base(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Search the knowledge base

        Args:
            query: Search query
            k: Number of results (default from config)

        Returns:
            List[Dict]: Search results
        """

    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """Get knowledge base information

        Returns:
            Dict: Knowledge base statistics
        """

    def clear_knowledge_base(self) -> Dict[str, Any]:
        """Clear all documents from knowledge base

        Returns:
            Dict: Operation result
        """

    def add_single_document(self, file_path: str) -> Dict[str, Any]:
        """Add single document to knowledge base

        Args:
            file_path: Path to document file

        Returns:
            Dict: Operation result
        """
```

## 🎭 Agent Templates

### Agent Templates

```python
# templates/agent_templates.py
class AgentTemplates:
    """Template configurations for different types of agents"""

    @staticmethod
    def create_rag_researcher_agent(llm, rag_tool, function_calling_llm=None) -> Agent:
        """Create RAG-enabled researcher agent

        Args:
            llm: Language model instance
            rag_tool: RAG retrieval tool
            function_calling_llm: Optional function calling LLM

        Returns:
            Agent: Configured researcher agent
        """

    @staticmethod
    def create_rag_analyst_agent(llm, rag_tool, function_calling_llm=None) -> Agent:
        """Create RAG-enabled analyst agent

        Args:
            llm: Language model instance
            rag_tool: RAG retrieval tool
            function_calling_llm: Optional function calling LLM

        Returns:
            Agent: Configured analyst agent
        """

    @staticmethod
    def create_custom_rag_agent(llm, rag_tool, role: str, goal: str, backstory: str,
                               additional_tools: List = None, function_calling_llm=None) -> Agent:
        """Create custom RAG-enabled agent

        Args:
            llm: Language model instance
            rag_tool: RAG retrieval tool
            role: Agent role description
            goal: Agent goal
            backstory: Agent backstory
            additional_tools: Optional additional tools
            function_calling_llm: Optional function calling LLM

        Returns:
            Agent: Configured custom agent
        """

class AgentConfiguration:
    """Configuration helper for agent creation"""

    ROLE_TEMPLATES = {
        "legal_researcher": "Senior Legal Research Specialist",
        "medical_researcher": "Medical Information Specialist",
        "technical_researcher": "Technical Documentation Specialist",
        # ... more templates
    }

    @staticmethod
    def get_template_config(agent_type: str) -> Dict[str, str]:
        """Get complete configuration template for agent types

        Args:
            agent_type: Type of agent ("legal_researcher", "medical_researcher", etc.)

        Returns:
            Dict: Configuration with keys:
                - role: Agent role
                - goal: Agent goal
                - backstory: Agent backstory
        """
```

### Task Templates

```python
# templates/task_templates.py
class TaskTemplates:
    """Template configurations for different types of tasks"""

    @staticmethod
    def create_research_task(agent, description: str = None, expected_output: str = None,
                           output_file: str = None) -> Task:
        """Create research task template

        Args:
            agent: Agent to assign task to
            description: Task description
            expected_output: Expected output format
            output_file: Output file name

        Returns:
            Task: Configured research task
        """

    @staticmethod
    def create_custom_task(agent, description: str, expected_output: str,
                         output_file: str = None) -> Task:
        """Create custom task

        Args:
            agent: Agent to assign task to
            description: Task description
            expected_output: Expected output
            output_file: Optional output file

        Returns:
            Task: Configured custom task
        """

class TaskConfiguration:
    """Configuration helper for task creation"""

    @staticmethod
    def get_template_config(task_type: str, topic: str = "[TOPIC]") -> Dict[str, str]:
        """Get complete configuration template for task types

        Args:
            task_type: Type of task
            topic: Topic to research

        Returns:
            Dict: Task configuration
        """
```

## ⚙️ Configuration

### RAG Configuration

```python
# config/rag_config_template.py
class RAGConfig:
    """RAG system configuration"""

    # Vector Database Options: "chroma", "faiss", "pinecone"
    VECTOR_DB = "chroma"

    # Embedding Model Options: "sentence-transformers", "openai", "huggingface"
    EMBEDDING_MODEL = "sentence-transformers"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Retrieval Settings
    RETRIEVAL_K = 3

    # Collection/Index Names
    COLLECTION_NAME = "your_collection_name"

    # File Processing
    SUPPORTED_FILE_TYPES = [".txt", ".pdf", ".docx", ".md"]
    DOCUMENTS_PATH = "data/documents"

    # Agent Template Placeholders
    AGENT_ROLES = {
        "researcher": {
            "role": "FILL_IN_RESEARCHER_ROLE",
            "goal": "FILL_IN_RESEARCHER_GOAL",
            "backstory": "FILL_IN_RESEARCHER_BACKSTORY"
        }
        # ... more roles
    }
```

## 🔧 Utility Functions

### Factory Functions

```python
# Vector store factory
from rag.vector_stores import get_vector_store

def get_vector_store(config: dict) -> BaseVectorStore:
    """Get vector store instance based on configuration

    Args:
        config: Configuration dictionary

    Returns:
        BaseVectorStore: Vector store instance

    Raises:
        ValueError: If unsupported vector database type
    """

# Embeddings factory
from rag.embeddings import get_embeddings

def get_embeddings(config: dict) -> BaseEmbeddings:
    """Get embeddings model based on configuration

    Args:
        config: Configuration dictionary

    Returns:
        BaseEmbeddings: Embeddings model instance

    Raises:
        ValueError: If unsupported embedding model type
    """
```

## 📊 Error Handling

### Common Exceptions

```python
# Custom exceptions for better error handling
class RAGSystemError(Exception):
    """Base exception for RAG system errors"""
    pass

class VectorStoreError(RAGSystemError):
    """Vector store related errors"""
    pass

class DocumentProcessingError(RAGSystemError):
    """Document processing related errors"""
    pass

class EmbeddingError(RAGSystemError):
    """Embedding model related errors"""
    pass

class ConfigurationError(RAGSystemError):
    """Configuration related errors"""
    pass
```

### Error Handling Examples

```python
# Example error handling in vector store operations
try:
    vector_store = get_vector_store(config)
    vector_store.add_documents(documents)
except VectorStoreError as e:
    print(f"Vector store error: {e}")
    # Handle vector store specific errors
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Handle configuration issues
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle other errors
```

## 🔍 Search and Filtering

### Advanced Search Options

```python
# Advanced search with filters (implementation varies by vector store)
def search_with_filters(query: str, filters: Dict[str, Any], k: int = 3):
    """Search with metadata filters

    Args:
        query: Search query
        filters: Metadata filters
        k: Number of results

    Returns:
        List[Dict]: Filtered search results
    """

# Example filter usage
filters = {
    "document_type": "legal",
    "jurisdiction": "federal",
    "year": {"$gte": 2020}  # Documents from 2020 or later
}

results = search_with_filters("contract law", filters, k=5)
```

## 📈 Performance Monitoring

### Performance Metrics

```python
# Built-in performance monitoring
class PerformanceMonitor:
    """Monitor system performance"""

    def record_search_time(self, duration: float):
        """Record search execution time"""

    def record_embedding_time(self, duration: float):
        """Record embedding generation time"""

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report

        Returns:
            Dict: Performance metrics including:
                - avg_search_time: Average search time
                - avg_embedding_time: Average embedding time
                - cache_hit_rate: Cache hit rate
                - total_requests: Total number of requests
        """
```

## 🚀 Usage Examples

### Basic Usage

```python
from rag.knowledge_base_manager import KnowledgeBaseManager
from rag.tools import RAGRetrievalToolFactory

# Initialize system
config = {"vector_db": "chroma", "embedding_model": "sentence-transformers"}
kb = KnowledgeBaseManager(config)

# Add documents
result = kb.add_documents_from_path("data/documents")

# Create RAG tool
rag_tool = RAGRetrievalToolFactory.create_tool(config)

# Search knowledge base
results = kb.search_knowledge_base("artificial intelligence", k=5)
```

### Advanced Usage

```python
from templates import AgentTemplates, TaskTemplates
from crewai import Crew

# Create agents with RAG capabilities
researcher = AgentTemplates.create_rag_researcher_agent(llm, rag_tool)
analyst = AgentTemplates.create_rag_analyst_agent(llm, rag_tool)

# Create tasks
research_task = TaskTemplates.create_research_task(researcher)
analysis_task = TaskTemplates.create_analysis_task(analyst)

# Create and run crew
crew = Crew(agents=[researcher, analyst], tasks=[research_task, analysis_task])
result = crew.kickoff()
```

This API reference provides comprehensive documentation for all major components of the RAG Multi-Agent Template. Use it as your definitive guide for building and customizing your RAG-enabled multi-agent systems! 🚀