# Multi-Agent RAG Template

A comprehensive template for creating RAG-enabled multi-agent systems using CrewAI and IBM WatsonxAI.

## Overview

This template provides both basic multi-agent functionality and advanced RAG (Retrieval-Augmented Generation) capabilities. Choose from two implementations:

### Basic Multi-Agent System (`agent.py`)
- **Researcher Agent** - Conducts web research using search tools
- **Writer Agent** - Creates written content based on research findings

### RAG-Enabled Multi-Agent System (`rag_agent.py`)
- **RAG-Powered Agents** - Access local knowledge bases for document-based research
- **Configurable Vector Databases** - Support for ChromaDB, FAISS, and Pinecone
- **Document Processing** - Handle PDF, DOCX, TXT, and Markdown files
- **Flexible Templates** - Pre-built configurations for legal, technical, and support use cases

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Set your environment variables:
```bash
export API_KEY="your_watsonx_api_key"
export SERPER_API_KEY="your_serper_api_key"  # Optional for web search
```

### 3. Choose Your Implementation

#### Basic Multi-Agent System
```bash
python agent.py
```

#### RAG-Enabled System
1. Configure your settings in `config/rag_config_template.py`
2. Add documents to `data/documents/`
3. Run the RAG system:
```bash
python rag_agent.py
```

## RAG Features

### Vector Database Support
- **ChromaDB** - Local vector database (default)
- **FAISS** - High-performance similarity search
- **Pinecone** - Cloud-based vector database

### Document Processing
- **PDF** - Extract text from PDF documents
- **DOCX** - Process Word documents
- **TXT/MD** - Plain text and Markdown files
- **Chunking** - Intelligent text splitting with overlap

### Embedding Models
- **Sentence Transformers** - Local embeddings (default)
- **OpenAI** - Cloud-based embeddings
- **Custom** - Easy integration of other models

## Configuration

### RAG Configuration (`config/rag_config_template.py`)
```python
class RAGConfig:
    VECTOR_DB = "chroma"  # or "faiss", "pinecone"
    EMBEDDING_MODEL = "sentence-transformers"
    CHUNK_SIZE = 500
    RETRIEVAL_K = 3
    # ... more options
```

### Agent Templates
Pre-built templates for common use cases:
- **Legal Research** - Legal document analysis and research
- **Technical Documentation** - API docs and technical guides
- **Customer Support** - FAQ and troubleshooting assistance
- **Medical Research** - Clinical literature review
- **Business Analysis** - Market research and intelligence

## Examples

### Legal Research System
```python
# See examples/legal_research_example.py
python examples/legal_research_example.py
```

### Technical Documentation
```python
# See examples/technical_docs_example.py
python examples/technical_docs_example.py
```

### Customer Support
```python
# See examples/customer_support_example.py
python examples/customer_support_example.py
```

## Project Structure

```
Multi Agent/
├── agent.py                 # Basic multi-agent system
├── rag_agent.py            # RAG-enabled system
├── config/
│   └── rag_config_template.py
├── rag/
│   ├── vector_stores/       # Vector database implementations
│   ├── document_processors/ # Document processing pipeline
│   ├── embeddings/         # Embedding model abstractions
│   ├── tools/              # RAG retrieval tools
│   └── knowledge_base_manager.py
├── templates/
│   ├── agent_templates.py   # Pre-built agent configurations
│   └── task_templates.py    # Pre-built task templates
├── examples/               # Complete use case examples
├── data/
│   └── documents/          # Your document storage
└── requirements.txt
```

## Customization

### Creating Custom Agents
```python
from templates import AgentTemplates

custom_agent = AgentTemplates.create_custom_rag_agent(
    llm=llm,
    rag_tool=rag_tool,
    role="Your Custom Role",
    goal="Your Custom Goal",
    backstory="Your Custom Backstory"
)
```

### Adding Custom Tasks
```python
from templates import TaskTemplates

custom_task = TaskTemplates.create_custom_task(
    agent=agent,
    description="Your task description",
    expected_output="Your expected output format",
    output_file="output.md"
)
```

### Knowledge Base Management
```python
from rag.knowledge_base_manager import KnowledgeBaseManager

kb = KnowledgeBaseManager(config)
result = kb.add_documents_from_path("path/to/documents")
search_results = kb.search_knowledge_base("query", k=5)
```

## Requirements

### Core Dependencies
- CrewAI >= 0.28.8
- langchain-ibm >= 0.1.0
- IBM WatsonxAI access

### Vector Database Options (choose one)
- chromadb >= 0.4.15 (local)
- faiss-cpu >= 1.7.4 (local)
- pinecone-client >= 2.2.4 (cloud)

### Document Processing
- PyPDF2 >= 3.0.1 (PDF support)
- python-docx >= 0.8.11 (DOCX support)
- sentence-transformers >= 2.2.2 (embeddings)

## Advanced Features

### Multiple Vector Databases
Switch between vector databases by changing configuration:
```python
config = {"vector_db": "chroma"}  # or "faiss", "pinecone"
```

### Custom Embedding Models
Implement custom embeddings:
```python
from rag.embeddings import BaseEmbeddings

class CustomEmbeddings(BaseEmbeddings):
    def embed_documents(self, texts):
        # Your implementation
        pass
```

### Batch Document Processing
Process multiple document types:
```python
from rag.document_processors import DocumentProcessorFactory

chunks = DocumentProcessorFactory.process_documents(file_paths, config)
```

## License

This template is provided as-is for educational and development purposes.