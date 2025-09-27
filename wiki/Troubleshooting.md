# Troubleshooting Guide

Comprehensive solutions for common issues with the RAG Multi-Agent Template.

## 🚨 Quick Diagnostic Checklist

Before diving into specific issues, run this quick diagnostic:

```bash
# 1. Check Python version
python --version  # Should be 3.8+

# 2. Check dependencies
pip list | grep -E "(crewai|chromadb|sentence-transformers)"

# 3. Test basic imports
python -c "import crewai, chromadb; print('Core imports OK')"

# 4. Check environment variables
echo $API_KEY | head -c 20  # Should show first 20 chars of your key
```

## 🔧 Installation Issues

### "Module not found" Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'crewai'
ModuleNotFoundError: No module named 'chromadb'
```

**Solutions:**

1. **Install requirements**:
```bash
pip install -r requirements.txt
```

2. **Check Python environment**:
```bash
which python
which pip
# Ensure you're in the right virtual environment
```

3. **Virtual environment setup**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

4. **Dependency conflicts**:
```bash
pip install --upgrade pip
pip install --force-reinstall -r requirements.txt
```

### ChromaDB Installation Issues

**Symptoms:**
```
ERROR: Failed building wheel for chroma-hnswlib
```

**Solutions:**

1. **Install build tools** (Linux/Mac):
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# macOS
xcode-select --install
```

2. **Use conda instead**:
```bash
conda install -c conda-forge chromadb
```

3. **Alternative installation**:
```bash
pip install chromadb --no-binary :all:
```

### FAISS Installation Issues

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement faiss-gpu
```

**Solutions:**

1. **Use CPU version**:
```bash
pip install faiss-cpu
```

2. **Platform-specific installation**:
```bash
# For M1/M2 Macs
conda install -c pytorch faiss-cpu

# For Windows
pip install faiss-cpu --no-cache
```

## 🔑 API Key Issues

### Invalid API Key Errors

**Symptoms:**
```
AuthenticationError: Invalid API key
Unauthorized: 401 Client Error
```

**Solutions:**

1. **Verify API key format**:
```bash
# IBM WatsonxAI keys typically start with specific patterns
echo $API_KEY | head -c 10
```

2. **Check environment variables**:
```bash
env | grep API
# Should show your API keys
```

3. **Test API connectivity**:
```python
import os
from langchain_ibm import WatsonxLLM

try:
    llm = WatsonxLLM(
        model_name="YOUR_MODEL",
        api_key=os.environ["API_KEY"],
        # ... other params
    )
    result = llm.invoke("Test")
    print("API connection successful")
except Exception as e:
    print(f"API error: {e}")
```

4. **Common fixes**:
```bash
# Remove quotes and spaces
export API_KEY=your_actual_key_here

# Reload environment
source ~/.bashrc  # or ~/.zshrc
```

### Service URL Issues

**Symptoms:**
```
ConnectionError: Failed to establish connection
Invalid service URL
```

**Solutions:**

1. **Verify service URL format**:
```python
# Should be like: https://us-south.ml.cloud.ibm.com
service_url = "https://your-region.ml.cloud.ibm.com"
```

2. **Check IBM Cloud service status**:
- Log into IBM Cloud console
- Verify WatsonxAI service is active
- Check service endpoints

3. **Test connectivity**:
```bash
curl -I https://your-service-url.com
# Should return 200 or 401, not connection errors
```

## 📁 Document Processing Issues

### PDF Processing Errors

**Symptoms:**
```
PdfReadError: Could not read PDF
UnicodeDecodeError: 'utf-8' codec can't decode
```

**Solutions:**

1. **Try alternative PDF library**:
```python
# If PyPDF2 fails, try pypdf
pip uninstall PyPDF2
pip install pypdf
```

2. **Handle corrupted PDFs**:
```python
def safe_pdf_processing(file_path):
    try:
        # Try primary method
        processor = PDFDocumentProcessor(config)
        return processor.process_document(file_path)
    except Exception as e:
        print(f"Primary PDF processing failed: {e}")
        # Try alternative or skip
        return []
```

3. **OCR for scanned PDFs**:
```bash
pip install pytesseract
# Then use OCR preprocessing
```

4. **File permissions**:
```bash
ls -la data/documents/
# Ensure Python can read the files
chmod 644 data/documents/*.pdf
```

### Document Not Found Issues

**Symptoms:**
```
FileNotFoundError: No such file or directory
No documents found to process
```

**Solutions:**

1. **Check file paths**:
```python
import os
docs_path = "data/documents"
print(f"Directory exists: {os.path.exists(docs_path)}")
print(f"Files found: {os.listdir(docs_path)}")
```

2. **Verify supported file types**:
```python
from rag.document_processors import DocumentProcessorFactory
print(f"Supported extensions: {DocumentProcessorFactory.get_supported_extensions()}")
```

3. **Create directory structure**:
```bash
mkdir -p data/documents
# Add your documents here
```

4. **File encoding issues**:
```python
# For text files with encoding problems
def read_with_fallback_encoding(file_path):
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise Exception(f"Could not decode {file_path}")
```

## 🗄️ Vector Database Issues

### ChromaDB Errors

**Symptoms:**
```
ValueError: Collection already exists
sqlite3.OperationalError: database is locked
```

**Solutions:**

1. **Collection management**:
```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")

# Delete existing collection
try:
    client.delete_collection("your_collection_name")
except:
    pass

# Create new collection
collection = client.create_collection("your_collection_name")
```

2. **Database lock issues**:
```bash
# Stop all Python processes using ChromaDB
pkill -f python

# Remove lock files
rm -f ./chroma_db/*.lock
```

3. **Permissions issues**:
```bash
chmod -R 755 ./chroma_db
```

### FAISS Errors

**Symptoms:**
```
AssertionError: Index not trained
RuntimeError: FAISS index is empty
```

**Solutions:**

1. **Index training**:
```python
import faiss
import numpy as np

# Ensure you have enough training data
if len(training_vectors) < 1000:
    print("Warning: Small training set, using flat index")
    index = faiss.IndexFlatIP(dimension)
else:
    # Train IVF index
    index.train(training_vectors)
```

2. **Empty index handling**:
```python
def safe_faiss_search(index, query_vector, k):
    if index.ntotal == 0:
        print("Warning: FAISS index is empty")
        return [], []
    return index.search(query_vector, min(k, index.ntotal))
```

### Pinecone Errors

**Symptoms:**
```
PineconeApiError: Index not found
PineconeApiError: Quota exceeded
```

**Solutions:**

1. **Index creation**:
```python
import pinecone

# Initialize with correct environment
pinecone.init(
    api_key="your-key",
    environment="your-environment"  # us-west1-gcp, etc.
)

# Check existing indexes
print(pinecone.list_indexes())

# Create if doesn't exist
if "your-index" not in pinecone.list_indexes():
    pinecone.create_index("your-index", dimension=384)
```

2. **Quota management**:
```python
# Check index stats
index = pinecone.Index("your-index")
stats = index.describe_index_stats()
print(f"Vector count: {stats['total_vector_count']}")
print(f"Dimension: {stats['dimension']}")
```

## 🤖 Agent Execution Issues

### Empty Agent Responses

**Symptoms:**
- Agents return empty strings
- No content in output files
- Agents seem to run but produce no results

**Solutions:**

1. **Check agent configuration**:
```python
# Ensure all placeholders are filled
researcher = Agent(
    llm=llm,
    role="Senior Research Specialist",  # Not "FILL_IN_ROLE"
    goal="Research and analyze topics thoroughly",  # Specific goal
    backstory="You are an expert researcher...",  # Detailed backstory
    tools=[rag_tool],
    verbose=1  # Enable verbose logging
)
```

2. **Verify LLM configuration**:
```python
# Test LLM directly
test_response = llm.invoke("What is artificial intelligence?")
print(f"LLM test response: {test_response}")
```

3. **Check task configuration**:
```python
task = Task(
    description="Research the latest developments in AI",  # Specific description
    expected_output="A comprehensive report with key findings",  # Clear output
    agent=researcher,
    output_file="ai_research.md"
)
```

4. **Tool availability**:
```python
# Verify RAG tool is working
try:
    test_results = rag_tool._run("test query")
    print(f"RAG tool test: {test_results}")
except Exception as e:
    print(f"RAG tool error: {e}")
```

### Agent Timeout Issues

**Symptoms:**
```
TimeoutError: Agent execution exceeded time limit
ConnectionTimeout: Request timed out
```

**Solutions:**

1. **Increase timeouts**:
```python
llm = WatsonxLLM(
    # ... other params
    request_timeout=120,  # Increase from default
    max_retries=5
)
```

2. **Simplify tasks**:
```python
# Break complex tasks into smaller ones
task1 = Task(description="Research topic A", agent=researcher)
task2 = Task(description="Analyze findings", agent=analyst, context=[task1])
```

3. **Optimize retrieval**:
```python
config = {
    "retrieval_k": 3,  # Reduce from 5 or 10
    "chunk_size": 500  # Smaller chunks = faster processing
}
```

### Agent Error Messages

**Symptoms:**
```
ValueError: Agent role cannot be empty
AttributeError: 'NoneType' object has no attribute
```

**Solutions:**

1. **Validate agent creation**:
```python
def create_safe_agent(llm, role, goal, backstory, tools):
    if not all([role, goal, backstory]):
        raise ValueError("Role, goal, and backstory must be provided")

    return Agent(
        llm=llm,
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools or [],
        verbose=1
    )
```

2. **Error handling in workflow**:
```python
try:
    result = crew.kickoff()
    print("Workflow completed successfully")
except Exception as e:
    print(f"Workflow error: {e}")
    # Implement fallback or retry logic
```

## 🚀 Performance Issues

### Slow Search Performance

**Symptoms:**
- Search takes > 5 seconds
- High memory usage during search
- System becomes unresponsive

**Solutions:**

1. **Optimize chunk size**:
```python
config = {
    "chunk_size": 300,     # Smaller chunks = faster search
    "chunk_overlap": 25,   # Reduce overlap
    "retrieval_k": 3       # Fewer results
}
```

2. **Use faster embeddings**:
```python
config = {
    "embedding_model_name": "all-MiniLM-L6-v2"  # Fast, small model
}
```

3. **Implement caching**:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_search(query, k):
    return knowledge_base.search_knowledge_base(query, k)
```

4. **Optimize vector database**:
```python
# For FAISS
config = {
    "vector_db": "faiss",
    "index_type": "IndexIVFFlat",  # Faster than flat for large datasets
    "nlist": 100
}
```

### Memory Issues

**Symptoms:**
```
MemoryError: Unable to allocate array
Process killed (out of memory)
```

**Solutions:**

1. **Batch processing**:
```python
def process_documents_in_batches(file_paths, batch_size=10):
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        chunks = DocumentProcessorFactory.process_documents(batch, config)
        vector_store.add_documents(chunks)
        print(f"Processed batch {i//batch_size + 1}")
```

2. **Use smaller models**:
```python
config = {
    "embedding_model_name": "all-MiniLM-L6-v2",  # 80MB instead of 500MB+
}
```

3. **Limit document size**:
```python
def filter_large_documents(file_paths, max_size_mb=10):
    filtered = []
    for path in file_paths:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb <= max_size_mb:
            filtered.append(path)
        else:
            print(f"Skipping large file: {path} ({size_mb:.1f}MB)")
    return filtered
```

## 🔍 Search Quality Issues

### Poor Search Results

**Symptoms:**
- Irrelevant documents returned
- Missing obvious matches
- Low-quality responses

**Solutions:**

1. **Improve chunking strategy**:
```python
config = {
    "chunk_size": 800,     # Larger chunks for better context
    "chunk_overlap": 100,  # More overlap for continuity
}
```

2. **Better embedding model**:
```python
config = {
    "embedding_model": "openai",  # More accurate than sentence-transformers
    "embedding_model_name": "text-embedding-ada-002"
}
```

3. **Metadata filtering**:
```python
def search_with_filters(query, document_type=None, date_range=None):
    filters = {}
    if document_type:
        filters["document_type"] = document_type
    # Apply filters in search
    return vector_store.search_with_filters(query, filters)
```

4. **Query preprocessing**:
```python
def preprocess_query(query):
    # Expand abbreviations, fix typos, etc.
    query = query.replace("AI", "artificial intelligence")
    query = query.replace("ML", "machine learning")
    return query
```

### No Search Results

**Symptoms:**
- Always returns "No documents found"
- Empty vector database
- Search never finds matches

**Solutions:**

1. **Verify document ingestion**:
```python
# Check if documents were actually added
info = knowledge_base.get_knowledge_base_info()
print(f"Document count: {info.get('count', 0)}")

if info.get('count', 0) == 0:
    print("No documents in knowledge base - run document ingestion")
```

2. **Test with simple queries**:
```python
# Try very simple, common words
test_queries = ["the", "and", "document", "information"]
for query in test_queries:
    results = knowledge_base.search_knowledge_base(query, k=1)
    print(f"Query '{query}': {len(results)} results")
```

3. **Check embedding dimensions**:
```python
# Ensure query and document embeddings have same dimensions
from rag.embeddings import get_embeddings

embeddings_model = get_embeddings(config)
test_embedding = embeddings_model.embed_query("test")
print(f"Embedding dimension: {len(test_embedding)}")
```

## 🔄 Recovery Procedures

### Reset Vector Database

```bash
# ChromaDB
rm -rf ./chroma_db
python -c "from rag.knowledge_base_manager import KnowledgeBaseManager; kb = KnowledgeBaseManager(config); kb.add_documents_from_path('data/documents')"

# FAISS
rm -rf ./faiss_db
# Re-run document processing

# Pinecone
python -c "import pinecone; pinecone.init(api_key='your-key'); pinecone.delete_index('your-index')"
```

### Rebuild Knowledge Base

```python
# Complete rebuild script
from rag.knowledge_base_manager import KnowledgeBaseManager

# Clear existing
kb = KnowledgeBaseManager(config)
kb.clear_knowledge_base()

# Rebuild from documents
result = kb.add_documents_from_path("data/documents")
print(f"Rebuild result: {result}")

# Verify
info = kb.get_knowledge_base_info()
print(f"New document count: {info.get('count', 0)}")
```

### Environment Reset

```bash
# Complete environment reset
deactivate  # Exit virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Clear caches
rm -rf __pycache__ */__pycache__ */*/__pycache__
rm -rf .chroma_db .faiss_db
```

## 🆘 When All Else Fails

### Diagnostic Script

Create `debug.py`:

```python
#!/usr/bin/env python3
"""Comprehensive diagnostic script"""

import sys
import os
import importlib

def check_python_version():
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✅ Python version OK")
    return True

def check_dependencies():
    required = [
        'crewai', 'chromadb', 'sentence_transformers',
        'langchain_ibm', 'numpy', 'pandas'
    ]

    for package in required:
        try:
            importlib.import_module(package)
            print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ {package} import failed: {e}")
            return False
    return True

def check_environment():
    api_key = os.getenv('API_KEY')
    if api_key:
        print(f"✅ API_KEY set (length: {len(api_key)})")
    else:
        print("❌ API_KEY not set")
        return False
    return True

def check_file_structure():
    required_paths = [
        'data/documents',
        'config/rag_config_template.py',
        'rag/__init__.py'
    ]

    for path in required_paths:
        if os.path.exists(path):
            print(f"✅ {path} exists")
        else:
            print(f"❌ {path} missing")
            return False
    return True

if __name__ == "__main__":
    print("=== RAG Multi-Agent Template Diagnostics ===\n")

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Environment", check_environment),
        ("File Structure", check_file_structure)
    ]

    all_passed = True
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        if not check_func():
            all_passed = False

    print(f"\n=== Summary ===")
    if all_passed:
        print("✅ All checks passed! Your environment looks good.")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
```

Run with:
```bash
python debug.py
```

### Getting Help

If you're still stuck:

1. **Gather information**:
   - Run the diagnostic script above
   - Copy full error messages
   - Note your operating system and Python version
   - List what you were trying to do when the error occurred

2. **Check documentation**:
   - [FAQ](FAQ) for common questions
   - [Getting Started](Getting-Started) for setup issues
   - [Advanced Configuration](Advanced-Configuration) for complex problems

3. **Community support**:
   - Search existing GitHub issues
   - Create new issue with diagnostic information
   - Join discussions for community help

4. **Professional support**:
   - IBM WatsonxAI support for API issues
   - Vector database vendor support for database-specific problems

Remember: The more specific information you provide, the faster we can help you solve the problem! 🚀