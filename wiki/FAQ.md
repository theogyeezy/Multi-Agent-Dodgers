# Frequently Asked Questions (FAQ)

Quick answers to the most common questions about the RAG Multi-Agent Template.

## 🚀 Getting Started

### Q: What's the difference between `agent.py` and `rag_agent.py`?

**A:** Two different templates for different needs:

- **`agent.py`** - Basic template with web search
  - Uses SerperDev for web search
  - Simpler setup, fewer dependencies
  - Great for learning and simple tasks

- **`rag_agent.py`** - Advanced template with document processing
  - Searches your own documents (PDF, DOCX, TXT)
  - Uses vector databases for intelligent retrieval
  - More powerful but requires more setup

### Q: Which template should I start with?

**A:**
- **Beginners**: Start with `agent.py` to learn the basics
- **Document-based work**: Use `rag_agent.py` for processing your own files
- **Production use**: `rag_agent.py` for enterprise applications

### Q: Do I need both IBM WatsonxAI and OpenAI?

**A:** No! You can use either:
- **IBM WatsonxAI**: Enterprise-focused, included in template
- **OpenAI**: Popular alternative, requires code changes
- **Local models**: Possible with additional configuration

## 🔧 Setup Issues

### Q: Getting "Module not found" errors?

**A:** Install dependencies:
```bash
pip install -r requirements.txt
```

Common missing modules:
- `crewai` - Core framework
- `chromadb` - Vector database
- `sentence-transformers` - Embeddings
- `PyPDF2` - PDF processing

### Q: API key errors - "Invalid API key"?

**A:** Check these steps:
1. **Get the right key**: From IBM Cloud WatsonxAI service
2. **Set environment variable**: `export API_KEY="your_key"`
3. **Check quotes**: Ensure no extra spaces or quotes
4. **Verify service**: Confirm WatsonxAI service is active

### Q: Vector database errors?

**A:** Each database has different setup requirements:

**ChromaDB** (easiest):
```bash
pip install chromadb
# No additional setup needed
```

**FAISS**:
```bash
pip install faiss-cpu  # or faiss-gpu for GPU
```

**Pinecone**:
```bash
pip install pinecone-client
export PINECONE_API_KEY="your_key"
```

## 📁 Document Processing

### Q: What file types are supported?

**A:** Currently supported:
- **PDF** - Research papers, reports, manuals
- **DOCX** - Microsoft Word documents
- **TXT** - Plain text files
- **MD** - Markdown documentation

**Coming soon**: PowerPoint, Excel, HTML

### Q: Why are my documents not being found?

**A:** Check these common issues:

1. **File location**: Documents in `data/documents/` folder?
2. **File extensions**: Using supported types (.pdf, .docx, .txt, .md)?
3. **File permissions**: Can Python read the files?
4. **Path configuration**: Check `DOCUMENTS_PATH` in config

### Q: How do I improve search quality?

**A:** Several optimization strategies:

**Chunk size tuning**:
```python
config = {
    "chunk_size": 800,     # Larger for academic papers
    "chunk_overlap": 100,  # More overlap for better context
}
```

**Retrieval tuning**:
```python
config = {
    "retrieval_k": 5,  # Get more results
}
```

**Better embeddings**:
```python
config = {
    "embedding_model": "openai",  # More accurate but requires API key
}
```

## 🤖 Agent Configuration

### Q: How do I customize agent roles?

**A:** Replace placeholder text in agent definitions:

```python
# Before (placeholder)
researcher = Agent(
    role="FILL_IN_RESEARCHER_ROLE",
    goal="FILL_IN_RESEARCHER_GOAL",
    backstory="FILL_IN_RESEARCHER_BACKSTORY"
)

# After (customized)
researcher = Agent(
    role="Senior Legal Research Specialist",
    goal="Research legal precedents and analyze case law for client matters",
    backstory="You are an expert legal researcher with 15 years experience..."
)
```

### Q: Can I add more than 2 agents?

**A:** Absolutely! Add as many as needed:

```python
# Create additional agents
analyst = Agent(llm=llm, role="Data Analyst", ...)
reviewer = Agent(llm=llm, role="Quality Reviewer", ...)
editor = Agent(llm=llm, role="Content Editor", ...)

# Add to crew
crew = Crew(
    agents=[researcher, analyst, reviewer, editor],
    tasks=[task1, task2, task3, task4]
)
```

### Q: How do I make agents work together?

**A:** Use task dependencies:

```python
task2 = Task(
    description="Analyze the research findings...",
    context=[task1],  # Waits for task1 to complete
    agent=analyst
)
```

## 🏗️ Vector Databases

### Q: Which vector database should I choose?

**A:** Depends on your needs:

**ChromaDB** - Best for beginners:
- ✅ Easy setup (no external services)
- ✅ Good for development and small datasets
- ✅ Local storage
- ❌ Limited scalability

**FAISS** - Best for performance:
- ✅ Very fast search
- ✅ Handles large datasets
- ✅ Local deployment
- ❌ More complex setup

**Pinecone** - Best for production:
- ✅ Fully managed (no maintenance)
- ✅ Scales automatically
- ✅ Built-in backups
- ❌ Requires internet connection
- ❌ Monthly costs

### Q: Can I switch vector databases later?

**A:** Yes! Just change the configuration:

```python
# Change from ChromaDB to FAISS
config = {
    "vector_db": "faiss",  # was "chroma"
    # ... other settings
}
```

Note: You'll need to re-process your documents.

### Q: How much data can each database handle?

**A:** Rough guidelines:

- **ChromaDB**: Up to 100k documents
- **FAISS**: Millions of documents
- **Pinecone**: Billions of documents

## 🚀 Performance

### Q: Search is slow - how to speed up?

**A:** Several optimization options:

**1. Reduce chunk size**:
```python
config = {"chunk_size": 300}  # Smaller chunks = faster
```

**2. Use faster embeddings**:
```python
config = {"embedding_model_name": "all-MiniLM-L6-v2"}  # Faster model
```

**3. Limit search results**:
```python
config = {"retrieval_k": 3}  # Fewer results = faster
```

**4. Upgrade vector database**:
```python
config = {"vector_db": "faiss"}  # Faster than ChromaDB
```

### Q: Memory usage is too high?

**A:** Memory optimization tips:

**1. Use smaller embedding models**:
```python
config = {"embedding_model_name": "all-MiniLM-L6-v2"}  # 80MB vs 500MB+
```

**2. Process documents in batches**:
```python
config = {"batch_size": 50}  # Process fewer documents at once
```

**3. Use FAISS with compression**:
```python
config = {
    "vector_db": "faiss",
    "index_type": "IndexIVFPQ"  # Compressed index
}
```

## 🔒 Security & Privacy

### Q: Is my data secure?

**A:** Security depends on your configuration:

**Local deployment** (ChromaDB/FAISS):
- ✅ Data stays on your machine
- ✅ No external API calls for search
- ⚠️ Still sends text to LLM APIs

**Cloud deployment** (Pinecone):
- ⚠️ Data stored in cloud
- ✅ Enterprise security features
- ⚠️ Subject to service provider policies

### Q: How to handle sensitive documents?

**A:** Several approaches:

**1. Local-only setup**:
```python
config = {
    "vector_db": "faiss",           # Local storage
    "embedding_model": "sentence-transformers"  # Local embeddings
}
# Still need local LLM for full privacy
```

**2. PII scrubbing**:
```python
# Remove sensitive info before processing
def scrub_document(text):
    # Remove SSNs, emails, etc.
    return cleaned_text
```

**3. On-premise deployment**:
- Deploy entire system on your infrastructure
- Use local LLM models
- No external API calls

### Q: Can I use this without internet?

**A:** Partially, with local components:

**Offline components**:
- ✅ ChromaDB/FAISS vector storage
- ✅ Sentence Transformers embeddings
- ✅ Document processing

**Requires internet**:
- ❌ IBM WatsonxAI API calls
- ❌ Web search (SerperDev)
- ❌ OpenAI embeddings

**Full offline solution**: Use local LLM models (like Ollama) with code modifications.

## 🛠️ Troubleshooting

### Q: Agents give empty or poor responses?

**A:** Common fixes:

**1. Check agent configuration**:
```python
# Ensure roles and goals are specific
role="Legal Research Specialist"  # Not just "Researcher"
goal="Find relevant case law for contract disputes"  # Not just "Research"
```

**2. Improve document quality**:
- Remove scanned/image PDFs (use OCR first)
- Ensure good document formatting
- Add metadata to files

**3. Tune search parameters**:
```python
config = {
    "retrieval_k": 5,      # Get more context
    "chunk_size": 800,     # Larger chunks for more context
}
```

### Q: Getting rate limit errors?

**A:** API rate limiting solutions:

**1. Add delays**:
```python
import time
time.sleep(1)  # Wait between API calls
```

**2. Batch requests**:
```python
# Process documents in smaller batches
config = {"batch_size": 10}
```

**3. Upgrade API plan**:
- IBM WatsonxAI: Higher tier plans
- OpenAI: Increase rate limits

### Q: Documents aren't being processed correctly?

**A:** Document processing issues:

**PDF problems**:
```python
# Try different PDF library
pip install pypdf  # Alternative to PyPDF2
```

**Encoding issues**:
```python
# The template handles common encoding issues automatically
# For custom processing, specify encoding:
with open(file, 'r', encoding='utf-8') as f:
    content = f.read()
```

**Large file issues**:
```python
config = {
    "chunk_size": 500,     # Smaller chunks for large files
    "max_file_size_mb": 50  # Skip very large files
}
```

## 📚 Use Cases

### Q: Can this work for my specific domain?

**A:** Very likely! The template is designed for any domain:

**Already supported examples**:
- Legal research
- Technical documentation
- Customer support
- Medical literature
- Business analysis

**Customization for other domains**:
1. Replace documents with domain-specific content
2. Customize agent roles for your field
3. Adjust prompts and expected outputs
4. Use domain-specific embedding models

### Q: How do I adapt for my language?

**A:** Multi-language support:

**1. Use multilingual embeddings**:
```python
config = {
    "embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}
```

**2. Configure LLM for your language**:
```python
# Update prompts and agent descriptions in your language
role="Especialista en Investigación Legal"  # Spanish example
```

**3. Process documents in your language**:
- Most document processors work with any language
- Ensure proper text encoding

## 💡 Advanced Features

### Q: Can I integrate with my existing systems?

**A:** Yes! Several integration options:

**1. API wrapper**:
```python
# Create REST API around the template
from fastapi import FastAPI
app = FastAPI()

@app.post("/research")
def research_endpoint(query: str):
    # Use your RAG system here
    return results
```

**2. Database integration**:
```python
# Load documents from database
def load_from_database():
    # Connect to your database
    # Fetch documents
    # Process with template
```

**3. Webhook integration**:
```python
# Trigger research on external events
# Process results automatically
# Send to other systems
```

### Q: How do I monitor performance?

**A:** Add monitoring:

```python
import time
import logging

# Log performance metrics
start_time = time.time()
results = knowledge_base.search_knowledge_base(query)
search_time = time.time() - start_time

logging.info(f"Search completed in {search_time:.2f}s")
```

## 🆘 Still Need Help?

### Getting More Support

1. **Check Documentation**:
   - [Getting Started](Getting-Started) - Basic setup
   - [Advanced Configuration](Advanced-Configuration) - Deep customization
   - [Troubleshooting](Troubleshooting) - Detailed problem solving

2. **Community Support**:
   - GitHub Issues - Bug reports and feature requests
   - GitHub Discussions - Questions and community help

3. **Professional Support**:
   - IBM WatsonxAI support for API issues
   - Vector database vendor support for database issues

### Before Asking for Help

Please include:
- [ ] Error messages (full stack trace)
- [ ] Your configuration settings
- [ ] Steps to reproduce the issue
- [ ] Environment details (OS, Python version)
- [ ] What you expected vs. what happened

This helps us help you faster! 🚀