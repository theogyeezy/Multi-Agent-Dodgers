# Advanced Configuration Guide

This comprehensive guide covers advanced configuration options for power users who want to customize every aspect of their RAG multi-agent system.

## 🎯 What You'll Master

- **Vector Database Optimization** - Fine-tune ChromaDB, FAISS, and Pinecone
- **Custom Embedding Models** - Implement your own embeddings
- **Advanced Agent Architectures** - Multi-tier agent systems
- **Performance Optimization** - Scale to production workloads
- **Security & Privacy** - Enterprise-grade configurations

## 📋 Prerequisites

- Completed [Getting Started](Getting-Started) guide
- Understanding of vector databases and embeddings
- Python experience with classes and inheritance
- Familiarity with AI/ML concepts

## 🏗️ Architecture Deep Dive

### System Components Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Agent Layer   │    │   Task Layer    │    │  Output Layer   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Research      │    │ • Analysis      │    │ • Reports       │
│ • Analysis      │──▶ │ • Generation    │──▶ │ • Summaries     │
│ • Writing       │    │ • Validation    │    │ • Documents     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RAG Layer     │    │  Vector Store   │    │   Knowledge     │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Retrieval     │    │ • ChromaDB      │    │ • Documents     │
│ • Embedding     │◀─▶ │ • FAISS         │◀─▶ │ • Metadata      │
│ • Ranking       │    │ • Pinecone      │    │ • Indices       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## ⚙️ Vector Database Configuration

### ChromaDB Advanced Setup

```python
# config/advanced_chroma_config.py
class AdvancedChromaConfig:
    def __init__(self):
        self.config = {
            "vector_db": "chroma",
            "persist_directory": "./advanced_chroma_db",
            "collection_name": "advanced_knowledge_base",

            # Performance settings
            "batch_size": 1000,
            "max_batch_size": 5000,
            "query_cache_size": 10000,

            # Distance metrics
            "distance_metric": "cosine",  # cosine, l2, ip

            # Collection settings
            "embedding_function": "custom",
            "metadata_schema": {
                "source": "str",
                "chunk_index": "int",
                "timestamp": "datetime",
                "category": "str",
                "importance": "float"
            }
        }

    def get_client_settings(self):
        return {
            "tenant": "default_tenant",
            "database": "default_database",
            "settings": {
                "chroma_db_impl": "duckdb+parquet",
                "chroma_api_impl": "chromadb.api.segment.SegmentAPI"
            }
        }
```

### FAISS Optimization

```python
# rag/vector_stores/optimized_faiss_store.py
import faiss
import numpy as np
from typing import Dict, Any, List

class OptimizedFAISSStore(FAISSVectorStore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.index_type = config.get("index_type", "IndexFlatIP")
        self.nlist = config.get("nlist", 100)  # Number of clusters
        self.nprobe = config.get("nprobe", 10)  # Number of clusters to search

    def _create_optimized_index(self):
        """Create optimized FAISS index based on data size and requirements"""
        dimension = self.dimension

        if self.index_type == "IndexIVFFlat":
            # For large datasets (>10k vectors)
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist)
            index.nprobe = self.nprobe

        elif self.index_type == "IndexIVFPQ":
            # For very large datasets with memory constraints
            quantizer = faiss.IndexFlatIP(dimension)
            m = 8  # Number of subquantizers
            nbits = 8  # Bits per subquantizer
            index = faiss.IndexIVFPQ(quantizer, dimension, self.nlist, m, nbits)
            index.nprobe = self.nprobe

        elif self.index_type == "IndexHNSW":
            # For fast similarity search
            M = 32  # Number of bi-directional links
            index = faiss.IndexHNSWFlat(dimension, M)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 100

        else:
            # Default flat index
            index = faiss.IndexFlatIP(dimension)

        return index

    def train_index(self, training_vectors: np.ndarray):
        """Train the index with a subset of data for optimal performance"""
        if hasattr(self._index, 'train'):
            print(f"Training index with {len(training_vectors)} vectors...")
            self._index.train(training_vectors)
            print("Index training completed")
```

### Pinecone Production Setup

```python
# rag/vector_stores/production_pinecone_store.py
class ProductionPineconeStore(PineconeVectorStore):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pod_type = config.get("pod_type", "p1.x1")
        self.replicas = config.get("replicas", 1)
        self.shards = config.get("shards", 1)

    def create_production_index(self):
        """Create a production-ready Pinecone index"""
        index_config = {
            "name": self.collection_name,
            "dimension": self.dimension,
            "metric": "cosine",
            "pod_type": self.pod_type,
            "replicas": self.replicas,
            "shards": self.shards,
            "metadata_config": {
                "indexed": ["source", "category", "timestamp"]
            }
        }

        if self.collection_name not in self._client.list_indexes():
            self._client.create_index(**index_config)

        return self._client.Index(self.collection_name)

    def bulk_upsert(self, vectors: List[Dict], batch_size: int = 100):
        """Efficiently upsert large amounts of data"""
        index = self._get_index()

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                index.upsert(vectors=batch, async_req=True)
                if i % 1000 == 0:
                    print(f"Upserted {i + len(batch)} vectors...")
            except Exception as e:
                print(f"Error upserting batch {i}: {e}")
                # Implement retry logic here
```

## 🧠 Custom Embedding Models

### Implementing Custom Embeddings

```python
# rag/embeddings/custom_embeddings.py
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
from .base_embeddings import BaseEmbeddings

class CustomTransformerEmbeddings(BaseEmbeddings):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model_name = config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = config.get("max_length", 512)
        self.batch_size = config.get("batch_size", 32)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with batching for efficiency"""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings.cpu().numpy().tolist())

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self.embed_documents([text])[0]

# Domain-specific embedding model
class LegalDocumentEmbeddings(CustomTransformerEmbeddings):
    def __init__(self, config: dict):
        # Use legal domain-specific model
        config["model_name"] = "nlpaueb/legal-bert-base-uncased"
        super().__init__(config)

    def preprocess_legal_text(self, text: str) -> str:
        """Legal document preprocessing"""
        # Remove citations, normalize case references, etc.
        import re

        # Remove case citations
        text = re.sub(r'\\b\\d+\\s+[A-Z][a-z]+\\s+\\d+\\b', '[CASE_CITATION]', text)

        # Normalize statute references
        text = re.sub(r'\\b\\d+\\s+U\\.S\\.C\\.\\s+§\\s+\\d+\\b', '[STATUTE_REF]', text)

        return text

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Override to add legal preprocessing"""
        processed_texts = [self.preprocess_legal_text(text) for text in texts]
        return super().embed_documents(processed_texts)
```

## 🎛️ Advanced Agent Configuration

### Multi-Tier Agent Architecture

```python
# templates/advanced_agent_templates.py
from crewai import Agent, Task, Crew
from typing import List, Dict, Any

class AdvancedAgentTemplates:
    @staticmethod
    def create_supervisor_agent(llm, tools: List) -> Agent:
        """Creates a supervisor agent that coordinates other agents"""
        return Agent(
            llm=llm,
            role="AI System Supervisor",
            goal="Coordinate and optimize the work of specialized agents to achieve complex tasks efficiently",
            backstory="""You are an advanced AI supervisor with expertise in task decomposition,
            agent coordination, and quality assurance. You excel at breaking down complex requests
            into subtasks and ensuring high-quality deliverables.""",
            tools=tools,
            allow_delegation=True,
            max_delegation_depth=2,
            verbose=1
        )

    @staticmethod
    def create_specialist_hierarchy(llm, rag_tool, domain: str) -> List[Agent]:
        """Creates a hierarchy of specialist agents"""

        if domain == "legal":
            return [
                # Senior specialist
                Agent(
                    llm=llm,
                    role="Senior Legal Research Director",
                    goal="Oversee complex legal research projects and ensure accuracy",
                    backstory="20+ years experience in legal research and analysis",
                    tools=[rag_tool],
                    allow_delegation=True,
                    verbose=1
                ),
                # Mid-level specialists
                Agent(
                    llm=llm,
                    role="Case Law Specialist",
                    goal="Research and analyze relevant case law and precedents",
                    backstory="Expert in legal precedent research and case analysis",
                    tools=[rag_tool],
                    verbose=1
                ),
                Agent(
                    llm=llm,
                    role="Statutory Analysis Specialist",
                    goal="Analyze statutes, regulations, and compliance requirements",
                    backstory="Specialist in regulatory compliance and statutory interpretation",
                    tools=[rag_tool],
                    verbose=1
                ),
                # Quality assurance
                Agent(
                    llm=llm,
                    role="Legal QA Reviewer",
                    goal="Review and validate legal research for accuracy and completeness",
                    backstory="Quality assurance expert with focus on legal accuracy",
                    tools=[rag_tool],
                    verbose=1
                )
            ]

        # Add other domain hierarchies here
        return []

class AdvancedWorkflowTemplates:
    @staticmethod
    def create_research_pipeline(agents: List[Agent]) -> List[Task]:
        """Creates a sophisticated research pipeline with validation"""

        tasks = []

        # Initial research task
        research_task = Task(
            description="""Conduct comprehensive research on the given topic using all available
            knowledge sources. Focus on finding the most relevant and authoritative information.""",
            expected_output="Structured research findings with source citations and confidence scores",
            agent=agents[1],  # Case law specialist
            output_file="research_findings.md"
        )
        tasks.append(research_task)

        # Analysis task
        analysis_task = Task(
            description="""Analyze the research findings to identify key insights, patterns,
            and implications. Provide structured analysis with recommendations.""",
            expected_output="Comprehensive analysis with actionable insights and recommendations",
            agent=agents[2],  # Statutory specialist
            context=[research_task],
            output_file="analysis_report.md"
        )
        tasks.append(analysis_task)

        # Quality review task
        review_task = Task(
            description="""Review all research and analysis for accuracy, completeness,
            and adherence to quality standards. Identify any gaps or issues.""",
            expected_output="Quality assessment report with validation results and recommendations",
            agent=agents[3],  # QA reviewer
            context=[research_task, analysis_task],
            output_file="quality_review.md"
        )
        tasks.append(review_task)

        # Supervisor synthesis task
        synthesis_task = Task(
            description="""Synthesize all findings into a final comprehensive report.
            Ensure quality, accuracy, and completeness of deliverables.""",
            expected_output="Final comprehensive report with executive summary and detailed findings",
            agent=agents[0],  # Supervisor
            context=tasks,
            output_file="final_report.md"
        )
        tasks.append(synthesis_task)

        return tasks
```

## 🚀 Performance Optimization

### Caching and Memory Management

```python
# rag/performance/caching.py
import pickle
import hashlib
from functools import lru_cache
from typing import Dict, Any, List
import redis
import os

class AdvancedCacheManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_type = config.get("cache_type", "memory")  # memory, redis, file
        self.cache_ttl = config.get("cache_ttl", 3600)  # 1 hour default

        if self.cache_type == "redis":
            self.redis_client = redis.Redis(
                host=config.get("redis_host", "localhost"),
                port=config.get("redis_port", 6379),
                db=config.get("redis_db", 0)
            )

    def _generate_cache_key(self, query: str, k: int, filters: Dict = None) -> str:
        """Generate deterministic cache key"""
        cache_data = f"{query}_{k}_{str(filters) if filters else ''}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    @lru_cache(maxsize=1000)
    def get_embedding_cache(self, text: str) -> List[float]:
        """Memory cache for embeddings"""
        # This will be populated by the embedding model
        return None

    def cache_search_results(self, query: str, k: int, results: List[Dict], filters: Dict = None):
        """Cache search results"""
        cache_key = self._generate_cache_key(query, k, filters)

        if self.cache_type == "redis":
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                pickle.dumps(results)
            )
        elif self.cache_type == "file":
            cache_dir = self.config.get("cache_dir", "./cache")
            os.makedirs(cache_dir, exist_ok=True)
            with open(f"{cache_dir}/{cache_key}.pkl", "wb") as f:
                pickle.dump(results, f)

    def get_cached_search_results(self, query: str, k: int, filters: Dict = None) -> List[Dict]:
        """Retrieve cached search results"""
        cache_key = self._generate_cache_key(query, k, filters)

        try:
            if self.cache_type == "redis":
                cached = self.redis_client.get(cache_key)
                if cached:
                    return pickle.loads(cached)
            elif self.cache_type == "file":
                cache_file = f"{self.config.get('cache_dir', './cache')}/{cache_key}.pkl"
                if os.path.exists(cache_file):
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
        except Exception as e:
            print(f"Cache retrieval error: {e}")

        return None

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "search_times": [],
            "embedding_times": [],
            "agent_execution_times": [],
            "cache_hit_rate": 0,
            "total_requests": 0,
            "cache_hits": 0
        }

    def record_search_time(self, duration: float):
        self.metrics["search_times"].append(duration)

    def record_cache_hit(self):
        self.metrics["cache_hits"] += 1
        self.metrics["total_requests"] += 1
        self.metrics["cache_hit_rate"] = self.metrics["cache_hits"] / self.metrics["total_requests"]

    def record_cache_miss(self):
        self.metrics["total_requests"] += 1
        self.metrics["cache_hit_rate"] = self.metrics["cache_hits"] / self.metrics["total_requests"]

    def get_performance_report(self) -> Dict[str, Any]:
        import statistics

        return {
            "avg_search_time": statistics.mean(self.metrics["search_times"]) if self.metrics["search_times"] else 0,
            "cache_hit_rate": self.metrics["cache_hit_rate"],
            "total_requests": self.metrics["total_requests"],
            "performance_grade": self._calculate_grade()
        }

    def _calculate_grade(self) -> str:
        """Simple performance grading"""
        avg_time = statistics.mean(self.metrics["search_times"]) if self.metrics["search_times"] else float('inf')
        cache_rate = self.metrics["cache_hit_rate"]

        if avg_time < 0.5 and cache_rate > 0.8:
            return "A+ (Excellent)"
        elif avg_time < 1.0 and cache_rate > 0.6:
            return "B+ (Good)"
        elif avg_time < 2.0 and cache_rate > 0.4:
            return "C+ (Average)"
        else:
            return "D (Needs Improvement)"
```

## 🔒 Security and Privacy

### Secure Configuration

```python
# config/security_config.py
import os
from cryptography.fernet import Fernet
from typing import Dict, Any

class SecurityConfig:
    def __init__(self):
        self.encryption_key = os.getenv("ENCRYPTION_KEY") or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)

    def encrypt_sensitive_data(self, data: str) -> bytes:
        """Encrypt sensitive configuration data"""
        return self.cipher_suite.encrypt(data.encode())

    def decrypt_sensitive_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive configuration data"""
        return self.cipher_suite.decrypt(encrypted_data).decode()

    def get_secure_config(self) -> Dict[str, Any]:
        """Get security-hardened configuration"""
        return {
            # API security
            "api_rate_limiting": {
                "requests_per_minute": 60,
                "burst_limit": 10
            },

            # Data privacy
            "data_privacy": {
                "anonymize_pii": True,
                "data_retention_days": 90,
                "encrypt_at_rest": True
            },

            # Access control
            "access_control": {
                "require_authentication": True,
                "session_timeout_minutes": 30,
                "max_concurrent_sessions": 5
            },

            # Audit logging
            "audit_logging": {
                "log_all_queries": True,
                "log_responses": False,  # May contain sensitive data
                "log_retention_days": 365
            }
        }

# PII scrubbing for sensitive documents
class PIIScrubber:
    def __init__(self):
        import re
        self.patterns = {
            "ssn": re.compile(r'\\b\\d{3}-\\d{2}-\\d{4}\\b'),
            "email": re.compile(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'),
            "phone": re.compile(r'\\b\\d{3}-\\d{3}-\\d{4}\\b'),
            "credit_card": re.compile(r'\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b')
        }

    def scrub_document(self, text: str) -> str:
        """Remove or mask PII from document text"""
        scrubbed = text

        for pii_type, pattern in self.patterns.items():
            scrubbed = pattern.sub(f"[{pii_type.upper()}_REDACTED]", scrubbed)

        return scrubbed
```

## 📊 Production Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-multi-agent
  labels:
    app: rag-multi-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-multi-agent
  template:
    metadata:
      labels:
        app: rag-multi-agent
    spec:
      containers:
      - name: rag-multi-agent
        image: your-registry/rag-multi-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## 🎯 Next Steps

After mastering advanced configuration:

1. **Production Deployment** - Deploy to cloud infrastructure
2. **Monitoring & Observability** - Set up comprehensive monitoring
3. **Custom Components** - Build domain-specific components
4. **API Development** - Create REST/GraphQL APIs
5. **Integration Patterns** - Connect with existing systems

## 📚 Related Documentation

- [Architecture Guide](Architecture-Guide) - Deep system understanding
- [Custom Components](Custom-Components) - Building extensions
- [Performance Optimization](Performance-Optimization) - Scaling strategies
- [API Reference](API-Reference) - Complete API documentation

Ready to build production-grade RAG systems? The advanced configuration gives you the tools to create enterprise-ready AI solutions!