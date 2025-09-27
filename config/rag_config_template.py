# RAG Configuration Template
# Customize these settings for your specific use case

class RAGConfig:
    # Vector Database Options: "chroma", "faiss", "pinecone"
    VECTOR_DB = "chroma"

    # Embedding Model Options: "sentence-transformers", "openai", "huggingface"
    EMBEDDING_MODEL = "sentence-transformers"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Default for sentence-transformers

    # Document Processing
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Retrieval Settings
    RETRIEVAL_K = 3  # Number of documents to retrieve

    # LLM Provider Options: "watsonx", "openai", "anthropic"
    LLM_PROVIDER = "watsonx"

    # Collection/Index Names
    COLLECTION_NAME = "your_collection_name"

    # File Processing
    SUPPORTED_FILE_TYPES = [".txt", ".pdf", ".docx", ".md"]
    DOCUMENTS_PATH = "data/documents"

    # API Keys (set in environment variables)
    REQUIRED_ENV_VARS = [
        "API_KEY",  # WatsonxAI API key
        "SERPER_API_KEY",  # Optional: for web search
        # Add other API keys as needed:
        # "OPENAI_API_KEY",
        # "PINECONE_API_KEY",
    ]

    # Agent Template Placeholders
    AGENT_ROLES = {
        "researcher": {
            "role": "FILL_IN_RESEARCHER_ROLE",
            "goal": "FILL_IN_RESEARCHER_GOAL",
            "backstory": "FILL_IN_RESEARCHER_BACKSTORY"
        },
        "writer": {
            "role": "FILL_IN_WRITER_ROLE",
            "goal": "FILL_IN_WRITER_GOAL",
            "backstory": "FILL_IN_WRITER_BACKSTORY"
        }
    }

    # Task Template Placeholders
    TASK_TEMPLATES = {
        "research_task": {
            "description": "FILL_IN_RESEARCH_TASK_DESCRIPTION",
            "expected_output": "FILL_IN_RESEARCH_EXPECTED_OUTPUT",
            "output_file": "research_output.md"
        },
        "writing_task": {
            "description": "FILL_IN_WRITING_TASK_DESCRIPTION",
            "expected_output": "FILL_IN_WRITING_EXPECTED_OUTPUT",
            "output_file": "final_output.md"
        }
    }