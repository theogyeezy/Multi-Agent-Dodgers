# Example: Legal Research RAG Multi-Agent System
from crewai import Crew
from langchain_ibm import WatsonxLLM
import os

# Import RAG components
from config.rag_config_template import RAGConfig
from rag.tools import RAGRetrievalToolFactory
from rag.knowledge_base_manager import KnowledgeBaseManager
from templates import AgentTemplates, TaskTemplates, AgentConfiguration, TaskConfiguration

# Configuration for Legal Research
config = {
    "vector_db": "chroma",  # or "faiss", "pinecone"
    "embedding_model": "sentence-transformers",
    "embedding_model_name": "all-MiniLM-L6-v2",
    "chunk_size": 800,  # Larger chunks for legal documents
    "chunk_overlap": 100,
    "retrieval_k": 5,  # More results for comprehensive legal research
    "collection_name": "legal_knowledge_base",
    "dimension": 384
}

# Set up your API keys
os.environ["API_KEY"] = "your_watsonx_api_key"

# Create LLM
llm = WatsonxLLM(
    model_name="your_model_name",
    deployment_id="your_deployment_id",
    service_url="your_service_url",
    api_key=os.environ["API_KEY"],
    max_tokens=2048  # Longer responses for legal analysis
)

def setup_legal_knowledge_base():
    """Set up knowledge base with legal documents"""
    knowledge_base = KnowledgeBaseManager(config)

    # Add legal documents from your documents folder
    # Place your legal documents (PDFs, DOCX, TXT) in data/documents/legal/
    legal_docs_path = "data/documents/legal"
    result = knowledge_base.add_documents_from_path(legal_docs_path)
    print(f"Legal knowledge base setup: {result}")

    return knowledge_base

def create_legal_research_crew():
    """Create a specialized legal research crew"""

    # Create RAG tool
    rag_tool = RAGRetrievalToolFactory.create_tool(config)

    # Get predefined legal researcher configuration
    legal_config = AgentConfiguration.get_template_config("legal_researcher")

    # Create legal research agent
    legal_researcher = AgentTemplates.create_custom_rag_agent(
        llm=llm,
        rag_tool=rag_tool,
        role=legal_config["role"],
        goal=legal_config["goal"],
        backstory=legal_config["backstory"]
    )

    # Create legal writer agent for drafting legal documents
    legal_writer = AgentTemplates.create_custom_rag_agent(
        llm=llm,
        rag_tool=rag_tool,
        role="Senior Legal Writer",
        goal="Draft clear, professional legal documents based on research findings",
        backstory="You are an experienced legal writer who specializes in translating complex legal research into clear, actionable legal documents and memoranda."
    )

    # Create legal research task
    legal_task_config = TaskConfiguration.get_template_config("legal_research", "contract dispute resolution")
    research_task = TaskTemplates.create_custom_task(
        agent=legal_researcher,
        description=legal_task_config["description"],
        expected_output=legal_task_config["expected_output"],
        output_file="legal_research_memo.md"
    )

    # Create legal writing task
    writing_task = TaskTemplates.create_custom_task(
        agent=legal_writer,
        description="Based on the legal research findings, draft a comprehensive legal memorandum that outlines the legal analysis, applicable precedents, and recommended legal strategy.",
        expected_output="A professional legal memorandum including: executive summary, factual background, legal analysis with case citations, risk assessment, and strategic recommendations. Format as formal legal document.",
        output_file="legal_memorandum.md"
    )

    # Create crew
    legal_crew = Crew(
        agents=[legal_researcher, legal_writer],
        tasks=[research_task, writing_task],
        verbose=1
    )

    return legal_crew

if __name__ == "__main__":
    print("Setting up Legal Research RAG System...")

    # Step 1: Set up knowledge base
    knowledge_base = setup_legal_knowledge_base()

    # Step 2: Create legal research crew
    crew = create_legal_research_crew()

    # Step 3: Run legal research workflow
    print("Starting legal research workflow...")
    result = crew.kickoff()
    print("Legal research completed!")
    print(result)

    # Optional: Test knowledge base search
    print("\nTesting legal knowledge base search:")
    test_query = "contract breach remedies"
    search_results = knowledge_base.search_knowledge_base(test_query, k=3)
    for i, result in enumerate(search_results, 1):
        print(f"\nLegal Source {i}:")
        print(f"Content: {result.get('content', '')[:300]}...")
        print(f"Source: {result.get('metadata', {}).get('source', 'Unknown')}")