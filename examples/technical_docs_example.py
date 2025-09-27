# Example: Technical Documentation RAG Multi-Agent System
from crewai import Crew
from langchain_ibm import WatsonxLLM
import os

# Import RAG components
from config.rag_config_template import RAGConfig
from rag.tools import RAGRetrievalToolFactory
from rag.knowledge_base_manager import KnowledgeBaseManager
from templates import AgentTemplates, TaskTemplates, AgentConfiguration, TaskConfiguration

# Configuration for Technical Documentation
config = {
    "vector_db": "chroma",  # or "faiss", "pinecone"
    "embedding_model": "sentence-transformers",
    "embedding_model_name": "all-MiniLM-L6-v2",
    "chunk_size": 600,  # Good size for technical content
    "chunk_overlap": 75,
    "retrieval_k": 4,
    "collection_name": "technical_knowledge_base",
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
    max_tokens=1536
)

def setup_technical_knowledge_base():
    """Set up knowledge base with technical documentation"""
    knowledge_base = KnowledgeBaseManager(config)

    # Add technical documents from your documents folder
    # Place your technical docs (PDFs, DOCX, TXT, MD) in data/documents/technical/
    tech_docs_path = "data/documents/technical"
    result = knowledge_base.add_documents_from_path(tech_docs_path)
    print(f"Technical knowledge base setup: {result}")

    return knowledge_base

def create_technical_documentation_crew():
    """Create a specialized technical documentation crew"""

    # Create RAG tool
    rag_tool = RAGRetrievalToolFactory.create_tool(config)

    # Create technical researcher agent
    technical_researcher = AgentTemplates.create_custom_rag_agent(
        llm=llm,
        rag_tool=rag_tool,
        role="Senior Technical Research Specialist",
        goal="Research technical specifications, APIs, and system documentation to provide comprehensive technical analysis",
        backstory="You are a senior technical researcher with expertise in software architecture, API documentation, and system design. You excel at finding and synthesizing technical information from various documentation sources."
    )

    # Create technical writer agent
    technical_writer = AgentTemplates.create_custom_rag_agent(
        llm=llm,
        rag_tool=rag_tool,
        role="Technical Documentation Specialist",
        goal="Create clear, comprehensive technical documentation that helps developers understand and implement solutions",
        backstory="You are an experienced technical writer who specializes in creating developer-friendly documentation. You excel at translating complex technical concepts into clear, actionable guides."
    )

    # Create code example agent
    code_specialist = AgentTemplates.create_custom_rag_agent(
        llm=llm,
        rag_tool=rag_tool,
        role="Code Example Specialist",
        goal="Generate practical code examples and implementation guides based on technical specifications",
        backstory="You are a senior developer who specializes in creating clear, working code examples and implementation patterns. You excel at bridging the gap between documentation and practical implementation."
    )

    # Create technical research task
    tech_task_config = TaskConfiguration.get_template_config("technical_analysis", "API integration patterns")
    research_task = TaskTemplates.create_custom_task(
        agent=technical_researcher,
        description=tech_task_config["description"],
        expected_output=tech_task_config["expected_output"],
        output_file="technical_research.md"
    )

    # Create documentation writing task
    documentation_task = TaskTemplates.create_custom_task(
        agent=technical_writer,
        description="Based on the technical research, create comprehensive developer documentation that includes setup instructions, configuration options, and best practices.",
        expected_output="Complete technical documentation including: overview, prerequisites, installation guide, configuration options, usage examples, troubleshooting section, and additional resources. Format as developer-friendly markdown.",
        output_file="technical_documentation.md"
    )

    # Create code examples task
    code_task = TaskTemplates.create_custom_task(
        agent=code_specialist,
        description="Create practical code examples and implementation guides based on the technical documentation. Include working code samples with explanations.",
        expected_output="Comprehensive code examples including: basic implementation, advanced usage patterns, error handling, and integration examples. Include inline comments and explanations. Format as executable code with documentation.",
        output_file="code_examples.md"
    )

    # Create crew
    technical_crew = Crew(
        agents=[technical_researcher, technical_writer, code_specialist],
        tasks=[research_task, documentation_task, code_task],
        verbose=1
    )

    return technical_crew

if __name__ == "__main__":
    print("Setting up Technical Documentation RAG System...")

    # Step 1: Set up knowledge base
    knowledge_base = setup_technical_knowledge_base()

    # Step 2: Create technical documentation crew
    crew = create_technical_documentation_crew()

    # Step 3: Run technical documentation workflow
    print("Starting technical documentation workflow...")
    result = crew.kickoff()
    print("Technical documentation completed!")
    print(result)

    # Optional: Test knowledge base search
    print("\nTesting technical knowledge base search:")
    test_queries = ["API authentication", "database configuration", "error handling patterns"]

    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        search_results = knowledge_base.search_knowledge_base(query, k=2)
        for i, result in enumerate(search_results, 1):
            print(f"  Result {i}: {result.get('content', '')[:200]}...")
            print(f"  Source: {result.get('metadata', {}).get('source', 'Unknown')}")