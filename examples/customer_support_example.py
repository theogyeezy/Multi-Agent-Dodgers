# Example: Customer Support RAG Multi-Agent System
from crewai import Crew
from langchain_ibm import WatsonxLLM
import os

# Import RAG components
from config.rag_config_template import RAGConfig
from rag.tools import RAGRetrievalToolFactory
from rag.knowledge_base_manager import KnowledgeBaseManager
from templates import AgentTemplates, TaskTemplates

# Configuration for Customer Support
config = {
    "vector_db": "chroma",
    "embedding_model": "sentence-transformers",
    "embedding_model_name": "all-MiniLM-L6-v2",
    "chunk_size": 400,  # Smaller chunks for quick support responses
    "chunk_overlap": 50,
    "retrieval_k": 3,
    "collection_name": "support_knowledge_base",
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
    max_tokens=1024
)

def setup_support_knowledge_base():
    """Set up knowledge base with support documents"""
    knowledge_base = KnowledgeBaseManager(config)

    # Add support documents from your documents folder
    # Place your support docs (FAQs, troubleshooting guides, etc.) in data/documents/support/
    support_docs_path = "data/documents/support"
    result = knowledge_base.add_documents_from_path(support_docs_path)
    print(f"Support knowledge base setup: {result}")

    return knowledge_base

def create_customer_support_crew():
    """Create a specialized customer support crew"""

    # Create RAG tool
    rag_tool = RAGRetrievalToolFactory.create_tool(config)

    # Create support specialist agent
    support_specialist = AgentTemplates.create_custom_rag_agent(
        llm=llm,
        rag_tool=rag_tool,
        role="Senior Customer Support Specialist",
        goal="Provide helpful, accurate customer support responses using the knowledge base to resolve customer issues quickly and effectively",
        backstory="You are an experienced customer support specialist with expertise in troubleshooting and problem resolution. You excel at finding relevant information from support documentation and providing clear, actionable solutions to customers."
    )

    # Create escalation agent for complex issues
    escalation_specialist = AgentTemplates.create_custom_rag_agent(
        llm=llm,
        rag_tool=rag_tool,
        role="Technical Escalation Specialist",
        goal="Handle complex technical issues by researching advanced solutions and providing detailed technical guidance",
        backstory="You are a technical escalation specialist who handles complex customer issues. You have deep technical knowledge and can provide detailed troubleshooting steps and advanced solutions."
    )

    # Create initial support response task
    support_task = TaskTemplates.create_custom_task(
        agent=support_specialist,
        description="Analyze the customer inquiry and provide a helpful response using information from the support knowledge base. Focus on providing clear, step-by-step solutions.",
        expected_output="A professional customer support response including: acknowledgment of the issue, step-by-step solution, relevant links or resources, and follow-up guidance. Format as friendly, professional customer communication.",
        output_file="support_response.md"
    )

    # Create escalation task for complex issues
    escalation_task = TaskTemplates.create_custom_task(
        agent=escalation_specialist,
        description="For complex technical issues that require advanced troubleshooting, research the knowledge base for detailed technical solutions and provide comprehensive guidance.",
        expected_output="Detailed technical support response including: issue analysis, advanced troubleshooting steps, technical explanations, alternative solutions, and escalation options. Format as technical support documentation.",
        output_file="escalation_response.md"
    )

    # Create crew
    support_crew = Crew(
        agents=[support_specialist, escalation_specialist],
        tasks=[support_task, escalation_task],
        verbose=1
    )

    return support_crew

def handle_customer_inquiry(customer_question: str):
    """Handle a specific customer inquiry"""
    knowledge_base = setup_support_knowledge_base()

    # Search knowledge base for relevant information
    search_results = knowledge_base.search_knowledge_base(customer_question, k=5)

    print(f"\nCustomer Question: {customer_question}")
    print("\nRelevant Knowledge Base Results:")
    for i, result in enumerate(search_results, 1):
        print(f"\nResult {i}:")
        print(f"Content: {result.get('content', '')[:300]}...")
        print(f"Source: {result.get('metadata', {}).get('source', 'Unknown')}")

    # Create and run support crew
    crew = create_customer_support_crew()

    print("\nGenerating customer support response...")
    response = crew.kickoff()
    return response

if __name__ == "__main__":
    print("Setting up Customer Support RAG System...")

    # Example customer inquiries
    sample_questions = [
        "How do I reset my password?",
        "My account is locked, what should I do?",
        "I'm having trouble with payment processing",
        "The application keeps crashing on startup",
        "How do I export my data?"
    ]

    # Set up knowledge base
    knowledge_base = setup_support_knowledge_base()

    # Handle a sample question
    sample_question = "How do I reset my password?"
    print(f"\nHandling sample question: '{sample_question}'")

    response = handle_customer_inquiry(sample_question)
    print("Support response generated!")
    print(response)

    # Optional: Test multiple questions
    print("\nTesting knowledge base with multiple questions:")
    for question in sample_questions[:3]:  # Test first 3 questions
        print(f"\nQuestion: {question}")
        results = knowledge_base.search_knowledge_base(question, k=2)
        for i, result in enumerate(results, 1):
            print(f"  Relevant info {i}: {result.get('content', '')[:150]}...")