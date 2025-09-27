# RAG-enabled Multi-Agent Template using CrewAI and WatsonxAI
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain_ibm import WatsonxLLM
import os

# Import RAG components
from config.rag_config_template import RAGConfig
from rag.tools import RAGRetrievalToolFactory
from rag.knowledge_base_manager import KnowledgeBaseManager
from templates import AgentTemplates, TaskTemplates, AgentConfiguration, TaskConfiguration

# Import api keys
os.environ["API_KEY"] = "your_api_key"
os.environ["SERPER_API_KEY"] = "your_serper_api_key"  # Optional for web search

# RAG Configuration - Customize these settings
config = {
    "vector_db": RAGConfig.VECTOR_DB,
    "embedding_model": RAGConfig.EMBEDDING_MODEL,
    "embedding_model_name": RAGConfig.EMBEDDING_MODEL_NAME,
    "chunk_size": RAGConfig.CHUNK_SIZE,
    "chunk_overlap": RAGConfig.CHUNK_OVERLAP,
    "retrieval_k": RAGConfig.RETRIEVAL_K,
    "collection_name": RAGConfig.COLLECTION_NAME,
    "dimension": 384  # Default for all-MiniLM-L6-v2
}

# LLM parameters
parameters = {"decoding_method": "greedy", "max_new_tokens": 256}

# Create LLM instance
llm = WatsonxLLM(
    model_name="LLM_NAME",
    deployment_id="deployment_id",
    service_url="api_url",
    api_key=os.environ["API_KEY"],
    max_retries=3,
    top_p=0.9,
    frequency_penalty=0,
    presence_penalty=0,
    max_tokens=1024
)

# Create 2nd LLM for function calling
function_calling_llm = WatsonxLLM(
    model_name="LLM_NAME",
    deployment_id="deployment_id",
    service_url="api_url",
    api_key=os.environ["API_KEY"],
    max_retries=3,
    top_p=0.9,
    frequency_penalty=0,
    presence_penalty=0,
    max_tokens=1024
)

# Initialize Knowledge Base Manager
knowledge_base = KnowledgeBaseManager(config)

# STEP 1: Add documents to knowledge base (uncomment and customize)
# documents_path = "data/documents"  # Path to your documents
# result = knowledge_base.add_documents_from_path(documents_path)
# print(f"Knowledge base setup: {result}")

# STEP 2: Create RAG retrieval tool
rag_tool = RAGRetrievalToolFactory.create_tool(config)

# Alternative: Create tool with documents in one step
# rag_tool = RAGRetrievalToolFactory.create_tool_with_documents(config, "data/documents")

# STEP 3: Create agents using templates

# Option 1: Use predefined templates with placeholders
researcher = AgentTemplates.create_rag_researcher_agent(
    llm=llm,
    rag_tool=rag_tool,
    function_calling_llm=function_calling_llm
)

writer = AgentTemplates.create_rag_writer_agent(
    llm=llm,
    rag_tool=rag_tool,
    function_calling_llm=function_calling_llm
)

# Option 2: Use predefined configurations for specific domains
# legal_config = AgentConfiguration.get_template_config("legal_researcher")
# legal_researcher = AgentTemplates.create_custom_rag_agent(
#     llm=llm,
#     rag_tool=rag_tool,
#     role=legal_config["role"],
#     goal=legal_config["goal"],
#     backstory=legal_config["backstory"],
#     function_calling_llm=function_calling_llm
# )

# Option 3: Create completely custom agents
# custom_agent = AgentTemplates.create_custom_rag_agent(
#     llm=llm,
#     rag_tool=rag_tool,
#     role="Your Custom Role",
#     goal="Your Custom Goal",
#     backstory="Your Custom Backstory",
#     additional_tools=[SerperDevTool()],  # Optional additional tools
#     function_calling_llm=function_calling_llm
# )

# STEP 4: Create tasks using templates

# Option 1: Use basic templates with placeholders
task1 = TaskTemplates.create_research_task(
    agent=researcher,
    description="FILL_IN_YOUR_RESEARCH_TASK_DESCRIPTION",
    expected_output="FILL_IN_YOUR_EXPECTED_OUTPUT",
    output_file="research_results.md"
)

task2 = TaskTemplates.create_writing_task(
    agent=writer,
    description="FILL_IN_YOUR_WRITING_TASK_DESCRIPTION",
    expected_output="FILL_IN_YOUR_EXPECTED_OUTPUT",
    output_file="final_document.md"
)

# Option 2: Use predefined task configurations
# legal_task_config = TaskConfiguration.get_template_config("legal_research", "contract law")
# legal_task = TaskTemplates.create_custom_task(
#     agent=legal_researcher,
#     description=legal_task_config["description"],
#     expected_output=legal_task_config["expected_output"],
#     output_file=legal_task_config["output_file"]
# )

# Option 3: Create completely custom tasks
# custom_task = TaskTemplates.create_custom_task(
#     agent=custom_agent,
#     description="Your custom task description with specific requirements",
#     expected_output="Your specific expected output format and content",
#     output_file="custom_output.md"
# )

# STEP 5: Create and run crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=1
)

# Run the crew
if __name__ == "__main__":
    # Print knowledge base info
    kb_info = knowledge_base.get_knowledge_base_info()
    print(f"Knowledge Base Info: {kb_info}")

    # Execute crew
    print("Starting RAG-enabled multi-agent workflow...")
    result = crew.kickoff()
    print("Workflow completed!")
    print(result)

    # Test individual components
    print("\nTesting RAG retrieval:")
    test_query = "your test query here"
    search_results = knowledge_base.search_knowledge_base(test_query)
    for i, result in enumerate(search_results, 1):
        print(f"Result {i}: {result.get('content', '')[:200]}...")

    # Test LLM
    print("\nTesting LLM:")
    test_response = llm.invoke("What can you help me with?")
    print(test_response)