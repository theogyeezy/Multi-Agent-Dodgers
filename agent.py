# Basic Multi-Agent Template using CrewAI and WatsonxAI
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain_ibm import WatsonxLLM
import os

# Import api keys
os.environ["API_KEY"] = "your_api_key"
os.environ["SERPER_API_KEY"] = "your_serper_api_key"

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

# Create search tool
search = SerperDevTool()

# Create researcher agent
researcher = Agent(
    llm=llm,
    function_calling_llm=function_calling_llm,
    role="FILL_IN_RESEARCHER_ROLE",
    goal="FILL_IN_RESEARCHER_GOAL",
    backstory="FILL_IN_RESEARCHER_BACKSTORY",
    allow_delegation=False,
    tools=[search],
    verbose=1
)

# Create writer agent
writer = Agent(
    llm=llm,
    role="FILL_IN_WRITER_ROLE",
    goal="FILL_IN_WRITER_GOAL",
    backstory="FILL_IN_WRITER_BACKSTORY",
    allow_delegation=False,
    verbose=1
)

# Create research task
task1 = Task(
    description="FILL_IN_RESEARCH_TASK_DESCRIPTION",
    expected_output="FILL_IN_RESEARCH_EXPECTED_OUTPUT",
    output_file="research_output.md",
    agent=researcher
)

# Create writing task
task2 = Task(
    description="FILL_IN_WRITING_TASK_DESCRIPTION",
    expected_output="FILL_IN_WRITING_EXPECTED_OUTPUT",
    output_file="final_output.md",
    agent=writer
)

# Create and run crew
crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=1)

if __name__ == "__main__":
    print("Starting basic multi-agent workflow...")
    result = crew.kickoff()
    print("Workflow completed!")
    print(result)

    # Test LLM
    print("\nTesting LLM:")
    test_response = llm.invoke("What can you help me with?")
    print(test_response)


