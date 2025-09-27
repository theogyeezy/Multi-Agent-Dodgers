# template for creating multi agent with crewai and langchain_ibm + WatsonxAI 
 
from crewai import Crew, Task, Agent
from crewai_tools import SerperDevTool
from langchain_ibm import WatsonxAI
import os 

# Import api keys
os.environ["API_KEY"] = "your_api_key"
os.environ["SERPER API KEY"] = "your api key"

# parameters
parameters = {"decoding method": "greedy", "max new tokens": 256}

# create llm instance
llm = WatsonxLLM(
    model_name="LLM NAME",
    deployment_id="deployment_id",
    service_url="api_url",
    api_key=os.environ["API_KEY"],
    max_retries=3,
    top_p=0.9,
    frequency_penalty=0,
    presence_penalty=0,
    max_tokens=1024
)

# create 2nd llm 
function_calling_llm = WatsonxLLM(
    model_name="LLM NAME",
    deployment_id="deployment_id",
    service_url="api_url",
    api_key=os.environ["API_KEY"],
    max_retries=3,
    top_p=0.9,
    frequency_penalty=0,
    presence_penalty=0,
    max_tokens=1024
)

# create agent 
researcher = Agent(
    llm=llm,
    function_calling_llm=function_calling_llm,
    role="fill in role",
    goal="fill in goal",
    backstory="fill in backstory",
    allow_delegation=False,
    tools=[search],
    verbose=1
)

# create second agent 
writer = Agent(
    llm=llm,
    role="fill in role",
    goal="fill in goal",
    backstory="fill in backstory",
    allow_delegation=False,
    verbose=1
)

# create a task 
task1 = Task(
    desccription="fill in task description",
    expected_output="fill in expected output",
    output_file="fill in output file",
    agent=researcher
)

# create a second task 
task2 = Task(
    desccription="fill in task description",
    expected_output="fill in expected output",
    output_file="fill in second output file",
    agent=writer
)

# put all together with the crew 
crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=1)
print(crew.kickoff()) 



# test out llm (run with python agent.py)
print(llm.invoke("ask a question to llm?"))


