# Getting Started - Your First RAG Multi-Agent System

Welcome! This guide will help you set up your first RAG-enabled multi-agent system in just a few steps. Perfect for beginners who are new to AI agents or RAG systems.

## 🎯 What You'll Learn

By the end of this guide, you'll have:
- ✅ A working multi-agent system
- ✅ Understanding of basic concepts
- ✅ Your first AI agent responding to queries
- ✅ Knowledge of next steps

## ⏱️ Time Required: 15-30 minutes

## 🧰 Prerequisites

### Required
- **Python 3.8+** (check with `python --version`)
- **Git** (for cloning the repository)
- **Text editor** (VS Code, PyCharm, or any editor)

### Recommended
- **IBM WatsonxAI account** (for AI models)
- **Basic Python knowledge** (variables, functions)

## 🚀 Step 1: Get the Template

### Option A: Use as Template (Recommended)
1. Go to the [repository page](https://github.com/theogyeezy/Multi-Agent-Template)
2. Click the green **"Use this template"** button
3. Create your own repository
4. Clone your new repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### Option B: Clone Directly
```bash
git clone https://github.com/theogyeezy/Multi-Agent-Template.git
cd Multi-Agent-Template
```

## 🔧 Step 2: Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

**What this installs:**
- `crewai` - Multi-agent framework
- `langchain-ibm` - IBM WatsonxAI integration
- `chromadb` - Vector database for RAG
- `sentence-transformers` - Text embeddings
- Other supporting libraries

## 🔑 Step 3: Configure API Keys

### Get IBM WatsonxAI Access
1. Sign up at [IBM Cloud](https://cloud.ibm.com/)
2. Create a WatsonxAI service instance
3. Get your API key and service URL

### Set Environment Variables
Create a `.env` file or set environment variables:

```bash
# Option 1: Environment variables
export API_KEY="your_watsonx_api_key"
export SERPER_API_KEY="your_serper_key"  # Optional for web search

# Option 2: Create .env file
echo 'API_KEY="your_watsonx_api_key"' > .env
echo 'SERPER_API_KEY="your_serper_key"' >> .env
```

## 🎯 Step 4: Choose Your Starting Point

You have two options:

### Option A: Basic Multi-Agent (Easier)
Perfect for learning the basics:

```bash
python agent.py
```

**What it does:**
- Web search research
- Content writing
- Simple agent interactions

### Option B: RAG-Enabled System (More Powerful)
For document-based AI:

```bash
python rag_agent.py
```

**What it does:**
- Searches your documents
- Knowledge-based responses
- Advanced AI capabilities

## 📝 Step 5: Customize Your First Agent

Let's start with the basic template. Open `agent.py` and customize:

### 1. Update LLM Configuration
```python
# Replace placeholders with your actual values
llm = WatsonxLLM(
    model_name="YOUR_MODEL_NAME",        # e.g., "ibm/granite-13b-chat-v2"
    deployment_id="YOUR_DEPLOYMENT_ID",  # From IBM Cloud
    service_url="YOUR_SERVICE_URL",      # From IBM Cloud
    api_key=os.environ["API_KEY"],
    max_tokens=1024
)
```

### 2. Customize Your Agents
```python
# Replace the placeholder text
researcher = Agent(
    llm=llm,
    role="Senior Research Specialist",                    # Your agent's role
    goal="Research topics thoroughly using web search",   # What it aims to do
    backstory="You are an expert researcher...",          # Agent's background
    tools=[search],
    verbose=1
)
```

### 3. Define Your Tasks
```python
task1 = Task(
    description="Research the latest trends in artificial intelligence", # What to do
    expected_output="A comprehensive report with key findings",          # What you want
    output_file="ai_research_report.md",                               # Where to save
    agent=researcher
)
```

## 🎉 Step 6: Run Your First Agent

```bash
python agent.py
```

**Expected output:**
- Agent initialization messages
- Research progress updates
- Generated report saved to file
- LLM test response

## 🔍 Understanding the Output

Your agent will:
1. **Initialize** - Set up the AI models and tools
2. **Research** - Use web search to find information
3. **Write** - Create content based on research
4. **Save** - Output results to markdown files

## 🎓 Next Steps

Congratulations! You've built your first multi-agent system. Here's what to explore next:

### Immediate Next Steps
1. **Customize Tasks** - Change the research topics
2. **Modify Agents** - Update roles and goals
3. **Try RAG** - Move to document-based AI with `rag_agent.py`

### Learning Resources
- [Basic Configuration](Basic-Configuration) - Understanding all settings
- [Your First Agent](Your-First-Agent) - Deep dive into agent creation
- [Use Case Examples](Legal-Research-Guide) - Real-world applications

### Advanced Features
- [RAG Setup](Advanced-Configuration) - Document processing
- [Vector Databases](Architecture-Guide) - Understanding storage
- [Custom Components](Custom-Components) - Building your own tools

## ❓ Common Issues

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### API key errors
- Double-check your API key is correct
- Ensure environment variables are set
- Verify IBM Cloud service is active

### Empty responses
- Check your model deployment is active
- Verify API permissions
- Test with simpler prompts first

## 🆘 Need Help?

- **Quick fixes**: Check [Troubleshooting](Troubleshooting)
- **Detailed setup**: See [Quick Setup Guide](Quick-Setup)
- **Configuration help**: Read [Basic Configuration](Basic-Configuration)
- **Community support**: Open a GitHub issue

## 🏆 Success Criteria

You know you're ready to move on when:
- ✅ Agents run without errors
- ✅ Output files are generated
- ✅ You understand basic agent concepts
- ✅ You can modify agent roles and tasks

Ready for more advanced features? Let's move to [Basic Configuration](Basic-Configuration) or try the [RAG system](Advanced-Configuration)!