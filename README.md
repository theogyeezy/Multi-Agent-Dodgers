# Multi-Agent System Template

A comprehensive template for creating multi-agent systems using CrewAI and IBM WatsonxAI.

## Overview

This template is designed for research and writing workflows but can be configured for any multi-agent use case. The default setup includes two agents:

1. **Researcher Agent** - Conducts research using search tools
2. **Writer Agent** - Creates written content based on the research

## How to Use

1. Configure your API keys in the environment variables:
   - `API_KEY` - Your IBM WatsonxAI API key
   - `SERPER API KEY` - Your Serper search API key

2. Customize the agents by filling in:
   - Role descriptions
   - Goals
   - Backstories
   - Task descriptions and expected outputs

3. Run the system:
   ```bash
   python agent.py
   ```

## Workflow

The system operates with a two-step process:
1. The first agent (researcher) performs research using available search tools
2. The second agent (writer) creates written content based on the research findings

## Configuration

The template can be easily adapted for other use cases by:
- Modifying agent roles and goals
- Changing task descriptions
- Adding or removing agents
- Incorporating different tools
- Adjusting LLM parameters

## Requirements

- CrewAI
- crewai-tools
- langchain-ibm
- IBM WatsonxAI access
- Serper API access (for search functionality)