from crewai import Agent
from typing import List, Dict, Any

class AgentTemplates:
    """Template configurations for different types of agents"""

    @staticmethod
    def create_rag_researcher_agent(llm, rag_tool, function_calling_llm=None) -> Agent:
        """Create a RAG-enabled researcher agent template"""
        return Agent(
            llm=llm,
            function_calling_llm=function_calling_llm,
            role="FILL_IN_RAG_RESEARCHER_ROLE",
            goal="FILL_IN_RAG_RESEARCHER_GOAL",
            backstory="FILL_IN_RAG_RESEARCHER_BACKSTORY",
            tools=[rag_tool],
            allow_delegation=False,
            verbose=1
        )

    @staticmethod
    def create_rag_analyst_agent(llm, rag_tool, function_calling_llm=None) -> Agent:
        """Create a RAG-enabled analyst agent template"""
        return Agent(
            llm=llm,
            function_calling_llm=function_calling_llm,
            role="FILL_IN_RAG_ANALYST_ROLE",
            goal="FILL_IN_RAG_ANALYST_GOAL",
            backstory="FILL_IN_RAG_ANALYST_BACKSTORY",
            tools=[rag_tool],
            allow_delegation=False,
            verbose=1
        )

    @staticmethod
    def create_rag_writer_agent(llm, rag_tool, function_calling_llm=None) -> Agent:
        """Create a RAG-enabled writer agent template"""
        return Agent(
            llm=llm,
            function_calling_llm=function_calling_llm,
            role="FILL_IN_RAG_WRITER_ROLE",
            goal="FILL_IN_RAG_WRITER_GOAL",
            backstory="FILL_IN_RAG_WRITER_BACKSTORY",
            tools=[rag_tool],
            allow_delegation=False,
            verbose=1
        )

    @staticmethod
    def create_custom_rag_agent(llm, rag_tool, role: str, goal: str, backstory: str,
                               additional_tools: List = None, function_calling_llm=None) -> Agent:
        """Create a custom RAG-enabled agent with specified parameters"""
        tools = [rag_tool]
        if additional_tools:
            tools.extend(additional_tools)

        return Agent(
            llm=llm,
            function_calling_llm=function_calling_llm,
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools,
            allow_delegation=False,
            verbose=1
        )

class AgentConfiguration:
    """Configuration helper for agent creation"""

    # Common role templates
    ROLE_TEMPLATES = {
        "legal_researcher": "Senior Legal Research Specialist",
        "medical_researcher": "Medical Information Specialist",
        "technical_researcher": "Technical Documentation Specialist",
        "business_analyst": "Business Intelligence Analyst",
        "content_writer": "Senior Content Creator",
        "data_analyst": "Data Analysis Expert",
        "customer_support": "Customer Support Specialist"
    }

    # Common goal templates
    GOAL_TEMPLATES = {
        "research": "Conduct comprehensive research using available knowledge base and provide accurate, well-sourced information",
        "analysis": "Analyze information from the knowledge base to identify patterns, insights, and recommendations",
        "writing": "Create well-structured, informative content based on research findings from the knowledge base",
        "support": "Provide helpful responses to user queries using the available knowledge base"
    }

    # Common backstory templates
    BACKSTORY_TEMPLATES = {
        "researcher": "You are an experienced researcher with expertise in finding and synthesizing information from various sources. You excel at using knowledge bases to provide comprehensive and accurate answers.",
        "analyst": "You are a skilled analyst with years of experience in data interpretation and insight generation. You can quickly identify patterns and draw meaningful conclusions from information.",
        "writer": "You are a professional writer with expertise in creating clear, engaging content. You excel at transforming research findings into well-structured documents.",
        "specialist": "You are a domain specialist with deep expertise in your field. You can quickly navigate through technical information and provide expert-level insights."
    }

    @staticmethod
    def get_template_config(agent_type: str) -> Dict[str, str]:
        """Get a complete configuration template for common agent types"""
        configs = {
            "legal_researcher": {
                "role": "Senior Legal Research Specialist",
                "goal": "Research legal precedents, statutes, and regulations using the legal knowledge base to provide accurate legal information and analysis",
                "backstory": "You are an experienced legal researcher with expertise in legal databases and document analysis. You excel at finding relevant case law, statutes, and legal principles to support legal arguments and provide comprehensive legal research."
            },
            "medical_researcher": {
                "role": "Medical Information Specialist",
                "goal": "Research medical literature, drug information, and clinical guidelines using the medical knowledge base to provide evidence-based medical information",
                "backstory": "You are a medical information specialist with extensive experience in medical literature review and evidence-based research. You excel at finding and synthesizing medical information from peer-reviewed sources."
            },
            "technical_writer": {
                "role": "Technical Documentation Specialist",
                "goal": "Create comprehensive technical documentation by researching and synthesizing information from technical knowledge bases",
                "backstory": "You are a technical writer with expertise in translating complex technical information into clear, user-friendly documentation. You excel at organizing and presenting technical information in an accessible format."
            },
            "business_analyst": {
                "role": "Business Intelligence Analyst",
                "goal": "Analyze business data and market information from the knowledge base to provide strategic insights and recommendations",
                "backstory": "You are a business analyst with extensive experience in market research and business intelligence. You excel at identifying trends, opportunities, and strategic insights from business data."
            }
        }

        return configs.get(agent_type, {
            "role": "FILL_IN_CUSTOM_ROLE",
            "goal": "FILL_IN_CUSTOM_GOAL",
            "backstory": "FILL_IN_CUSTOM_BACKSTORY"
        })