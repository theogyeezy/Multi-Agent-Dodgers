from crewai import Task
from typing import Dict, Any

class TaskTemplates:
    """Template configurations for different types of tasks"""

    @staticmethod
    def create_research_task(agent, description: str = None, expected_output: str = None,
                           output_file: str = None) -> Task:
        """Create a research task template"""
        return Task(
            description=description or "FILL_IN_RESEARCH_TASK_DESCRIPTION",
            expected_output=expected_output or "FILL_IN_RESEARCH_EXPECTED_OUTPUT",
            output_file=output_file or "research_output.md",
            agent=agent
        )

    @staticmethod
    def create_analysis_task(agent, description: str = None, expected_output: str = None,
                           output_file: str = None) -> Task:
        """Create an analysis task template"""
        return Task(
            description=description or "FILL_IN_ANALYSIS_TASK_DESCRIPTION",
            expected_output=expected_output or "FILL_IN_ANALYSIS_EXPECTED_OUTPUT",
            output_file=output_file or "analysis_output.md",
            agent=agent
        )

    @staticmethod
    def create_writing_task(agent, description: str = None, expected_output: str = None,
                          output_file: str = None) -> Task:
        """Create a writing task template"""
        return Task(
            description=description or "FILL_IN_WRITING_TASK_DESCRIPTION",
            expected_output=expected_output or "FILL_IN_WRITING_EXPECTED_OUTPUT",
            output_file=output_file or "final_output.md",
            agent=agent
        )

    @staticmethod
    def create_custom_task(agent, description: str, expected_output: str,
                         output_file: str = None) -> Task:
        """Create a custom task with specified parameters"""
        return Task(
            description=description,
            expected_output=expected_output,
            output_file=output_file,
            agent=agent
        )

class TaskConfiguration:
    """Configuration helper for task creation"""

    # Common task description templates
    DESCRIPTION_TEMPLATES = {
        "legal_research": "Research relevant legal precedents, statutes, and regulations related to [TOPIC]. Use the legal knowledge base to find applicable case law and provide comprehensive legal analysis.",
        "medical_research": "Research medical literature and clinical guidelines related to [CONDITION/TREATMENT]. Use the medical knowledge base to find evidence-based information and current best practices.",
        "technical_documentation": "Create comprehensive technical documentation for [SYSTEM/FEATURE]. Research existing documentation and technical specifications from the knowledge base.",
        "market_analysis": "Analyze market trends and competitive landscape for [INDUSTRY/PRODUCT]. Use business intelligence data from the knowledge base to identify opportunities and threats.",
        "content_creation": "Create engaging content about [TOPIC] based on research findings. Use information from the knowledge base to ensure accuracy and comprehensiveness.",
        "customer_support": "Provide detailed response to customer inquiry about [ISSUE]. Use the knowledge base to find relevant product information and troubleshooting steps."
    }

    # Common expected output templates
    OUTPUT_TEMPLATES = {
        "research_report": "A comprehensive research report including: executive summary, key findings, detailed analysis, supporting evidence from sources, and recommendations. Format as structured markdown document.",
        "legal_brief": "A legal brief including: issue identification, applicable law analysis, case law citations, statutory references, and legal conclusions. Format as professional legal document.",
        "technical_guide": "A technical guide including: overview, step-by-step instructions, code examples (if applicable), troubleshooting section, and additional resources. Format as user-friendly documentation.",
        "market_analysis": "A market analysis report including: market overview, competitive analysis, trend identification, SWOT analysis, and strategic recommendations. Format as business presentation outline.",
        "content_piece": "A well-structured content piece including: engaging introduction, main content sections, supporting data and examples, and clear conclusion. Format as publishable article.",
        "support_response": "A helpful customer support response including: problem acknowledgment, step-by-step solution, relevant links/resources, and follow-up guidance. Format as professional customer communication."
    }

    @staticmethod
    def get_template_config(task_type: str, topic: str = "[TOPIC]") -> Dict[str, str]:
        """Get a complete configuration template for common task types"""
        configs = {
            "legal_research": {
                "description": f"Research relevant legal precedents, statutes, and regulations related to {topic}. Use the legal knowledge base to find applicable case law, analyze legal principles, and provide comprehensive legal analysis with proper citations.",
                "expected_output": "A comprehensive legal research report including: case law analysis, statutory interpretation, regulatory compliance requirements, potential legal risks, and actionable legal recommendations. Format as professional legal memorandum.",
                "output_file": "legal_research_report.md"
            },
            "medical_research": {
                "description": f"Research medical literature and clinical guidelines related to {topic}. Use the medical knowledge base to find evidence-based information, current treatment protocols, and best practices.",
                "expected_output": "A detailed medical research summary including: clinical evidence review, treatment guidelines, contraindications, efficacy data, and clinical recommendations. Format as medical literature review.",
                "output_file": "medical_research_summary.md"
            },
            "technical_analysis": {
                "description": f"Analyze technical documentation and specifications related to {topic}. Use the technical knowledge base to understand system architecture, identify implementation details, and assess technical feasibility.",
                "expected_output": "A technical analysis report including: system overview, architecture assessment, implementation recommendations, potential challenges, and technical specifications. Format as technical specification document.",
                "output_file": "technical_analysis.md"
            },
            "market_research": {
                "description": f"Research market trends, competitive landscape, and business opportunities related to {topic}. Use the business knowledge base to analyze market data and identify strategic insights.",
                "expected_output": "A market research report including: market size analysis, competitive positioning, trend identification, opportunity assessment, and strategic recommendations. Format as business intelligence report.",
                "output_file": "market_research_report.md"
            },
            "content_creation": {
                "description": f"Create comprehensive content about {topic} based on research from the knowledge base. Ensure accuracy, engagement, and proper sourcing of information.",
                "expected_output": "A well-researched content piece including: engaging introduction, informative main sections, supporting data and examples, practical insights, and compelling conclusion. Format as publishable article.",
                "output_file": "content_piece.md"
            }
        }

        return configs.get(task_type, {
            "description": f"FILL_IN_CUSTOM_TASK_DESCRIPTION_FOR_{topic.upper()}",
            "expected_output": "FILL_IN_CUSTOM_EXPECTED_OUTPUT",
            "output_file": "custom_output.md"
        })