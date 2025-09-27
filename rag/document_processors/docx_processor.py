from typing import Dict, Any
from .base_processor import BaseDocumentProcessor

class DOCXDocumentProcessor(BaseDocumentProcessor):
    """DOCX file processor"""

    def load_document(self, file_path: str) -> str:
        """Load DOCX document from file"""
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx not installed. Run: pip install python-docx")

        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Error loading DOCX file {file_path}: {str(e)}")

    def process_document(self, file_path: str) -> list:
        """Process DOCX document with additional metadata"""
        chunks = super().process_document(file_path)

        # Add file-specific metadata
        for chunk in chunks:
            chunk["metadata"]["file_type"] = "docx"
            chunk["metadata"]["processor"] = "DOCXDocumentProcessor"

        return chunks