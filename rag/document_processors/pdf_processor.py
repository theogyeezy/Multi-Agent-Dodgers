from typing import Dict, Any
from .base_processor import BaseDocumentProcessor

class PDFDocumentProcessor(BaseDocumentProcessor):
    """PDF file processor"""

    def load_document(self, file_path: str) -> str:
        """Load PDF document from file"""
        try:
            import PyPDF2
        except ImportError:
            try:
                import pypdf
                PyPDF2 = pypdf
            except ImportError:
                raise ImportError("PyPDF2 or pypdf not installed. Run: pip install PyPDF2 or pip install pypdf")

        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Error loading PDF file {file_path}: {str(e)}")

    def process_document(self, file_path: str) -> list:
        """Process PDF document with additional metadata"""
        chunks = super().process_document(file_path)

        # Add file-specific metadata
        for chunk in chunks:
            chunk["metadata"]["file_type"] = "pdf"
            chunk["metadata"]["processor"] = "PDFDocumentProcessor"

        return chunks