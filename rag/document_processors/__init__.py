import os
from typing import List, Dict, Any
from .base_processor import BaseDocumentProcessor
from .text_processor import TextDocumentProcessor
from .pdf_processor import PDFDocumentProcessor
from .docx_processor import DOCXDocumentProcessor

class DocumentProcessorFactory:
    """Factory for creating document processors based on file extension"""

    @staticmethod
    def get_processor(file_path: str, config: Dict[str, Any]) -> BaseDocumentProcessor:
        """Get appropriate processor for file type"""
        _, ext = os.path.splitext(file_path.lower())

        if ext in ['.txt', '.md']:
            return TextDocumentProcessor(config)
        elif ext == '.pdf':
            return PDFDocumentProcessor(config)
        elif ext in ['.docx', '.doc']:
            return DOCXDocumentProcessor(config)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def process_documents(file_paths: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process multiple documents and return all chunks"""
        all_chunks = []

        for file_path in file_paths:
            try:
                processor = DocumentProcessorFactory.get_processor(file_path, config)
                chunks = processor.process_document(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

        return all_chunks

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Get list of supported file extensions"""
        return ['.txt', '.md', '.pdf', '.docx', '.doc']

__all__ = [
    "BaseDocumentProcessor",
    "TextDocumentProcessor",
    "PDFDocumentProcessor",
    "DOCXDocumentProcessor",
    "DocumentProcessorFactory"
]