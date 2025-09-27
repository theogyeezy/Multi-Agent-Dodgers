from typing import Dict, Any
from .base_processor import BaseDocumentProcessor

class TextDocumentProcessor(BaseDocumentProcessor):
    """Text file processor (.txt, .md)"""

    def load_document(self, file_path: str) -> str:
        """Load text document from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            raise ValueError(f"Error loading text file {file_path}: {str(e)}")

    def process_document(self, file_path: str) -> list:
        """Process text document with additional metadata"""
        chunks = super().process_document(file_path)

        # Add file-specific metadata
        for chunk in chunks:
            chunk["metadata"]["file_type"] = "text"
            chunk["metadata"]["processor"] = "TextDocumentProcessor"

        return chunks