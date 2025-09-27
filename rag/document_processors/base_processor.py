from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseDocumentProcessor(ABC):
    """Abstract base class for document processors"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config.get("chunk_size", 500)
        self.chunk_overlap = config.get("chunk_overlap", 50)

    @abstractmethod
    def load_document(self, file_path: str) -> str:
        """Load document from file path"""
        pass

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if not text:
            return []

        chunks = []
        words = text.split()

        if len(words) <= self.chunk_size:
            return [text]

        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            if end >= len(words):
                break

            start = end - self.chunk_overlap

        return chunks

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process document and return chunks with metadata"""
        text = self.load_document(file_path)
        chunks = self.chunk_text(text)

        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                "content": chunk,
                "metadata": {
                    "source": file_path,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })

        return processed_chunks