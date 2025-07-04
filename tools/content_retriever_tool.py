from smolagents import Tool
from typing import Optional
from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker
from sentence_transformers import SentenceTransformer, util
import torch


class ContentRetrieverTool(Tool):
    name = "retrieve_content"
    description = """Retrieve the content of a webpage or document in markdown format. Supports PDF, DOCX, XLSX, HTML, images, and more."""
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL or local path of the webpage or document to retrieve.",
        },
        "query": {
            "type": "string",
            "description": "The subject on the page you are looking for. The shorter the more relevant content is returned.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float = 0.2,
        **kwargs,
    ):
        self.threshold = threshold
        self._document_converter = DocumentConverter()
        self._model = SentenceTransformer(
            model_name if model_name is not None else "all-MiniLM-L6-v2"
        )
        self._chunker = HierarchicalChunker()
        
        self._state_question = None

        super().__init__(**kwargs)
        self._has_docling = self._check_docling_availability()
        self._initialized = False
    
    def _check_docling_availability(self):
        """Check if docling is available"""
        try:
            import docling
            return True
        except ImportError:
            return False
    
    def setup(self):
        """Initialize the tool"""
        if not hasattr(self, '_has_docling'):
            self._has_docling = self._check_docling_availability()
        self._initialized = True
    
    def configure_from_state(self, question: str):
        """Store question for potential query enhancement"""
        self._state_question = question
        print(f"🔧 ContentRetriever noted question context: {question[:50]}...")

    def forward(self, url: str, query: Optional[str] = None) -> str:
        """
        OFFICIAL WORKAROUND: Use Optional[str] = None in signature
        and handle nullable validation at runtime
        """
        # Runtime validation is handled by the parameter being optional
        # query=None is perfectly valid for this tool
        
        if not url or not url.strip():
            raise ValueError("url parameter is required and cannot be empty")
        
        # Handle None query (which is valid)
        if query is None:
            query = ""
        
        try:
            if self._has_docling:
                return self._process_with_docling(url, query)
            else:
                return self._process_basic(url, query)
                
        except Exception as e:
            return f"Error retrieving content: {str(e)}"
    
    def _process_with_docling(self, url: str, query: str) -> str:
        """Process content using docling (advanced)"""
        import torch
        from sentence_transformers import util
        
        document = self._document_converter.convert(url).document
        chunks = list(self._chunker.chunk(dl_doc=document))
        
        if len(chunks) == 0:
            return "No content found."

        chunks_text = [chunk.text for chunk in chunks]
        
        # If no query, return first few chunks
        if not query.strip():
            chunks_with_context = [self._chunker.contextualize(chunk) for chunk in chunks[:3]]
            return "\n\n".join(chunks_with_context)
        
        # Process with query
        chunks_with_context = [self._chunker.contextualize(chunk) for chunk in chunks]
        chunks_context = [
            chunks_with_context[i].replace(chunks_text[i], "").strip()
            for i in range(len(chunks))
        ]

        chunk_embeddings = self._model.encode(chunks_text, convert_to_tensor=True)
        context_embeddings = self._model.encode(chunks_context, convert_to_tensor=True)
        query_embedding = self._model.encode(
            [term.strip() for term in query.split(",") if term.strip()],
            convert_to_tensor=True,
        )

        selected_indices = []
        for embeddings in [context_embeddings, chunk_embeddings]:
            for cos_scores in util.pytorch_cos_sim(query_embedding, embeddings):
                probabilities = torch.nn.functional.softmax(cos_scores, dim=0)
                sorted_indices = torch.argsort(probabilities, descending=True)
                
                cumulative = 0.0
                for i in sorted_indices:
                    cumulative += probabilities[i].item()
                    selected_indices.append(i.item())
                    if cumulative >= self.threshold:
                        break

        selected_indices = list(dict.fromkeys(selected_indices))
        selected_indices = selected_indices[::-1]

        if len(selected_indices) == 0:
            return "No relevant content found."

        return "\n\n".join([chunks_with_context[idx] for idx in selected_indices])
    
    def _process_basic(self, url: str, query: str) -> str:
        """Basic content processing without docling"""
        import requests
        import re
        
        if url.startswith('http'):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            content = response.text
            
            # Basic HTML cleaning
            content = re.sub(r'<script.*?</script>', '', content, flags=re.DOTALL)
            content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL)
            content = re.sub(r'<[^>]+>', '', content)
            content = re.sub(r'\s+', ' ', content).strip()
        else:
            # Local file
            import os
            if os.path.exists(url):
                with open(url, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            else:
                return f"File not found: {url}"
        
        # Limit content length
        if len(content) > 5000:
            content = content[:5000] + "..."
        
        return content
