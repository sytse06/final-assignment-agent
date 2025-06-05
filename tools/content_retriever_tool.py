# tools/content_retriever_tool.py  
# Document content retrieval with semantic search

from smolagents import Tool
import torch
from typing import Optional, List
import traceback


from smolagents import Tool
from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker
from sentence_transformers import SentenceTransformer, util
import torch


class ContentRetrieverTool(Tool):
    """
    Advanced document content retrieval with semantic search and intelligent chunking.
    
    Uses IBM's docling for professional-grade document processing and semantic similarity
    for targeted content extraction. Supports PDF, DOCX, XLSX, HTML, images, and more.
    """
    
    name = "retrieve_content"
    description = """Retrieve relevant content from documents or webpages using semantic search.

This tool processes various document formats (PDF, DOCX, XLSX, HTML, images) and returns 
only the most relevant sections based on your query. Use this when you need to extract 
specific information from large documents or find relevant sections in PDFs."""

    inputs = {
        "url": {
            "type": "string",
            "description": "The URL or local file path of the document to process.",
        },
        "query": {
            "type": "string", 
            "description": "What you're looking for in the document. Be specific but concise.",
        },
    }
    output_type = "string"

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float = 0.2,
        **kwargs,
    ):
        """
        Initialize ContentRetrieverTool.
        
        Args:
            model_name: Sentence transformer model for embeddings
            threshold: Minimum relevance threshold (0.1-0.5, lower = more content)
        """
        self.threshold = threshold
        self._document_converter = DocumentConverter()
        self._model = SentenceTransformer(
            model_name if model_name is not None else "all-MiniLM-L6-v2"
        )
        self._chunker = HierarchicalChunker()

        super().__init__(**kwargs)

    def forward(self, url: str, query: str) -> str:
        """
        Retrieve relevant content from document based on query.
        
        Args:
            url: Document URL or path
            query: Search query for relevant content
            
        Returns:
            Relevant document content
        """
        try:
            # Convert document
            document = self._document_converter.convert(url).document

            # Chunk document
            chunks = list(self._chunker.chunk(dl_doc=document))
            if len(chunks) == 0:
                return "No content found."

            # Extract text and context
            chunks_text = [chunk.text for chunk in chunks]
            chunks_with_context = [self._chunker.contextualize(chunk) for chunk in chunks]
            chunks_context = [
                chunks_with_context[i].replace(chunks_text[i], "").strip()
                for i in range(len(chunks))
            ]

            # Create embeddings
            chunk_embeddings = self._model.encode(chunks_text, convert_to_tensor=True)
            context_embeddings = self._model.encode(chunks_context, convert_to_tensor=True)
            query_embedding = self._model.encode(
                [term.strip() for term in query.split(",") if term.strip()],
                convert_to_tensor=True,
            )

            # Find relevant content using semantic similarity
            selected_indices = []  # aggregate indexes across chunks and context matches and for all queries
            for embeddings in [context_embeddings, chunk_embeddings]:
                # Compute cosine similarities (returns 1D tensor)
                for cos_scores in util.pytorch_cos_sim(query_embedding, embeddings):
                    # Convert to softmax probabilities
                    probabilities = torch.nn.functional.softmax(cos_scores, dim=0)
                    # Sort by probability descending
                    sorted_indices = torch.argsort(probabilities, descending=True)
                    # Accumulate until total probability reaches threshold

                    cumulative = 0.0
                    for i in sorted_indices:
                        cumulative += probabilities[i].item()
                        selected_indices.append(i.item())
                        if cumulative >= self.threshold:
                            break

            # Remove duplicates and preserve order
            selected_indices = list(dict.fromkeys(selected_indices))
            # Make most relevant items last for better focus
            selected_indices = selected_indices[::-1]

            if len(selected_indices) == 0:
                return "No content found."

            return "\n\n".join([chunks_with_context[idx] for idx in selected_indices])
            
        except Exception as e:
            return f"Error processing document: {str(e)}"

    def __repr__(self):
        return f"ContentRetrieverTool(threshold={self.threshold})"


# Convenience functions for testing
def test_content_retriever(
    test_url: str = None,
    test_query: str = "test content"
) -> ContentRetrieverTool:
    """Test ContentRetrieverTool functionality"""
    tool = ContentRetrieverTool()
    
    if test_url:
        print(f"🧪 Testing with URL: {test_url}")
        result = tool.forward(test_url, test_query)
        print(f"Result length: {len(result)} characters")
        print(f"First 200 chars: {result[:200]}...")
    else:
        print("ℹ️  ContentRetrieverTool loaded - provide test_url to test functionality")
    
    return tool


if __name__ == "__main__":
    # Basic test
    print("🧪 Testing ContentRetrieverTool...")
    test_tool = test_content_retriever()
    print("✅ ContentRetrieverTool loaded successfully")