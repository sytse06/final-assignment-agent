from smolagents import Tool
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
    
    def configure_from_state(self, question: str):
        """Store question for potential query enhancement"""
        self._state_question = question
        print(f"🔧 ContentRetriever noted question context: {question[:50]}...")

    def forward(self, url: str, query: str) -> str:
        document = self._document_converter.convert(url).document

        chunks = list(self._chunker.chunk(dl_doc=document))
        if len(chunks) == 0:
            return "No content found."

        chunks_text = [chunk.text for chunk in chunks]
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

        selected_indices = []  # aggregate indexes across chunks and context matches and for all queries
        for embeddings in [
            context_embeddings,
            chunk_embeddings,
        ]:
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

        selected_indices = list(
            dict.fromkeys(selected_indices)
        )  # remove duplicates and preserve order
        selected_indices = selected_indices[
            ::-1
        ]  # make most relevant items last for better focus

        if len(selected_indices) == 0:
            return "No content found."

        return "\n\n".join([chunks_with_context[idx] for idx in selected_indices])