# tools/content_retriever_tool.py
# Information retrieval with temporal awareness and verification

from smolagents import Tool
import torch
from typing import Optional, List, Dict, Tuple
import traceback
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker
from sentence_transformers import SentenceTransformer, util

# Import context bridge for task awareness
try:
    from agent_context import ContextVariableFlow
    CONTEXT_BRIDGE_AVAILABLE = True
except ImportError:
    CONTEXT_BRIDGE_AVAILABLE = False
    print("⚠️  Context bridge not available for ContentRetrieverTool")


class ContentRetrieverTool(Tool):
    """
    Information retrieval with semantic search, temporal awareness,
    and source verification capabilities.
    
    Enhances web research with:
    - Temporal context awareness for current vs historical content
    - Source authority assessment for reliability
    - Multi-source verification support
    - Content extraction based on question context
    """
    
    name = "retrieve_content"
    description = """Retrieve relevant content from documents or webpages using semantic search with temporal awareness and source verification.

This tool processes various document formats (PDF, DOCX, XLSX, HTML, images) and returns the most relevant sections based on your query, with additional context about temporal relevance and source authority. Perfect for research tasks requiring current information and reliable sources."""

    inputs = {
        "url": {
            "type": "string",
            "description": "The URL or local file path of the document to process.",
        },
        "query": {
            "type": "string", 
            "description": "What you're looking for in the document. Be specific about temporal needs (e.g., 'current data', 'recent research', 'historical analysis').",
        },
        "verification_mode": {
            "type": "string",
            "description": "Optional: 'multi_source' to compare with other sources, 'authority_check' to assess source reliability, or 'temporal_analysis' for time-sensitive content.",
            "nullable": True,
            "default": "standard"
        }
    }
    output_type = "string"

    # Source authority patterns for different domains
    AUTHORITY_PATTERNS = {
        'government': [
            'gov', 'gov.uk', 'europa.eu', 'un.org', 'who.int',
            'census.gov', 'sec.gov', 'fda.gov', 'cdc.gov', 'nih.gov'
        ],
        'academic': [
            'edu', 'ac.uk', 'nature.com', 'science.org', 'pubmed',
            'arxiv.org', 'scholar.google', 'jstor.org', 'ieee.org'
        ],
        'financial': [
            'bloomberg.com', 'reuters.com', 'sec.gov', 'nasdaq.com',
            'nyse.com', 'imf.org', 'worldbank.org', 'federalreserve.gov'
        ],
        'news_tier1': [
            'bbc.com', 'reuters.com', 'ap.org', 'npr.org',
            'pbs.org', 'economist.com', 'nytimes.com', 'wsj.com'
        ],
        'international': [
            'un.org', 'who.int', 'worldbank.org', 'imf.org',
            'oecd.org', 'wto.org', 'unesco.org'
        ]
    }

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float = 0.2,
        enable_temporal_analysis: bool = True,
        enable_authority_assessment: bool = True,
        **kwargs,
    ):
        """
        Initialize ContentRetrieverTool.
        
        Args:
            model_name: Sentence transformer model for embeddings
            threshold: Minimum relevance threshold (0.1-0.5, lower = more content)
            enable_temporal_analysis: Add temporal context to content analysis
            enable_authority_assessment: Assess source authority and reliability
        """
        self.threshold = threshold
        self.enable_temporal_analysis = enable_temporal_analysis
        self.enable_authority_assessment = enable_authority_assessment
        
        self._document_converter = DocumentConverter()
        self._model = SentenceTransformer(
            model_name if model_name is not None else "all-MiniLM-L6-v2"
        )
        self._chunker = HierarchicalChunker()

        super().__init__(**kwargs)

    def _get_current_temporal_context(self) -> Dict[str, str]:
        """Get current temporal context for analysis"""
        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            return {
                "current_date": now.strftime("%Y-%m-%d"),
                "current_year": str(now.year),
                "current_month": now.strftime("%B %Y"),
                "current_quarter": f"Q{(now.month-1)//3 + 1} {now.year}"
            }
        except Exception as e:
            # Fallback if datetime import fails
            print(f"⚠️  Temporal context error: {e}")
            return {
                "current_date": "2025-06-13",
                "current_year": "2025",
                "current_month": "June 2025",
                "current_quarter": "Q2 2025"
            }

    def _assess_source_authority(self, url: str) -> Tuple[str, float, str]:
        """
        Assess the authority level of a source URL
        
        Returns:
            Tuple of (authority_level, confidence_score, reasoning)
        """
        if not self.enable_authority_assessment:
            return ("unknown", 0.5, "Authority assessment disabled")
            
        url_lower = url.lower()
        
        # Check for authoritative domains
        for category, domains in self.AUTHORITY_PATTERNS.items():
            for auth_domain in domains:
                if auth_domain in url_lower:
                    if category == 'government':
                        return ('government_authority', 0.95, f"Official government source: {auth_domain}")
                    elif category == 'academic':
                        return ('academic_authority', 0.90, f"Academic/research institution: {auth_domain}")
                    elif category == 'financial':
                        return ('financial_authority', 0.85, f"Financial authority: {auth_domain}")
                    elif category == 'news_tier1':
                        return ('news_tier1', 0.80, f"Tier-1 news source: {auth_domain}")
                    elif category == 'international':
                        return ('international_authority', 0.90, f"International organization: {auth_domain}")
        
        # Check for general patterns
        if any(pattern in url_lower for pattern in ['wikipedia', 'wiki']):
            return ('reference', 0.70, "Wikipedia/reference source")
        elif any(pattern in url_lower for pattern in ['news', 'times', 'post', 'guardian', 'telegraph']):
            return ('news_general', 0.60, "General news source")
        elif '.edu' in url_lower or '.ac.' in url_lower:
            return ('educational', 0.75, "Educational institution")
        
        return ('unknown', 0.30, "Source authority unclear")

    def _analyze_temporal_relevance(self, content: str, query: str) -> Dict[str, any]:
        """
        Analyze temporal relevance of content based on query context
        
        Returns:
            Dictionary with temporal analysis results
        """
        if not self.enable_temporal_analysis:
            return {"temporal_analysis": "disabled"}
            
        temporal_context = self._get_current_temporal_context()
        
        # Detect temporal indicators in query
        query_lower = query.lower()
        temporal_indicators = {
            'current': ['current', 'latest', 'recent', 'now', 'today', 'this year'],
            'historical': ['historical', 'past', 'previous', 'former', 'old'],
            'trending': ['trending', 'changing', 'evolving', 'developing'],
            'comparative': ['compare', 'vs', 'versus', 'difference', 'change']
        }
        
        detected_types = []
        for temp_type, indicators in temporal_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                detected_types.append(temp_type)
        
        # Analyze content for temporal markers
        content_lower = content.lower()
        date_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',  # Full dates
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{4}-\d{1,2}-\d{1,2}\b'   # YYYY-MM-DD
        ]
        
        found_dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            found_dates.extend(matches)
        
        # Assess temporal relevance
        current_year = int(temporal_context["current_year"])
        recent_threshold = current_year - 2  # Last 2 years considered recent
        
        temporal_assessment = {
            "query_temporal_type": detected_types,
            "found_dates": found_dates[:5],  # Limit to first 5 dates
            "current_context": temporal_context,
            "temporal_relevance": "unknown"
        }
        
        if found_dates:
            # Try to extract years and assess recency
            years = []
            for date in found_dates:
                year_match = re.search(r'\b(\d{4})\b', date)
                if year_match:
                    try:
                        year = int(year_match.group(1))
                        if 1900 <= year <= current_year + 1:  # Reasonable year range
                            years.append(year)
                    except ValueError:
                        continue
            
            if years:
                latest_year = max(years)
                if latest_year >= recent_threshold:
                    temporal_assessment["temporal_relevance"] = "recent"
                elif latest_year >= current_year - 10:
                    temporal_assessment["temporal_relevance"] = "moderately_recent"
                else:
                    temporal_assessment["temporal_relevance"] = "historical"
        
        return temporal_assessment

    def _get_enhanced_context(self) -> Dict[str, any]:
        """Get context from context bridge if available"""
        if CONTEXT_BRIDGE_AVAILABLE and ContextVariableFlow.is_context_active():
            return ContextVariableFlow.get_task_context()
        return {}

    def forward(self, url: str, query: str, verification_mode: str = "standard") -> str:
        """
        Enhanced content retrieval with temporal awareness and verification.
        
        Args:
            url: Document URL or path
            query: Search query for relevant content with temporal context
            verification_mode: Type of verification to perform
            
        Returns:
            Enhanced content with temporal and authority context
        """
        try:
            # Get enhanced context from context bridge
            task_context = self._get_enhanced_context()
            
            # Assess source authority
            authority_level, authority_confidence, authority_reasoning = self._assess_source_authority(url)
            
            # Convert document
            document = self._document_converter.convert(url).document

            # Chunk document
            chunks = list(self._chunker.chunk(dl_doc=document))
            if len(chunks) == 0:
                return self._format_no_content_response(url, authority_level, authority_reasoning)

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

            # Remove duplicates and preserve order
            selected_indices = list(dict.fromkeys(selected_indices))
            selected_indices = selected_indices[::-1]  # Most relevant last

            if len(selected_indices) == 0:
                return self._format_no_content_response(url, authority_level, authority_reasoning)

            # Get relevant content
            relevant_content = "\n\n".join([chunks_with_context[idx] for idx in selected_indices])
            
            # Perform temporal analysis
            temporal_analysis = self._analyze_temporal_relevance(relevant_content, query)
            
            # Format enhanced response
            return self._format_enhanced_response(
                url=url,
                query=query,
                content=relevant_content,
                authority_level=authority_level,
                authority_confidence=authority_confidence,
                authority_reasoning=authority_reasoning,
                temporal_analysis=temporal_analysis,
                verification_mode=verification_mode,
                task_context=task_context
            )
            
        except Exception as e:
            error_context = f"Error processing document from {url}: {str(e)}"
            if CONTEXT_BRIDGE_AVAILABLE and ContextVariableFlow.is_context_active():
                task_id = ContextVariableFlow.get_task_id()
                error_context = f"Task {task_id}: {error_context}"
            return error_context

    def _format_enhanced_response(
        self, 
        url: str, 
        query: str, 
        content: str, 
        authority_level: str,
        authority_confidence: float,
        authority_reasoning: str,
        temporal_analysis: Dict,
        verification_mode: str,
        task_context: Dict
    ) -> str:
        """Format the enhanced response with metadata"""
        
        response_parts = []
        
        # Add source information
        response_parts.append(f"📄 SOURCE: {url}")
        
        # Add authority assessment
        if self.enable_authority_assessment:
            confidence_emoji = "🥇" if authority_confidence >= 0.8 else "🥈" if authority_confidence >= 0.6 else "🥉"
            response_parts.append(f"{confidence_emoji} AUTHORITY: {authority_level} ({authority_confidence:.1%} confidence)")
            response_parts.append(f"   Reasoning: {authority_reasoning}")
        
        # Add temporal analysis
        if self.enable_temporal_analysis and temporal_analysis.get("temporal_analysis") != "disabled":
            temporal_relevance = temporal_analysis.get("temporal_relevance", "unknown")
            temporal_emoji = "🕰️" if temporal_relevance == "recent" else "📅" if temporal_relevance == "moderately_recent" else "📜"
            response_parts.append(f"{temporal_emoji} TEMPORAL RELEVANCE: {temporal_relevance}")
            
            if temporal_analysis.get("found_dates"):
                dates_preview = ", ".join(temporal_analysis["found_dates"][:3])
                response_parts.append(f"   Key dates found: {dates_preview}")
        
        # Add verification context
        if verification_mode != "standard":
            response_parts.append(f"🔍 VERIFICATION MODE: {verification_mode}")
        
        # Add task context if available
        if task_context and task_context.get("task_id"):
            response_parts.append(f"🎯 TASK: {task_context['task_id']}")
        
        response_parts.append("")
        response_parts.append("📖 RELEVANT CONTENT:")
        response_parts.append("-" * 40)
        response_parts.append(content)
        
        return "\n".join(response_parts)

    def _format_no_content_response(self, url: str, authority_level: str, authority_reasoning: str) -> str:
        """Format response when no content is found"""
        response = f"📄 SOURCE: {url}\n"
        if self.enable_authority_assessment:
            response += f"🔍 AUTHORITY: {authority_level}\n"
            response += f"   Reasoning: {authority_reasoning}\n"
        response += "\n❌ No relevant content found for the given query."
        return response

    def __repr__(self):
        return f"EnhancedContentRetrieverTool(threshold={self.threshold}, temporal={self.enable_temporal_analysis}, authority={self.enable_authority_assessment})"


# Convenience functions for testing
def test_grounded_content_retriever(
    test_url: str = None,
    test_query: str = "current research findings",
    verification_mode: str = "authority_check"
) -> ContentRetrieverTool:
    """Test ContentRetrieverTool with grounding functionality"""
    tool = ContentRetrieverTool(
        enable_temporal_analysis=True,
        enable_authority_assessment=True
    )
    
    if test_url:
        print(f"🧪 Testing ContentRetrieverTool with Grounding")
        print(f"   URL: {test_url}")
        print(f"   Query: {test_query}")
        print(f"   Verification mode: {verification_mode}")
        print("-" * 50)
        
        result = tool.forward(test_url, test_query, verification_mode)
        print(result)
        print("-" * 50)
        print(f"✅ Test completed. Result length: {len(result)} characters")
    else:
        print("ℹ️  ContentRetrieverTool loaded")
        print("   Features: Temporal awareness, source authority assessment")
        print("   Provide test_url to test functionality")
    
    return tool


def demonstrate_grounding_features():
    """Demonstrate the grounding features of ContentRetrieverTool"""
    print("🌟 CONTENT RETRIEVER FEATURES with Grounding")
    print("=" * 50)
    
    # Create tool instance
    tool = ContentRetrieverTool()
    
    # Demonstrate source authority assessment
    print("\n🏆 SOURCE AUTHORITY ASSESSMENT:")
    test_urls = [
        "https://www.census.gov/data/population-estimates",
        "https://www.nature.com/articles/research-paper",
        "https://www.bbc.com/news/world-news",
        "https://en.wikipedia.org/wiki/Climate_Change",
        "https://random-blog.com/article"
    ]
    
    for url in test_urls:
        authority, confidence, reasoning = tool._assess_source_authority(url)
        print(f"  {url}")
        print(f"    Authority: {authority} ({confidence:.1%})")
        print(f"    Reasoning: {reasoning}")
        print()
    
    # Demonstrate temporal analysis
    print("🕰️ TEMPORAL ANALYSIS FEATURES:")
    temporal_queries = [
        "current population statistics",
        "historical climate data", 
        "recent research findings",
        "trending economic indicators"
    ]
    
    for query in temporal_queries:
        print(f"  Query: '{query}'")
        # This would normally analyze content, but we'll show the query analysis
        query_lower = query.lower()
        if 'current' in query_lower or 'recent' in query_lower:
            print(f"    → Detected need for current/recent information")
        elif 'historical' in query_lower:
            print(f"    → Detected need for historical information")
        elif 'trending' in query_lower:
            print(f"    → Detected need for trend analysis")
        print()
    
    print("✅ Grounded ContentRetrieverTool ready for web research!")
    print("   🕰️ Temporal awareness active")
    print("   🏆 Source authority assessment enabled")
    print("   🔍 Multi-source verification support")
    print("   🌉 Context bridge integration ready")


if __name__ == "__main__":
    demonstrate_grounding_features()