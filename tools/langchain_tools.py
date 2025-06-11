# tools/langchain_tools.py

from smolagents import tool
import requests
from urllib.parse import urljoin
import tempfile
import base64
from typing import Optional, Dict, Any
import os
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.utilities import GoogleSerperAPIWrapper

@tool
def search_web_serper(query: str, num_results: int = 5) -> str:
    """Search the web using Serper API for current information and real-time data.
    
    Perfect for current events, recent news, real-time information.
    
    Args:
        query: Search query for current/recent information
        num_results: Number of results to return (1-10)
    """
    try:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return "❌ SERPER_API_KEY not set"

        # Validate and limit num_results
        num_results = min(max(1, num_results), 3)
        
        search = GoogleSerperAPIWrapper()
        
        # Get structured results
        results = search.results(query)
        formatted_results = []
        
        # Add knowledge graph, answer box, and search results
        if "knowledgeGraph" in results:
            kg = results["knowledgeGraph"]
            formatted_results.append(f"Knowledge: {kg.get('description', '')}")
        
        if "answerBox" in results:
            ab = results["answerBox"]
            formatted_results.append(f"Answer: {ab.get('answer', ab.get('snippet', ''))}")
        
        if "organic" in results:
            for i, result in enumerate(results["organic"][:num_results], 1):
                formatted_results.append(f"{i}. {result.get('title')}: {result.get('snippet')}")
        
        if not formatted_results:
            return f"No web search results found for: {query}"
        
        return f"Web search for '{query}':\n" + "\n".join(formatted_results)
        
    except Exception as e:
        return f"Web search error: {str(e)}"

@tool
def search_wikipedia(query: str, max_docs: int = 2) -> str:
    """Search Wikipedia for reliable information.
    
    Perfect for general knowledge questions, historical facts, scientific concepts.
    
    Args:
        query: What to search for on Wikipedia
        max_docs: Maximum number of articles to return (1-5)
    """
    try:
        
        max_docs = min(max(1, max_docs), 5)  # Limit between 1-5
        search_docs = WikipediaLoader(query=query, load_max_docs=max_docs).load()
        
        if not search_docs:
            return f"No Wikipedia articles found for: {query}"
        
        formatted_results = "\n\n---\n\n".join([
            f'<Document source="{doc.metadata["source"]}" title="{doc.metadata.get("title", "")}">\n{doc.page_content[:1500]}\n</Document>'
            for doc in search_docs
        ])
        
        return f"Wikipedia search results for '{query}':\n\n{formatted_results}"
        
    except ImportError:
        return "WikipediaLoader not available. Install: pip install langchain-community"
    except Exception as e:
        return f"Wikipedia search error: {str(e)}"


@tool
def search_arxiv(query: str, max_papers: int = 3) -> str:
    """Search ArXiv for scientific papers and research.
    
    Excellent for academic questions, scientific methodology, recent research.
    
    Args:
        query: Scientific topic or paper to search for
        max_papers: Maximum number of papers to return (1-5)
    """
    try:        
        max_papers = min(max(1, max_papers), 5)  # Limit between 1-5
        search_docs = ArxivLoader(query=query, load_max_docs=max_papers).load()
        
        if not search_docs:
            return f"No ArXiv papers found for: {query}"
        
        formatted_results = "\n\n---\n\n".join([
            f'<Paper source="{doc.metadata["source"]}" authors="{doc.metadata.get("authors", "")}">\n'
            f'Title: {doc.metadata.get("title", "")}\n'
            f'Summary: {doc.page_content[:1000]}\n'
            f'</Paper>'
            for doc in search_docs
        ])
        
        return f"ArXiv search results for '{query}':\n\n{formatted_results}"
        
    except ImportError:
        return "ArxivLoader not available. Install: pip install langchain-community"
    except Exception as e:
        return f"ArXiv search error: {str(e)}"
    
@tool
def final_answer(answer: str) -> str:
    """Provide the final answer to the GAIA question.
    
    Use this tool when you have gathered all necessary information and are ready 
    to provide the definitive answer to the question.
    
    Args:
        answer: The final answer to the GAIA question
    """
    return f"FINAL ANSWER: {answer}"

    
# Export all tools
ALL_LANGCHAIN_TOOLS = [
    search_wikipedia,
    search_web_serper,
    search_arxiv,
    final_answer
]