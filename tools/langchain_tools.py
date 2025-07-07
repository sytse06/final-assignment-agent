# tools/langchain_tools.py
# Simple SmolagAgents tools using @tool decorator (back to working approach)

import os
from smolagents import Tool, tool

# LangChain imports for utilities
try:
    from langchain_community.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper
    from langchain_community.document_loaders import ArxivLoader
    LANGCHAIN_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LangChain imports failed: {e}")
    LANGCHAIN_IMPORTS_AVAILABLE = False

# ============================================================================
# NATIVE SMOLAGENTS TOOLS (using @tool decorator - your working approach)
# ============================================================================

@tool
def search_web_serper(query: str, num_results: int = 3) -> str:
    """Search the web using Serper API for current information and real-time data.
    
    Perfect for current events, recent news, real-time information.
    
    Args:
        query: Search query for current/recent information
        num_results: Number of results to return (1-5)
    """
    try:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return "❌ SERPER_API_KEY not set"

        # Validate and limit num_results
        num_results = min(max(1, int(num_results)), 3)
        
        if not LANGCHAIN_IMPORTS_AVAILABLE:
            return "❌ GoogleSerperAPIWrapper not available - install langchain-community"
        
        search = GoogleSerperAPIWrapper(serper_api_key=api_key)
        
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
        max_docs: Maximum number of articles to return (1-3)
    """
    try:
        if not LANGCHAIN_IMPORTS_AVAILABLE:
            return "❌ WikipediaAPIWrapper not available - install langchain-community"
        
        max_docs = min(max(1, int(max_docs)), 3)  # Limit between 1-3
        
        wikipedia = WikipediaAPIWrapper()
        
        # Search for articles
        try:
            result = wikipedia.run(query)
            
            if not result or result.strip() == "":
                return f"No Wikipedia articles found for: {query}"
            
            # Limit result length
            if len(result) > 1500:
                result = result[:1500] + "..."
            
            return f"Wikipedia search results for '{query}':\n\n{result}"
            
        except Exception as search_error:
            return f"Wikipedia search error: {str(search_error)}"
        
    except Exception as e:
        return f"Wikipedia tool error: {str(e)}"

@tool
def search_arxiv(query: str, max_papers: int = 2) -> str:
    """Search ArXiv for scientific papers and research.
    
    Excellent for academic questions, scientific methodology, recent research.
    
    Args:
        query: Scientific topic or paper to search for
        max_papers: Maximum number of papers to return (1-3)
    """
    try:
        # Check for required dependencies
        try:
            import arxiv
            import fitz  # PyMuPDF
        except ImportError as dep_error:
            missing = []
            try:
                import arxiv
            except ImportError:
                missing.append("arxiv")
            try:
                import fitz
            except ImportError:
                missing.append("pymupdf")
            
            return f"❌ ArXiv tool missing dependencies: {', '.join(missing)}. Install with: pip install {' '.join(missing)}"
        
        if not LANGCHAIN_IMPORTS_AVAILABLE:
            return "❌ ArxivLoader not available - install langchain-community"
        
        max_papers = min(max(1, int(max_papers)), 3)  # Limit between 1-3
        
        try:
            loader = ArxivLoader(query=query, load_max_docs=max_papers)
            docs = loader.load()
            
            if not docs:
                return f"No ArXiv papers found for: {query}"
            
            formatted_results = []
            for i, doc in enumerate(docs, 1):
                title = doc.metadata.get("title", "Unknown Title")
                authors = doc.metadata.get("authors", "Unknown Authors")
                summary = doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content
                
                formatted_results.append(f"{i}. Title: {title}\n   Authors: {authors}\n   Summary: {summary}")
            
            return f"ArXiv search results for '{query}':\n\n" + "\n\n".join(formatted_results)
            
        except Exception as search_error:
            return f"ArXiv search error: {str(search_error)}"
        
    except Exception as e:
        return f"ArXiv tool error: {str(e)}"

# ============================================================================
# TOOL COLLECTION AND AVAILABILITY CHECKING
# ============================================================================

def check_tool_availability():
    """Check which tools are available based on environment and dependencies"""
    available_tools = []
    status = {}
    
    # Check Wikipedia
    if LANGCHAIN_IMPORTS_AVAILABLE:
        available_tools.append(search_wikipedia)
        status['wikipedia'] = True
        print("✅ Wikipedia tool available")
    else:
        status['wikipedia'] = False
        print("❌ Wikipedia tool unavailable (missing langchain-community)")
    
    # Check Serper
    if os.getenv("SERPER_API_KEY") and LANGCHAIN_IMPORTS_AVAILABLE:
        available_tools.append(search_web_serper)
        status['serper'] = True
        print("✅ Serper web search tool available")
    else:
        status['serper'] = False
        if not os.getenv("SERPER_API_KEY"):
            print("❌ Serper tool unavailable (SERPER_API_KEY not set)")
        else:
            print("❌ Serper tool unavailable (missing langchain-community)")
    
    # Check ArXiv
    arxiv_available = False
    try:
        import arxiv
        import fitz
        if LANGCHAIN_IMPORTS_AVAILABLE:
            available_tools.append(search_arxiv)
            arxiv_available = True
            print("✅ ArXiv tool available")
    except ImportError:
        print("❌ ArXiv tool unavailable (missing arxiv or pymupdf)")
    
    status['arxiv'] = arxiv_available
    
    return available_tools, status

# Initialize tools
ALL_LANGCHAIN_TOOLS, TOOL_STATUS = check_tool_availability()

print(f"🎯 SmolagAgents research tools loaded: {len(ALL_LANGCHAIN_TOOLS)} total")

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def get_langchain_tools():
    """Get all available research tools"""
    return ALL_LANGCHAIN_TOOLS

def get_tool_status():
    """Get detailed tool status"""
    return {
        'total_tools': len(ALL_LANGCHAIN_TOOLS),
        'research_tools_available': len(ALL_LANGCHAIN_TOOLS) > 1,
        'tool_details': TOOL_STATUS
    }

# For backwards compatibility
get_all_langchain_tools = get_langchain_tools

# Export everything
__all__ = [
    'search_web_serper',
    'search_wikipedia', 
    'search_arxiv',
    'ALL_LANGCHAIN_TOOLS',
    'get_langchain_tools',
    'get_tool_status'
]

# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_tools():
    """Test all available tools"""
    print("🧪 TESTING RESEARCH TOOLS")
    print("=" * 25)
    
    status = get_tool_status()
    print(f"Status: {status}")
    
    # Test environment
    print(f"\nEnvironment:")
    print(f"  SERPER_API_KEY: {'✅ Set' if os.getenv('SERPER_API_KEY') else '❌ Not set'}")
    print(f"  LangChain imports: {'✅ Available' if LANGCHAIN_IMPORTS_AVAILABLE else '❌ Missing'}")
    
    # Test tool access
    print(f"\nAvailable tools:")
    for i, tool in enumerate(ALL_LANGCHAIN_TOOLS):
        print(f"  {i+1}. {tool.name} - {tool.description[:50]}...")
    
    return len(ALL_LANGCHAIN_TOOLS) > 1

if __name__ == "__main__":
    test_tools()