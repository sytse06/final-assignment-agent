# tools/langchain_tools.py
# Simple LangChain tools using load_tools() approach

import os
from smolagents import Tool, tool

# LangChain imports
try:
    from langchain.agents import load_tools
    from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
    from langchain_community.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper
    from langchain_community.tools import WikipediaQueryRun
    from langchain_core.tools import Tool as LangChainTool
    LANGCHAIN_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LangChain imports failed: {e}")
    LANGCHAIN_IMPORTS_AVAILABLE = False

# ============================================================================
# SIMPLE LANGCHAIN TOOLS USING load_tools()
# ============================================================================

def create_simple_langchain_tools():
    """Create LangChain tools using the simple load_tools() approach"""
    tools = []
    
    if not LANGCHAIN_IMPORTS_AVAILABLE:
        print("❌ LangChain imports not available - skipping tool creation")
        return tools
    
    try:
        # 1. Wikipedia tool (no API key required)
        try:
            wikipedia_tools = load_tools(["wikipedia"])
            if wikipedia_tools:
                wikipedia_tool = Tool.from_langchain(wikipedia_tools[0])
                tools.append(wikipedia_tool)
                print("✅ Wikipedia tool loaded successfully")
        except Exception as e:
            print(f"⚠️ Wikipedia tool failed: {e}")
        
        # 2. SerpAPI tool (requires SERPAPI_API_KEY or SERPER_API_KEY)
        serpapi_key = os.getenv("SERPERAPI_API_KEY") or os.getenv("SERP_API_KEY")
        if serpapi_key:
            try:
                serpapi_tools = load_tools(["serpapi"])
                if serpapi_tools:
                    search_tool = Tool.from_langchain(serpapi_tools[0])
                    tools.append(search_tool)
                    print("✅ SerpAPI search tool loaded successfully")
            except Exception as e:
                print(f"⚠️ SerpAPI tool failed: {e}")
        else:
            print("⚠️ SERPAPI_API_KEY or SERPER_API_KEY not set, skipping web search tool")
        
        # 3. ArXiv tool (requires arxiv + pymupdf packages)
        try:
            # Check for required dependencies
            import arxiv
            import fitz  # PyMuPDF
            
            arxiv_tools = load_tools(["arxiv"])
            if arxiv_tools:
                arxiv_tool = Tool.from_langchain(arxiv_tools[0])
                tools.append(arxiv_tool)
                print("✅ ArXiv tool loaded successfully")
        except ImportError as e:
            missing_deps = []
            try:
                import arxiv
            except ImportError:
                missing_deps.append("arxiv")
            
            try:
                import fitz
            except ImportError:
                missing_deps.append("pymupdf")
            
            print(f"⚠️ ArXiv tool failed - missing dependencies: {', '.join(missing_deps)}")
            print(f"   Install with: pip install arxiv pymupdf langchain-community")
        except Exception as e:
            print(f"⚠️ ArXiv tool failed: {e}")
            
    except ImportError as e:
        print(f"❌ langchain.agents not available: {e}")
        print("   Install with: pip install langchain")
    
    return tools

# ============================================================================
# NATIVE SMOLAGENTS TOOL
# ============================================================================

@tool
def final_answer(answer: str) -> str:
    """Provide the final answer to the GAIA question.
    
    Use this tool when you have gathered all necessary information and are ready 
    to provide the definitive answer to the question.
    
    Args:
        answer: The final answer to the GAIA question
    """
    return f"FINAL ANSWER: {answer}"

# ============================================================================
# INITIALIZE AND EXPORT
# ============================================================================

# Create tools on import
ALL_LANGCHAIN_TOOLS = []

try:
    langchain_tools = create_simple_langchain_tools()
    ALL_LANGCHAIN_TOOLS = langchain_tools + [final_answer]
    print(f"🎯 Simple LangChain tools loaded: {len(ALL_LANGCHAIN_TOOLS)} total")
except Exception as e:
    print(f"❌ Failed to load LangChain tools: {e}")
    ALL_LANGCHAIN_TOOLS = [final_answer]
    print("🔄 Fallback: Using only final_answer tool")

# For backwards compatibility
def get_langchain_tools():
    """Get all available LangChain tools"""
    return ALL_LANGCHAIN_TOOLS

# Export the list
__all__ = ['ALL_LANGCHAIN_TOOLS', 'get_langchain_tools', 'final_answer']

# ============================================================================
# ALTERNATIVE: Manual tool creation for specific providers
# ============================================================================

def create_serper_tool():
    """Create Serper API tool if available"""
    if not LANGCHAIN_IMPORTS_AVAILABLE:
        return None
        
    serper_key = os.getenv("SERPER_API_KEY") or os.getenv("SERPAPI_API_KEY")
    if not serper_key:
        return None
        
    try:
        search = GoogleSerperAPIWrapper(serper_api_key=serper_key)
        
        langchain_tool = LangChainTool(
            name="search_serper",
            description="Search the web for current information",
            func=search.run
        )
        
        return Tool.from_langchain(langchain_tool)
    except Exception as e:
        print(f"⚠️ Serper tool creation failed: {e}")
        return None

def create_manual_wikipedia_tool():
    """Create Wikipedia tool manually if load_tools fails"""
    if not LANGCHAIN_IMPORTS_AVAILABLE:
        return None
        
    try:
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return Tool.from_langchain(wikipedia)
    except Exception as e:
        print(f"⚠️ Manual Wikipedia tool creation failed: {e}")
        return None

# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_simple_approach():
    """Test the simple load_tools approach"""
    print("🧪 Testing simple LangChain tools approach")
    print("=" * 45)
    
    # Test environment
    print("Environment check:")
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    serper_key = os.getenv("SERPER_API_KEY") 
    print(f"  SERPAPI_API_KEY: {'✅ Set' if serpapi_key else '❌ Not set'}")
    print(f"  SERPER_API_KEY: {'✅ Set' if serper_key else '❌ Not set'}")
    
    if serpapi_key or serper_key:
        print(f"  🔑 Using: {'SERPAPI_API_KEY' if serpapi_key else 'SERPER_API_KEY'}")
    else:
        print(f"  ⚠️ No web search API key available")
        print(f"     Get free key at: https://serpapi.com/")
    
    # Test tool creation
    tools = create_simple_langchain_tools()
    print(f"\nTools created: {len(tools)}")
    
    for i, tool in enumerate(tools):
        print(f"  {i+1}. {tool.name} - {tool.description[:60]}...")
    
    # Test imports
    print(f"\nALL_LANGCHAIN_TOOLS: {len(ALL_LANGCHAIN_TOOLS)} tools")
    
    return len(tools) > 0

if __name__ == "__main__":
    test_simple_approach()