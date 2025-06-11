# tools/__init__.py
# Simple tool collection for GAIA agent system

# Core GAIA tools
try:
    from .get_attachment_tool import GetAttachmentTool
    print("✅ GetAttachmentTool loaded successfully")
except ImportError as e:
    print(f"⚠️  GetAttachmentTool import failed: {e}")
    GetAttachmentTool = None

try:
    from .content_retriever_tool import ContentRetrieverTool
    print("✅ ContentRetrieverTool loaded successfully")
except ImportError as e:
    print(f"⚠️  ContentRetrieverTool import failed: {e}")
    ContentRetrieverTool = None

# Research tools (optional)
try:
    from .langchain_tools import (
        search_wikipedia, 
        search_arxiv,
        search_web_serper,
        final_answer,
        ALL_LANGCHAIN_TOOLS
    )
    print("✅ Langchain research tools (Serper/Wikipedia/ArXiv) loaded successfully")
    LANGCHAIN_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Langchain research tools not available: {e}")
    search_wikipedia = None
    search_arxiv = None
    search_web_serper = None
    final_answer= None
    ALL_LANGCHAIN_TOOLS = []
    LANGCHAIN_TOOLS_AVAILABLE = False

# Export available tools
__all__ = []

if GetAttachmentTool:
    __all__.append('GetAttachmentTool')

if ContentRetrieverTool:
    __all__.append('ContentRetrieverTool')

if LANGCHAIN_TOOLS_AVAILABLE:
    __all__.extend(['search_wikipedia', 'search_arxiv', 'search_web_serper', 'ALL_LANGCHAIN_TOOLS'])

# Tool status for debugging
def get_tool_status():
    """Returns status of tool availability"""
    return {
        'GetAttachmentTool': GetAttachmentTool is not None,
        'ContentRetrieverTool': ContentRetrieverTool is not None,
        'research_tools': LANGCHAIN_TOOLS_AVAILABLE,
        'total_core_tools': sum([GetAttachmentTool is not None, ContentRetrieverTool is not None]),
        'total_research_tools': 4 if LANGCHAIN_TOOLS_AVAILABLE else 0
    }

print(f"🔧 GAIA Tools Status: {get_tool_status()}")