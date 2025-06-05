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
    from .langchain_tools import search_wikipedia, search_arxiv
    print("✅ Research tools (Wikipedia/ArXiv) loaded successfully")
    RESEARCH_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Research tools not available: {e}")
    search_wikipedia = None
    search_arxiv = None
    RESEARCH_TOOLS_AVAILABLE = False

# Export available tools
__all__ = []

if GetAttachmentTool:
    __all__.append('GetAttachmentTool')

if ContentRetrieverTool:
    __all__.append('ContentRetrieverTool')

if RESEARCH_TOOLS_AVAILABLE:
    __all__.extend(['search_wikipedia', 'search_arxiv'])

# Tool status for debugging
def get_tool_status():
    """Returns status of tool availability"""
    return {
        'GetAttachmentTool': GetAttachmentTool is not None,
        'ContentRetrieverTool': ContentRetrieverTool is not None,
        'research_tools': RESEARCH_TOOLS_AVAILABLE,
        'total_core_tools': sum([GetAttachmentTool is not None, ContentRetrieverTool is not None]),
        'total_research_tools': 2 if RESEARCH_TOOLS_AVAILABLE else 0
    }

print(f"🔧 GAIA Tools Status: {get_tool_status()}")