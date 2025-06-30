# tools/__init__.py
# Import custom tools
try:
    from .get_attachment_tool import GetAttachmentTool
    from .content_retriever_tool import ContentRetrieverTool
    from .youtube_content_tool import YouTubeContentTool  # 🔥 NEW
    print("✅ Custom tools loaded in __init__.py")
except ImportError as e:
    print(f"⚠️ Custom tools failed to load in __init__.py: {e}")
    GetAttachmentTool = None
    ContentRetrieverTool = None
    YouTubeContentTool = None  # 🔥 NEW

# Import ContentGroundingTool for web_researcher
try:
    from .content_grounding_tool import ContentGroundingTool
    CONTENT_GROUNDING_AVAILABLE = True
    print("✅ ContentGroundingTool loaded in __init__.py")
except ImportError as e:
    print(f"⚠️ ContentGroundingTool failed to load in __init__.py: {e}")
    ContentGroundingTool = None
    CONTENT_GROUNDING_AVAILABLE = False

# Import LangChain tools - ONLY what actually exists
try:
    from .langchain_tools import ALL_LANGCHAIN_TOOLS, get_langchain_tools
    LANGCHAIN_TOOLS_AVAILABLE = len(ALL_LANGCHAIN_TOOLS) > 1  # More than just final_answer
    print(f"✅ LangChain tools re-exported in __init__.py: {len(ALL_LANGCHAIN_TOOLS)} tools")
except ImportError as e:
    print(f"⚠️ LangChain tools failed to load in __init__.py: {e}")
    ALL_LANGCHAIN_TOOLS = []
    get_langchain_tools = lambda: []
    LANGCHAIN_TOOLS_AVAILABLE = False

# Define what gets exported - ONLY what actually exists
__all__ = [
    # Custom tools
    'GetAttachmentTool',
    'ContentRetrieverTool',
    'YouTubeContentTool',  # 🔥 NEW
    'ContentGroundingTool',
    
    # LangChain tools - only the list, not individual tools
    'ALL_LANGCHAIN_TOOLS',
    'get_langchain_tools',
    'LANGCHAIN_TOOLS_AVAILABLE',
    'CONTENT_GROUNDING_AVAILABLE'
]

def get_tool_status():
    """Get tool availability status"""
    return {
        'GetAttachmentTool': GetAttachmentTool is not None,
        'ContentRetrieverTool': ContentRetrieverTool is not None,
        'YouTubeContentTool': YouTubeContentTool is not None,  # 🔥 NEW
        'ContentGroundingTool': ContentGroundingTool is not None,
        'research_tools': LANGCHAIN_TOOLS_AVAILABLE,
        'total_core_tools': sum([
            GetAttachmentTool is not None,
            ContentRetrieverTool is not None,
            YouTubeContentTool is not None,  # 🔥 NEW
            ContentGroundingTool is not None
        ]),
        'total_research_tools': len(ALL_LANGCHAIN_TOOLS) - 1 if LANGCHAIN_TOOLS_AVAILABLE else 0  # Exclude final_answer
    }

# Print status on import
print(f"🔧 GAIA Tools Status: {get_tool_status()}")

def get_all_tools():
    """Get all available tools (custom + langchain)"""
    tools = []
    
    # Add custom tools
    if GetAttachmentTool:
        tools.append(GetAttachmentTool())
    if ContentRetrieverTool:
        tools.append(ContentRetrieverTool())
    if YouTubeContentTool:  # 🔥 NEW
        tools.append(YouTubeContentTool())
    if ContentGroundingTool:
        tools.append(ContentGroundingTool())
    
    # Add LangChain tools
    tools.extend(ALL_LANGCHAIN_TOOLS)
    
    return tools

def get_web_researcher_tools():
    """Get tools specifically for web_researcher agent"""
    tools = []
    
    # Add LangChain research tools (search capabilities)
    tools.extend(ALL_LANGCHAIN_TOOLS)
    
    # Add content processing tools
    if ContentRetrieverTool:
        tools.append(ContentRetrieverTool())
    
    # 🔥 NEW: Add YouTube content extraction for web research
    if YouTubeContentTool:
        tools.append(YouTubeContentTool())
    
    # Add content grounding for verification
    if ContentGroundingTool:
        tools.append(ContentGroundingTool())
    
    # Add file access for document research
    if GetAttachmentTool:
        tools.append(GetAttachmentTool())
    
    return tools

def get_document_processor_tools():
    """Get tools specifically for document_processor agent"""  # 🔥 NEW FUNCTION
    tools = []
    
    # Add document processing tools
    if ContentRetrieverTool:
        tools.append(ContentRetrieverTool())
    
    # Add YouTube content extraction for video analysis
    if YouTubeContentTool:
        tools.append(YouTubeContentTool())
    
    # Add file access for documents
    if GetAttachmentTool:
        tools.append(GetAttachmentTool())
    
    return tools

print(f"🔧 Tools package initialized successfully")