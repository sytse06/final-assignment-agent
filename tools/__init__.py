# Import custom tools
try:
    from .get_attachment_tool import GetAttachmentTool
    from .content_retriever_tool import ContentRetrieverTool
    print("✅ Custom tools loaded in __init__.py")
except ImportError as e:
    print(f"⚠️ Custom tools failed to load in __init__.py: {e}")
    GetAttachmentTool = None
    ContentRetrieverTool = None

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
    
    # LangChain tools - only the list, not individual tools
    'ALL_LANGCHAIN_TOOLS',
    'get_langchain_tools',
    'LANGCHAIN_TOOLS_AVAILABLE'
]

def get_tool_status():
    """Get tool availability status"""
    return {
        'GetAttachmentTool': GetAttachmentTool is not None,
        'ContentRetrieverTool': ContentRetrieverTool is not None,
        'research_tools': LANGCHAIN_TOOLS_AVAILABLE,
        'total_core_tools': sum([
            GetAttachmentTool is not None,
            ContentRetrieverTool is not None
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
    
    # Add LangChain tools
    tools.extend(ALL_LANGCHAIN_TOOLS)
    
    return tools

print(f"🔧 Tools package initialized successfully")