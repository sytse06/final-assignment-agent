# tools/__init__.py
# Tool collection for GAIA agent system

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

# Export available tools
__all__ = []

if GetAttachmentTool:
    __all__.append('GetAttachmentTool')

if ContentRetrieverTool:
    __all__.append('ContentRetrieverTool')

# Convenience function to get all available tools
def get_available_gaia_tools():
    """Returns list of available GAIA-specific tools"""
    tools = []
    
    try:
        if GetAttachmentTool:
            tools.append(GetAttachmentTool())
            
        if ContentRetrieverTool:
            tools.append(ContentRetrieverTool())
    except Exception as e:
        print(f"⚠️  Error creating tool instances: {e}")
    
    return tools

# Tool status for debugging
def get_tool_status():
    """Returns status of tool availability"""
    try:
        available_tools = get_available_gaia_tools()
        return {
            'GetAttachmentTool': GetAttachmentTool is not None,
            'ContentRetrieverTool': ContentRetrieverTool is not None,
            'total_available': len(available_tools)
        }
    except Exception as e:
        print(f"⚠️  Error getting tool status: {e}")
        return {
            'GetAttachmentTool': False,
            'ContentRetrieverTool': False,
            'total_available': 0
        }

# Only print status if tools are being imported (not just for status check)
if __name__ != "__main__":
    try:
        status = get_tool_status()
        print(f"🔧 GAIA Tools Status: {status}")
    except Exception as e:
        print(f"⚠️  Could not check tool status: {e}")