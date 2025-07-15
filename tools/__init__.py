# tools/__init__.py
# Cleaned up tools initialization for GAIA Agent with enhanced diagnostics

import os
import smolagents
from pathlib import Path

# Import core custom tools (keeping only what's relevant for GAIA)
try:
    from .content_retriever_tool import ContentRetrieverTool
    print("✅ ContentRetrieverTool loaded")
except ImportError as e:
    print(f"⚠️ ContentRetrieverTool failed to load: {e}")
    ContentRetrieverTool = None
# Import SpeechToTextTool with proper error handling
try:
    from smolagents import SpeechToTextTool
    SPEECH_TO_TEXT_AVAILABLE = True
    print("✅ SpeechToTextTool loaded")
except ImportError as e:
    print(f"⚠️ SpeechToTextTool failed to load: {e}")
    SpeechToTextTool = None
    SPEECH_TO_TEXT_AVAILABLE = False

# Import YouTube tool if available
try:
    from .youtube_content_tool import YouTubeContentTool
    YOUTUBE_TOOL_AVAILABLE = True
    print("✅ YouTubeContentTool loaded")
except ImportError as e:
    YouTubeContentTool = None
    YOUTUBE_TOOL_AVAILABLE = False
    
    # More specific error messages based on the actual error
    error_msg = str(e).lower()
    if "no module named 'tools.youtube_content_tool'" in error_msg:
        print("⚠️ YouTubeContentTool not found: tools/youtube_content_tool.py missing")
        print("💡 Create tools/youtube_content_tool.py file")
    elif "yt_dlp" in error_msg or "yt-dlp" in error_msg:
        print(f"⚠️ YouTubeContentTool dependency issue: {e}")
        print("💡 Install: pip install yt-dlp")
    elif "requests" in error_msg:
        print(f"⚠️ YouTubeContentTool dependency issue: {e}")
        print("💡 Install: pip install requests")
    else:
        print(f"⚠️ YouTubeContentTool failed to load: {e}")
        print("💡 Check tools/youtube_content_tool.py for syntax errors")

# Import VisionWebBrowserTool with enhanced error detection
try:
    from .vision_browser_tool import VisionWebBrowserTool
    VISION_BROWSER_AVAILABLE = True
    print("✅ VisionWebBrowserTool loaded")
except ImportError as e:
    VisionWebBrowserTool = None
    VISION_BROWSER_AVAILABLE = False
    
    # More specific error messages based on the actual error
    error_msg = str(e).lower()
    if "no module named 'tools.vision_browser_tool'" in error_msg:
        print("⚠️ VisionWebBrowserTool not found: tools/vision_browser_tool.py missing")
        print("💡 Create tools/vision_browser_tool.py file")
    elif "helium" in error_msg:
        print(f"⚠️ VisionWebBrowserTool dependency issue: {e}")
        print("💡 Install: pip install helium selenium")
    elif "selenium" in error_msg:
        print(f"⚠️ VisionWebBrowserTool dependency issue: {e}")
        print("💡 Install: pip install selenium")
    else:
        print(f"⚠️ VisionWebBrowserTool failed to load: {e}")
        print("💡 Check tools/vision_browser_tool.py for syntax errors")

# Import LangChain research tools
try:
    from .langchain_tools import ALL_LANGCHAIN_TOOLS, get_langchain_tools, get_tool_status as get_langchain_status
    LANGCHAIN_TOOLS_AVAILABLE = len(ALL_LANGCHAIN_TOOLS) > 1  # More than just final_answer
    print(f"✅ LangChain research tools loaded: {len(ALL_LANGCHAIN_TOOLS)} tools")
except ImportError as e:
    print(f"⚠️ LangChain tools failed to load: {e}")
    ALL_LANGCHAIN_TOOLS = []
    get_langchain_tools = lambda: []
    get_langchain_status = lambda: {'research_tools_available': False}
    LANGCHAIN_TOOLS_AVAILABLE = False

# Define what gets exported
__all__ = [
    # Core GAIA tools
    'ContentRetrieverTool',
    'YouTubeContentTool', 
    'VisionWebBrowserTool',
    'SpeechToTextTool',
    
    # Availability flags
    'YOUTUBE_TOOL_AVAILABLE',
    'VISION_BROWSER_AVAILABLE',
    'SPEECH_TO_TEXT_AVAILABLE',
    'LANGCHAIN_TOOLS_AVAILABLE',
    
    # LangChain research tools
    'ALL_LANGCHAIN_TOOLS',
    'get_langchain_tools',
    
    # Utility functions
    'get_tool_status',
    'get_content_processor_tools',
    'get_web_researcher_tools',
    
    # Diagnostic functions
    'diagnose_vision_tool',
    'diagnose_youtube_tool',
    'validate_tool_dependencies',
    'check_vision_status',
    'check_youtube_status'
]

def get_tool_status():
    """Get comprehensive tool availability status for GAIA Agent"""
    # Get LangChain tool status
    langchain_status = get_langchain_status() if LANGCHAIN_TOOLS_AVAILABLE else {'research_tools_available': False}
    
    return {
        # Core tools
        'ContentRetrieverTool': ContentRetrieverTool is not None,
        'YouTubeContentTool': YouTubeContentTool is not None,
        'VisionWebBrowserTool': VisionWebBrowserTool is not None,
        'SpeechToTextTool': SPEECH_TO_TEXT_AVAILABLE,
        
        # Research capabilities
        'research_tools_available': langchain_status.get('research_tools_available', False),
        'langchain_tools_count': len(ALL_LANGCHAIN_TOOLS) if LANGCHAIN_TOOLS_AVAILABLE else 0,
        
        # Summary
        'total_core_tools': sum([
            ContentRetrieverTool is not None,
            YouTubeContentTool is not None,
            VisionWebBrowserTool is not None,
            SPEECH_TO_TEXT_AVAILABLE
        ]),
        'total_research_tools': len(ALL_LANGCHAIN_TOOLS) if LANGCHAIN_TOOLS_AVAILABLE else 0,
        
        # Capability assessment
        'content_processing_capable': ContentRetrieverTool is not None,
        'web_navigation_capable': VisionWebBrowserTool is not None,
        'multimedia_capable': YouTubeContentTool is not None,
        'audio_processing_capable': SPEECH_TO_TEXT_AVAILABLE,
        'research_capable': langchain_status.get('research_tools_available', False)
    }

def diagnose_vision_tool():
    """Comprehensive diagnosis of VisionWebBrowserTool status"""
    print("🔍 Diagnosing VisionWebBrowserTool...")
    
    # Check file existence
    current_dir = Path(__file__).parent
    vision_tool_path = current_dir / "vision_browser_tool.py"
    
    print(f"📁 Looking for: {vision_tool_path}")
    print(f"📁 Absolute path: {vision_tool_path.absolute()}")
    
    if vision_tool_path.exists():
        print("✅ vision_browser_tool.py file exists")
        
        # Check file size and basic content
        file_size = vision_tool_path.stat().st_size
        print(f"📄 File size: {file_size} bytes")
        
        if file_size == 0:
            print("❌ File is empty!")
            return False
        
        # Try to read the file and check for basic class definition
        try:
            with open(vision_tool_path, 'r') as f:
                content = f.read()
                if 'class VisionWebBrowserTool' in content:
                    print("✅ VisionWebBrowserTool class found in file")
                else:
                    print("❌ VisionWebBrowserTool class not found in file")
                    print("💡 File content preview:")
                    print(content[:200] + "..." if len(content) > 200 else content)
                    return False
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return False
        
        # Check dependencies
        try:
            import helium
            print("✅ helium dependency available")
        except ImportError:
            print("❌ helium dependency missing")
            return False
        
        try:
            import selenium
            print("✅ selenium dependency available")
        except ImportError:
            print("❌ selenium dependency missing")
            return False
        
        # Try importing the tool
        try:
            from .vision_browser_tool import VisionWebBrowserTool
            print("✅ VisionWebBrowserTool import successful")
            
            # Try instantiating
            tool = VisionWebBrowserTool()
            print("✅ VisionWebBrowserTool instantiation successful")
            return True
            
        except Exception as e:
            print(f"❌ VisionWebBrowserTool import/instantiation failed: {e}")
            print(f"💡 Error details: {type(e).__name__}: {e}")
            return False
    else:
        print("❌ vision_browser_tool.py file does not exist")
        print(f"💡 Create the file at: {vision_tool_path}")
        
        # Show what files do exist in the tools directory
        tools_files = list(current_dir.glob("*.py"))
        print(f"📁 Files in tools directory: {[f.name for f in tools_files]}")
        return False

def diagnose_youtube_tool():
    """Comprehensive diagnosis of YouTubeContentTool status"""
    print("🎥 Diagnosing YouTubeContentTool...")
    
    # Check file existence
    current_dir = Path(__file__).parent
    youtube_tool_path = current_dir / "youtube_content_tool.py"
    
    print(f"📁 Looking for: {youtube_tool_path}")
    print(f"📁 Absolute path: {youtube_tool_path.absolute()}")
    
    if youtube_tool_path.exists():
        print("✅ youtube_content_tool.py file exists")
        
        # Check file size and basic content
        file_size = youtube_tool_path.stat().st_size
        print(f"📄 File size: {file_size} bytes")
        
        if file_size == 0:
            print("❌ File is empty!")
            return False
        
        # Try to read the file and check for basic class definition
        try:
            with open(youtube_tool_path, 'r') as f:
                content = f.read()
                if 'class YouTubeContentTool' in content:
                    print("✅ YouTubeContentTool class found in file")
                else:
                    print("❌ YouTubeContentTool class not found in file")
                    print("💡 File content preview:")
                    print(content[:200] + "..." if len(content) > 200 else content)
                    return False
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return False
        
        # Check dependencies
        try:
            import yt_dlp
            print("✅ yt-dlp dependency available")
        except ImportError:
            print("❌ yt-dlp dependency missing")
            print("💡 Install: pip install yt-dlp")
            return False
        
        try:
            import requests
            print("✅ requests dependency available")
        except ImportError:
            print("❌ requests dependency missing")
            print("💡 Install: pip install requests")
            return False
        
        # Try importing the tool
        try:
            from .youtube_content_tool import YouTubeContentTool
            print("✅ YouTubeContentTool import successful")
            
            # Try instantiating
            tool = YouTubeContentTool()
            print("✅ YouTubeContentTool instantiation successful")
            return True
            
        except Exception as e:
            print(f"❌ YouTubeContentTool import/instantiation failed: {e}")
            print(f"💡 Error details: {type(e).__name__}: {e}")
            return False
    else:
        print("❌ youtube_content_tool.py file does not exist")
        print(f"💡 Create the file at: {youtube_tool_path}")
        
        # Show what files do exist in the tools directory
        tools_files = list(current_dir.glob("*.py"))
        print(f"📁 Files in tools directory: {[f.name for f in tools_files]}")
        return False

def validate_tool_dependencies():
    """Enhanced validation with specific error detection for each tool"""
    issues = []
    recommendations = []
    
    # Check helium and selenium availability
    helium_available = False
    selenium_available = False
    
    try:
        import helium
        helium_available = True
        print("✅ helium package available")
    except ImportError:
        print("❌ helium package not available")
    
    try:
        import selenium
        selenium_available = True
        print("✅ selenium package available")
    except ImportError:
        print("❌ selenium package not available")
    
    # Check YouTube tool dependencies
    yt_dlp_available = False
    requests_available = False
    
    try:
        import yt_dlp
        yt_dlp_available = True
        print("✅ yt-dlp package available")
    except ImportError:
        print("❌ yt-dlp package not available")
    
    try:
        import requests
        requests_available = True
        print("✅ requests package available")
    except ImportError:
        print("❌ requests package not available")
    
    # Check VisionWebBrowserTool with specific diagnosis
    if VisionWebBrowserTool is None:
        if not helium_available or not selenium_available:
            issues.append("VisionWebBrowserTool dependencies missing")
            missing_deps = []
            if not helium_available:
                missing_deps.append("helium")
            if not selenium_available:
                missing_deps.append("selenium")
            recommendations.append(f"Install missing dependencies: pip install {' '.join(missing_deps)}")
        else:
            # Dependencies are available but tool still failed to load
            issues.append("VisionWebBrowserTool file missing or has errors")
            recommendations.append("Check if tools/vision_browser_tool.py exists and has no syntax errors")
    
    # Check YouTubeContentTool with specific diagnosis
    if YouTubeContentTool is None:
        if not yt_dlp_available or not requests_available:
            issues.append("YouTubeContentTool dependencies missing")
            missing_deps = []
            if not yt_dlp_available:
                missing_deps.append("yt-dlp")
            if not requests_available:
                missing_deps.append("requests")
            recommendations.append(f"Install missing dependencies: pip install {' '.join(missing_deps)}")
        else:
            # Dependencies are available but tool still failed to load
            issues.append("YouTubeContentTool file missing or has errors")
            recommendations.append("Check if tools/youtube_content_tool.py exists and has no syntax errors")
    
    # Enhanced DuckDuckGo check
    ddgs_available = False
    old_package_available = False
    
    try:
        import ddgs
        ddgs_available = True
        print("✅ ddgs package available")
    except ImportError:
        pass
    
    try:
        import duckduckgo_search
        old_package_available = True
        if not ddgs_available:
            print("⚠️ Using deprecated duckduckgo_search package")
    except ImportError:
        pass
    
    if not ddgs_available and not old_package_available:
        issues.append("No DuckDuckGo search package available")
        recommendations.append("Run: pip install ddgs")
    elif old_package_available and not ddgs_available:
        issues.append("Using deprecated 'duckduckgo-search' package")
        recommendations.append("Run: pip uninstall duckduckgo-search && pip install ddgs")
    
    # Check for content processing dependencies
    if ContentRetrieverTool is None:
        issues.append("ContentRetrieverTool not available")
        recommendations.append("Check tools/content_retriever_tool.py exists and dependencies")
    
    return {
        'issues': issues,
        'recommendations': recommendations,
        'status': 'healthy' if not issues else 'needs_attention',
        'dependency_details': {
            'helium_available': helium_available,
            'selenium_available': selenium_available,
            'yt_dlp_available': yt_dlp_available,
            'requests_available': requests_available,
            'ddgs_available': ddgs_available,
            'vision_tool_available': VisionWebBrowserTool is not None,
            'youtube_tool_available': YouTubeContentTool is not None
        }
    }

def check_vision_status():
    """Quick status check for vision capabilities"""
    print("🔍 Vision Tool Status Check:")
    success = diagnose_vision_tool()
    
    print(f"\n🔍 Dependency Status:")
    deps = validate_tool_dependencies()
    print(f"Overall Status: {deps['status']}")
    
    if deps['issues']:
        print("\n❌ Issues found:")
        for issue in deps['issues']:
            print(f"   - {issue}")
    
    if deps['recommendations']:
        print("\n💡 Recommendations:")
        for rec in deps['recommendations']:
            print(f"   - {rec}")
    
    print(f"\n📊 Dependency Details:")
    for key, value in deps['dependency_details'].items():
        status = "✅" if value else "❌"
        print(f"   {key}: {status}")
    
    return success and deps['status'] == 'healthy'

def check_youtube_status():
    """Quick status check for YouTube capabilities"""
    print("🎥 YouTube Tool Status Check:")
    success = diagnose_youtube_tool()
    
    print(f"\n🔍 Dependency Status:")
    deps = validate_tool_dependencies()
    print(f"Overall Status: {deps['status']}")
    
    if deps['issues']:
        print("\n❌ Issues found:")
        for issue in deps['issues']:
            print(f"   - {issue}")
    
    if deps['recommendations']:
        print("\n💡 Recommendations:")
        for rec in deps['recommendations']:
            print(f"   - {rec}")
    
    print(f"\n📊 YouTube-specific Details:")
    youtube_deps = {k: v for k, v in deps['dependency_details'].items() 
                   if 'youtube' in k.lower() or k in ['yt_dlp_available', 'requests_available']}
    for key, value in youtube_deps.items():
        status = "✅" if value else "❌"
        print(f"   {key}: {status}")
    
    return success and deps['status'] == 'healthy'

def get_content_processor_tools():
    """Get tools specifically for content_processor specialist"""
    tools = []
    
    # Core content processing
    if ContentRetrieverTool:
        tools.append(ContentRetrieverTool())
        print("✓ Added ContentRetrieverTool to content_processor")

    # Speech_to_text processing
    if SpeechToTextTool:
        tools.append(SpeechToTextTool())
        print("✓ Added SpeechToTextTool to content_processor")        
    
    # Multimedia content processing (NEW: YouTube support)
    if YouTubeContentTool:
        tools.append(YouTubeContentTool())
        print("✓ Added YouTubeContentTool to content_processor")
    
    print(f"📦 Content processor tools: {len(tools)} available")
    return tools

def get_web_researcher_tools():
    """Get tools specifically for web_researcher specialist - FOCUSED ON SEARCH & DISCOVERY"""
    tools = []
    
    # Add LangChain research tools (PRIMARY for web research)
    if LANGCHAIN_TOOLS_AVAILABLE:
        langchain_tools = get_langchain_tools()
        tools.extend(langchain_tools)
        print(f"✓ Added {len(langchain_tools)} LangChain research tools to web_researcher (PRIMARY)")
    
    # Always add VisitWebpageTool (essential for web research)
    try:
        from smolagents import VisitWebpageTool
        tools.append(VisitWebpageTool())
        print("✓ Added VisitWebpageTool to web_researcher (essential)")
    except Exception as e:
        print(f"⚠️ Failed to add VisitWebpageTool: {e}")
    
    print(f"🔍 Web researcher tools: {len(tools)} available (focused on search & discovery)")
    return tools

# Print initialization status
print(f"🔧 GAIA Tools Status: {get_tool_status()}")

# Validate dependencies and show recommendations
dependency_status = validate_tool_dependencies()
if dependency_status['issues']:
    print("⚠️ Dependency issues detected:")
    for issue in dependency_status['issues']:
        print(f"   - {issue}")
    print("💡 Recommendations:")
    for rec in dependency_status['recommendations']:
        print(f"   - {rec}")
else:
    print("✅ All tool dependencies validated successfully")

print("🔧 Tools package initialized for GAIA Agent")