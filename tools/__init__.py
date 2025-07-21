# tools/__init__.py
# Updated tools initialization for GAIA Agent with BrowserProfileTool and smolagents integration

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

# Import BrowserProfileTool for authenticated browser automation
try:
    from .BrowserProfileTool import BrowserProfileTool, get_authenticated_browser_instructions, HELIUM_AVAILABLE
    BROWSER_PROFILE_AVAILABLE = True
    print("✅ BrowserProfileTool loaded")
except ImportError as e:
    BrowserProfileTool = None
    get_authenticated_browser_instructions = None
    BROWSER_PROFILE_AVAILABLE = False
    HELIUM_AVAILABLE = False
    
    # Specific error messages
    error_msg = str(e).lower()
    if "no module named 'tools.browserprofiletool'" in error_msg:
        print("⚠️ BrowserProfileTool not found: tools/BrowserProfileTool.py missing")
        print("💡 Create tools/BrowserProfileTool.py file")
    elif "undetected_chromedriver" in error_msg:
        print(f"⚠️ BrowserProfileTool dependency issue: {e}")
        print("💡 Install: pip install undetected-chromedriver selenium")
    elif "selenium" in error_msg:
        print(f"⚠️ BrowserProfileTool dependency issue: {e}")
        print("💡 Install: pip install selenium")
    else:
        print(f"⚠️ BrowserProfileTool failed to load: {e}")
        print("💡 Check tools/BrowserProfileTool.py for syntax errors")

# Import smolagents vision browser components
try:
    from smolagents.vision_web_browser import (
        go_back, close_popups, search_item_ctrl_f, 
        save_screenshot, helium_instructions
    )
    SMOLAGENTS_VISION_AVAILABLE = True
    print("✅ smolagents vision browser components loaded")
except ImportError as e:
    go_back = None
    close_popups = None
    search_item_ctrl_f = None
    save_screenshot = None
    helium_instructions = None
    SMOLAGENTS_VISION_AVAILABLE = False
    print(f"⚠️ smolagents vision browser not available: {e}")
    print("💡 Update smolagents: pip install --upgrade smolagents")

# Import standard smolagents tools
try:
    from smolagents import VisitWebpageTool, WikipediaSearchTool
    SMOLAGENTS_STANDARD_AVAILABLE = True
    print("✅ smolagents standard tools loaded")
except ImportError as e:
    VisitWebpageTool = None
    WikipediaSearchTool = None
    SMOLAGENTS_STANDARD_AVAILABLE = False
    print(f"⚠️ smolagents standard tools not available: {e}")

# DEPRECATED: Remove VisionWebBrowserTool - replaced by BrowserProfileTool + smolagents components
# The old VisionWebBrowserTool approach has been replaced by the modular approach

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
    'BrowserProfileTool',  # NEW: Authentication tool
    'SpeechToTextTool',
    
    # smolagents vision browser components
    'go_back',
    'close_popups', 
    'search_item_ctrl_f',
    'save_screenshot',
    'helium_instructions',
    'get_authenticated_browser_instructions',  # NEW: Authentication instructions
    
    # smolagents standard tools
    'VisitWebpageTool',
    'WikipediaSearchTool',
    
    # Availability flags
    'YOUTUBE_TOOL_AVAILABLE',
    'BROWSER_PROFILE_AVAILABLE',  # NEW: Authentication availability
    'SMOLAGENTS_VISION_AVAILABLE',  # NEW: Vision browser availability
    'SMOLAGENTS_STANDARD_AVAILABLE',  # NEW: Standard tools availability
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
    'diagnose_browser_profile_tool',  # NEW: Browser profile diagnostics
    'diagnose_youtube_tool',
    'validate_tool_dependencies',
    'check_browser_profile_status',  # NEW: Browser profile status
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
        'BrowserProfileTool': BrowserProfileTool is not None,  # NEW
        'SpeechToTextTool': SPEECH_TO_TEXT_AVAILABLE,
        
        # smolagents components
        'smolagents_vision_available': SMOLAGENTS_VISION_AVAILABLE,  # NEW
        'smolagents_standard_available': SMOLAGENTS_STANDARD_AVAILABLE,  # NEW
        
        # Research capabilities
        'research_tools_available': langchain_status.get('research_tools_available', False),
        'langchain_tools_count': len(ALL_LANGCHAIN_TOOLS) if LANGCHAIN_TOOLS_AVAILABLE else 0,
        
        # Summary
        'total_core_tools': sum([
            ContentRetrieverTool is not None,
            YouTubeContentTool is not None,
            BrowserProfileTool is not None,  # NEW
            SPEECH_TO_TEXT_AVAILABLE
        ]),
        'total_research_tools': len(ALL_LANGCHAIN_TOOLS) if LANGCHAIN_TOOLS_AVAILABLE else 0,
        
        # Capability assessment
        'content_processing_capable': ContentRetrieverTool is not None,
        'authenticated_browsing_capable': BrowserProfileTool is not None and SMOLAGENTS_VISION_AVAILABLE,  # NEW
        'multimedia_capable': YouTubeContentTool is not None,
        'audio_processing_capable': SPEECH_TO_TEXT_AVAILABLE,
        'research_capable': langchain_status.get('research_tools_available', False)
    }

def diagnose_browser_profile_tool():
    """Comprehensive diagnosis of BrowserProfileTool status"""
    print("🔍 Diagnosing BrowserProfileTool...")
    
    # Check file existence
    current_dir = Path(__file__).parent
    tool_path = current_dir / "BrowserProfileTool.py"
    
    print(f"📁 Looking for: {tool_path}")
    print(f"📁 Absolute path: {tool_path.absolute()}")
    
    if tool_path.exists():
        print("✅ BrowserProfileTool.py file exists")
        
        # Check file size and basic content
        file_size = tool_path.stat().st_size
        print(f"📄 File size: {file_size} bytes")
        
        if file_size == 0:
            print("❌ File is empty!")
            return False
        
        # Try to read the file and check for basic class definition
        try:
            with open(tool_path, 'r') as f:
                content = f.read()
                if 'class BrowserProfileTool' in content:
                    print("✅ BrowserProfileTool class found in file")
                else:
                    print("❌ BrowserProfileTool class not found in file")
                    print("💡 File content preview:")
                    print(content[:200] + "..." if len(content) > 200 else content)
                    return False
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return False
        
        # Check dependencies
        try:
            import undetected_chromedriver as uc
            print("✅ undetected-chromedriver dependency available")
        except ImportError:
            print("❌ undetected-chromedriver dependency missing")
            print("💡 Install: pip install undetected-chromedriver")
            return False
        
        try:
            import selenium
            print("✅ selenium dependency available")
        except ImportError:
            print("❌ selenium dependency missing")
            print("💡 Install: pip install selenium")
            return False
        
        try:
            import helium
            print("✅ helium dependency available")
        except ImportError:
            print("⚠️ helium dependency missing (optional)")
            print("💡 Install: pip install helium")
        
        # Try importing the tool
        try:
            from .BrowserProfileTool import BrowserProfileTool
            print("✅ BrowserProfileTool import successful")
            
            # Try instantiating
            tool = BrowserProfileTool()
            print("✅ BrowserProfileTool instantiation successful")
            return True
            
        except Exception as e:
            print(f"❌ BrowserProfileTool import/instantiation failed: {e}")
            print(f"💡 Error details: {type(e).__name__}: {e}")
            return False
    else:
        print("❌ BrowserProfileTool.py file does not exist")
        print(f"💡 Create the file at: {tool_path}")
        
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
    """Enhanced validation with authentication and smolagents components"""
    issues = []
    recommendations = []
    
    # Check authentication dependencies
    auth_deps_available = True
    try:
        import undetected_chromedriver as uc
        print("✅ undetected-chromedriver available")
    except ImportError:
        print("❌ undetected-chromedriver not available")
        auth_deps_available = False
    
    try:
        import selenium
        print("✅ selenium available")
    except ImportError:
        print("❌ selenium not available")
        auth_deps_available = False
    
    try:
        import helium
        print("✅ helium available")
    except ImportError:
        print("⚠️ helium not available (optional for enhanced browser automation)")
    
    # Check smolagents availability
    try:
        import smolagents
        print("✅ smolagents base package available")
    except ImportError:
        print("❌ smolagents not available")
        issues.append("smolagents package missing")
        recommendations.append("Install: pip install smolagents")
    
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
    
    # Analyze tool availability
    if BrowserProfileTool is None and auth_deps_available:
        issues.append("BrowserProfileTool file missing despite dependencies being available")
        recommendations.append("Check if tools/BrowserProfileTool.py exists and has no syntax errors")
    elif BrowserProfileTool is None and not auth_deps_available:
        issues.append("BrowserProfileTool dependencies missing")
        missing_deps = []
        try:
            import undetected_chromedriver
        except ImportError:
            missing_deps.append("undetected-chromedriver")
        try:
            import selenium
        except ImportError:
            missing_deps.append("selenium")
        if missing_deps:
            recommendations.append(f"Install missing dependencies: pip install {' '.join(missing_deps)}")
    
    if YouTubeContentTool is None and (yt_dlp_available and requests_available):
        issues.append("YouTubeContentTool file missing despite dependencies being available")
        recommendations.append("Check if tools/youtube_content_tool.py exists and has no syntax errors")
    elif YouTubeContentTool is None and not (yt_dlp_available and requests_available):
        issues.append("YouTubeContentTool dependencies missing")
        missing_deps = []
        if not yt_dlp_available:
            missing_deps.append("yt-dlp")
        if not requests_available:
            missing_deps.append("requests")
        recommendations.append(f"Install missing dependencies: pip install {' '.join(missing_deps)}")
    
    if not SMOLAGENTS_VISION_AVAILABLE:
        issues.append("smolagents vision browser components not available")
        recommendations.append("Update smolagents: pip install --upgrade smolagents")
    
    # Check for content processing dependencies
    if ContentRetrieverTool is None:
        issues.append("ContentRetrieverTool not available")
        recommendations.append("Check tools/content_retriever_tool.py exists and dependencies")
    
    return {
        'issues': issues,
        'recommendations': recommendations,
        'status': 'healthy' if not issues else 'needs_attention',
        'dependency_details': {
            'undetected_chromedriver_available': auth_deps_available,
            'helium_available': HELIUM_AVAILABLE,
            'smolagents_vision_available': SMOLAGENTS_VISION_AVAILABLE,
            'smolagents_standard_available': SMOLAGENTS_STANDARD_AVAILABLE,
            'yt_dlp_available': yt_dlp_available,
            'requests_available': requests_available,
            'browser_profile_tool_available': BrowserProfileTool is not None,
            'youtube_tool_available': YouTubeContentTool is not None
        }
    }

def check_browser_profile_status():
    """Quick status check for browser profile capabilities"""
    print("🔍 Browser Profile Tool Status Check:")
    success = diagnose_browser_profile_tool()
    
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
    
    print(f"\n📊 Browser Profile Details:")
    profile_deps = {k: v for k, v in deps['dependency_details'].items() 
                   if 'browser' in k.lower() or k in ['undetected_chromedriver_available', 'helium_available']}
    for key, value in profile_deps.items():
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

    # Speech to text processing
    if SpeechToTextTool:
        tools.append(SpeechToTextTool())
        print("✓ Added SpeechToTextTool to content_processor")        
    
    # Multimedia content processing (YouTube support)
    if YouTubeContentTool:
        tools.append(YouTubeContentTool())
        print("✓ Added YouTubeContentTool to content_processor")
    
    # Authentication for restricted content
    if BrowserProfileTool:
        tools.append(BrowserProfileTool())
        print("✓ Added BrowserProfileTool to content_processor (for authenticated content)")
    
    print(f"📦 Content processor tools: {len(tools)} available")
    return tools

def get_web_researcher_tools():
    """Get tools specifically for web_researcher specialist with authentication and smolagents integration"""
    tools = []
    
    # Add LangChain research tools (PRIMARY for web research)
    if LANGCHAIN_TOOLS_AVAILABLE:
        langchain_tools = get_langchain_tools()
        tools.extend(langchain_tools)
        print(f"✓ Added {len(langchain_tools)} LangChain research tools to web_researcher")
    
    # Add standard smolagents tools
    if SMOLAGENTS_STANDARD_AVAILABLE:
        if VisitWebpageTool:
            tools.append(VisitWebpageTool())
            print("✓ Added VisitWebpageTool to web_researcher")
        if WikipediaSearchTool:
            tools.append(WikipediaSearchTool())
            print("✓ Added WikipediaSearchTool to web_researcher")
    
    # Add smolagents vision browser tools (these are simple tools, not class instances)
    vision_tools_added = 0
    if SMOLAGENTS_VISION_AVAILABLE:
        if go_back:
            tools.append(go_back)
            vision_tools_added += 1
        if close_popups:
            tools.append(close_popups)
            vision_tools_added += 1
        if search_item_ctrl_f:
            tools.append(search_item_ctrl_f)
            vision_tools_added += 1
        
        if vision_tools_added > 0:
            print(f"✓ Added {vision_tools_added} smolagents vision browser tools to web_researcher")
    
    # Add authentication capability
    if BrowserProfileTool:
        tools.append(BrowserProfileTool())
        print("✓ Added BrowserProfileTool to web_researcher (for authenticated browsing)")
    
    print(f"🔍 Web researcher tools: {len(tools)} available")
    return tools

# Print initialization status
print(f"\n🔧 GAIA Tools Status: {get_tool_status()}")

# Validate dependencies and show recommendations
dependency_status = validate_tool_dependencies()
if dependency_status['issues']:
    print("\n⚠️ Dependency issues detected:")
    for issue in dependency_status['issues']:
        print(f"   - {issue}")
    print("💡 Recommendations:")
    for rec in dependency_status['recommendations']:
        print(f"   - {rec}")
else:
    print("\n✅ All tool dependencies validated successfully")

print("\n🔧 Tools package initialized for GAIA Agent with authentication and smolagents integration")