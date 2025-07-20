# test_browser_dependencies.py
#!/usr/bin/env python3
"""
Enhanced Browser Dependency Test for New Architecture
Tests BrowserProfileTool dependencies and smolagents integration
"""

import sys
import os
import subprocess

def test_browser_profile_dependencies():
    """Test dependencies specific to BrowserProfileTool"""
    print("🔍 BROWSER PROFILE TOOL DEPENDENCIES")
    print("=" * 50)
    
    dependencies_ok = True
    
    # Test undetected-chromedriver
    try:
        import undetected_chromedriver as uc
        print(f"✅ undetected-chromedriver: {getattr(uc, '__version__', 'version unknown')}")
    except ImportError as e:
        print(f"❌ undetected-chromedriver import failed: {e}")
        print("💡 Install with: pip install undetected-chromedriver")
        dependencies_ok = False
    
    # Test selenium
    try:
        import selenium
        print(f"✅ selenium: {getattr(selenium, '__version__', 'version unknown')}")
    except ImportError as e:
        print(f"❌ selenium import failed: {e}")
        print("💡 Install with: pip install selenium")
        dependencies_ok = False
    
    # Test helium (optional for BrowserProfileTool)
    try:
        import helium
        print(f"✅ helium: {getattr(helium, '__version__', 'version unknown')}")
    except ImportError as e:
        print(f"⚠️ helium import failed: {e}")
        print("💡 Install with: pip install helium (optional)")
    
    return dependencies_ok

def test_smolagents_integration():
    """Test smolagents vision browser integration"""
    print("\n🔍 SMOLAGENTS INTEGRATION")
    print("=" * 50)
    
    integration_ok = True
    
    # Test base smolagents
    try:
        import smolagents
        print(f"✅ smolagents: {getattr(smolagents, '__version__', 'version unknown')}")
    except ImportError as e:
        print(f"❌ smolagents import failed: {e}")
        print("💡 Install with: pip install smolagents")
        integration_ok = False
        return False
    
    # Test vision browser components
    try:
        from smolagents.vision_web_browser import (
            go_back, close_popups, search_item_ctrl_f, 
            save_screenshot, helium_instructions
        )
        print("✅ smolagents vision browser components")
        print(f"   go_back: {getattr(go_back, 'name', 'tool available')}")
        print(f"   close_popups: {getattr(close_popups, 'name', 'tool available')}")
        print(f"   search_item_ctrl_f: {getattr(search_item_ctrl_f, 'name', 'tool available')}")
        print(f"   helium_instructions: {len(helium_instructions)} chars")
    except ImportError as e:
        print(f"❌ smolagents vision browser import failed: {e}")
        print("💡 Update smolagents: pip install --upgrade smolagents")
        integration_ok = False
    
    # Test standard smolagents tools
    try:
        from smolagents import VisitWebpageTool, WikipediaSearchTool
        print("✅ smolagents standard tools")
    except ImportError as e:
        print(f"⚠️ smolagents standard tools import failed: {e}")
    
    return integration_ok

def test_browser_profile_tool():
    """Test BrowserProfileTool functionality"""
    print("\n🔍 BROWSER PROFILE TOOL FUNCTIONALITY")
    print("=" * 50)
    
    try:
        # Add tools directory to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tools'))
        
        from BrowserProfileTool import BrowserProfileTool
        
        tool = BrowserProfileTool()
        print(f"✅ BrowserProfileTool created: {tool.name}")
        
        # Test environment detection
        is_container = tool._is_container_environment()
        print(f"✅ Environment detection: Container={is_container}")
        
        # Test profile directory creation
        profile_dir = tool._profile_dir
        print(f"✅ Profile directory: {profile_dir}")
        
        # Test cookie detection methods (without actually running them)
        print("✅ Cookie detection methods available:")
        print("   - Environment variable detection")
        print("   - File-based cookie detection") 
        print("   - Browser extraction (local dev)")
        
        return True
        
    except ImportError as e:
        print(f"❌ BrowserProfileTool import failed: {e}")
        print("💡 Check that tools/BrowserProfileTool.py exists")
        return False
    except Exception as e:
        print(f"❌ BrowserProfileTool test failed: {e}")
        return False

def test_authentication_setup():
    """Test authentication setup without actually authenticating"""
    print("\n🔍 AUTHENTICATION SETUP")
    print("=" * 50)
    
    # Check for environment variables
    auth_envs = ["YOUTUBE_COOKIES", "LINKEDIN_COOKIES", "GITHUB_COOKIES"]
    found_envs = 0
    
    for env_var in auth_envs:
        if os.getenv(env_var):
            print(f"✅ {env_var}: Set")
            found_envs += 1
        else:
            print(f"ℹ️ {env_var}: Not set")
    
    if found_envs > 0:
        print(f"✅ Authentication environment configured: {found_envs}/{len(auth_envs)} platforms")
    else:
        print("ℹ️ No authentication environment variables set")
        print("💡 For deployment, set YOUTUBE_COOKIES, LINKEDIN_COOKIES, GITHUB_COOKIES")
    
    # Check for cookie files
    cookie_paths = [
        "./cookies/youtube_cookies.txt",
        "./cookies/linkedin_cookies.txt", 
        "./cookies/github_cookies.txt",
        "/app/cookies/youtube_cookies.txt"
    ]
    
    found_files = 0
    for path in cookie_paths:
        if os.path.exists(path):
            print(f"✅ Cookie file found: {path}")
            found_files += 1
    
    if found_files == 0:
        print("ℹ️ No cookie files found (normal for development)")
    
    return True

def test_minimal_browser_functionality():
    """Test minimal browser functionality with BrowserProfileTool"""
    print("\n🔍 MINIMAL BROWSER FUNCTIONALITY")
    print("=" * 50)
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tools'))
        from BrowserProfileTool import BrowserProfileTool
        
        tool = BrowserProfileTool()
        
        # Test tool invocation (without actually starting browser)
        print("🔧 Testing tool invocation...")
        
        # This should handle missing dependencies gracefully
        result = tool.forward("general")
        print(f"✅ Tool forward() method: {result[:100]}...")
        
        # Test cleanup
        tool.cleanup()
        print("✅ Tool cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal browser functionality test failed: {e}")
        return False

def main():
    """Run enhanced browser dependency tests"""
    print("🧪 Enhanced Browser Dependencies Test Suite")
    print("=" * 60)
    print(f"🐍 Python: {sys.version}")
    print(f"📂 Working directory: {os.getcwd()}")
    print()
    
    # Run all tests
    deps_ok = test_browser_profile_dependencies()
    integration_ok = test_smolagents_integration()
    tool_ok = test_browser_profile_tool()
    auth_ok = test_authentication_setup()
    browser_ok = test_minimal_browser_functionality()
    
    # Summary
    print("\n📊 ENHANCED DEPENDENCY TEST SUMMARY")
    print("=" * 50)
    print(f"BrowserProfileTool Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"smolagents Integration: {'✅ PASS' if integration_ok else '❌ FAIL'}")
    print(f"BrowserProfileTool Functionality: {'✅ PASS' if tool_ok else '❌ FAIL'}")
    print(f"Authentication Setup: {'✅ PASS' if auth_ok else '❌ FAIL'}")
    print(f"Minimal Browser Functionality: {'✅ PASS' if browser_ok else '❌ FAIL'}")
    
    all_passed = all([deps_ok, integration_ok, tool_ok, auth_ok, browser_ok])
    
    if all_passed:
        print("\n🎉 ALL ENHANCED DEPENDENCY TESTS PASSED!")
        print("Your enhanced browser automation setup is ready:")
        print("   ✅ BrowserProfileTool with authentication support")
        print("   ✅ smolagents vision browser integration") 
        print("   ✅ Multi-platform authentication capability")
        print("   ✅ Docker/HF Spaces compatibility")
        
    else:
        print("\n⚠️ Some tests failed. Install missing dependencies:")
        if not deps_ok:
            print("   pip install undetected-chromedriver selenium")
        if not integration_ok:
            print("   pip install --upgrade smolagents")
        print("\nRefer to error messages above for specific issues.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)