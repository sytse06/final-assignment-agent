#!/usr/bin/env python3
"""
Local test script for YouTubeContentTool
Run this to test your YouTube tool locally
"""

import os
import sys
from pathlib import Path

# Add tool class
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tools'))

try:
    from youtube_content_tool import YouTubeContentTool
except ImportError:
    print("❌ Could not import YouTubeContentTool")
    print("   Make sure youtube_content_tool.py is in the same directory")
    print("   Or adjust the import path above")
    sys.exit(1)


def create_youtube_tool_for_hf_spaces():
    """Create YouTube tool configured for HF Spaces deployment"""
    tool = YouTubeContentTool()
    # Cookies will be automatically loaded from YOUTUBE_COOKIES secret
    return tool


def create_youtube_tool_for_local_dev(method="browser", browser="chrome", cookies_file=None):
    """
    Create YouTube tool for local development with multiple auth options
    
    Args:
        method: "browser", "file", or "env"
        browser: "chrome", "firefox", "safari", "edge" (for browser method)
        cookies_file: Path to cookies.txt (for file method)
    """
    tool = YouTubeContentTool()
    
    if method == "browser":
        tool.set_cookies_from_browser(browser)
        print(f"🔧 Configured for {browser} browser cookies")
    elif method == "file":
        if cookies_file:
            tool.set_cookies_from_file(cookies_file)
            print(f"🔧 Configured for cookies file: {cookies_file}")
        else:
            print("❌ cookies_file path required for file method")
    elif method == "env":
        print("🔧 Configured for environment/secret cookies (same as HF Spaces)")
    
    return tool


def test_youtube_tool_locally(test_video_id="dQw4w9WgXcQ"):
    """
    Test YouTube tool with different authentication methods locally
    
    Args:
        test_video_id: YouTube video ID to test with
    """
    test_url = f"https://youtube.com/watch?v={test_video_id}"
    
    print("🧪 Testing YouTube tool with hybrid extraction approach...\n")
    print(f"🎯 Test video: {test_url}")
    print("=" * 60)
    
    # Test the hybrid approach (simple first, then authenticated)
    print("\n📋 Test: Hybrid approach (simple → authenticated fallback)")
    try:
        tool = YouTubeContentTool()  # Default configuration
        result = tool.forward(test_url)
        
        # Check which method worked
        if "🎬 **Method:** public_simple" in result:
            print("✅ SUCCESS: Simple public extraction worked!")
        elif "🍪 **Method:** authenticated" in result:
            print("✅ SUCCESS: Authenticated extraction worked!")
        else:
            print("✅ SUCCESS: Some extraction method worked!")
            
        print(f"\n📊 Result preview:")
        print("-" * 40)
        print(result[:500] + "..." if len(result) > 500 else result)
        print("-" * 40)
        
        return True  # Success
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


def test_specific_methods(test_video_id="dQw4w9WgXcQ"):
    """Test specific authentication methods"""
    test_url = f"https://youtube.com/watch?v={test_video_id}"
    
    print("\n📋 Testing Specific Authentication Methods:")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Browser cookie authentication
    print("\n🔹 Testing browser cookie authentication (Chrome):")
    try:
        tool_browser = create_youtube_tool_for_local_dev("browser", "chrome")
        result_browser = tool_browser.forward(test_url)
        success = "📺" in result_browser and "Error" not in result_browser
        results['browser_chrome'] = success
        print(f"   {'✅ SUCCESS' if success else '❌ FAILED'}")
        if success:
            print(f"   Preview: {result_browser[:100]}...")
    except Exception as e:
        results['browser_chrome'] = False
        print(f"   ❌ FAILED: {e}")
    
    # Test 2: Firefox browser cookies
    print("\n🔹 Testing browser cookie authentication (Firefox):")
    try:
        tool_firefox = create_youtube_tool_for_local_dev("browser", "firefox")
        result_firefox = tool_firefox.forward(test_url)
        success = "📺" in result_firefox and "Error" not in result_firefox
        results['browser_firefox'] = success
        print(f"   {'✅ SUCCESS' if success else '❌ FAILED'}")
        if success:
            print(f"   Preview: {result_firefox[:100]}...")
    except Exception as e:
        results['browser_firefox'] = False
        print(f"   ❌ FAILED: {e}")
    
    # Test 3: Cookies file (if available)
    print("\n🔹 Testing cookies file:")
    cookies_paths = ["cookies.txt", "youtube_cookies.txt", "./cookies.txt", "../cookies.txt"]
    cookies_file_found = None
    
    for path in cookies_paths:
        if os.path.exists(path):
            cookies_file_found = path
            break
    
    if cookies_file_found:
        try:
            tool_file = create_youtube_tool_for_local_dev("file", cookies_file=cookies_file_found)
            result_file = tool_file.forward(test_url)
            success = "📺" in result_file and "Error" not in result_file
            results['cookies_file'] = success
            print(f"   ✅ SUCCESS: Using {cookies_file_found}" if success else f"   ❌ FAILED: Using {cookies_file_found}")
            if success:
                print(f"   Preview: {result_file[:100]}...")
        except Exception as e:
            results['cookies_file'] = False
            print(f"   ❌ FAILED: {e}")
    else:
        results['cookies_file'] = None
        print("   ⚠️ No cookies.txt file found in common locations")
        print("   Searched: " + ", ".join(cookies_paths))
    
    # Test 4: No authentication (basic fallback)
    print("\n🔹 Testing no authentication (fallback):")
    try:
        tool_basic = YouTubeContentTool()
        # Force no authentication by not setting any cookies
        result_basic = tool_basic._extract_public_video_content(test_url)
        if result_basic:
            success = True
            results['no_auth'] = True
            print("   ✅ SUCCESS: Public extraction worked")
            print(f"   Title: {result_basic.get('title', 'Unknown')}")
            print(f"   Duration: {result_basic.get('duration', 0)} seconds")
            print(f"   Has transcript: {result_basic.get('has_transcript', False)}")
        else:
            success = False
            results['no_auth'] = False
            print("   ❌ FAILED: No content extracted")
    except Exception as e:
        results['no_auth'] = False
        print(f"   ❌ FAILED: {e}")
    
    return results


def test_different_videos():
    """Test with different types of videos"""
    print("\n📋 Testing Different Video Types:")
    print("=" * 60)
    
    test_videos = [
        ("dQw4w9WgXcQ", "Rick Astley - Never Gonna Give You Up (Classic, very public)"),
        ("9bZkp7q19f0", "PSY - GANGNAM STYLE (Very popular, public)"),
        ("fJ9rUzIMcZQ", "Queen - Bohemian Rhapsody (Music, public)"),
        ("L1vXCYZAYYM", "Penguin chicks stand up to giant Petrel")
    ]
    
    results = {}
    
    for video_id, description in test_videos:
        print(f"\n🎬 Testing: {description}")
        print(f"   Video ID: {video_id}")
        
        try:
            tool = YouTubeContentTool()
            result = tool.forward(f"https://youtube.com/watch?v={video_id}")
            
            if "Error" not in result and "📺" in result:
                results[video_id] = True
                print("   ✅ SUCCESS")
                # Extract title from result
                lines = result.split('\n')
                for line in lines:
                    if line.startswith('📺'):
                        print(f"   {line}")
                        break
            else:
                results[video_id] = False
                print("   ❌ FAILED")
                
        except Exception as e:
            results[video_id] = False
            print(f"   ❌ FAILED: {e}")
    
    return results


def print_summary(hybrid_success, method_results, video_results):
    """Print test summary"""
    print("\n🔍 TEST SUMMARY")
    print("=" * 60)
    
    # Hybrid test result
    print(f"🎯 Hybrid Approach: {'✅ SUCCESS' if hybrid_success else '❌ FAILED'}")
    
    # Method results
    print("\n📊 Authentication Methods:")
    for method, result in method_results.items():
        if result is None:
            status = "⚠️ SKIPPED"
        elif result:
            status = "✅ SUCCESS"
        else:
            status = "❌ FAILED"
        print(f"   {method}: {status}")
    
    # Video results
    print(f"\n🎬 Different Videos: {sum(video_results.values())}/{len(video_results)} successful")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if hybrid_success:
        print("• ✅ Your tool is working! The hybrid approach succeeded.")
        print("• 🚀 You can integrate this tool into your GAIA agent system.")
    else:
        print("• ⚠️ Hybrid approach failed. Try these solutions:")
        
        if method_results.get('browser_chrome') or method_results.get('browser_firefox'):
            print("  - Browser cookies work! Use create_youtube_tool_for_local_dev('browser')")
        
        if method_results.get('cookies_file'):
            print("  - Cookies file works! Use create_youtube_tool_for_local_dev('file')")
        
        if not any(method_results.values()):
            print("  - All methods failed. This might be due to:")
            print("    1. Network issues")
            print("    2. YouTube blocking your IP")
            print("    3. Missing dependencies (yt-dlp, requests)")
            print("    4. Firewall/proxy issues")
    
    print("\n🔧 NEXT STEPS:")
    print("• For HF Spaces: Set YOUTUBE_COOKIES secret with your browser cookies")
    print("• For local dev: Use the method that worked in the tests above")
    print("• For GAIA integration: Add YouTubeContentTool() to your agent's toolbox")


def main():
    """Main test function"""
    print("🚀 YouTube Content Tool - Local Testing")
    print("=" * 60)
    print("This script will test your YouTube tool with different configurations")
    print("to see what works in your environment.\n")
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    try:
        import yt_dlp
        import requests
        print("✅ yt-dlp and requests are available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install yt-dlp requests")
        return
    
    # Run tests
    print("\n🧪 Starting tests...")
    
    # Test 1: Hybrid approach
    hybrid_success = test_youtube_tool_locally()
    
    # Test 2: Specific methods
    method_results = test_specific_methods()
    
    # Test 3: Different videos
    video_results = test_different_videos()
    
    # Print summary
    print_summary(hybrid_success, method_results, video_results)


if __name__ == "__main__":
    main()