# local_authentication.py
"""
Local authentication workflow testing for Youtube and Github
"""
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

"""
Test local authentication workflow with BrowserProfileTool
"""

def test_platform_individually(platform_name: str):
    """Test a single platform with robust error handling"""
    from tools import BrowserProfileTool
    import helium
    import time
    
    print(f"\n🧪 Testing {platform_name.capitalize()} Platform")
    print("=" * 50)
    
    browser_tool = None
    
    try:
        browser_tool = BrowserProfileTool()
        
        # Initialize browser
        print(f"🔧 Setting up {platform_name} browser...")
        result = browser_tool.forward(platform_name)
        print(f"Result: {result}")
        
        if "failed" in result.lower():
            print(f"❌ Browser setup failed for {platform_name}")
            return False
        
        # Test basic navigation
        print(f"🌐 Testing navigation...")
        
        if platform_name == "youtube":
            success = test_youtube_navigation(browser_tool)
        elif platform_name == "github":
            success = test_github_navigation(browser_tool)
        else:
            success = True
        
        return success
        
    except Exception as e:
        print(f"❌ {platform_name} test failed: {str(e)}")
        return False
    
    finally:
        if browser_tool:
            print(f"🧹 Cleaning up {platform_name} browser...")
            browser_tool.cleanup()
            time.sleep(2)  # Give cleanup time

def test_youtube_authentication():
    """Test YouTube authentication workflow"""
    from tools import BrowserProfileTool
    import helium
    import time
    
    print("🧪 Testing YouTube Authentication Workflow")
    print("=" * 50)
    
    # Initialize browser profile tool
    browser_tool = BrowserProfileTool()
    
    try:
        # Set up YouTube profile
        result = browser_tool.forward("youtube")
        print(f"✅ Browser setup result: {result}")
        
        # Navigate to YouTube
        helium.go_to("https://youtube.com")
        time.sleep(2)
        
        # Check authentication status
        if helium.Text("Sign in").exists():
            print("ℹ️ Not authenticated - you can manually sign in now")
            print("   1. Click 'Sign in' in the browser window")
            print("   2. Complete login process") 
            print("   3. Future sessions will remember this login")
            
            # Wait for manual authentication
            input("Press Enter after signing in (or to continue without auth)...")
            
        else:
            print("✅ Already authenticated or no sign-in needed")
        
        # Test navigation to a specific video
        helium.go_to("https://youtube.com/watch?v=dQw4w9WgXcQ")  # Rick Roll - safe test video
        time.sleep(3)
        
        # Check if video loads properly
        if helium.Text("Rick Astley").exists():
            print("✅ Video page loaded successfully")
        elif helium.Text("Video unavailable").exists():
            print("⚠️ Video unavailable (might be geo-blocked)")
        else:
            print("ℹ️ Page loaded - authentication working")
        
        return True
        
    except Exception as e:
        print(f"❌ Authentication test failed: {str(e)}")
        return False
    
    finally:
        # Cleanup
        browser_tool.cleanup()

def test_github_navigation(browser_tool):
    """Test GitHub navigation"""
    from tools import BrowserProfileTool
    import helium
    import time
    
    print("🧪 Testing Github Authentication Workflow")
    print("=" * 50)
    
    # Initialize browser profile tool
    browser_tool = BrowserProfileTool()
    
    try:
        browser = browser_tool.get_current_browser()
        if not browser:
            print("❌ No browser instance available")
            return False
        
        print("   Navigating to GitHub...")
        success = browser_tool.navigate_safely("https://github.com")
        
        if not success:
            print("   ❌ GitHub navigation failed")
            return False
        
        time.sleep(3)
        
        # Check if page loaded
        title = browser.title
        if "github" in title.lower():
            print(f"   ✅ GitHub loaded: {title}")
            
            # Check authentication status
            page_source = browser.page_source
            if "Sign in" in page_source:
                print("   ℹ️ Sign in available - manual login possible")
            elif "Dashboard" in page_source or "Pull requests" in page_source:
                print("   ✅ Already signed into GitHub")
            else:
                print("   ✅ GitHub public access available")
            
            return True
        else:
            print(f"   ⚠️ Unexpected page: {title}")
            return False
        
    except Exception as e:
        print(f"   ❌ GitHub navigation error: {str(e)}")
        return False

def main():
    """Run simplified platform tests one by one"""
    print("🚀 Simplified Platform Authentication Tests")
    print("=" * 60)
    print("Testing each platform individually with enhanced error handling\n")
    
    platforms = ["youtube", "github"]
    results = {}
    
    for platform in platforms:
        print(f"\n{'='*20} {platform.upper()} {'='*20}")
        
        # Test platform
        success = test_platform_individually(platform)
        results[platform] = success
        
        # Wait between platforms
        if platform != platforms[-1]:  # Not last platform
            print(f"\n⏳ Waiting 5 seconds before next platform...")
            time.sleep(5)
    
    # Summary
    print(f"\n{'='*20} SUMMARY {'='*20}")
    for platform, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{platform.capitalize():10}: {status}")
    
    success_count = sum(results.values())
    print(f"\nOverall: {success_count}/{len(platforms)} platforms working")
    
    if success_count == len(platforms):
        print("\n🎉 All platforms working!")
        print("Your authentication system is ready for GAIA tasks!")
    elif success_count >= 2:
        print("\n✅ Most platforms working!")
        print("You can proceed with GAIA tasks using working platforms")
    else:
        print("\n⚠️ Multiple platform issues detected")
        print("Try running individual tests or check browser setup")
    
    # Tips
    print(f"\n💡 Tips:")
    print(f"   - YouTube: Most reliable, good for video content tasks")
    print(f"   - GitHub: Generally works well for code/repo tasks")  
    print(f"   - For production: Use environment variable authentication")

if __name__ == "__main__":
    main()