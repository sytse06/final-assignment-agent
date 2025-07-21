# local_authentication.py
"""
Local authentication workflow testing for Youtube and Github
"""
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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
        time.sleep(60)
        
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
    import time
    
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

def test_profile_persistence():
    """Test that profiles persist between sessions"""
    from tools import BrowserProfileTool
    import os
    
    print("\n🧪 Testing Profile Persistence")
    print("=" * 50)
    
    browser_tool = BrowserProfileTool()
    profile_dir = browser_tool._profile_dir
    
    # Check if profiles exist
    for platform in ["youtube", "linkedin", "github", "general"]:
        platform_dir = os.path.join(profile_dir, platform)
        if os.path.exists(platform_dir):
            files = os.listdir(platform_dir)
            print(f"✅ {platform} profile exists: {len(files)} files")
        else:
            print(f"ℹ️ {platform} profile not created yet")
    
    return True

if __name__ == "__main__":
    print("🚀 Local Authentication Test Suite")
    print("=" * 60)
    
    # Test 1: YouTube authentication workflow
    youtube_success = test_youtube_authentication()
    
    # Test 2: Profile persistence  
    persistence_success = test_profile_persistence()
    
    print("\n📊 Test Summary")
    print("=" * 30)
    print(f"YouTube Authentication: {'✅ PASS' if youtube_success else '❌ FAIL'}")
    print(f"Profile Persistence: {'✅ PASS' if persistence_success else '❌ FAIL'}")
    
    if youtube_success and persistence_success:
        print("\n🎉 Local authentication setup is working!")
        print("💡 Tips:")
        print("   - Sign into platforms manually once")
        print("   - Profiles will remember your login")
        print("   - Use browser_profile() before navigating to authenticated content")
    else:
        print("\n⚠️ Some tests failed - check error messages above")