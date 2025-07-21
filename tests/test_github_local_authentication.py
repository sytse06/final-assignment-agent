# github_local_authentication.py.py
"""
GitHub authentication test specifically for smolagents repository access
"""
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def manual_github_login():
    """Open GitHub in browser for manual login"""
    from tools import BrowserProfileTool
    
    print("🚀 Manual GitHub Login")
    print("=" * 40)
    
    browser_tool = None
    
    try:
        # Initialize browser profile tool
        browser_tool = BrowserProfileTool()
        
        # Set up GitHub browser profile
        print("🔧 Setting up GitHub browser...")
        result = browser_tool.forward("github")
        print(f"Result: {result}")
        
        if "failed" in result.lower():
            print("❌ Browser setup failed")
            return False
        
        # Get the browser instance
        browser = browser_tool.get_current_browser()
        if not browser:
            print("❌ No browser instance available")
            return False
        
        # Navigate to GitHub logout first (clean state)
        print("🚪 Logging out any existing session...")
        browser.get("https://github.com/logout")
        time.sleep(3)
        
        # Navigate to GitHub login
        print("🔐 Opening GitHub login page...")
        browser.get("https://github.com/login")
        time.sleep(2)
        
        print("\n" + "="*50)
        print("🔐 MANUAL LOGIN TIME")
        print("="*50)
        print("✅ Browser is open with GitHub login page")
        print("✅ Complete your login manually in the browser")
        print("✅ Take your time - no rush!")
        print("\n📋 Steps:")
        print("   1. Enter your GitHub username/email")
        print("   2. Enter your password")
        print("   3. Complete 2FA if prompted")
        print("   4. Wait for your dashboard to load")
        
        print("\n⏸️ Login when ready...")
        input("Press Enter when you've finished logging in (or just to close): ")
        
        # Check if login worked
        print("\n🔍 Checking login status...")
        browser.get("https://github.com/notifications")
        time.sleep(3)
        
        if "Notifications" in browser.page_source:
            print("✅ Login successful! Your GitHub profile is authenticated.")
            print("🎉 Future browser sessions will remember this login.")
        else:
            print("ℹ️ Login status unclear, but that's okay!")
            print("💡 If you logged in, the profile should be saved.")
        
        print("\n✅ Manual login session complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False
    
    finally:
        if browser_tool:
            print("🧹 Cleaning up...")
            browser_tool.cleanup()

if __name__ == "__main__":
    print("🎯 Simple GitHub Manual Login Tool")
    print("This opens GitHub in a browser for you to login manually\n")
    
    success = manual_github_login()
    
    if success:
        print("\n🎉 Done! Your GitHub authentication should now be saved.")
        print("💡 Future GAIA agent runs will use this authentication.")
    else:
        print("\n⚠️ Something went wrong, but you can try again.")