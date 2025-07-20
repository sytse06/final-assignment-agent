# test_browser_profile_tool.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tools'))

def test_browser_profile_tool():
    """Test the new BrowserProfileTool"""
    print("🔧 Testing BrowserProfileTool...")
    
    try:
        from BrowserProfileTool import BrowserProfileTool
        
        tool = BrowserProfileTool()
        print(f"✅ BrowserProfileTool created: {tool.name}")
        
        # Test tool configuration
        print(f"   Inputs: {list(tool.inputs.keys())}")
        print(f"   Output type: {tool.output_type}")
        
        # Test environment detection
        is_container = tool._is_container_environment()
        print(f"   Container environment: {is_container}")
        
        profile_dir = tool._get_profile_directory()
        print(f"   Profile directory: {profile_dir}")
        
        return True
        
    except ImportError as e:
        print(f"❌ BrowserProfileTool import failed: {e}")
        print("💡 Check that tools/BrowserProfileTool.py exists")
        return False
    except Exception as e:
        print(f"❌ BrowserProfileTool test failed: {e}")
        return False

def test_smolagents_vision_components():
    """Test smolagents vision browser components"""
    print("🔧 Testing smolagents vision browser components...")
    
    try:
        from smolagents.vision_web_browser import (
            go_back, close_popups, search_item_ctrl_f, 
            save_screenshot, helium_instructions
        )
        
        print("✅ All smolagents vision components imported successfully:")
        print(f"   go_back: {go_back.name if hasattr(go_back, 'name') else 'Available'}")
        print(f"   close_popups: {close_popups.name if hasattr(close_popups, 'name') else 'Available'}")
        print(f"   search_item_ctrl_f: {search_item_ctrl_f.name if hasattr(search_item_ctrl_f, 'name') else 'Available'}")
        print(f"   save_screenshot: {'Available' if save_screenshot else 'None'}")
        print(f"   helium_instructions: {len(helium_instructions)} characters")
        
        return True
        
    except ImportError as e:
        print(f"❌ smolagents vision components import failed: {e}")
        print("💡 Update smolagents: pip install --upgrade smolagents")
        return False
    except Exception as e:
        print(f"❌ smolagents vision components test failed: {e}")
        return False

def test_browser_profile_functionality():
    """Test actual browser profile functionality"""
    print("🔧 Testing browser profile functionality...")
    
    try:
        from BrowserProfileTool import BrowserProfileTool
        
        tool = BrowserProfileTool()
        
        # Test profile setup (without actually starting browser)
        print("   Testing cookie detection methods...")
        
        # Test environment variable detection
        env_cookies = tool._get_cookies_from_environment("youtube")
        print(f"   Environment cookies for YouTube: {'Found' if env_cookies else 'Not found'}")
        
        # Test file-based cookies
        file_cookies = tool._get_cookies_from_file("youtube")
        print(f"   File cookies for YouTube: {'Found' if file_cookies else 'Not found'}")
        
        # Test browser extraction (should return None in most environments)
        browser_cookies = tool._get_cookies_from_browser("youtube")
        print(f"   Browser cookies for YouTube: {'Found' if browser_cookies else 'Not found'}")
        
        print("✅ Cookie detection methods working")
        return True
        
    except Exception as e:
        print(f"❌ Browser profile functionality test failed: {e}")
        return False

if __name__ == "__main__":
    test_browser_profile_tool()
    test_smolagents_vision_components()  
    test_browser_profile_functionality()