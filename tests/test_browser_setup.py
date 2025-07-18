#!/usr/bin/env python3
"""
Browser Diagnostic Script
Run this to test browser setup before using VisionWebBrowserTool

Usage:
    poetry run python browser_diagnostic.py
    
Or:
    python browser_diagnostic.py
"""

import sys
import os
import subprocess

def diagnose_browser_setup():
    """
    Complete browser diagnostic for VisionWebBrowserTool
    """
    print("🔍 BROWSER SETUP DIAGNOSTIC")
    print("=" * 50)
    print(f"🐍 Python: {sys.version}")
    print(f"📂 Working directory: {os.getcwd()}")
    print()
    
    # === STEP 1: Import Tests ===
    print("📦 TESTING IMPORTS...")
    print("-" * 30)
    
    # Test helium import
    try:
        import helium
        helium_version = getattr(helium, '__version__', 'version unknown')
        print(f"✅ Helium: {helium_version}")
    except ImportError as e:
        print(f"❌ Helium import failed: {e}")
        print("💡 Install with: poetry add helium")
        return False
    
    # Test selenium import
    try:
        import selenium
        selenium_version = getattr(selenium, '__version__', 'version unknown')
        print(f"✅ Selenium: {selenium_version}")
    except ImportError as e:
        print(f"❌ Selenium import failed: {e}")
        print("💡 Install with: poetry add selenium")
        return False
    
    # Test webdriver import
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        print("✅ Selenium webdriver components")
    except ImportError as e:
        print(f"❌ Selenium webdriver import failed: {e}")
        return False
    
    print()
    
    # === STEP 2: System Dependencies ===
    print("🖥️  TESTING SYSTEM DEPENDENCIES...")
    print("-" * 30)
    
    # Test Chrome
    try:
        result = subprocess.run(['google-chrome', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Chrome: {result.stdout.strip()}")
        else:
            print(f"⚠️ Chrome command failed: {result.stderr}")
    except FileNotFoundError:
        print("❌ Chrome not found in PATH")
        print("💡 Install Chrome browser")
    except subprocess.TimeoutExpired:
        print("⚠️ Chrome command timed out")
    except Exception as e:
        print(f"⚠️ Chrome check failed: {e}")
    
    # Test ChromeDriver (alternative check)
    try:
        result = subprocess.run(['chromedriver', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ ChromeDriver: {result.stdout.strip()}")
        else:
            print(f"⚠️ ChromeDriver in PATH but failed: {result.stderr}")
    except FileNotFoundError:
        print("ℹ️ ChromeDriver not in PATH (helium will handle this)")
    except Exception as e:
        print(f"ℹ️ ChromeDriver check failed: {e}")
    
    print()
    
    # === STEP 3: Environment Check ===
    print("🌍 TESTING ENVIRONMENT...")
    print("-" * 30)
    
    env_info = {
        "DISPLAY": os.getenv('DISPLAY', 'None'),
        "SSH_CLIENT": "Yes" if os.getenv('SSH_CLIENT') else "No",
        "Container": "Yes" if os.path.exists('/.dockerenv') else "No",
        "HF_Spaces": "Yes" if os.getenv('SPACE_ID') else "No",
    }
    
    for key, value in env_info.items():
        print(f"🔍 {key}: {value}")
    
    # Determine expected mode
    headless_required = env_info["DISPLAY"] == "None" or env_info["SSH_CLIENT"] == "Yes"
    print(f"🎯 Recommended mode: {'Headless' if headless_required else 'Headless (for consistency)'}")
    print()
    
    # === STEP 4: Basic Helium Test ===
    print("🚀 TESTING HELIUM BROWSER START...")
    print("-" * 30)
    
    try:
        print("🔧 Attempting helium.start_chrome(headless=True)...")
        driver = helium.start_chrome(headless=True)
        
        print("✅ Browser started successfully!")
        
        # Test basic navigation
        try:
            current_url = driver.current_url
            print(f"✅ Current URL: {current_url}")
            
            # Test navigation to a simple page
            print("🔧 Testing navigation to example.com...")
            helium.go_to("https://example.com")
            
            # Wait a moment
            import time
            time.sleep(2)
            
            # Get page title
            title = driver.title
            print(f"✅ Page title: {title}")
            
            # Test screenshot capability
            print("🔧 Testing screenshot capability...")
            screenshot_data = driver.get_screenshot_as_png()
            print(f"✅ Screenshot captured: {len(screenshot_data)} bytes")
            
        except Exception as nav_error:
            print(f"⚠️ Navigation test failed: {nav_error}")
        
        # Clean up
        print("🧹 Cleaning up browser...")
        helium.kill_browser()
        print("✅ Browser closed successfully")
        
        print()
        print("🎉 SUCCESS! Browser automation is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Helium browser start failed: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        
        # === STEP 5: Fallback Test with Selenium ===
        print()
        print("🔧 TESTING FALLBACK: SELENIUM DIRECTLY...")
        print("-" * 30)
        
        try:
            from selenium import webdriver
            
            print("🔧 Attempting selenium webdriver.Chrome(headless=True)...")
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            driver = webdriver.Chrome(options=options)
            print("✅ Selenium Chrome driver started!")
            
            # Test basic functionality
            driver.get("https://example.com")
            title = driver.title
            print(f"✅ Selenium navigation works: {title}")
            
            driver.quit()
            print("✅ Selenium cleanup successful")
            
            print()
            print("🤔 MIXED RESULTS:")
            print("   ✅ Selenium works")
            print("   ❌ Helium has issues")
            print("   💡 This suggests a helium-specific configuration problem")
            
            return False
            
        except Exception as selenium_error:
            print(f"❌ Selenium also failed: {selenium_error}")
            print()
            print("💥 COMPLETE FAILURE:")
            print("   ❌ Both Helium and Selenium failed")
            print("   💡 This suggests a system-level browser/driver issue")
            
            return False

def print_recommendations():
    """Print recommendations based on diagnostic results"""
    print()
    print("🎯 RECOMMENDATIONS")
    print("=" * 50)
    print()
    print("If diagnostic PASSED:")
    print("  ✅ Your browser setup is working correctly")
    print("  ✅ VisionWebBrowserTool should work")
    print("  ✅ Use headless=True for consistent behavior")
    print()
    print("If diagnostic FAILED:")
    print("  🔧 Install missing dependencies:")
    print("     poetry add helium selenium")
    print("  🔧 Install Chrome browser if missing")
    print("  🔧 Try webdriver-manager for auto-driver management:")
    print("     poetry add webdriver-manager")
    print()
    print("For HF Spaces deployment:")
    print("  🚀 The headless configuration tested here will work")
    print("  🚀 Add Chrome installation to your Dockerfile")
    print()

if __name__ == "__main__":
    print("🧪 Starting browser diagnostic...")
    print()
    
    success = diagnose_browser_setup()
    print_recommendations()
    
    if success:
        print("🎉 DIAGNOSTIC PASSED - Ready for browser automation!")
        sys.exit(0)
    else:
        print("❌ DIAGNOSTIC FAILED - Check recommendations above")
        sys.exit(1)