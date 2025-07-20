import os
import tempfile
from typing import Optional
from smolagents import Tool

try:
    import undetected_chromedriver as uc
    from selenium import webdriver
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

try:
    import helium
    HELIUM_AVAILABLE = True
except ImportError:
    HELIUM_AVAILABLE = False

class BrowserProfileTool(Tool):
    """
    Simple browser profile setup tool for authenticated web access.
    
    Sets up undetected Chrome browser with persistent profile directory
    that preserves login sessions across tasks. Does not handle any
    browser interactions - those are done via helium commands.
    """
    
    name = "browser_profile"
    description = """Set up authenticated browser session with saved profile.

Initializes browser with persistent profile directory that maintains
login sessions for platforms like YouTube, LinkedIn, GitHub.

Usage: browser_profile("youtube") or browser_profile("linkedin")

Available profiles: youtube, linkedin, github, general
Browser will load with saved cookies and authentication state.
All subsequent browser interactions use helium commands directly.
"""
    
    inputs = {
        "profile_name": {
            "type": "string", 
            "description": "Profile name: 'youtube', 'linkedin', 'github', 'general'",
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._profile_dir = os.path.expanduser("~/.gaia_browser_profiles")
        self._screenshot_dir = tempfile.mkdtemp(prefix="gaia_screenshots_")
        os.makedirs(self._profile_dir, exist_ok=True)
        self._active_browser = None
        
    def forward(self, profile_name: str) -> str:
        if not DEPENDENCIES_AVAILABLE:
            return "Error: Missing dependencies. Install: pip install undetected-chromedriver selenium"
        
        try:
            # Close existing browser if different profile
            if self._active_browser and hasattr(self._active_browser, 'current_profile'):
                if self._active_browser.current_profile != profile_name:
                    self._active_browser.quit()
                    self._active_browser = None
                else:
                    return f"Browser already active with {profile_name} profile"
            
            # Create profile directory
            profile_path = os.path.join(self._profile_dir, profile_name)
            os.makedirs(profile_path, exist_ok=True)
            
            # Configure Chrome options
            options = uc.ChromeOptions()
            options.add_argument(f"--user-data-dir={profile_path}")
            options.add_argument("--no-first-run")
            options.add_argument("--no-default-browser-check")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--no-sandbox")
            options.add_argument("--window-size=1920,1080")
            
            # Platform-specific optimizations
            if profile_name == "linkedin":
                options.add_experimental_option("useAutomationExtension", False)
                options.add_experimental_option("excludeSwitches", ["enable-automation"])
            elif profile_name == "youtube":
                options.add_argument("--disable-features=VizDisplayCompositor")
            
            # Initialize browser
            self._active_browser = uc.Chrome(options=options, version_main=None)
            self._active_browser.current_profile = profile_name
            self._active_browser.implicitly_wait(10)
            
            # Set up helium integration
            if HELIUM_AVAILABLE:
                helium.set_driver(self._active_browser)
            
            return f"Browser ready with {profile_name} profile. Saved sessions and cookies loaded."
            
        except Exception as e:
            return f"Browser setup failed: {str(e)}"
    
    def cleanup(self):
        """Clean up browser session"""
        if self._active_browser:
            try:
                self._active_browser.quit()
            except:
                pass
            self._active_browser = None

def get_authenticated_browser_instructions() -> str:
    """
    Complete browser automation instructions for CodeAgent with authentication support.
    
    Teaches CodeAgent how to handle authentication, browser automation, and 
    complex web workflows using helium commands and browser profiles.
    """
    return """
AUTHENTICATED BROWSER AUTOMATION INSTRUCTIONS

AUTHENTICATION WORKFLOW:
Use browser_profile() to set up authenticated sessions, then helium commands for all interactions.

PROFILE SETUP:
```py
browser_profile("youtube")    # For YouTube content, age-restricted videos
browser_profile("linkedin")   # For LinkedIn profiles, company data
browser_profile("github")     # For GitHub repositories, private access
browser_profile("general")    # For standard web browsing
```

AUTHENTICATION DETECTION AND HANDLING:
```py
# Check authentication status
go_to('youtube.com/watch?v=restricted_video')
if Text('Sign in').exists():
    click('Sign in')
    # Browser automatically uses saved credentials from profile
    wait_until(Text('Your account').exists)

# Age verification workflow
if Text('The following content has been identified').exists():
    scroll_down(400)
    if Text('I understand and wish to proceed').exists():
        click('I understand and wish to proceed')
        
# Cookie consent handling
if Text('Accept cookies').exists():
    click('Accept all')
elif Text('Cookie consent').exists():
    press(ESC)  # Close cookie banner
```

PLATFORM-SPECIFIC AUTHENTICATION PATTERNS:

YOUTUBE AUTHENTICATION:
```py
browser_profile("youtube")
go_to('youtube.com/watch?v=video_id')

# Handle sign-in
if Text('Sign in').exists():
    click('Sign in')
    wait_until(Text('Library').exists)  # Wait for authenticated state

# Handle age verification
if Text('I understand and wish to proceed').exists():
    click('I understand and wish to proceed')

# Handle restricted content
if Text('This video is not available').exists():
    # Try refreshing with authenticated session
    refresh()
    wait_until(lambda: not Text('Loading').exists())
```

LINKEDIN AUTHENTICATION:
```py
browser_profile("linkedin")
go_to('linkedin.com/company/company-name')

# Check if login required
if Text('Sign in').exists():
    click('Sign in')
    wait_until(Text('My Network').exists)  # Wait for authenticated navbar

# Handle private profiles
if Text('You need to sign in').exists():
    refresh()  # Reload with authenticated session
    wait_until(Text('About').exists)
```

GITHUB AUTHENTICATION:
```py
browser_profile("github")
go_to('github.com/org/private-repo')

# Handle private repository access
if Text('This repository is private').exists():
    # Profile should have access - refresh if needed
    refresh()
    wait_until(Text('Code').exists)

# Handle organization access
if Text('You need to be a member').exists():
    # Check if profile has organization access
    go_to('github.com/settings/organizations')
```

MULTI-PLATFORM WORKFLOWS:
```py
# Research workflow across platforms
browser_profile("linkedin")
go_to('linkedin.com/company/openai')
company_data = extract_company_info()

browser_profile("github")
go_to('github.com/openai')
code_data = extract_repository_info()

browser_profile("youtube")
go_to('youtube.com/c/openai')
video_data = extract_channel_info()
```

AUTHENTICATION ERROR HANDLING:
```py
# Robust authentication with retries
def authenticate_platform(platform, url):
    browser_profile(platform)
    go_to(url)
    
    # Check if authentication worked
    if Text('Sign in').exists():
        click('Sign in')
        time.sleep(3)
        
        # Verify authentication success
        if Text('Sign in').exists():
            # Authentication failed - may need manual intervention
            return False
    return True

# Usage
if not authenticate_platform("youtube", "youtube.com/watch?v=xyz"):
    # Handle authentication failure
    print("Authentication required - check profile setup")
```

SESSION MANAGEMENT:
```py
# Switch between platforms within same task
browser_profile("linkedin")
linkedin_data = research_company_on_linkedin()

browser_profile("youtube") 
youtube_data = research_company_videos()

# Combine data from multiple authenticated sources
combined_research = merge_platform_data(linkedin_data, youtube_data)
```

GENERAL BROWSER AUTOMATION (works with any profile):
```py
# Navigation
go_to('example.com')
refresh()
go_back()

# Element interaction
click("Button Text")
click(Link("Link Text"))
write("text", into="Field Name")
select("Option", from_="Dropdown")

# Conditional logic
if Text('Element exists').exists():
    click('Element exists')
else:
    scroll_down(500)
    click('Alternative element')

# Form handling
write("search query", into="Search")
select("United States", from_="Country")
press(ENTER)

# Page navigation
scroll_down(1200)
scroll_up(800)
press(PAGE_DOWN)
press(ESC)  # Close modals

# Waiting and timing
wait_until(Text('Results loaded').exists)
time.sleep(2)  # Explicit wait when needed
```

ADVANCED PATTERNS:
```py
# Dynamic authentication detection
def ensure_authenticated(platform):
    if Text('Sign in').exists() or Text('Log in').exists():
        browser_profile(platform)
        refresh()
        wait_until(lambda: not Text('Sign in').exists())

# Platform-specific data extraction
def extract_youtube_transcript(video_url):
    browser_profile("youtube")
    go_to(video_url)
    ensure_authenticated("youtube")
    
    if Text('Show transcript').exists():
        click('Show transcript')
        wait_until(Text('Transcript').exists)
        return get_transcript_text()
    
# Error recovery with profile switching
def robust_platform_access(platform, url, max_retries=3):
    for attempt in range(max_retries):
        try:
            browser_profile(platform)
            go_to(url)
            
            if not Text('Sign in').exists():
                return True  # Successfully authenticated
                
            time.sleep(2)  # Wait between retries
        except:
            continue
    return False  # Authentication failed
```

SCREENSHOT AND DEBUGGING:
```py
# Take screenshots for debugging (browser_profile sets up screenshot capability)
import time
timestamp = int(time.time())
screenshot_path = f"/tmp/debug_{timestamp}.png"
# Browser automatically captures state

# Debug authentication issues
def debug_auth_state():
    if Text('Sign in').exists():
        print("Not authenticated - sign in button present")
    elif Text('Account').exists() or Text('Profile').exists():
        print("Authenticated - user elements present")
    else:
        print("Authentication state unclear")
```

BEST PRACTICES:
1. Always use browser_profile() before navigating to authenticated content
2. Check for sign-in indicators and handle them programmatically
3. Use wait_until() for dynamic content loading after authentication
4. Handle platform-specific authentication flows (age verification, 2FA, etc.)
5. Write retry logic for failed authentication attempts
6. Combine multiple authenticated sources in single workflows
7. Use refresh() to reload pages with updated authentication state

REMEMBER:
- browser_profile() sets up authentication - helium commands do the work
- Each platform has different authentication indicators to check
- Profiles preserve login state between tasks and browser restarts
- Always verify authentication worked before proceeding with task
- Write robust error handling for authentication failures
"""