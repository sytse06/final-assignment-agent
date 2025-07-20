import os
import tempfile
import json
from typing import Optional, Dict
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
    Browser profile setup tool with multiple cookie authentication strategies.
    
    Supports cookie authentication from environment variables, browser extraction,
    or file-based cookies. Designed to work in both local development and 
    Docker/HF Spaces deployment environments.
    """
    
    name = "browser_profile"
    description = """Set up browser session with authentication support.

Initializes browser with persistent profile and cookie authentication.
Supports multiple cookie sources for deployment flexibility.

Usage: browser_profile("youtube") or browser_profile("linkedin")

Available profiles: youtube, linkedin, github, general
Automatically detects and uses available authentication methods.
"""
    
    inputs = {
        "profile_name": {
            "type": "string", 
            "description": "Platform profile: 'youtube', 'linkedin', 'github', 'general'",
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._profile_dir = self._get_profile_directory()
        self._active_browser = None
        self._temp_cookie_files = []
        os.makedirs(self._profile_dir, exist_ok=True)
        
    def _get_profile_directory(self) -> str:
        """Determine appropriate profile directory based on environment"""
        if self._is_container_environment():
            return "/tmp/browser_profiles"
        else:
            return os.path.expanduser("~/.gaia_browser_profiles")
    
    def _is_container_environment(self) -> bool:
        """Detect if running in container/cloud environment"""
        return (os.path.exists('/.dockerenv') or 
                'SPACE_ID' in os.environ or 
                'CONTAINER_NAME' in os.environ or
                'CODESPACE_NAME' in os.environ)
        
    def forward(self, profile_name: str) -> str:
        if not DEPENDENCIES_AVAILABLE:
            return "Error: Missing dependencies. Install: pip install undetected-chromedriver selenium"
        
        try:
            # Close existing browser if different profile
            if self._active_browser and hasattr(self._active_browser, 'current_profile'):
                if self._active_browser.current_profile != profile_name:
                    self._cleanup_browser()
                else:
                    return f"Browser already active with {profile_name} profile"
            
            # Try multiple cookie authentication strategies
            cookie_file = self._get_cookies_for_platform(profile_name)
            auth_method = "none"
            
            if cookie_file:
                if cookie_file.startswith("/tmp/"):
                    auth_method = "environment"
                else:
                    auth_method = "file"
            
            # Initialize browser with or without cookies
            result = self._initialize_browser(profile_name, cookie_file)
            
            if result:
                return f"Browser ready with {profile_name} profile. Authentication: {auth_method}"
            else:
                return f"Browser setup failed for {profile_name} profile"
                
        except Exception as e:
            return f"Browser setup failed: {str(e)}"
    
    def _get_cookies_for_platform(self, platform: str) -> Optional[str]:
        """Get cookies using multiple strategies"""
        
        # Strategy 1: Environment variable cookies (HF Spaces/Docker)
        env_cookies = self._get_cookies_from_environment(platform)
        if env_cookies:
            return env_cookies
        
        # Strategy 2: File-based cookies (CI/CD)
        file_cookies = self._get_cookies_from_file(platform)
        if file_cookies:
            return file_cookies
        
        # Strategy 3: Browser extraction (local development)
        browser_cookies = self._get_cookies_from_browser(platform)
        if browser_cookies:
            return browser_cookies
        
        return None
    
    def _get_cookies_from_environment(self, platform: str) -> Optional[str]:
        """Extract cookies from environment variables"""
        try:
            env_var_name = f"{platform.upper()}_COOKIES"
            cookie_data = os.environ.get(env_var_name)
            
            if not cookie_data:
                return None
            
            # Create temporary cookies file
            temp_dir = tempfile.gettempdir()
            cookies_path = os.path.join(temp_dir, f'{platform}_cookies_{os.getpid()}.txt')
            
            with open(cookies_path, 'w') as f:
                f.write(cookie_data)
            
            # Track for cleanup
            self._temp_cookie_files.append(cookies_path)
            
            return cookies_path
            
        except Exception:
            return None
    
    def _get_cookies_from_file(self, platform: str) -> Optional[str]:
        """Get cookies from predefined file locations"""
        try:
            # Common cookie file locations
            possible_paths = [
                f"/app/cookies/{platform}_cookies.txt",
                f"./cookies/{platform}_cookies.txt",
                f"~/.gaia_cookies/{platform}_cookies.txt",
                f"/secrets/{platform}_cookies.txt"
            ]
            
            for path in possible_paths:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path) and os.path.getsize(expanded_path) > 0:
                    return expanded_path
            
            return None
            
        except Exception:
            return None
    
    def _get_cookies_from_browser(self, platform: str) -> Optional[str]:
        """Extract cookies from local browser (development only)"""
        try:
            # Only attempt in non-container environments
            if self._is_container_environment():
                return None
            
            # This would require browser cookie extraction
            # For now, return None - can be implemented with browser_cookie3 library
            return None
            
        except Exception:
            return None
    
    def _initialize_browser(self, profile_name: str, cookie_file: Optional[str] = None) -> bool:
        """Initialize browser with profile and optional cookies"""
        try:
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
            
            # Container-specific options
            if self._is_container_environment():
                options.add_argument("--headless")
                options.add_argument("--disable-gpu")
                options.add_argument("--disable-software-rasterizer")
            
            # Initialize browser
            self._active_browser = uc.Chrome(options=options, version_main=None)
            self._active_browser.current_profile = profile_name
            self._active_browser.implicitly_wait(10)
            
            # Load cookies if available
            if cookie_file and os.path.exists(cookie_file):
                self._load_cookies_into_browser(cookie_file)
            
            # Set up helium integration
            if HELIUM_AVAILABLE:
                helium.set_driver(self._active_browser)
            
            return True
            
        except Exception:
            return False
    
    def _load_cookies_into_browser(self, cookie_file: str):
        """Load cookies from file into browser session"""
        try:
            # Navigate to base domain first
            current_profile = getattr(self._active_browser, 'current_profile', 'general')
            
            if current_profile == "youtube":
                self._active_browser.get("https://youtube.com")
            elif current_profile == "linkedin":
                self._active_browser.get("https://linkedin.com")
            elif current_profile == "github":
                self._active_browser.get("https://github.com")
            else:
                self._active_browser.get("https://google.com")
            
            # Parse and add cookies
            with open(cookie_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            parts = line.split('\t')
                            if len(parts) >= 7:
                                domain = parts[0]
                                name = parts[5]
                                value = parts[6]
                                
                                # Add cookie if domain matches
                                if any(d in domain for d in ['.youtube.com', '.linkedin.com', '.github.com', '.google.com']):
                                    cookie_dict = {
                                        'name': name,
                                        'value': value,
                                        'domain': domain
                                    }
                                    self._active_browser.add_cookie(cookie_dict)
                        except:
                            continue
            
        except Exception:
            pass
    
    def _cleanup_browser(self):
        """Clean up browser session"""
        if self._active_browser:
            try:
                self._active_browser.quit()
            except:
                pass
            self._active_browser = None
    
    def cleanup(self):
        """Clean up browser session and temporary files"""
        self._cleanup_browser()
        
        # Clean up temporary cookie files
        for temp_file in self._temp_cookie_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        self._temp_cookie_files.clear()

def get_authenticated_browser_instructions() -> str:
    """
    Browser automation instructions for CodeAgent with authentication support.
    
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

DEPLOYMENT COOKIE SETUP:

For HF Spaces deployment, set environment variables:
- YOUTUBE_COOKIES: Netscape format cookies for YouTube
- LINKEDIN_COOKIES: Netscape format cookies for LinkedIn  
- GITHUB_COOKIES: Netscape format cookies for GitHub

Export cookies from browser using browser extensions or developer tools.
Tool automatically detects and uses available cookie sources.

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
- Tool automatically adapts to deployment environment (local vs Docker)
"""

def get_cookie_export_instructions() -> str:
    """Instructions for exporting cookies for deployment"""
    return """
COOKIE EXPORT FOR DEPLOYMENT

LOCAL DEVELOPMENT:
1. Log into platforms manually in Chrome
2. Browser profiles automatically save authentication state
3. No additional setup required

HF SPACES / DOCKER DEPLOYMENT:
1. Export cookies from authenticated browser sessions:
   - Install browser extension: "Get cookies.txt" or "cookies.txt"
   - Visit youtube.com, linkedin.com, github.com (while logged in)
   - Export cookies in Netscape format for each platform

2. Set environment variables in HF Spaces:
   - YOUTUBE_COOKIES: Contents of youtube cookies.txt file
   - LINKEDIN_COOKIES: Contents of linkedin cookies.txt file  
   - GITHUB_COOKIES: Contents of github cookies.txt file

3. Deploy application:
   - Tool automatically detects environment variables
   - Creates temporary cookie files for browser sessions
   - Falls back to public access if no cookies available

COOKIE FILE FORMAT:
Netscape HTTP Cookie File format:
# domain    flag    path    secure    expiration    name    value
.youtube.com    TRUE    /    FALSE    1234567890    session_token    abc123...

Tool supports multiple cookie sources and automatically selects best available option.
"""