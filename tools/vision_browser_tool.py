# tools/vision_web_browser_tool.py

import time
from io import BytesIO
from typing import Optional, Dict, Any
import logging

import helium
import PIL.Image
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import WebDriverException

from smolagents import Tool

logger = logging.getLogger(__name__)

# Helium instructions for agentic browser automation
HELIUM_INSTRUCTIONS = """
Use your web_search tool when you want to get Google search results.
Then you can use helium to access websites. Don't use helium for Google search, only for navigating websites!
Don't bother about the helium driver, it's already managed.
We've already ran "from helium import *"
Then you can go to pages!
<code>
go_to('github.com/trending')
</code>

You can directly click clickable elements by inputting the text that appears on them.
<code>
click("Top products")
</code>

If it's a link:
<code>
click(Link("Top products"))
</code>

If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your screenshot.
Never try to login in a page.

To scroll up or down, use scroll_down or scroll_up with as an argument the number of pixels to scroll from.
<code>
scroll_down(num_pixels=1200) # This will scroll one viewport down
</code>

When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
<code>
close_popups()
</code>

You can use .exists() to check for the existence of an element. For example:
<code>
if Text('Accept cookies?').exists():
    click('I accept')
</code>

Proceed in several steps rather than trying to solve the task in one shot.
And at the end, only when you have your answer, return your final answer.
<code>
final_answer("YOUR_ANSWER_HERE")
</code>

If pages seem stuck on loading, you might have to wait, for instance `import time` and run `time.sleep(5.0)`. But don't overuse this!
To list elements on page, DO NOT try code-based element searches like 'contributors = find_all(S("ol > li"))': just look at the latest screenshot you have and read it visually, or use your tool search_item_ctrl_f.
Of course, you can act on buttons like a user would do when navigating.
After each code blob you write, you will be automatically provided with an updated screenshot of the browser and the current browser url.
But beware that the screenshot will only be taken at the end of the whole action, it won't see intermediate states.
Don't kill the browser.
When you have modals or cookie banners on screen, you should get rid of them before you can click anything else.
"""

class VisionBrowserTool(Tool):
    """
    Vision-enabled browser tool that provides both:
    Agentic helium environment setup for CodeAgent direct access
    """
    
    name = "vision_browser"
    description = "Vision browser with agentic helium support and standard browsing methods"
    
    inputs = {
        "action": {
            "type": "string",
            "description": "Action: 'setup_agentic' for helium environment, or standard actions: 'navigate_to', 'click_element', 'scroll_page', 'close_popups', 'get_page_info'"
        },
        "parameters": {
            "type": "object", 
            "description": "Parameters for the action (e.g., {'url': 'https://example.com'})"
        }
    }
    
    output_type = "string"
    
    def __init__(self, headless: bool = False, window_size: tuple = (1000, 1350)):
        super().__init__()
        self.headless = headless
        self.window_size = window_size
        self.driver: Optional[webdriver.Chrome] = None
        self.is_initialized = False
    
    def forward(self, action: str, parameters: Dict[str, Any] = None) -> str:
        """Execute browser action."""
        if parameters is None:
            parameters = {}
        
        if action == "setup_agentic":
            return self._setup_agentic_environment()
        
        # Ensure browser is initialized for standard actions
        self._ensure_initialized()
        
        if action == "navigate_to":
            return self._navigate_to(parameters.get("url", ""))
        elif action == "click_element":
            return self._click_element(parameters.get("text", ""))
        elif action == "scroll_page":
            direction = parameters.get("direction", "down")
            pixels = parameters.get("pixels", 1200)
            return self._scroll_page(direction, pixels)
        elif action == "close_popups":
            return self._close_popups()
        elif action == "get_page_info":
            return self._get_page_info()
        else:
            return f"Unknown action: {action}. Available: setup_agentic, navigate_to, click_element, scroll_page, close_popups, get_page_info"
    
    def _setup_agentic_environment(self) -> str:
        """Initialize browser and return helium instructions for agentic use."""
        self._ensure_initialized()
        
        current_url = "about:blank"
        try:
            current_url = self.driver.current_url
        except:
            pass
        
        return f"""✅ AGENTIC HELIUM BROWSER READY!

🌐 Current page: {current_url}
🖼️ Window size: {self.window_size}

{HELIUM_INSTRUCTIONS}"""
    
    def _ensure_initialized(self):
        """Ensure browser is initialized."""
        if self.is_initialized and self.driver:
            try:
                _ = self.driver.current_url
                return
            except WebDriverException:
                pass
        
        self._initialize_browser()
    
    def _initialize_browser(self):
        """Initialize Chrome browser with helium."""
        try:
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--force-device-scale-factor=1")
            chrome_options.add_argument(f"--window-size={self.window_size[0]},{self.window_size[1]}")
            chrome_options.add_argument("--disable-pdf-viewer")
            chrome_options.add_argument("--window-position=0,0")
            
            if self.headless:
                chrome_options.add_argument("--headless")
            
            self.driver = helium.start_chrome(headless=self.headless, options=chrome_options)
            self.is_initialized = True
            
        except Exception as e:
            raise Exception(f"Browser initialization failed: {e}")
    
    def _navigate_to(self, url: str) -> str:
        """Navigate to URL."""
        if not url:
            return "Error: URL required"
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        try:
            helium.go_to(url)
            time.sleep(2)
            current_url = self.driver.current_url
            title = self.driver.title
            return f"✅ Navigated to {current_url} - Title: {title}"
        except Exception as e:
            return f"❌ Navigation failed: {e}"
    
    def _click_element(self, text: str) -> str:
        """Click element by text."""
        if not text:
            return "Error: Text required"
        
        try:
            helium.click(text)
            return f"✅ Clicked: '{text}'"
        except Exception as e:
            try:
                helium.click(helium.Link(text))
                return f"✅ Clicked link: '{text}'"
            except Exception as e2:
                return f"❌ Click failed for '{text}': {e2}"
    
    def _scroll_page(self, direction: str, pixels: int) -> str:
        """Scroll page up or down."""
        try:
            if direction.lower() == "down":
                helium.scroll_down(num_pixels=pixels)
                return f"✅ Scrolled down {pixels}px"
            elif direction.lower() == "up":
                helium.scroll_up(num_pixels=pixels)
                return f"✅ Scrolled up {pixels}px"
            else:
                return f"❌ Invalid direction: {direction}"
        except Exception as e:
            return f"❌ Scroll failed: {e}"
    
    def _close_popups(self) -> str:
        """Close popups using ESC key."""
        try:
            webdriver.ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(1)
            return "✅ Closed popups with ESC"
        except Exception as e:
            return f"❌ Close popups failed: {e}"
    
    def _get_page_info(self) -> str:
        """Get current page info."""
        try:
            url = self.driver.current_url
            title = self.driver.title
            return f"📄 {title} | {url}"
        except Exception as e:
            return f"❌ Get page info failed: {e}"
    
    def cleanup(self):
        """Clean up browser."""
        try:
            if self.driver and self.is_initialized:
                helium.kill_browser()
                self.driver = None
                self.is_initialized = False
        except Exception:
            pass

# Individual tools that CodeAgent needs as callable functions
@tool
def close_popups() -> str:
    """Close any visible modal or pop-up using ESC key. Use this instead of trying to click X buttons."""
    try:
        from selenium import webdriver
        from selenium.webdriver.common.keys import Keys
        import helium
        
        driver = helium.get_driver()
        if driver:
            webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
            return "✅ Closed popups with ESC key"
        else:
            return "❌ No browser driver available"
    except Exception as e:
        return f"❌ Close popups failed: {e}"

@tool  
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Search for text on current page and jump to nth occurrence.
    Args:
        text: Text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    try:
        import helium
        from selenium.webdriver.common.by import By
        
        driver = helium.get_driver()
        if not driver:
            return "❌ No browser driver available"
            
        elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
        
        if nth_result > len(elements):
            return f"❌ Match n°{nth_result} not found (only {len(elements)} matches found)"
        
        if len(elements) == 0:
            return f"❌ No matches found for '{text}'"
        
        # Scroll to the nth element
        elem = elements[nth_result - 1]
        driver.execute_script("arguments[0].scrollIntoView(true);", elem)
        
        return f"✅ Found {len(elements)} matches for '{text}'. Focused on element {nth_result} of {len(elements)}"
        
    except Exception as e:
        return f"❌ Search failed: {e}"

# Setup function for agentic browser use
def setup_agentic_browser(agent, headless: bool = False) -> str:
    """
    Set up CodeAgent for agentic browser automation with required tools.
    
    Usage:
        instructions = setup_agentic_browser(web_researcher)
        result = agent.run(f"{task}\n\n{instructions}")
    """
    try:
        # Initialize browser
        browser_tool = VisionWebBrowserTool(headless=headless)
        browser_tool._ensure_initialized()
        
        # Import helium into agent's environment
        agent.python_executor("from helium import *")
        agent.python_executor("import time")
        
        # Add the required tools to agent if not already present
        if close_popups not in agent.tools:
            agent.tools.append(close_popups)
        if search_item_ctrl_f not in agent.tools:
            agent.tools.append(search_item_ctrl_f)
        
        current_url = "about:blank"
        try:
            current_url = browser_tool.driver.current_url
        except:
            pass
        
        return f"""✅ AGENTIC HELIUM BROWSER READY!

🌐 Current page: {current_url}
🔧 Tools available: close_popups(), search_item_ctrl_f()

{HELIUM_INSTRUCTIONS}"""
        
    except Exception as e:
        return f"❌ Agentic browser setup failed: {e}"

# Get vision browser tools for CodeAgent
def get_vision_browser_tools():
    """Get the individual tools needed for CodeAgent vision browser automation."""
    return [close_popups, search_item_ctrl_f]