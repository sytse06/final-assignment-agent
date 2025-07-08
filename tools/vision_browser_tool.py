import os
import time
import tempfile
import shutil
import json
from typing import Optional, Dict, List
from dataclasses import dataclass
from PIL import Image
from io import BytesIO
from pathlib import Path
import requests
from urllib.parse import urlparse

from smolagents import Tool

try:
    import helium
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

@dataclass
class ContentResult:
    content_type: str
    file_path: Optional[str] = None
    url: Optional[str] = None
    content_text: Optional[str] = None
    metadata: Dict = None
    handoff_instructions: str = ""
    success: bool = True
    error_message: str = ""
    screenshot_path: Optional[str] = None

class BrowserSession:
    def __init__(self, download_dir: Optional[str] = None):
        self.driver = None
        self.is_active = False
        self.download_dir = download_dir or tempfile.mkdtemp(prefix="browser_")
        self.downloaded_files = []
        self.screenshots = []
        
    def initialize(self, headless: bool = False):
        if self.is_active:
            return "Session already active"
            
        try:
            # 🔥 IMPROVED: Better Chrome options based on HF tutorial
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--force-device-scale-factor=1")
            chrome_options.add_argument("--window-size=1000,1350")
            chrome_options.add_argument("--window-position=0,0")  # NEW: Position control
            chrome_options.add_argument("--disable-pdf-viewer")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--disable-popup-blocking")  # NEW: Better popup handling
            
            prefs = {
                "download.default_directory": self.download_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True,
                "plugins.always_open_pdf_externally": True
            }
            chrome_options.add_experimental_option("prefs", prefs)
            
            if headless:
                chrome_options.add_argument("--headless")
            
            # 🔥 IMPROVED: Use helium's start_chrome directly
            self.driver = helium.start_chrome(headless=headless, options=chrome_options)
            self.is_active = True
            os.makedirs(self.download_dir, exist_ok=True)
            
            print(f"✅ Browser session initialized: {self.driver.current_url}")
            return "Session initialized"
            
        except Exception as e:
            return f"Initialization failed: {str(e)}"
    
    def take_screenshot(self, description: str = "screenshot") -> str:
        """🔥 NEW: Take screenshot with consistent naming"""
        try:
            if not self.is_active:
                return None
                
            # Let animations complete
            time.sleep(1.0)
            
            driver = helium.get_driver()
            png_bytes = driver.get_screenshot_as_png()
            
            timestamp = int(time.time())
            screenshot_path = os.path.join(self.download_dir, f"{description}_{timestamp}.png")
            
            with open(screenshot_path, 'wb') as f:
                f.write(png_bytes)
            
            self.screenshots.append(screenshot_path)
            print(f"📸 Screenshot saved: {screenshot_path}")
            
            return screenshot_path
            
        except Exception as e:
            print(f"⚠️ Screenshot failed: {e}")
            return None
    
    def detect_content_type(self) -> str:
        try:
            current_url = helium.get_driver().current_url.lower()
            page_source = helium.get_driver().page_source.lower()
            
            if current_url.endswith('.pdf') or 'content-type: application/pdf' in page_source:
                return "direct_pdf"
            
            if any(term in page_source for term in ['pdf', 'download pdf', 'view pdf', '.pdf"']):
                return "embedded_pdf"
            
            if '<table' in page_source:
                return "table_content"
            
            if any(term in page_source for term in ['<img', 'image', 'figure']):
                return "image_content"
            
            return "web_content"
            
        except Exception:
            return "unknown"
    
    def smart_click(self, element_text: str) -> bool:
        """🔥 NEW: Smart clicking with fallbacks based on HF tutorial"""
        try:
            # Try direct helium click first
            helium.click(element_text)
            return True
        except:
            try:
                # Try as link
                helium.click(helium.Link(element_text))
                return True
            except:
                try:
                    # Try as button
                    helium.click(helium.Button(element_text))
                    return True
                except:
                    return False
    
    def close_popups(self) -> str:
        """🔥 NEW: Close popups using HF tutorial method"""
        try:
            driver = helium.get_driver()
            webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(0.5)
            return "Popups closed with ESC key"
        except Exception as e:
            return f"Popup close failed: {str(e)}"
    
    def search_text_on_page(self, text: str, nth_result: int = 1) -> str:
        """🔥 NEW: Search text on page based on HF tutorial"""
        try:
            driver = helium.get_driver()
            elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
            
            if not elements:
                return f"No matches found for '{text}'"
            
            if nth_result > len(elements):
                return f"Match n°{nth_result} not found (only {len(elements)} matches found)"
            
            elem = elements[nth_result - 1]
            driver.execute_script("arguments[0].scrollIntoView(true);", elem)
            
            return f"Found {len(elements)} matches for '{text}'. Focused on element {nth_result} of {len(elements)}"
            
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    def scroll_page(self, direction: str = "down", pixels: int = 1200) -> str:
        """🔥 NEW: Scroll functionality based on HF tutorial"""
        try:
            if direction.lower() == "down":
                helium.scroll_down(pixels)
                return f"Scrolled down {pixels} pixels"
            elif direction.lower() == "up":
                helium.scroll_up(pixels)
                return f"Scrolled up {pixels} pixels"
            else:
                return "Invalid direction. Use 'up' or 'down'"
        except Exception as e:
            return f"Scroll failed: {str(e)}"
    
    def get_page_info(self) -> Dict:
        """🔥 NEW: Get comprehensive page information"""
        try:
            driver = helium.get_driver()
            return {
                "url": driver.current_url,
                "title": driver.title,
                "page_source_length": len(driver.page_source),
                "content_type": self.detect_content_type(),
                "timestamp": time.time()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def wait_for_download(self, timeout: int = 30) -> Optional[str]:
        initial_files = set(os.listdir(self.download_dir))
        
        for _ in range(timeout):
            time.sleep(1)
            current_files = set(os.listdir(self.download_dir))
            new_files = current_files - initial_files
            
            if new_files:
                for file in new_files:
                    if not file.endswith('.crdownload') and not file.endswith('.tmp'):
                        file_path = os.path.join(self.download_dir, file)
                        self.downloaded_files.append(file_path)
                        return file_path
        
        return None
    
    def extract_pdf_url(self) -> Optional[str]:
        try:
            driver = helium.get_driver()
            
            pdf_links = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf')]")
            if pdf_links:
                return pdf_links[0].get_attribute('href')
            
            current_url = driver.current_url
            if current_url.endswith('.pdf'):
                return current_url
            
            return None
            
        except Exception:
            return None
    
    def download_url_directly(self, url: str) -> Optional[str]:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            content_disposition = response.headers.get('content-disposition', '')
            if 'filename=' in content_disposition:
                filename = content_disposition.split('filename=')[1].strip('"')
            else:
                parsed = urlparse(url)
                filename = os.path.basename(parsed.path) or "downloaded_file"
                if not filename.endswith('.pdf') and 'pdf' in response.headers.get('content-type', ''):
                    filename += '.pdf'
            
            file_path = os.path.join(self.download_dir, filename)
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            self.downloaded_files.append(file_path)
            return file_path
            
        except Exception:
            return None
    
    def close(self):
        try:
            if self.is_active:
                helium.kill_browser()
                self.driver = None
                self.is_active = False
                return "Session closed"
            return "No active session"
        except Exception as e:
            return f"Error closing session: {str(e)}"
    
    def cleanup(self):
        try:
            if os.path.exists(self.download_dir):
                shutil.rmtree(self.download_dir)
            self.downloaded_files = []
            self.screenshots = []
        except Exception:
            pass

# Global session
_browser_session = BrowserSession()

class VisionWebBrowserTool(Tool):
    name = "vision_browser_tool"
    description = """🔥 IMPROVED: Navigate web pages with advanced interaction capabilities. 
    
    Based on SmolagAgents best practices for browser automation:
    - Navigate to web pages with automatic screenshots
    - Smart element clicking with multiple fallback strategies
    - Text search and scrolling capabilities
    - Popup handling and content extraction
    - File download management
    - Visual analysis with screenshots
    
    Actions: navigate, click_element, search_text, scroll_page, close_popups, take_screenshot, extract_content"""
    
    inputs = {
        "action": {
            "type": "string",
            "description": "Action to perform: 'navigate', 'click_element', 'search_text', 'scroll_page', 'close_popups', 'take_screenshot', 'extract_content', 'close_browser'",
        },
        "url": {
            "type": "string", 
            "description": "Target URL for navigation action.",
            "nullable": True,
        },
        "element_text": {
            "type": "string", 
            "description": "Text of element to click (for click_element action).",
            "nullable": True,
        },
        "search_text": {
            "type": "string",
            "description": "Text to search for on page (for search_text action).",
            "nullable": True,
        },
        "scroll_direction": {
            "type": "string",
            "description": "Direction to scroll: 'up' or 'down' (for scroll_page action).",
            "nullable": True,
        },
        "scroll_pixels": {
            "type": "integer",
            "description": "Number of pixels to scroll (default: 1200).",
            "nullable": True,
        },
        "nth_result": {
            "type": "integer",
            "description": "Which search result to focus on (default: 1).",
            "nullable": True,
        },
        "wait_seconds": {
            "type": "integer",
            "description": "Seconds to wait after action (default: 2).",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._session = None
        self._initialized = False
    
    def setup(self):
        """Initialize the browser session"""
        self._session = _browser_session
        self._initialized = True
    
    def forward(
        self, 
        action: str,
        url: Optional[str] = None,
        element_text: Optional[str] = None,
        search_text: Optional[str] = None,
        scroll_direction: Optional[str] = "down",
        scroll_pixels: Optional[int] = 1200,
        nth_result: Optional[int] = 1,
        wait_seconds: Optional[int] = 2
    ) -> str:
        
        if not DEPENDENCIES_AVAILABLE:
            return "Dependencies not available. Install: pip install helium selenium"
        
        try:
            # Initialize browser if needed
            if not _browser_session.is_active:
                init_result = _browser_session.initialize()
                if "failed" in init_result:
                    return init_result
            
            # Execute action
            if action == "navigate":
                result = self._navigate(url, wait_seconds)
                
            elif action == "click_element":
                result = self._click_element(element_text, wait_seconds)
                
            elif action == "search_text":
                result = self._search_text(search_text, nth_result)
                
            elif action == "scroll_page":
                result = self._scroll_page(scroll_direction, scroll_pixels)
                
            elif action == "close_popups":
                result = self._close_popups()
                
            elif action == "take_screenshot":
                result = self._take_screenshot()
                
            elif action == "extract_content":
                result = self._extract_content()
                
            elif action == "close_browser":
                result = ContentResult(
                    content_type="session",
                    handoff_instructions=_browser_session.close(),
                    success=True
                )
                
            else:
                result = ContentResult(
                    content_type="error",
                    success=False,
                    error_message=f"Unknown action: {action}"
                )
            
            return self._format_result(result)
            
        except Exception as e:
            return f"Browser error: {str(e)}"
    
    def _navigate(self, url: str, wait_seconds: int) -> ContentResult:
        """🔥 IMPROVED: Navigation with automatic screenshot"""
        if not url:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="URL required for navigation"
            )
        
        try:
            helium.go_to(url)
            time.sleep(wait_seconds)
            
            # Take automatic screenshot
            screenshot_path = _browser_session.take_screenshot("navigation")
            
            # Get page info
            page_info = _browser_session.get_page_info()
            
            return ContentResult(
                content_type="navigation",
                url=page_info.get("url"),
                screenshot_path=screenshot_path,
                content_text=f"Navigation complete. Title: {page_info.get('title', 'Unknown')}",
                metadata=page_info,
                handoff_instructions=f"Successfully navigated to {url}. Screenshot available for visual analysis.",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Navigation failed: {str(e)}"
            )
    
    def _click_element(self, element_text: str, wait_seconds: int) -> ContentResult:
        """🔥 IMPROVED: Smart clicking with fallbacks"""
        if not element_text:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="element_text required for clicking"
            )
        
        try:
            # Smart click with fallbacks
            click_success = _browser_session.smart_click(element_text)
            
            if click_success:
                time.sleep(wait_seconds)
                
                # Take screenshot after click
                screenshot_path = _browser_session.take_screenshot("after_click")
                page_info = _browser_session.get_page_info()
                
                return ContentResult(
                    content_type="click_result",
                    url=page_info.get("url"),
                    screenshot_path=screenshot_path,
                    content_text=f"Successfully clicked '{element_text}'",
                    metadata=page_info,
                    handoff_instructions=f"Clicked '{element_text}'. Check screenshot for result. Current URL: {page_info.get('url')}",
                    success=True
                )
            else:
                return ContentResult(
                    content_type="error",
                    success=False,
                    error_message=f"Could not find clickable element: '{element_text}'"
                )
                
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Click failed: {str(e)}"
            )
    
    def _search_text(self, search_text: str, nth_result: int) -> ContentResult:
        """🔥 NEW: Text search based on HF tutorial"""
        if not search_text:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="search_text required"
            )
        
        try:
            search_result = _browser_session.search_text_on_page(search_text, nth_result)
            screenshot_path = _browser_session.take_screenshot("search_result")
            
            return ContentResult(
                content_type="search_result",
                screenshot_path=screenshot_path,
                content_text=search_result,
                handoff_instructions=f"Search completed for '{search_text}'. Result: {search_result}",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Search failed: {str(e)}"
            )
    
    def _scroll_page(self, direction: str, pixels: int) -> ContentResult:
        """🔥 NEW: Page scrolling based on HF tutorial"""
        try:
            scroll_result = _browser_session.scroll_page(direction, pixels)
            screenshot_path = _browser_session.take_screenshot("after_scroll")
            
            return ContentResult(
                content_type="scroll_result",
                screenshot_path=screenshot_path,
                content_text=scroll_result,
                handoff_instructions=f"Scroll completed: {scroll_result}",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Scroll failed: {str(e)}"
            )
    
    def _close_popups(self) -> ContentResult:
        """🔥 NEW: Close popups based on HF tutorial"""
        try:
            close_result = _browser_session.close_popups()
            screenshot_path = _browser_session.take_screenshot("after_popup_close")
            
            return ContentResult(
                content_type="popup_close",
                screenshot_path=screenshot_path,
                content_text=close_result,
                handoff_instructions=f"Popup close attempted: {close_result}",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Popup close failed: {str(e)}"
            )
    
    def _take_screenshot(self) -> ContentResult:
        """🔥 NEW: Manual screenshot capture"""
        try:
            screenshot_path = _browser_session.take_screenshot("manual")
            page_info = _browser_session.get_page_info()
            
            return ContentResult(
                content_type="screenshot",
                screenshot_path=screenshot_path,
                url=page_info.get("url"),
                content_text=f"Screenshot captured of {page_info.get('title', 'current page')}",
                metadata=page_info,
                handoff_instructions=f"Manual screenshot captured. Available for visual analysis.",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Screenshot failed: {str(e)}"
            )
    
    def _extract_content(self) -> ContentResult:
        """🔥 IMPROVED: Extract page content with metadata"""
        try:
            driver = helium.get_driver()
            page_text = driver.find_element(By.TAG_NAME, "body").text
            page_info = _browser_session.get_page_info()
            
            # Truncate if too long
            if len(page_text) > 5000:
                page_text = page_text[:5000] + "... [truncated]"
            
            screenshot_path = _browser_session.take_screenshot("content_extract")
            
            return ContentResult(
                content_type="web_content",
                content_text=page_text,
                url=page_info.get("url"),
                screenshot_path=screenshot_path,
                metadata=page_info,
                handoff_instructions=f"Page content extracted from {page_info.get('title', 'current page')}. {len(page_text)} characters available.",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Content extraction failed: {str(e)}"
            )
    
    def _format_result(self, result: ContentResult) -> str:
        """🔥 IMPROVED: Enhanced result formatting"""
        result_dict = {
            "success": result.success,
            "content_type": result.content_type,
            "handoff_instructions": result.handoff_instructions,
        }
        
        if result.file_path:
            result_dict["file_path"] = result.file_path
        
        if result.url:
            result_dict["url"] = result.url
            
        if result.screenshot_path:
            result_dict["screenshot_path"] = result.screenshot_path
        
        if result.content_text:
            text = result.content_text[:500] + "..." if len(result.content_text) > 500 else result.content_text
            result_dict["content_text"] = text
        
        if result.metadata:
            result_dict["metadata"] = result.metadata
        
        if result.error_message:
            result_dict["error_message"] = result.error_message
        
        # Enhanced session info
        result_dict["session_info"] = {
            "files_downloaded": len(_browser_session.downloaded_files),
            "screenshots_taken": len(_browser_session.screenshots),
            "browser_active": _browser_session.is_active
        }
        
        return json.dumps(result_dict, indent=2)

def cleanup_browser_session():
    """Clean up browser session and files"""
    global _browser_session
    _browser_session.close()
    _browser_session.cleanup()
    return "Browser session cleaned up"

# 🔥 NEW: Additional utility functions for SmolagAgent integration
def get_browser_instructions() -> str:
    """Get Helium usage instructions for SmolagAgent integration"""
    return """
BROWSER AUTOMATION INSTRUCTIONS:

The vision_browser tool provides these actions:
- navigate: Go to a URL (automatically takes screenshot)
- click_element: Click on page elements by text
- search_text: Find and focus on text within the page  
- scroll_page: Scroll up or down by pixels
- close_popups: Close modal dialogs and popups
- take_screenshot: Capture current page state
- extract_content: Get page text content
- close_browser: Clean up session

USAGE PATTERNS:
1. Navigate first: {"action": "navigate", "url": "https://example.com"}
2. Interact: {"action": "click_element", "element_text": "Login"}  
3. Search: {"action": "search_text", "search_text": "Chicago", "nth_result": 1}
4. Extract: {"action": "extract_content"}

All actions automatically take screenshots for visual confirmation.
Use the screenshots and handoff_instructions to chain actions effectively.
"""