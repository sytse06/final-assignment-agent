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
    from selenium.webdriver.support.ui import WebDriverWait, Select
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
        
        self.workflow_state = {
            "current_site": None,
            "navigation_history": [],
            "form_interactions": [],
            "search_history": [],
            "error_count": 0,
            "retry_strategies": [],
            "extracted_data": {},
            "performance_metrics": {
                "total_actions": 0,
                "successful_actions": 0,
                "average_response_time": 0
            }
        }
        
    def initialize(self, headless: bool = False):
        if self.is_active:
            return "Session already active"
            
        try:
            # Better Chrome options based on HF tutorial
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--force-device-scale-factor=1")
            chrome_options.add_argument("--window-size=1200,1000")  # Slightly larger for better visibility
            chrome_options.add_argument("--window-position=0,0")
            chrome_options.add_argument("--disable-pdf-viewer")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--disable-popup-blocking")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # NEW: Avoid detection
            chrome_options.add_argument("--no-sandbox")  # NEW: Better compatibility
            
            prefs = {
                "download.default_directory": self.download_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True,
                "plugins.always_open_pdf_externally": True,
                "profile.default_content_setting_values.notifications": 2,
                "profile.default_content_settings.popups": 0,
                "profile.password_manager_enabled": False,
                "credentials_enable_service": False,
            }
            chrome_options.add_experimental_option("prefs", prefs)
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            if headless:
                chrome_options.add_argument("--headless")
            
            # Try to use environment functions first
            try:
                if hasattr(globals(), 'start_chrome'):
                    self.driver = start_chrome()
                else:
                    # Fallback to direct helium with options
                    self.driver = helium.start_chrome(headless=headless, options=chrome_options)
            except:
                # Final fallback to direct helium
                self.driver = helium.start_chrome(headless=headless, options=chrome_options)
            
            self.is_active = True
            os.makedirs(self.download_dir, exist_ok=True)
            
            # Set implicit wait for better element detection
            self.driver.implicitly_wait(3)
            
            print(f"✅ Enhanced browser session initialized: {self.driver.current_url}")
            return "Enhanced session initialized with text input capabilities"
            
        except Exception as e:
            return f"Initialization failed: {str(e)}"
    
    def take_screenshot(self, description: str = "screenshot") -> str:
        """Take screenshot with consistent naming and metadata"""
        try:
            if not self.is_active:
                return None
                
            time.sleep(1.0)  # Let animations complete
            
            driver = helium.get_driver()
            png_bytes = driver.get_screenshot_as_png()
            
            timestamp = int(time.time())
            screenshot_path = os.path.join(self.download_dir, f"{description}_{timestamp}.png")
            
            with open(screenshot_path, 'wb') as f:
                f.write(png_bytes)
            
            self.screenshots.append({
                "path": screenshot_path,
                "description": description,
                "timestamp": timestamp,
                "url": driver.current_url,
                "title": driver.title
            })
            
            print(f"📸 Screenshot saved: {screenshot_path}")
            return screenshot_path
            
        except Exception as e:
            print(f"⚠️ Screenshot failed: {e}")
            return None
    
    def detect_content_type(self) -> str:
        """Content type detection"""
        try:
            current_url = helium.get_driver().current_url.lower()
            page_source = helium.get_driver().page_source.lower()
            
            if "github.com" in current_url:
                if "/issues" in current_url:
                    return "github_issues"
                elif "/pull" in current_url:
                    return "github_pull_request"
                else:
                    return "github_content"
            
            if "stackoverflow.com" in current_url:
                return "stackoverflow_content"
            
            if any(doc_term in current_url for doc_term in ["docs", "documentation", "wiki"]):
                return "documentation"
            
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
        """Smart clicking with fallbacks based on HF tutorial"""
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
        """Close popups using HF tutorial method"""
        try:
            driver = helium.get_driver()
            webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(0.5)
            return "Popups closed with ESC key"
        except Exception as e:
            return f"Popup close failed: {str(e)}"
    
    def search_text_on_page(self, text: str, nth_result: int = 1) -> str:
        """Search text on page based on HF tutorial"""
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
    
    def get_page_info(self) -> Dict:
        """Optimal page information"""
        try:
            driver = helium.get_driver()
            return {
                "url": driver.current_url,
                "title": driver.title,
                "page_source_length": len(driver.page_source),
                "content_type": self.detect_content_type(),
                "timestamp": time.time(),
                "viewport_size": driver.execute_script("return {width: window.innerWidth, height: window.innerHeight};"),
                "scroll_position": driver.execute_script("return {x: window.scrollX, y: window.scrollY};")
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _track_action(self, action_type: str, result: str, duration: float, success: bool):
        """Track action performance"""
        self.workflow_state["performance_metrics"]["total_actions"] += 1
        if success:
            self.workflow_state["performance_metrics"]["successful_actions"] += 1
        
        # Update average response time
        current_avg = self.workflow_state["performance_metrics"]["average_response_time"]
        total_actions = self.workflow_state["performance_metrics"]["total_actions"]
        new_avg = ((current_avg * (total_actions - 1)) + duration) / total_actions
        self.workflow_state["performance_metrics"]["average_response_time"] = new_avg
        
        # Log the action
        action_log = {
            "action_type": action_type,
            "result": result,
            "duration": duration,
            "success": success,
            "timestamp": time.time(),
            "url": self.driver.current_url if self.is_active else None
        }
        
        if action_type not in self.workflow_state:
            self.workflow_state[action_type] = []
        self.workflow_state[action_type].append(action_log)
    
    def get_workflow_summary(self) -> Dict:
        """Get workflow performance summary"""
        metrics = self.workflow_state["performance_metrics"]
        
        success_rate = 0
        if metrics["total_actions"] > 0:
            success_rate = metrics["successful_actions"] / metrics["total_actions"]
        
        return {
            "total_actions": metrics["total_actions"],
            "successful_actions": metrics["successful_actions"],
            "success_rate": f"{success_rate:.2%}",
            "average_response_time": f"{metrics['average_response_time']:.2f}s",
            "sites_visited": len(set(log.get("url", "") for actions in self.workflow_state.values() 
                                   if isinstance(actions, list) for log in actions)),
            "screenshots_taken": len(self.screenshots),
            "searches_performed": len(self.workflow_state["search_history"]),
            "current_site": self.workflow_state["current_site"]
        }
    
    def scroll_page(self, direction: str = "down", pixels: int = 1200) -> str:
        """Page scrolling with better control"""
        try:
            if direction.lower() == "down":
                helium.scroll_down(pixels)
                result = f"Scrolled down {pixels} pixels"
            elif direction.lower() == "up":
                helium.scroll_up(pixels)
                result = f"Scrolled up {pixels} pixels"
            elif direction.lower() == "top":
                helium.get_driver().execute_script("window.scrollTo(0, 0);")
                result = "Scrolled to top of page"
            elif direction.lower() == "bottom":
                helium.get_driver().execute_script("window.scrollTo(0, document.body.scrollHeight);")
                result = "Scrolled to bottom of page"
            else:
                result = "Invalid direction. Use 'up', 'down', 'top', or 'bottom'"
                
            self._track_action("scroll_page", result, 0, True)
            return result
            
        except Exception as e:
            result = f"Scroll failed: {str(e)}"
            self._track_action("scroll_page", result, 0, False)
            return result
    
    def search_text_on_page(self, text: str, nth_result: int = 1) -> str:
        """Enhanced text search with better element detection"""
        try:
            start_time = time.time()
            driver = helium.get_driver()
            
            # Use Ctrl+F to open browser search
            webdriver.ActionChains(driver).key_down(Keys.CONTROL).send_keys('f').key_up(Keys.CONTROL).perform()
            time.sleep(0.5)
            
            # Type the search term
            webdriver.ActionChains(driver).send_keys(text).perform()
            time.sleep(0.5)
            
            # Navigate to nth result if needed
            if nth_result > 1:
                for _ in range(nth_result - 1):
                    webdriver.ActionChains(driver).send_keys(Keys.F3).perform()
                    time.sleep(0.2)
            
            # Close search box
            webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
            
            # Alternative method: Find elements containing text
            elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
            
            if elements and nth_result <= len(elements):
                elem = elements[nth_result - 1]
                driver.execute_script("arguments[0].scrollIntoView(true);", elem)
                
                # Highlight the element briefly
                driver.execute_script("""
                    arguments[0].style.backgroundColor = 'yellow';
                    arguments[0].style.border = '2px solid red';
                    setTimeout(function() {
                        arguments[0].style.backgroundColor = '';
                        arguments[0].style.border = '';
                    }, 2000);
                """, elem)
                
                result = f"Found {len(elements)} matches for '{text}'. Focused on element {nth_result} of {len(elements)}"
                self._track_action("search_text", result, time.time() - start_time, True)
                return result
            else:
                result = f"Match n°{nth_result} not found (only {len(elements)} matches found)"
                self._track_action("search_text", result, time.time() - start_time, False)
                return result
                
        except Exception as e:
            result = f"Search failed: {str(e)}"
            self._track_action("search_text", result, 0, False)
            return result
    
    def wait_for_element(self, selector: str, timeout: int = 10) -> str:
        """Wait for element to appear"""
        try:
            start_time = time.time()
            driver = helium.get_driver()
            
            wait = WebDriverWait(driver, timeout)
            element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            
            result = f"Element '{selector}' found after {time.time() - start_time:.2f}s"
            self._track_action("wait_for_element", result, time.time() - start_time, True)
            return result
            
        except TimeoutException:
            result = f"Element '{selector}' not found within {timeout}s"
            self._track_action("wait_for_element", result, time.time() - start_time, False)
            return result
        except Exception as e:
            result = f"Wait for element failed: {str(e)}"
            self._track_action("wait_for_element", result, 0, False)
            return result
    
    def wait_for_page_load(self, timeout: int = 30) -> str:
        """Wait for page to fully load"""
        try:
            start_time = time.time()
            driver = helium.get_driver()
            
            wait = WebDriverWait(driver, timeout)
            
            # Wait for document ready state
            wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
            
            # Wait for jQuery if present
            try:
                wait.until(lambda d: d.execute_script("return jQuery.active == 0") if d.execute_script("return typeof jQuery != 'undefined'") else True)
            except:
                pass
            
            result = f"Page loaded completely after {time.time() - start_time:.2f}s"
            self._track_action("wait_for_page_load", result, time.time() - start_time, True)
            return result
            
        except TimeoutException:
            result = f"Page load timeout after {timeout}s"
            self._track_action("wait_for_page_load", result, time.time() - start_time, False)
            return result
        except Exception as e:
            result = f"Wait for page load failed: {str(e)}"
            self._track_action("wait_for_page_load", result, 0, False)
            return result
    
    def extract_structured_data(self) -> Dict:
        """Extract structured data from page"""
        try:
            start_time = time.time()
            driver = helium.get_driver()
            
            structured_data = {
                "tables": [],
                "lists": [],
                "forms": [],
                "links": [],
                "images": [],
                "headings": []
            }
            
            # Extract tables
            tables = driver.find_elements(By.TAG_NAME, "table")
            for table in tables:
                rows = table.find_elements(By.TAG_NAME, "tr")
                table_data = []
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td") + row.find_elements(By.TAG_NAME, "th")
                    table_data.append([cell.text.strip() for cell in cells])
                structured_data["tables"].append(table_data)
            
            # Extract lists
            lists = driver.find_elements(By.TAG_NAME, "ul") + driver.find_elements(By.TAG_NAME, "ol")
            for lst in lists:
                items = lst.find_elements(By.TAG_NAME, "li")
                structured_data["lists"].append([item.text.strip() for item in items])
            
            # Extract forms
            forms = driver.find_elements(By.TAG_NAME, "form")
            for form in forms:
                inputs = form.find_elements(By.TAG_NAME, "input") + form.find_elements(By.TAG_NAME, "textarea") + form.find_elements(By.TAG_NAME, "select")
                form_data = []
                for inp in inputs:
                    form_data.append({
                        "type": inp.get_attribute("type"),
                        "name": inp.get_attribute("name"),
                        "placeholder": inp.get_attribute("placeholder"),
                        "value": inp.get_attribute("value")
                    })
                structured_data["forms"].append(form_data)
            
            # Extract links
            links = driver.find_elements(By.TAG_NAME, "a")
            for link in links[:20]:  # Limit to first 20 links
                href = link.get_attribute("href")
                if href:
                    structured_data["links"].append({
                        "text": link.text.strip(),
                        "href": href
                    })
            
            # Extract images
            images = driver.find_elements(By.TAG_NAME, "img")
            for img in images[:10]:  # Limit to first 10 images
                src = img.get_attribute("src")
                if src:
                    structured_data["images"].append({
                        "alt": img.get_attribute("alt"),
                        "src": src
                    })
            
            # Extract headings
            for level in range(1, 7):
                headings = driver.find_elements(By.TAG_NAME, f"h{level}")
                for heading in headings:
                    structured_data["headings"].append({
                        "level": level,
                        "text": heading.text.strip()
                    })
            
            result = f"Extracted structured data: {len(structured_data['tables'])} tables, {len(structured_data['lists'])} lists, {len(structured_data['links'])} links"
            self._track_action("extract_structured_data", result, time.time() - start_time, True)
            
            return structured_data
            
        except Exception as e:
            result = f"Extract structured data failed: {str(e)}"
            self._track_action("extract_structured_data", result, 0, False)
            return {"error": result}    
    
    
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
        """Session closing with cleanup"""
        try:
            if self.is_active:
                # Save final workflow summary
                summary = self.get_workflow_summary()
                summary_path = os.path.join(self.download_dir, "workflow_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                helium.kill_browser()
                self.driver = None
                self.is_active = False
                
                print(f"📊 Session closed. Summary saved to {summary_path}")
                print(f"📈 Performance: {summary['success_rate']} success rate, {summary['total_actions']} total actions")
                
                return f"Enhanced session closed. {summary['success_rate']} success rate."
            return "No active session"
            
        except Exception as e:
            return f"Error closing session: {str(e)}"
    
    def cleanup(self):
        """Cleanup with optional file preservation"""
        try:
            # Ask if user wants to preserve important files
            important_files = [f for f in self.downloaded_files if f.endswith(('.pdf', '.xlsx', '.csv', '.json'))]
            
            if important_files:
                print(f"🔄 Found {len(important_files)} potentially important files:")
                for file in important_files:
                    print(f"  - {os.path.basename(file)}")
            
            # Always preserve workflow summary and screenshots
            preserved_files = []
            if os.path.exists(self.download_dir):
                for file in os.listdir(self.download_dir):
                    if file.endswith(('.json', '.png')) and ('workflow' in file or 'screenshot' in file):
                        preserved_files.append(file)
                
                # Clean up everything except preserved files
                for file in os.listdir(self.download_dir):
                    if file not in preserved_files:
                        try:
                            os.remove(os.path.join(self.download_dir, file))
                        except:
                            pass
            
            self.downloaded_files = []
            self.screenshots = []
            self.workflow_state = {
                "current_site": None,
                "navigation_history": [],
                "form_interactions": [],
                "search_history": [],
                "error_count": 0,
                "retry_strategies": [],
                "extracted_data": {},
                "performance_metrics": {
                    "total_actions": 0,
                    "successful_actions": 0,
                    "average_response_time": 0
                }
            }
            
            print(f"🧹 Cleanup completed. Preserved {len(preserved_files)} important files.")
            
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
        
    def type_text(self, text: str, element_selector: str = None, clear_first: bool = True, press_enter: bool = False) -> str:
        """
        Type text into input fields with multiple strategies
        
        Args:
            text: Text to type
            element_selector: CSS selector for specific element (optional)
            clear_first: Whether to clear field before typing
            press_enter: Whether to press Enter after typing
        """
        try:
            start_time = time.time()
            driver = helium.get_driver()
            
            # Strategy 1: Use specific selector
            if element_selector:
                try:
                    element = driver.find_element(By.CSS_SELECTOR, element_selector)
                    if clear_first:
                        element.clear()
                    element.send_keys(text)
                    if press_enter:
                        element.send_keys(Keys.RETURN)
                    
                    result = f"Typed '{text}' into {element_selector}"
                    self._track_action("type_text", result, time.time() - start_time, True)
                    return result
                except Exception as e:
                    print(f"Selector strategy failed: {e}")
            
            # Strategy 2: Find active input field
            try:
                search_selectors = [
                    "input[type='search']",
                    "input[placeholder*='search' i]",
                    "input[placeholder*='Search']",
                    "input[name*='search' i]",
                    "input[id*='search' i]",
                    "input[class*='search' i]",
                    "textarea[placeholder*='search' i]",
                    "input[type='text']:focus",
                    "input[type='text']",
                    "textarea"
                ]
                
                for selector in search_selectors:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        element = elements[0]
                        if clear_first:
                            element.clear()
                        element.send_keys(text)
                        if press_enter:
                            element.send_keys(Keys.RETURN)
                        
                        result = f"Typed '{text}' into {selector}"
                        self._track_action("type_text", result, time.time() - start_time, True)
                        return result
                        
            except Exception as e:
                print(f"Auto-detection strategy failed: {e}")
            
            # Strategy 3: Use helium's write function
            try:
                if clear_first:
                    # Try to clear current field
                    driver.execute_script("document.activeElement.value = '';")
                
                helium.write(text)
                if press_enter:
                    helium.press(helium.ENTER)
                
                result = f"Typed '{text}' using helium write"
                self._track_action("type_text", result, time.time() - start_time, True)
                return result
                
            except Exception as e:
                print(f"Helium strategy failed: {e}")
            
            # Strategy 4: Send keys to active element
            try:
                active_element = driver.switch_to.active_element
                if clear_first:
                    active_element.clear()
                active_element.send_keys(text)
                if press_enter:
                    active_element.send_keys(Keys.RETURN)
                
                result = f"Typed '{text}' into active element"
                self._track_action("type_text", result, time.time() - start_time, True)
                return result
                
            except Exception as e:
                result = f"All typing strategies failed: {str(e)}"
                self._track_action("type_text", result, time.time() - start_time, False)
                return result
                
        except Exception as e:
            result = f"Type text failed: {str(e)}"
            self._track_action("type_text", result, 0, False)
            return result
    
    def fill_form_field(self, field_name: str, value: str, clear_first: bool = True) -> str:
        """
        Fill form fields by name, placeholder, or label
        
        Args:
            field_name: Name, placeholder, or label text of the field
            value: Value to enter
            clear_first: Whether to clear field before filling
        """
        try:
            start_time = time.time()
            driver = helium.get_driver()
            
            # Strategy 1: Find by various attributes
            selectors = [
                f"input[name='{field_name}']",
                f"input[placeholder*='{field_name}' i]",
                f"input[id*='{field_name}' i]",
                f"input[class*='{field_name}' i]",
                f"textarea[name='{field_name}']",
                f"textarea[placeholder*='{field_name}' i]",
                f"textarea[id*='{field_name}' i]",
                f"select[name='{field_name}']",
                f"select[id*='{field_name}' i]"
            ]
            
            for selector in selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        element = elements[0]
                        
                        # Handle different input types
                        tag_name = element.tag_name.lower()
                        if tag_name == "select":
                            # Dropdown selection
                            select = Select(element)
                            try:
                                select.select_by_visible_text(value)
                            except:
                                select.select_by_value(value)
                        else:
                            # Text input
                            if clear_first:
                                element.clear()
                            element.send_keys(value)
                        
                        result = f"Filled '{field_name}' with '{value}' using {selector}"
                        self._track_action("fill_form", result, time.time() - start_time, True)
                        return result
                        
                except Exception as e:
                    continue
            
            # Strategy 2: Use helium's write with 'into' parameter
            try:
                if clear_first:
                    helium.write("", into=field_name, clear=True)
                helium.write(value, into=field_name)
                
                result = f"Filled '{field_name}' with '{value}' using helium"
                self._track_action("fill_form", result, time.time() - start_time, True)
                return result
                
            except Exception as e:
                result = f"Could not find field '{field_name}': {str(e)}"
                self._track_action("fill_form", result, time.time() - start_time, False)
                return result
                
        except Exception as e:
            result = f"Fill form field failed: {str(e)}"
            self._track_action("fill_form", result, 0, False)
            return result
    
    def perform_search(self, query: str, search_button_text: str = "Search") -> str:
        """
        Perform search with query and button click
        
        Args:
            query: Search query to enter
            search_button_text: Text on search button (default: "Search")
        """
        try:
            start_time = time.time()
            
            # Step 1: Type in search box
            type_result = self.type_text(query, clear_first=True)
            
            # Step 2: Try to click search button
            search_clicked = False
            
            # Try multiple search button strategies
            search_buttons = [
                search_button_text,
                "Search",
                "Go",
                "Submit",
                "Find",
                "🔍"  # Search icon
            ]
            
            for button_text in search_buttons:
                try:
                    if self.smart_click(button_text):
                        search_clicked = True
                        break
                except:
                    continue
            
            # Step 3: If no button found, try pressing Enter
            if not search_clicked:
                try:
                    driver = helium.get_driver()
                    webdriver.ActionChains(driver).send_keys(Keys.RETURN).perform()
                    search_clicked = True
                except:
                    pass
            
            # Step 4: Wait for results to load
            time.sleep(2)
            
            if search_clicked:
                result = f"Searched for '{query}' successfully"
                self._track_action("perform_search", result, time.time() - start_time, True)
                self.workflow_state["search_history"].append({
                    "query": query,
                    "timestamp": time.time(),
                    "success": True
                })
                return result
            else:
                result = f"Typed '{query}' but couldn't find search button"
                self._track_action("perform_search", result, time.time() - start_time, False)
                return result
                
        except Exception as e:
            result = f"Search failed: {str(e)}"
            self._track_action("perform_search", result, 0, False)
            return result
    
    def select_dropdown_option(self, dropdown_identifier: str, option_text: str) -> str:
        """
        Select option from dropdown
        
        Args:
            dropdown_identifier: CSS selector, name, or id of dropdown
            option_text: Text of option to select
        """
        try:
            start_time = time.time()
            driver = helium.get_driver()
            
            # Strategy 1: Use helium's select
            try:
                helium.select(option_text, from_=dropdown_identifier)
                result = f"Selected '{option_text}' from '{dropdown_identifier}' using helium"
                self._track_action("select_dropdown", result, time.time() - start_time, True)
                return result
                
            except Exception as e:
                print(f"Helium select failed: {e}")
            
            # Strategy 2: Find dropdown and use Selenium Select
            selectors = [
                f"select[name='{dropdown_identifier}']",
                f"select[id='{dropdown_identifier}']",
                f"select[class*='{dropdown_identifier}']",
                dropdown_identifier if dropdown_identifier.startswith(('.', '#', '[')) else f"#{dropdown_identifier}"
            ]
            
            for selector in selectors:
                try:
                    dropdown = driver.find_element(By.CSS_SELECTOR, selector)
                    select = Select(dropdown)
                    
                    # Try multiple selection strategies
                    try:
                        select.select_by_visible_text(option_text)
                    except:
                        select.select_by_value(option_text)
                    
                    result = f"Selected '{option_text}' from dropdown using {selector}"
                    self._track_action("select_dropdown", result, time.time() - start_time, True)
                    return result
                    
                except Exception as e:
                    continue
            
            result = f"Could not find dropdown '{dropdown_identifier}'"
            self._track_action("select_dropdown", result, time.time() - start_time, False)
            return result
            
        except Exception as e:
            result = f"Dropdown selection failed: {str(e)}"
            self._track_action("select_dropdown", result, 0, False)
            return result
    
    def press_key_combination(self, keys: str) -> str:
        """
       Press keyboard shortcuts
        
        Args:
            keys: Key combination like "ctrl+f", "ctrl+a", "escape", etc.
        """
        try:
            start_time = time.time()
            driver = helium.get_driver()
            
            # Parse key combination
            keys_lower = keys.lower()
            
            if keys_lower == "ctrl+f":
                webdriver.ActionChains(driver).key_down(Keys.CONTROL).send_keys('f').key_up(Keys.CONTROL).perform()
                result = "Pressed Ctrl+F"
            elif keys_lower == "ctrl+a":
                webdriver.ActionChains(driver).key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).perform()
                result = "Pressed Ctrl+A"
            elif keys_lower == "escape":
                webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                result = "Pressed Escape"
            elif keys_lower == "enter":
                webdriver.ActionChains(driver).send_keys(Keys.RETURN).perform()
                result = "Pressed Enter"
            elif keys_lower == "tab":
                webdriver.ActionChains(driver).send_keys(Keys.TAB).perform()
                result = "Pressed Tab"
            else:
                # Try to send the keys directly
                webdriver.ActionChains(driver).send_keys(keys).perform()
                result = f"Pressed '{keys}'"
            
            self._track_action("press_keys", result, time.time() - start_time, True)
            return result
            
        except Exception as e:
            result = f"Key press failed: {str(e)}"
            self._track_action("press_keys", result, 0, False)
            return result

# Global session
_browser_session = BrowserSession()

class VisionWebBrowserTool(Tool):
    name = "vision_browser_tool"
    description = """Navigate web pages with advanced interaction capabilities 
    including text input, form filling, and search functionality. The Vision Web 
    Browser Tool inherits from BrowserSession class via inheritance pattern. 
    The tool contains and uses the session rather than being the session 
    (composition = containment + delegation + lifecycle management). 
    
    Based on SmolagAgents best practices for browser automation:
    - Navigate to web pages with automatic screenshots
    - Smart element clicking with multiple fallback strategies
    - Text input in any field with robust fallback strategies
    - Form filling by field name, placeholder, or label
    - Search functionality with query + button click
    - Keyboard shortcuts (Ctrl+F, Escape, Enter, etc.)
    - Dropdown selection with multiple strategies
    - Advanced waiting for elements and page loads
    - Structured data extraction (tables, lists, forms, links)
    - Performance tracking and workflow analytics
    - Visual analysis with screenshots at each step
    """
    
    inputs = {
        "action": {
            "type": "string",
            "description": "Action to perform: 'navigate', 'click_element', 'search_text', 'scroll_page', 'close_popups', 'take_screenshot', 'extract_content', 'type_text', 'fill_form', 'perform_search', 'select_dropdown', 'press_key_combination', 'wait_for_element', 'wait_for_page_load', 'extract_structured_data', 'close_browser'",
        },
        "url": {
            "type": "string", 
            "description": "Target URL for navigation action.",
            "nullable": True,
        },
        "element_text": {
            "type": "string", 
            "description": "Text of element to click or field name for form filling.",
            "nullable": True,
        },
        "text_input": {
            "type": "string",
            "description": "Text to type, search query, or form field value.",
            "nullable": True,
        },
        "element_selector": {
            "type": "string", 
            "description": "CSS selector for specific element (for type_text, wait_for_element actions).",
            "nullable": True,
        },
        "dropdown_selector": {
            "type": "string",
            "description": "CSS selector for dropdown (for select_dropdown action).",
            "nullable": True,
        },
        "search_text": {
            "type": "string",
            "description": "Text to search for on page (for search_text action).",
            "nullable": True,
        },
        "key_combination": {
            "type": "string",
            "description": "Key combination to press (for press_key_combination action): 'ctrl+f', 'escape', 'enter', 'tab', etc.",
            "nullable": True,
        },
        "scroll_direction": {
            "type": "string",
            "description": "Direction to scroll: 'up', 'down', 'top', 'bottom' (for scroll_page action).",
            "nullable": True,
        },
        "scroll_pixels": {
            "type": "integer",
            "description": "Number of pixels to scroll (default: 1200).",
            "nullable": True,
        },
        "clear_first": {
            "type": "boolean",
            "description": "Whether to clear field before typing (default: True).",
            "nullable": True,
        },
        "press_enter": {
            "type": "boolean",
            "description": "Whether to press Enter after typing (default: False).",
            "nullable": True,
        },
        "nth_result": {
            "type": "integer",
            "description": "Which search result to focus on (default: 1).",
            "nullable": True,
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds for wait actions (default: 10).",
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
    
    @property
    def session(self) -> BrowserSession:
        """
        🔥 Lazy session creation using composition pattern
        
        Creates and initializes browser session only when needed.
        Each tool instance gets its own session for isolation.
        """
        if self._session is None:
            self._session = BrowserSession()
            init_result = self._session.initialize()
            if "failed" in init_result:
                raise RuntimeError(f"Browser initialization failed: {init_result}")
            self._initialized = True
            print(f"🔥 Browser session created for {self.name}")
        return self._session
    
    def forward(
        self, 
        action: str,
        url: Optional[str] = None,
        element_text: Optional[str] = None,
        text_input: Optional[str] = None,
        element_selector: Optional[str] = None,
        dropdown_selector: Optional[str] = None,
        search_text: Optional[str] = None,
        key_combination: Optional[str] = None,
        scroll_direction: Optional[str] = "down",
        scroll_pixels: Optional[int] = 1200,
        clear_first: Optional[bool] = True,
        press_enter: Optional[bool] = False,
        nth_result: Optional[int] = 1,
        timeout: Optional[int] = 10,
        wait_seconds: Optional[int] = 2
    ) -> str:
        
        if not DEPENDENCIES_AVAILABLE:
            return self._format_error("Dependencies not available. Install: pip install helium selenium")
        
        try:
            # 🔥 Direct method calls to session - no duplication!
            if action == "navigate":
                result = self._navigate(url, wait_seconds)
                
            elif action == "click_element":
                result = self._click_element(element_text, wait_seconds)
                
            elif action == "type_text":
                result = self._type_text(text_input, element_selector, clear_first, press_enter)
                
            elif action == "fill_form":
                result = self._fill_form(element_text, text_input, clear_first)
                
            elif action == "perform_search":
                result = self._perform_search(text_input, element_text or "Search")
                
            elif action == "select_dropdown":
                result = self._select_dropdown(dropdown_selector, text_input)
                
            elif action == "press_key_combination":
                result = self._press_key_combination(key_combination)
                
            elif action == "search_text":
                result = self._search_text(search_text, nth_result)
                
            elif action == "scroll_page":
                result = self._scroll_page(scroll_direction, scroll_pixels)
                
            elif action == "wait_for_element":
                result = self._wait_for_element(element_selector, timeout)
                
            elif action == "wait_for_page_load":
                result = self._wait_for_page_load(timeout)
                
            elif action == "extract_structured_data":
                result = self._extract_structured_data()
                
            elif action == "close_popups":
                result = self._close_popups()
                
            elif action == "take_screenshot":
                result = self._take_screenshot()
                
            elif action == "extract_content":
                result = self._extract_content()
                
            elif action == "close_browser":
                result = self._close_browser()
                
            else:
                result = ContentResult(
                    content_type="error",
                    success=False,
                    error_message=f"Unknown action: {action}. Available actions: navigate, click_element, type_text, fill_form, perform_search, select_dropdown, press_key_combination, search_text, scroll_page, wait_for_element, wait_for_page_load, extract_structured_data, close_popups, take_screenshot, extract_content, close_browser"
                )
            
            return self._format_result(result)
            
        except Exception as e:
            return self._format_error(f"Browser error: {str(e)}")
    
    # 🔥 Action methods using composition - direct calls to session methods
    
    def _navigate(self, url: str, wait_seconds: int) -> ContentResult:
        """Navigate to URL with automatic screenshot"""
        if not url:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="URL required for navigation"
            )
        
        try:
            helium.go_to(url)
            time.sleep(wait_seconds)
            
            # Use session methods directly
            screenshot_path = self.session.take_screenshot("navigation")
            page_info = self.session.get_page_info()
            
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
        """Click element with smart fallbacks"""
        if not element_text:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="element_text required for clicking"
            )
        
        try:
            # Direct call to session method
            click_success = self.session.smart_click(element_text)
            
            if click_success:
                time.sleep(wait_seconds)
                
                screenshot_path = self.session.take_screenshot("after_click")
                page_info = self.session.get_page_info()
                
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
    
    def _type_text(self, text: str, selector: str = None, clear_first: bool = True, press_enter: bool = False) -> ContentResult:
        """Type text using session method"""
        if not text:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="text_input required for typing"
            )
        
        try:
            # Direct call to session method - no duplication!
            type_result = self.session.type_text(text, selector, clear_first, press_enter)
            time.sleep(1)
            
            screenshot_path = self.session.take_screenshot("after_typing")
            page_info = self.session.get_page_info()
            
            return ContentResult(
                content_type="text_input",
                url=page_info.get("url"),
                screenshot_path=screenshot_path,
                content_text=type_result,
                metadata={"text_typed": text, "selector": selector},
                handoff_instructions=f"Typed '{text}'. Check screenshot for result.",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Type text failed: {str(e)}"
            )
    
    def _fill_form(self, field_name: str, value: str, clear_first: bool = True) -> ContentResult:
        """Fill form field using session method"""
        if not field_name or not value:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="field_name and value required"
            )
        
        try:
            # Direct call to session method
            fill_result = self.session.fill_form_field(field_name, value, clear_first)
            time.sleep(1)
            
            screenshot_path = self.session.take_screenshot("after_form_fill")
            page_info = self.session.get_page_info()
            
            return ContentResult(
                content_type="form_fill",
                url=page_info.get("url"),
                screenshot_path=screenshot_path,
                content_text=fill_result,
                metadata={"field_name": field_name, "value": value},
                handoff_instructions=f"Filled '{field_name}' with '{value}'. Check screenshot for result.",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Form fill failed: {str(e)}"
            )
    
    def _perform_search(self, query: str, search_button: str = "Search") -> ContentResult:
        """Perform search using session method"""
        if not query:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="query required for search"
            )
        
        try:
            # Direct call to session method
            search_result = self.session.perform_search(query, search_button)
            time.sleep(2)
            
            screenshot_path = self.session.take_screenshot("search_results")
            page_info = self.session.get_page_info()
            
            return ContentResult(
                content_type="search_results",
                url=page_info.get("url"),
                screenshot_path=screenshot_path,
                content_text=search_result,
                metadata={"query": query, "search_button": search_button},
                handoff_instructions=f"Search completed for '{query}'. Check screenshot for results.",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Search failed: {str(e)}"
            )
    
    def _select_dropdown(self, dropdown_selector: str, option_text: str) -> ContentResult:
        """Select dropdown option using session method"""
        if not dropdown_selector or not option_text:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="dropdown_selector and option_text required"
            )
        
        try:
            # Direct call to session method
            select_result = self.session.select_dropdown_option(dropdown_selector, option_text)
            time.sleep(1)
            
            screenshot_path = self.session.take_screenshot("after_dropdown_select")
            page_info = self.session.get_page_info()
            
            return ContentResult(
                content_type="dropdown_select",
                url=page_info.get("url"),
                screenshot_path=screenshot_path,
                content_text=select_result,
                metadata={"dropdown_selector": dropdown_selector, "option_text": option_text},
                handoff_instructions=f"Selected '{option_text}' from dropdown. Check screenshot for result.",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Dropdown select failed: {str(e)}"
            )
    
    def _press_key_combination(self, keys: str) -> ContentResult:
        """Press key combination using session method"""
        if not keys:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="key_combination required"
            )
        
        try:
            # Direct call to session method
            key_result = self.session.press_key_combination(keys)
            time.sleep(1)
            
            screenshot_path = self.session.take_screenshot("after_key_press")
            page_info = self.session.get_page_info()
            
            return ContentResult(
                content_type="key_press",
                url=page_info.get("url"),
                screenshot_path=screenshot_path,
                content_text=key_result,
                metadata={"keys": keys},
                handoff_instructions=f"Pressed '{keys}'. Check screenshot for result.",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Key press failed: {str(e)}"
            )
    
    def _search_text(self, search_text: str, nth_result: int) -> ContentResult:
        """Search text on page using session method"""
        if not search_text:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="search_text required"
            )
        
        try:
            # Direct call to session method
            search_result = self.session.search_text_on_page(search_text, nth_result)
            screenshot_path = self.session.take_screenshot("search_result")
            
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
        """Scroll page using session method"""
        try:
            # Direct call to session method
            scroll_result = self.session.scroll_page(direction, pixels)
            screenshot_path = self.session.take_screenshot("after_scroll")
            
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
    
    def _wait_for_element(self, selector: str, timeout: int) -> ContentResult:
        """Wait for element using session method"""
        if not selector:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="element_selector required"
            )
        
        try:
            # Direct call to session method
            wait_result = self.session.wait_for_element(selector, timeout)
            screenshot_path = self.session.take_screenshot("after_element_wait")
            page_info = self.session.get_page_info()
            
            return ContentResult(
                content_type="element_wait",
                url=page_info.get("url"),
                screenshot_path=screenshot_path,
                content_text=wait_result,
                metadata={"selector": selector, "timeout": timeout},
                handoff_instructions=f"Wait for element '{selector}' completed. Check screenshot.",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Element wait failed: {str(e)}"
            )
    
    def _wait_for_page_load(self, timeout: int) -> ContentResult:
        """Wait for page load using session method"""
        try:
            # Direct call to session method
            load_result = self.session.wait_for_page_load(timeout)
            screenshot_path = self.session.take_screenshot("after_page_load")
            page_info = self.session.get_page_info()
            
            return ContentResult(
                content_type="page_load",
                url=page_info.get("url"),
                screenshot_path=screenshot_path,
                content_text=load_result,
                metadata={"timeout": timeout},
                handoff_instructions=f"Page load wait completed. Check screenshot.",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Page load wait failed: {str(e)}"
            )
    
    def _extract_structured_data(self) -> ContentResult:
        """Extract structured data using session method"""
        try:
            # Direct call to session method
            structured_data = self.session.extract_structured_data()
            screenshot_path = self.session.take_screenshot("structured_data_extract")
            page_info = self.session.get_page_info()
            
            return ContentResult(
                content_type="structured_data",
                url=page_info.get("url"),
                screenshot_path=screenshot_path,
                content_text=json.dumps(structured_data, indent=2),
                metadata=structured_data,
                handoff_instructions=f"Structured data extracted. Available in metadata.",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Structured data extraction failed: {str(e)}"
            )
    
    def _close_popups(self) -> ContentResult:
        """Close popups using session method"""
        try:
            # Direct call to session method
            close_result = self.session.close_popups()
            screenshot_path = self.session.take_screenshot("after_popup_close")
            
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
        """Take screenshot using session method"""
        try:
            # Direct call to session method
            screenshot_path = self.session.take_screenshot("manual")
            page_info = self.session.get_page_info()
            
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
        """Extract page content using session method"""
        try:
            driver = helium.get_driver()
            page_text = driver.find_element(By.TAG_NAME, "body").text
            page_info = self.session.get_page_info()
            
            # Truncate if too long
            if len(page_text) > 5000:
                page_text = page_text[:5000] + "... [truncated]"
            
            screenshot_path = self.session.take_screenshot("content_extract")
            
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
    
    def _close_browser(self) -> ContentResult:
        """Close browser using session method"""
        try:
            if self._session:
                close_result = self._session.close()
                self._session = None
                self._initialized = False
                
                return ContentResult(
                    content_type="session_close",
                    content_text=close_result,
                    handoff_instructions="Browser session closed successfully.",
                    success=True
                )
            else:
                return ContentResult(
                    content_type="session_close",
                    content_text="No active session to close",
                    handoff_instructions="No browser session was active.",
                    success=True
                )
                
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Browser close failed: {str(e)}"
            )
    
    def _format_result(self, result: ContentResult) -> str:
        """Format ContentResult as JSON string"""
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
        
        # Session info
        if self._session:
            result_dict["session_info"] = {
                "files_downloaded": len(self._session.downloaded_files),
                "screenshots_taken": len(self._session.screenshots),
                "browser_active": self._session.is_active,
                "workflow_summary": self._session.get_workflow_summary()
            }
        
        return json.dumps(result_dict, indent=2)
    
    def _format_error(self, error_message: str) -> str:
        """Format error message as JSON string"""
        return json.dumps({
            "success": False,
            "content_type": "error",
            "error_message": error_message,
            "handoff_instructions": "Action failed. Check error message for details."
        }, indent=2)
    
    def cleanup(self):
        """Clean up browser session"""
        if self._session:
            self._session.close()
            self._session.cleanup()
            self._session = None
            self._initialized = False
            print("🧹 VisionWebBrowserTool cleaned up")

# Additional utility functions for SmolagAgent integration
def get_browser_instructions() -> str:
    """Get Helium usage instructions for SmolagAgent integration"""
    return """
ACTIONS:
- navigate: Go to a URL (automatically takes screenshot)
- click_element: Click on page elements by text
- scroll_page: Scroll up/down/top/bottom
- close_popups: Close modal dialogs and popups  
- take_screenshot: Capture current page state
- extract_content: Get page text content
- wait_for_element: Wait for element to appear
- wait_for_page_load: Wait for page to fully load
- extract_structured_data: Get tables, lists, forms, links
- close_browser: Clean up session

TEXT INPUT ACTIONS:
- type_text: Type text into any input field
- fill_form: Fill form fields by name/placeholder/label
- perform_search: Search with query + button click
- select_dropdown: Select option from dropdown
- press_key_combination: Press keyboard shortcuts (ctrl+f, escape, etc.)

USAGE PATTERNS:
1. Navigate: {"action": "navigate", "url": "https://github.com/numpy/numpy"}
2. Search: {"action": "perform_search", "text_input": "is:closed polynomial label:Regression"}
3. Type: {"action": "type_text", "text_input": "search query", "press_enter": true}
4. Fill Form: {"action": "fill_form", "element_text": "username", "text_input": "myuser"}
5. Select: {"action": "select_dropdown", "element_selector": "#sort-select", "text_input": "oldest"}

All actions automatically take screenshots and track performance metrics.
Use the workflow_state and performance tracking for optimization.
"""