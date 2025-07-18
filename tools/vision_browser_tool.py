import os
import time
import tempfile
import shutil
import json
from typing import Optional, Dict, List, Tuple
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
    from selenium.webdriver.common.action_chains import ActionChains
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

class VisionWebBrowserTool(Tool):
    """
    This tool provides you with intelligent browsing capabilities:
    - Smart multi-strategy interactions
    - Content type detection and site-specific behavior  
    - Structured data extraction
    - Performance tracking and analytics
    - Agent-friendly result formatting
    """
    name = "vision_browser_tool"
    description = """Web browser automation tool for navigating websites, visual understanding and extracting data.

    PRIMARY NAVIGATION: Screenshots taken automatically after each action for visual confirmation.

    AVAILABLE ACTIONS:
    - navigate: Go to URL (takes screenshot automatically)
    - click_element: Click buttons, links, or text on page
    - search_text: Find specific text on current page and focus on it
    - extract_content: Get all text from page
    - extract_structured_data: Get tables, lists, forms, links from page
    - fill_form: Fill input fields by name or placeholder
    - type_text: Type text into input fields with auto-detection
    - perform_search: Complete search workflow (type query + submit)
    - select_dropdown: Choose option from dropdown menu
    - scroll_page: Scroll up/down on page
    - take_screenshot: Capture current page view manually
    - close_popups: Close modal dialogs with ESC key
    - wait_for_element: Wait for element to appear on page
    - wait_for_page_load: Wait for page to fully load
    - press_key_combination: Send keyboard shortcuts (ctrl+f, escape, etc.)
    - close_browser: End browser session

    CONTEXT STRATEGIES TO FIND YOUR WAY:
    - Multi-strategy clicking with automatic fallbacks
    - Smart form field detection across multiple attributes
    - Site-specific content type detection (GitHub, USGS, databases)
    - Performance tracking and error recovery
    - Download management and file tracking

    USAGE PATTERN:
    1. navigate to URL first
    2. extract_content to see what's on page
    3. search_text to find specific elements
    4. click_element or fill_form to interact
    5. extract_structured_data to get tables/lists
    6. Check handoff_instructions for next steps

    Use one action at a time and check results before proceeding.
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
        
        # 🎯 AGENT INTELLIGENCE: Performance tracking and workflow state
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
    
    @property
    def session(self) -> 'BrowserSession':
        """
        🔥 Lazy session creation - only basic browser management
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
            # 🎯 AGENT INTELLIGENCE: All smart capabilities stay in this layer
            if action == "navigate":
                result = self._navigate(url, wait_seconds)
                
            elif action == "click_element":
                result = self._smart_click_element(element_text, wait_seconds)
                
            elif action == "type_text":
                result = self._smart_type_text(text_input, element_selector, clear_first, press_enter)
                
            elif action == "fill_form":
                result = self._smart_fill_form(element_text, text_input, clear_first)
                
            elif action == "perform_search":
                result = self._smart_perform_search(text_input, element_text or "Search")
                
            elif action == "select_dropdown":
                result = self._smart_select_dropdown(dropdown_selector, text_input)
                
            elif action == "press_key_combination":
                result = self._press_key_combination(key_combination)
                
            elif action == "search_text":
                result = self._smart_search_text(search_text, nth_result)
                
            elif action == "scroll_page":
                result = self._smart_scroll_page(scroll_direction, scroll_pixels)
                
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

    def get_browser_instructions() -> str:
        """
        Comprehensive browser usage instructions for SmolagAgent integration
        """
        return """
        🌐 VISION WEB BROWSER USAGE INSTRUCTIONS

        🎯 WORKFLOW PATTERNS:
        1. BASIC NAVIGATION: navigate → extract_content → search_text → click_element
        2. FORM INTERACTION: navigate → fill_form → perform_search → extract_structured_data
        3. DATA EXTRACTION: navigate → search_text → extract_structured_data → scroll_page
        4. ERROR RECOVERY: close_popups → take_screenshot → wait_for_page_load → retry_action

        FEATURES:
        - Automatic screenshots after each action to check your status
        - Each action returns JSON with success status, handoff_instructions, and context for decision-making
        - Multiple fallback strategies for clicking and form interaction
        - Performance tracking with success rates and timing
        - Content-aware detection of page types (GitHub, databases, PDFs)
        - Intelligent error recovery with retry logic
        - Session state persistence with cleanup management

        BEST PRACTICES:
        1. Always check handoff_instructions in response for next step guidance
        2. Use extract_content first to understand page structure
        3. Use search_text to locate elements before clicking
        4. Take screenshots when debugging interaction issues
        5. Use wait_for_element for dynamic content loading
        6. Handle popups immediately when they appear
        7. Extract structured data when you find relevant tables/forms

        📋 ACTION EXAMPLES:

        NAVIGATION & DISCOVERY:
        • {"action": "navigate", "url": "https://nas.er.usgs.gov/default.aspx"}
        • {"action": "extract_content"} - Get page text to understand structure
        • {"action": "search_text", "search_text": "Search by State", "nth_result": 1}
        • {"action": "take_screenshot"} - Visual confirmation of current state

        INTERACTION & FORM FILLING:
        • {"action": "click_element", "element_text": "Search by State"}
        • {"action": "fill_form", "element_text": "State", "text_input": "Florida"}
        • {"action": "select_dropdown", "dropdown_selector": "#state-select", "text_input": "Florida"}
        • {"action": "perform_search", "text_input": "crocodile", "element_text": "Search"}

        DATA EXTRACTION:
        • {"action": "extract_structured_data"} - Get tables, lists, forms, links
        • {"action": "search_text", "search_text": "crocodile", "nth_result": 2} - Find specific mentions
        • {"action": "scroll_page", "scroll_direction": "down", "scroll_pixels": 1200}

        UTILITY & RECOVERY:
        • {"action": "close_popups"} - Handle modal dialogs
        • {"action": "wait_for_element", "element_selector": ".results-table", "timeout": 10}
        • {"action": "press_key_combination", "key_combination": "ctrl+f"} - Browser search
        • {"action": "wait_for_page_load", "timeout": 30} - Ensure page fully loaded

        Use ONE action at a time, analyze results and handoff_instructions, then proceed strategically.
        """

    def _navigate(self, url: str, wait_seconds: int) -> ContentResult:
        """🎯 AGENT INTELLIGENCE: Smart navigation with comprehensive error handling"""
        
        # Input validation
        if not url:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="URL required for navigation",
                handoff_instructions="Please provide a valid URL for navigation."
            )
        
        if not isinstance(url, str):
            return ContentResult(
                content_type="error", 
                success=False,
                error_message=f"URL must be a string, got {type(url)}",
                handoff_instructions="Please provide a valid URL string."
            )
        
        try:
            start_time = time.time()
            
            # Use basic session navigation
            success = self.session.navigate_to(url)
            
            if not success:
                self._track_action("navigate", f"Failed to navigate to {url}", time.time() - start_time, False)
                return ContentResult(
                    content_type="error",
                    success=False,
                    error_message=f"Navigation failed to {url}",
                    handoff_instructions=f"Could not navigate to {url}. Check URL validity and network connection."
                )
            
            # Wait for page load
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            
            # 🎯 AGENT INTELLIGENCE: Smart screenshot and page analysis
            screenshot_path = self._take_smart_screenshot("navigation")
            page_info = self._analyze_page_content()
            
            self._track_action("navigate", f"Successfully navigated to {url}", time.time() - start_time, True)
            
            return ContentResult(
                content_type="navigation",
                url=page_info.get("url", url),
                screenshot_path=screenshot_path,
                content_text=f"Navigation complete to {page_info.get('url', url)}",
                metadata=page_info,
                handoff_instructions=f"Successfully navigated to {page_info.get('url', url)}. Content type: {page_info.get('content_type', 'unknown')}. Screenshot available for visual analysis.",
                success=True
            )
            
        except Exception as e:
            self._track_action("navigate", f"Navigation error: {str(e)}", 0, False)
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Navigation error: {str(e)}",
                handoff_instructions="Navigation error occurred. Check URL and browser state."
            )

    def _smart_click_element(self, element_text: str, wait_seconds: int) -> ContentResult:
        """🎯 AGENT INTELLIGENCE: Multi-strategy clicking with intelligent fallbacks"""
        if not element_text:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="element_text required for clicking",
                handoff_instructions="Provide the text of the element you want to click."
            )
        
        try:
            start_time = time.time()
            print(f"🎯 Smart click attempting: '{element_text}'")
            
            # Strategy 1: Try helium if available
            if DEPENDENCIES_AVAILABLE:
                try:
                    import helium
                    helium.click(element_text)
                    print(f"✅ Helium click successful")
                    success = True
                    strategy = "helium"
                except ImportError:
                    success = False
                    print("⚠️ Helium not available, trying selenium")
                except Exception as helium_error:
                    success = False
                    print(f"⚠️ Helium click failed: {helium_error}, trying selenium")
            else:
                success = False
            
            # Strategy 2: Selenium fallbacks with multiple selectors
            if not success:
                try:
                    driver = self.session.get_driver()
                    wait = WebDriverWait(driver, 10)
                    
                    # Multiple intelligent selector strategies
                    selectors = [
                        (By.XPATH, f"//button[contains(text(), '{element_text}')]"),
                        (By.XPATH, f"//a[contains(text(), '{element_text}')]"),
                        (By.XPATH, f"//input[@value='{element_text}']"),
                        (By.XPATH, f"//input[@placeholder='{element_text}']"),
                        (By.XPATH, f"//*[contains(text(), '{element_text}')]"),
                        (By.LINK_TEXT, element_text),
                        (By.PARTIAL_LINK_TEXT, element_text),
                        (By.XPATH, f"//span[contains(text(), '{element_text}')]"),
                        (By.XPATH, f"//div[contains(text(), '{element_text}')]"),
                    ]
                    
                    for strategy_name, selector in selectors:
                        try:
                            element = wait.until(EC.element_to_be_clickable((strategy_name, selector)))
                            element.click()
                            print(f"✅ Selenium click successful with {strategy_name}")
                            success = True
                            strategy = f"selenium_{strategy_name}"
                            break
                        except:
                            continue
                            
                except Exception as selenium_error:
                    print(f"❌ Selenium strategies failed: {selenium_error}")
            
            if success:
                time.sleep(wait_seconds)
                
                # 🎯 AGENT INTELLIGENCE: Smart post-click analysis
                screenshot_path = self._take_smart_screenshot("after_click")
                page_info = self._analyze_page_content()
                
                self._track_action("click_element", f"Successfully clicked '{element_text}' using {strategy}", time.time() - start_time, True)
                
                return ContentResult(
                    content_type="click_result",
                    url=page_info.get("url"),
                    screenshot_path=screenshot_path,
                    content_text=f"Successfully clicked '{element_text}' using {strategy}",
                    metadata=page_info,
                    handoff_instructions=f"Clicked '{element_text}'. Page may have changed - check screenshot. Current URL: {page_info.get('url')}. Content type: {page_info.get('content_type')}",
                    success=True
                )
            else:
                self._track_action("click_element", f"Failed to find '{element_text}'", time.time() - start_time, False)
                return ContentResult(
                    content_type="error",
                    success=False,
                    error_message=f"Could not find clickable element: '{element_text}'",
                    handoff_instructions="Element not found. Try search_text to locate it first, or check if the text is exact."
                )
                
        except Exception as e:
            self._track_action("click_element", f"Click error: {str(e)}", 0, False)
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Click failed: {str(e)}"
            )

    def _smart_type_text(self, text: str, element_selector: str = None, clear_first: bool = True, press_enter: bool = False) -> ContentResult:
        """Multi-strategy text input with smart field detection"""
        if not text:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="text_input required for typing"
            )
        
        try:
            start_time = time.time()
            driver = self.session.get_driver()
            strategy_used = "none"
            
            # Strategy 1: Use specific selector if provided
            if element_selector:
                try:
                    element = driver.find_element(By.CSS_SELECTOR, element_selector)
                    if clear_first:
                        element.clear()
                    element.send_keys(text)
                    if press_enter:
                        element.send_keys(Keys.RETURN)
                    strategy_used = f"selector: {element_selector}"
                    success = True
                except Exception as e:
                    print(f"Selector strategy failed: {e}")
                    success = False
            else:
                success = False
            
            # Strategy 2: Smart field auto-detection
            if not success:
                try:
                    # Intelligent field detection selectors
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
                        "textarea",
                        "input[type='email']",
                        "input[type='password']"
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
                            strategy_used = f"auto-detected: {selector}"
                            success = True
                            break
                            
                except Exception as e:
                    print(f"Auto-detection strategy failed: {e}")
            
            # Strategy 3: Use helium's write function
            if not success and DEPENDENCIES_AVAILABLE:
                try:
                    if clear_first:
                        driver.execute_script("document.activeElement.value = '';")
                    
                    helium.write(text)
                    if press_enter:
                        helium.press(helium.ENTER)
                    strategy_used = "helium write"
                    success = True
                    
                except Exception as e:
                    print(f"Helium strategy failed: {e}")
            
            # Strategy 4: Active element fallback
            if not success:
                try:
                    active_element = driver.switch_to.active_element
                    if clear_first:
                        active_element.clear()
                    active_element.send_keys(text)
                    if press_enter:
                        active_element.send_keys(Keys.RETURN)
                    strategy_used = "active element"
                    success = True
                    
                except Exception as e:
                    print(f"Active element strategy failed: {e}")
            
            if success:
                time.sleep(1)
                screenshot_path = self._take_smart_screenshot("after_typing")
                page_info = self._analyze_page_content()
                
                self._track_action("type_text", f"Typed '{text}' using {strategy_used}", time.time() - start_time, True)
                
                return ContentResult(
                    content_type="text_input",
                    url=page_info.get("url"),
                    screenshot_path=screenshot_path,
                    content_text=f"Typed '{text}' using {strategy_used}",
                    metadata={"text_typed": text, "strategy": strategy_used},
                    handoff_instructions=f"Typed '{text}' successfully. Check screenshot for result.",
                    success=True
                )
            else:
                self._track_action("type_text", f"Failed to type '{text}' - no input field found", time.time() - start_time, False)
                return ContentResult(
                    content_type="error",
                    success=False,
                    error_message=f"Could not find input field to type '{text}'",
                    handoff_instructions="No input field found. Try clicking on a text field first or provide an element_selector."
                )
            
        except Exception as e:
            self._track_action("type_text", f"Type error: {str(e)}", 0, False)
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Type text failed: {str(e)}"
            )

    def _smart_fill_form(self, field_name: str, value: str, clear_first: bool = True) -> ContentResult:
        """Smart form field detection and filling"""
        if not field_name or not value:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="field_name and value required"
            )
        
        try:
            start_time = time.time()
            driver = self.session.get_driver()
            
            # Field detection strategies
            selectors = [
                f"input[name='{field_name}']",
                f"input[placeholder*='{field_name}' i]",
                f"input[id*='{field_name}' i]",
                f"input[class*='{field_name}' i]",
                f"textarea[name='{field_name}']",
                f"textarea[placeholder*='{field_name}' i]",
                f"textarea[id*='{field_name}' i]",
                f"select[name='{field_name}']",
                f"select[id*='{field_name}' i]",
                f"input[aria-label*='{field_name}' i]",
                f"textarea[aria-label*='{field_name}' i]"
            ]
            
            strategy_used = "none"
            
            for selector in selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        element = elements[0]
                        
                        # Handle different input types intelligently
                        tag_name = element.tag_name.lower()
                        if tag_name == "select":
                            # Smart dropdown selection
                            select = Select(element)
                            try:
                                select.select_by_visible_text(value)
                                strategy_used = f"dropdown by text: {selector}"
                            except:
                                try:
                                    select.select_by_value(value)
                                    strategy_used = f"dropdown by value: {selector}"
                                except:
                                    select.select_by_index(0)  # Fallback to first option
                                    strategy_used = f"dropdown by index: {selector}"
                        else:
                            # Text input
                            if clear_first:
                                element.clear()
                            element.send_keys(value)
                            strategy_used = f"text input: {selector}"
                        
                        break
                        
                except Exception as e:
                    continue
            
            # Fallback: Use helium if available
            if strategy_used == "none" and DEPENDENCIES_AVAILABLE:
                try:
                    if clear_first:
                        helium.write("", into=field_name, clear=True)
                    helium.write(value, into=field_name)
                    strategy_used = "helium form fill"
                except Exception as e:
                    pass
            
            if strategy_used != "none":
                time.sleep(1)
                screenshot_path = self._take_smart_screenshot("after_form_fill")
                page_info = self._analyze_page_content()
                
                self._track_action("fill_form", f"Filled '{field_name}' with '{value}' using {strategy_used}", time.time() - start_time, True)
                
                return ContentResult(
                    content_type="form_fill",
                    url=page_info.get("url"),
                    screenshot_path=screenshot_path,
                    content_text=f"Filled '{field_name}' with '{value}' using {strategy_used}",
                    metadata={"field_name": field_name, "value": value, "strategy": strategy_used},
                    handoff_instructions=f"Filled '{field_name}' with '{value}'. Check screenshot for result.",
                    success=True
                )
            else:
                self._track_action("fill_form", f"Could not find field '{field_name}'", time.time() - start_time, False)
                return ContentResult(
                    content_type="error",
                    success=False,
                    error_message=f"Could not find field '{field_name}'",
                    handoff_instructions=f"Field '{field_name}' not found. Try extract_structured_data to see available form fields."
                )
            
        except Exception as e:
            self._track_action("fill_form", f"Form fill error: {str(e)}", 0, False)
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Form fill failed: {str(e)}"
            )

    def _smart_perform_search(self, query: str, search_button_text: str = "Search") -> ContentResult:
        """Search workflow with button detection"""
        if not query:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="query required for search"
            )
        
        try:
            start_time = time.time()
            
            # Step 1: Smart type in search box
            type_result = self._smart_type_text(query, clear_first=True)
            if not type_result.success:
                return type_result
            
            # Step 2: Intelligent search button detection and clicking
            search_clicked = False
            strategy_used = "none"
            
            # Multiple search button strategies
            search_buttons = [
                search_button_text,
                "Search",
                "Go",
                "Submit",
                "Find",
                "🔍",  # Search icon
                "search",
                "SEARCH"
            ]
            
            driver = self.session.get_driver()
            
            for button_text in search_buttons:
                try:
                    # Try different button selectors
                    button_selectors = [
                        (By.XPATH, f"//button[contains(text(), '{button_text}')]"),
                        (By.XPATH, f"//input[@value='{button_text}']"),
                        (By.XPATH, f"//input[@type='submit' and contains(@value, '{button_text}')]"),
                        (By.XPATH, f"//a[contains(text(), '{button_text}')]"),
                        (By.XPATH, f"//*[@role='button' and contains(text(), '{button_text}')]")
                    ]
                    
                    for selector_type, selector in button_selectors:
                        try:
                            elements = driver.find_elements(selector_type, selector)
                            if elements and elements[0].is_enabled():
                                elements[0].click()
                                search_clicked = True
                                strategy_used = f"button click: {button_text}"
                                break
                        except:
                            continue
                    
                    if search_clicked:
                        break
                        
                except:
                    continue
            
            # Step 3: Fallback to Enter key
            if not search_clicked:
                try:
                    ActionChains(driver).send_keys(Keys.RETURN).perform()
                    search_clicked = True
                    strategy_used = "enter key"
                except:
                    pass
            
            # Step 4: Wait for results and analyze
            time.sleep(2)
            
            if search_clicked:
                screenshot_path = self._take_smart_screenshot("search_results")
                page_info = self._analyze_page_content()
                
                self._track_action("perform_search", f"Search '{query}' completed using {strategy_used}", time.time() - start_time, True)
                
                # Track search in workflow state
                self.workflow_state["search_history"].append({
                    "query": query,
                    "timestamp": time.time(),
                    "success": True,
                    "strategy": strategy_used
                })
                
                return ContentResult(
                    content_type="search_results",
                    url=page_info.get("url"),
                    screenshot_path=screenshot_path,
                    content_text=f"Search completed for '{query}' using {strategy_used}",
                    metadata={"query": query, "strategy": strategy_used, "page_info": page_info},
                    handoff_instructions=f"Search completed for '{query}'. Check screenshot for results. Use extract_structured_data to get detailed results.",
                    success=True
                )
            else:
                self._track_action("perform_search", f"Search '{query}' failed - no button found", time.time() - start_time, False)
                return ContentResult(
                    content_type="error",
                    success=False,
                    error_message=f"Typed '{query}' but couldn't find search button",
                    handoff_instructions="Search query entered but no search button found. Try pressing Enter manually with press_key_combination."
                )
                
        except Exception as e:
            self._track_action("perform_search", f"Search error: {str(e)}", 0, False)
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Search failed: {str(e)}"
            )

    def _smart_select_dropdown(self, dropdown_selector: str, option_text: str) -> ContentResult:
        """Smart dropdown selection with multiple strategies"""
        if not dropdown_selector or not option_text:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="dropdown_selector and option_text required"
            )
        
        try:
            start_time = time.time()
            driver = self.session.get_driver()
            strategy_used = "none"
            
            # Strategy 1: Use helium's select if available
            if DEPENDENCIES_AVAILABLE:
                try:
                    helium.select(option_text, from_=dropdown_selector)
                    strategy_used = "helium select"
                    success = True
                except Exception as e:
                    print(f"Helium select failed: {e}")
                    success = False
            else:
                success = False
            
            # Strategy 2: Intelligent dropdown detection
            if not success:
                selectors = [
                    f"select[name='{dropdown_selector}']",
                    f"select[id='{dropdown_selector}']",
                    f"select[class*='{dropdown_selector}']",
                    dropdown_selector if dropdown_selector.startswith(('.', '#', '[')) else f"#{dropdown_selector}"
                ]
                
                for selector in selectors:
                    try:
                        dropdown = driver.find_element(By.CSS_SELECTOR, selector)
                        select = Select(dropdown)
                        
                        # Try multiple selection strategies
                        try:
                            select.select_by_visible_text(option_text)
                            strategy_used = f"by visible text: {selector}"
                            success = True
                            break
                        except:
                            try:
                                select.select_by_value(option_text)
                                strategy_used = f"by value: {selector}"
                                success = True
                                break
                            except:
                                # Try partial match
                                for option in select.options:
                                    if option_text.lower() in option.text.lower():
                                        select.select_by_visible_text(option.text)
                                        strategy_used = f"by partial match: {selector}"
                                        success = True
                                        break
                                if success:
                                    break
                                    
                    except Exception as e:
                        continue
            
            if success:
                time.sleep(1)
                screenshot_path = self._take_smart_screenshot("after_dropdown_select")
                page_info = self._analyze_page_content()
                
                self._track_action("select_dropdown", f"Selected '{option_text}' using {strategy_used}", time.time() - start_time, True)
                
                return ContentResult(
                    content_type="dropdown_select",
                    url=page_info.get("url"),
                    screenshot_path=screenshot_path,
                    content_text=f"Selected '{option_text}' using {strategy_used}",
                    metadata={"dropdown_selector": dropdown_selector, "option_text": option_text, "strategy": strategy_used},
                    handoff_instructions=f"Selected '{option_text}' from dropdown. Check screenshot for result.",
                    success=True
                )
            else:
                self._track_action("select_dropdown", f"Could not find dropdown '{dropdown_selector}'", time.time() - start_time, False)
                return ContentResult(
                    content_type="error",
                    success=False,
                    error_message=f"Could not find dropdown '{dropdown_selector}' or option '{option_text}'",
                    handoff_instructions="Dropdown not found. Try extract_structured_data to see available dropdowns."
                )
            
        except Exception as e:
            self._track_action("select_dropdown", f"Dropdown error: {str(e)}", 0, False)
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Dropdown select failed: {str(e)}"
            )

    def _smart_search_text(self, search_text: str, nth_result: int) -> ContentResult:
        """Enhanced text search with highlighting and focus"""
        if not search_text:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="search_text required"
            )
        
        try:
            start_time = time.time()
            driver = self.session.get_driver()
            
            # Use browser's built-in search (Ctrl+F)
            ActionChains(driver).key_down(Keys.CONTROL).send_keys('f').key_up(Keys.CONTROL).perform()
            time.sleep(0.5)
            
            # Type the search term
            ActionChains(driver).send_keys(search_text).perform()
            time.sleep(0.5)
            
            # Navigate to nth result if needed
            if nth_result > 1:
                for _ in range(nth_result - 1):
                    ActionChains(driver).send_keys(Keys.F3).perform()
                    time.sleep(0.2)
            
            # Close search box
            ActionChains(driver).send_keys(Keys.ESCAPE).perform()
            
            # 🎯 AGENT INTELLIGENCE: Find and highlight elements
            elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{search_text}')]")
            
            if elements and nth_result <= len(elements):
                elem = elements[nth_result - 1]
                driver.execute_script("arguments[0].scrollIntoView(true);", elem)
                
                # Smart highlighting with temporary visual feedback
                driver.execute_script("""
                    arguments[0].style.backgroundColor = 'yellow';
                    arguments[0].style.border = '2px solid red';
                    arguments[0].style.boxShadow = '0 0 10px red';
                    setTimeout(function() {
                        arguments[0].style.backgroundColor = '';
                        arguments[0].style.border = '';
                        arguments[0].style.boxShadow = '';
                    }, 3000);
                """, elem)
                
                screenshot_path = self._take_smart_screenshot("search_result")
                
                result_text = f"Found {len(elements)} matches for '{search_text}'. Focused on element {nth_result} of {len(elements)}"
                self._track_action("search_text", result_text, time.time() - start_time, True)
                
                return ContentResult(
                    content_type="search_result",
                    screenshot_path=screenshot_path,
                    content_text=result_text,
                    metadata={"search_text": search_text, "total_matches": len(elements), "focused_result": nth_result},
                    handoff_instructions=f"Found and highlighted '{search_text}' (result {nth_result} of {len(elements)}). Element is highlighted in red. You can now click on it.",
                    success=True
                )
            else:
                result_text = f"Match n°{nth_result} not found (only {len(elements)} matches found)"
                self._track_action("search_text", result_text, time.time() - start_time, False)
                
                return ContentResult(
                    content_type="search_result",
                    content_text=result_text,
                    handoff_instructions=f"Only found {len(elements)} matches for '{search_text}'. Try a different search term or use nth_result <= {len(elements)}.",
                    success=True if len(elements) > 0 else False
                )
                
        except Exception as e:
            result_text = f"Search failed: {str(e)}"
            self._track_action("search_text", result_text, 0, False)
            return ContentResult(
                content_type="error",
                success=False,
                error_message=result_text
            )

    def _smart_scroll_page(self, direction: str, pixels: int) -> ContentResult:
        """Scrolling with content analysis"""
        try:
            start_time = time.time()
            
            # Get page info before scrolling
            old_scroll_pos = self.session.get_driver().execute_script("return window.scrollY;")
            
            # Perform scroll using basic session method
            if direction.lower() == "down":
                self.session.scroll_down(pixels)
                result_text = f"Scrolled down {pixels} pixels"
            elif direction.lower() == "up":
                self.session.scroll_up(pixels)
                result_text = f"Scrolled up {pixels} pixels"
            elif direction.lower() == "top":
                self.session.scroll_to_top()
                result_text = "Scrolled to top of page"
            elif direction.lower() == "bottom":
                self.session.scroll_to_bottom()
                result_text = "Scrolled to bottom of page"
            else:
                return ContentResult(
                    content_type="error",
                    success=False,
                    error_message="Invalid direction. Use 'up', 'down', 'top', or 'bottom'"
                )
            
            # 🎯 AGENT INTELLIGENCE: Analyze what's now visible
            new_scroll_pos = self.session.get_driver().execute_script("return window.scrollY;")
            viewport_height = self.session.get_driver().execute_script("return window.innerHeight;")
            page_height = self.session.get_driver().execute_script("return document.body.scrollHeight;")
            
            scroll_percentage = (new_scroll_pos / (page_height - viewport_height)) * 100 if page_height > viewport_height else 100
            
            screenshot_path = self._take_smart_screenshot("after_scroll")
            
            self._track_action("scroll_page", result_text, time.time() - start_time, True)
            
            return ContentResult(
                content_type="scroll_result",
                screenshot_path=screenshot_path,
                content_text=result_text,
                metadata={
                    "old_position": old_scroll_pos,
                    "new_position": new_scroll_pos,
                    "scroll_percentage": f"{scroll_percentage:.1f}%",
                    "page_height": page_height,
                    "viewport_height": viewport_height
                },
                handoff_instructions=f"Scroll completed: {result_text}. Now at {scroll_percentage:.1f}% of page. Check screenshot for new content.",
                success=True
            )
            
        except Exception as e:
            self._track_action("scroll_page", f"Scroll error: {str(e)}", 0, False)
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Scroll failed: {str(e)}"
            )

    def _extract_structured_data(self) -> ContentResult:
        """Comprehensive structured data extraction with smart analysis"""
        try:
            start_time = time.time()
            driver = self.session.get_driver()
            
            structured_data = {
                "tables": [],
                "lists": [],
                "forms": [],
                "links": [],
                "images": [],
                "headings": [],
                "buttons": [],
                "inputs": []
            }
            
            # Table content extraction
            tables = driver.find_elements(By.TAG_NAME, "table")
            for i, table in enumerate(tables):
                try:
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    table_data = []
                    for row in rows:
                        cells = row.find_elements(By.TAG_NAME, "td") + row.find_elements(By.TAG_NAME, "th")
                        if cells:  # Only add non-empty rows
                            table_data.append([cell.text.strip() for cell in cells])
                    
                    if table_data:  # Only add tables with data
                        structured_data["tables"].append({
                            "index": i,
                            "rows": len(table_data),
                            "columns": len(table_data[0]) if table_data else 0,
                            "data": table_data[:5],  # First 5 rows only
                            "has_more": len(table_data) > 5
                        })
                except:
                    continue
            
            # List extraction
            lists = driver.find_elements(By.TAG_NAME, "ul") + driver.find_elements(By.TAG_NAME, "ol")
            for i, lst in enumerate(lists[:10]):  # Limit to 10 lists
                try:
                    items = lst.find_elements(By.TAG_NAME, "li")
                    list_items = [item.text.strip() for item in items if item.text.strip()]
                    if list_items:
                        structured_data["lists"].append({
                            "index": i,
                            "type": lst.tag_name,
                            "items": list_items[:10],  # First 10 items
                            "total_items": len(list_items),
                            "has_more": len(list_items) > 10
                        })
                except:
                    continue
            
            # Form analysis
            forms = driver.find_elements(By.TAG_NAME, "form")
            for i, form in enumerate(forms):
                try:
                    inputs = form.find_elements(By.TAG_NAME, "input") + form.find_elements(By.TAG_NAME, "textarea") + form.find_elements(By.TAG_NAME, "select")
                    form_fields = []
                    for inp in inputs:
                        field_info = {
                            "type": inp.get_attribute("type"),
                            "name": inp.get_attribute("name"),
                            "id": inp.get_attribute("id"),
                            "placeholder": inp.get_attribute("placeholder"),
                            "value": inp.get_attribute("value"),
                            "required": inp.get_attribute("required") is not None,
                            "tag": inp.tag_name
                        }
                        # Only add if it has useful attributes
                        if any([field_info["name"], field_info["id"], field_info["placeholder"]]):
                            form_fields.append(field_info)
                    
                    if form_fields:
                        structured_data["forms"].append({
                            "index": i,
                            "action": form.get_attribute("action"),
                            "method": form.get_attribute("method"),
                            "fields": form_fields
                        })
                except:
                    continue
            
            # Link extraction
            links = driver.find_elements(By.TAG_NAME, "a")
            for link in links[:20]:  # Limit to first 20 links
                try:
                    href = link.get_attribute("href")
                    text = link.text.strip()
                    if href and text:  # Only links with both href and text
                        structured_data["links"].append({
                            "text": text,
                            "href": href,
                            "title": link.get_attribute("title")
                        })
                except:
                    continue
            
            # Button detection
            buttons = driver.find_elements(By.TAG_NAME, "button") + driver.find_elements(By.XPATH, "//input[@type='submit']") + driver.find_elements(By.XPATH, "//input[@type='button']")
            for button in buttons[:15]:  # Limit to 15 buttons
                try:
                    text = button.text.strip() or button.get_attribute("value") or button.get_attribute("title")
                    if text:
                        structured_data["buttons"].append({
                            "text": text,
                            "type": button.get_attribute("type"),
                            "id": button.get_attribute("id"),
                            "class": button.get_attribute("class")
                        })
                except:
                    continue
            
            # Input field detection
            inputs = driver.find_elements(By.TAG_NAME, "input") + driver.find_elements(By.TAG_NAME, "textarea")
            for inp in inputs[:15]:  # Limit to 15 inputs
                try:
                    field_type = inp.get_attribute("type")
                    if field_type not in ["submit", "button", "hidden"]:  # Skip non-input types
                        structured_data["inputs"].append({
                            "type": field_type,
                            "name": inp.get_attribute("name"),
                            "id": inp.get_attribute("id"),
                            "placeholder": inp.get_attribute("placeholder"),
                            "class": inp.get_attribute("class")
                        })
                except:
                    continue
            
            # Extract headings
            for level in range(1, 7):
                headings = driver.find_elements(By.TAG_NAME, f"h{level}")
                for heading in headings[:10]:  # Limit per level
                    text = heading.text.strip()
                    if text:
                        structured_data["headings"].append({
                            "level": level,
                            "text": text
                        })
            
            screenshot_path = self._take_smart_screenshot("structured_data_extract")
            page_info = self._analyze_page_content()
            
            # Create summary for handoff instructions
            summary = []
            if structured_data["tables"]:
                summary.append(f"{len(structured_data['tables'])} tables")
            if structured_data["forms"]:
                summary.append(f"{len(structured_data['forms'])} forms")
            if structured_data["buttons"]:
                summary.append(f"{len(structured_data['buttons'])} buttons")
            if structured_data["inputs"]:
                summary.append(f"{len(structured_data['inputs'])} input fields")
            if structured_data["links"]:
                summary.append(f"{len(structured_data['links'])} links")
            
            summary_text = ", ".join(summary) if summary else "No structured data found"
            
            self._track_action("extract_structured_data", f"Extracted: {summary_text}", time.time() - start_time, True)
            
            return ContentResult(
                content_type="structured_data",
                url=page_info.get("url"),
                screenshot_path=screenshot_path,
                content_text=f"Structured data extracted: {summary_text}",
                metadata=structured_data,
                handoff_instructions=f"Extracted {summary_text}. Use this data to understand available interactions. All details in metadata.",
                success=True
            )
            
        except Exception as e:
            self._track_action("extract_structured_data", f"Extraction error: {str(e)}", 0, False)
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Structured data extraction failed: {str(e)}"
            )

    def _extract_content(self) -> ContentResult:
        """Content extraction with analysis"""
        try:
            driver = self.session.get_driver()
            
            # Get full page text
            page_text = driver.find_element(By.TAG_NAME, "body").text
            page_info = self._analyze_page_content()
            
            # Content analysis
            content_stats = {
                "total_characters": len(page_text),
                "total_words": len(page_text.split()),
                "total_lines": len(page_text.split('\n')),
                "content_type": page_info.get("content_type", "unknown")
            }
            
            # Truncate if too long but provide stats
            display_text = page_text
            if len(page_text) > 5000:
                display_text = page_text[:5000] + f"\n... [TRUNCATED - showing first 5000 of {len(page_text)} characters]"
            
            screenshot_path = self._take_smart_screenshot("content_extract")
            
            return ContentResult(
                content_type="web_content",
                content_text=display_text,
                url=page_info.get("url"),
                screenshot_path=screenshot_path,
                metadata={**page_info, **content_stats},
                handoff_instructions=f"Extracted {content_stats['total_words']} words from {page_info.get('title', 'current page')}. Content type: {content_stats['content_type']}. Use search_text to find specific information.",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Content extraction failed: {str(e)}"
            )

    def _take_smart_screenshot(self, description: str = "screenshot") -> Optional[str]:
        """Take screenshot with metadata"""
        try:
            # Use basic session screenshot but add intelligent naming
            timestamp = int(time.time())
            action_count = self.workflow_state["performance_metrics"]["total_actions"]
            smart_filename = f"{description}_{action_count}_{timestamp}"
            
            return self.session.take_screenshot(smart_filename)
        except Exception as e:
            print(f"⚠️ Smart screenshot failed: {e}")
            return None

    def _analyze_page_content(self) -> Dict:
        """Comprehensive page analysis"""
        try:
            driver = self.session.get_driver()
            
            # Basic page info
            page_info = {
                "url": driver.current_url,
                "title": driver.title,
                "page_source_length": len(driver.page_source),
                "timestamp": time.time(),
                "viewport_size": driver.execute_script("return {width: window.innerWidth, height: window.innerHeight};"),
                "scroll_position": driver.execute_script("return {x: window.scrollX, y: window.scrollY};")
            }
            
            # 🎯 AGENT INTELLIGENCE: Smart content type detection
            page_info["content_type"] = self._detect_content_type()
            
            # Additional smart analysis
            try:
                # Count interactive elements
                page_info["interactive_elements"] = {
                    "buttons": len(driver.find_elements(By.TAG_NAME, "button")),
                    "links": len(driver.find_elements(By.TAG_NAME, "a")),
                    "inputs": len(driver.find_elements(By.TAG_NAME, "input")),
                    "forms": len(driver.find_elements(By.TAG_NAME, "form")),
                    "tables": len(driver.find_elements(By.TAG_NAME, "table"))
                }
                
                # Check for common indicators
                page_source_lower = driver.page_source.lower()
                page_info["page_indicators"] = {
                    "has_search": "search" in page_source_lower,
                    "has_login": any(term in page_source_lower for term in ["login", "sign in", "password"]),
                    "has_forms": page_info["interactive_elements"]["forms"] > 0,
                    "has_tables": page_info["interactive_elements"]["tables"] > 0,
                    "is_responsive": "viewport" in page_source_lower
                }
                
            except Exception as e:
                print(f"Additional analysis failed: {e}")
            
            return page_info
            
        except Exception as e:
            return {"error": str(e), "timestamp": time.time()}

    def _detect_content_type(self) -> str:
        """Content type detection"""
        try:
            driver = self.session.get_driver()
            current_url = driver.current_url.lower()
            page_source = driver.page_source.lower()
            
            # Site-specific detection
            if "github.com" in current_url:
                if "/issues" in current_url:
                    return "github_issues"
                elif "/pull" in current_url:
                    return "github_pull_request"
                else:
                    return "github_content"
            
            if "stackoverflow.com" in current_url:
                return "stackoverflow_content"
                
            if "usgs.gov" in current_url:
                return "usgs_database"
            
            # Content-based detection
            if any(doc_term in current_url for doc_term in ["docs", "documentation", "wiki"]):
                return "documentation"
            
            if current_url.endswith('.pdf') or 'content-type: application/pdf' in page_source:
                return "direct_pdf"
            
            if any(term in page_source for term in ['pdf', 'download pdf', 'view pdf', '.pdf"']):
                return "embedded_pdf"
            
            if '<table' in page_source and page_source.count('<table') > 2:
                return "data_table_content"
            
            if any(term in page_source for term in ['<form', 'input type=', 'search']):
                return "interactive_form_content"
            
            if any(term in page_source for term in ['<img', 'image', 'figure', 'gallery']):
                return "image_content"
                
            if any(term in page_source for term in ['article', 'blog', 'post', 'news']):
                return "article_content"
            
            return "web_content"
            
        except Exception:
            return "unknown"

    def _track_action(self, action_type: str, result: str, duration: float, success: bool):
        """Performance tracking and analytics"""
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
            "url": self.session.get_driver().current_url if self._session and self._session.is_active else None
        }
        
        # Store in workflow state
        if action_type not in self.workflow_state:
            self.workflow_state[action_type] = []
        self.workflow_state[action_type].append(action_log)

    def _get_workflow_summary(self) -> Dict:
        """🎯Workflow performance summary"""
        metrics = self.workflow_state["performance_metrics"]
        
        success_rate = 0
        if metrics["total_actions"] > 0:
            success_rate = metrics["successful_actions"] / metrics["total_actions"]
        
        return {
            "total_actions": metrics["total_actions"],
            "successful_actions": metrics["successful_actions"],
            "success_rate": f"{success_rate:.2%}",
            "average_response_time": f"{metrics['average_response_time']:.2f}s",
            "searches_performed": len(self.workflow_state["search_history"]),
            "current_site": self.workflow_state["current_site"]
        }

    # Basic action methods that delegate to session

    def _press_key_combination(self, keys: str) -> ContentResult:
        """Press key combination using basic session method"""
        if not keys:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="key_combination required"
            )
        
        try:
            result = self.session.press_keys(keys)
            screenshot_path = self._take_smart_screenshot("after_key_press")
            page_info = self._analyze_page_content()
            
            return ContentResult(
                content_type="key_press",
                url=page_info.get("url"),
                screenshot_path=screenshot_path,
                content_text=result,
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

    def _wait_for_element(self, selector: str, timeout: int) -> ContentResult:
        """Wait for element using basic session method"""
        if not selector:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="element_selector required"
            )
        
        try:
            result = self.session.wait_for_element(selector, timeout)
            screenshot_path = self._take_smart_screenshot("after_element_wait")
            page_info = self._analyze_page_content()
            
            return ContentResult(
                content_type="element_wait",
                url=page_info.get("url"),
                screenshot_path=screenshot_path,
                content_text=result,
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
        """Wait for page load using basic session method"""
        try:
            result = self.session.wait_for_page_load(timeout)
            screenshot_path = self._take_smart_screenshot("after_page_load")
            page_info = self._analyze_page_content()
            
            return ContentResult(
                content_type="page_load",
                url=page_info.get("url"),
                screenshot_path=screenshot_path,
                content_text=result,
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

    def _close_popups(self) -> ContentResult:
        """Close popups using basic session method"""
        try:
            result = self.session.close_popups()
            screenshot_path = self._take_smart_screenshot("after_popup_close")
            
            return ContentResult(
                content_type="popup_close",
                screenshot_path=screenshot_path,
                content_text=result,
                handoff_instructions=f"Popup close attempted: {result}",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Popup close failed: {str(e)}"
            )

    def _take_screenshot(self) -> ContentResult:
        """Take screenshot using smart method"""
        try:
            screenshot_path = self._take_smart_screenshot("manual")
            page_info = self._analyze_page_content()
            
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

    def _close_browser(self) -> ContentResult:
        """Close browser and provide summary"""
        try:
            if self._session:
                summary = self._get_workflow_summary()
                close_result = self._session.close()
                self._session = None
                self._initialized = False
                
                return ContentResult(
                    content_type="session_close",
                    content_text=f"Browser session closed. {close_result}",
                    metadata=summary,
                    handoff_instructions=f"Browser session closed successfully. Performance: {summary['success_rate']} success rate, {summary['total_actions']} total actions.",
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
        """Result formatting with session info"""
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
        
        # Include workflow performance info
        if self._session:
            result_dict["session_info"] = {
                "browser_active": self._session.is_active,
                "workflow_summary": self._get_workflow_summary()
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


class BrowserSession:
    """
    🔧 LAYER 2: BASIC BROWSER MANAGEMENT
    
    This layer only handles:
    - Chrome driver initialization and configuration
    - Basic navigation and operations  
    - Session lifecycle management
    - Resource cleanup
    - Simple screenshots and downloads
    
    NO INTELLIGENCE HERE - just basic browser operations for the tool layer to use.
    """
    
    def __init__(self, download_dir: Optional[str] = None):
        self.driver = None
        self.is_active = False
        self.download_dir = download_dir or tempfile.mkdtemp(prefix="browser_")
        self.downloaded_files = []
        self.screenshots = []
        
    def initialize(self, headless: bool = True) -> str:
        """🔧 Basic browser initialization"""
        if self.is_active:
            return "Session already active"
        
        try:
            print(f"🔧 Initializing Chrome browser (headless={headless})")
            
            # Basic Chrome configuration
            chrome_options = webdriver.ChromeOptions()
            
            if headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-popup-blocking")
            chrome_options.add_argument("--log-level=3")
            
            # Download preferences
            os.makedirs(self.download_dir, exist_ok=True)
            prefs = {
                "download.default_directory": self.download_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True
            }
            chrome_options.add_experimental_option("prefs", prefs)
            
            # Initialize driver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(60)
            self.is_active = True
            
            # Set up helium if available
            if DEPENDENCIES_AVAILABLE:
                try:
                    helium.set_driver(self.driver)
                except:
                    pass
            
            print(f"✅ Chrome browser initialized successfully")
            return "Session initialized successfully"
            
        except Exception as e:
            error_msg = f"Browser initialization failed: {e}"
            print(f"❌ {error_msg}")
            return f"Initialization failed: {error_msg}"

    def navigate_to(self, url: str) -> bool:
        """🔧 Basic navigation"""
        if not self.is_active:
            return False
            
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            self.driver.get(url)
            return True
            
        except Exception as e:
            print(f"❌ Navigation failed: {e}")
            return False

    def get_driver(self):
        """🔧 Get raw driver for advanced operations"""
        return self.driver

    def take_screenshot(self, filename: str) -> Optional[str]:
        """🔧 Basic screenshot capture"""
        try:
            if not self.is_active:
                return None
                
            time.sleep(1.0)  # Let page settle
            
            timestamp = int(time.time())
            screenshot_path = os.path.join(self.download_dir, f"{filename}_{timestamp}.png")
            
            self.driver.save_screenshot(screenshot_path)
            
            self.screenshots.append({
                "path": screenshot_path,
                "timestamp": timestamp,
                "url": self.driver.current_url,
                "title": self.driver.title
            })
            
            return screenshot_path
            
        except Exception as e:
            print(f"⚠️ Screenshot failed: {e}")
            return None

    def scroll_down(self, pixels: int = 1200):
        """🔧 Basic scroll down"""
        try:
            if DEPENDENCIES_AVAILABLE and 'helium' in globals():
                helium.scroll_down(pixels)
            else:
                self.driver.execute_script(f"window.scrollBy(0, {pixels});")
        except Exception as e:
            print(f"Scroll down failed: {e}")

    def scroll_up(self, pixels: int = 1200):
        """🔧 Basic scroll up"""
        try:
            if DEPENDENCIES_AVAILABLE and 'helium' in globals():
                helium.scroll_up(pixels)
            else:
                self.driver.execute_script(f"window.scrollBy(0, -{pixels});")
        except Exception as e:
            print(f"Scroll up failed: {e}")

    def scroll_to_top(self):
        """🔧 Basic scroll to top"""
        try:
            self.driver.execute_script("window.scrollTo(0, 0);")
        except Exception as e:
            print(f"Scroll to top failed: {e}")

    def scroll_to_bottom(self):
        """🔧 Basic scroll to bottom"""
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        except Exception as e:
            print(f"Scroll to bottom failed: {e}")

    def press_keys(self, keys: str) -> str:
        """🔧 Basic key press"""
        try:
            keys_lower = keys.lower()
            
            if keys_lower == "ctrl+f":
                ActionChains(self.driver).key_down(Keys.CONTROL).send_keys('f').key_up(Keys.CONTROL).perform()
                return "Pressed Ctrl+F"
            elif keys_lower == "ctrl+a":
                ActionChains(self.driver).key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).perform()
                return "Pressed Ctrl+A"
            elif keys_lower == "escape":
                ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
                return "Pressed Escape"
            elif keys_lower == "enter":
                ActionChains(self.driver).send_keys(Keys.RETURN).perform()
                return "Pressed Enter"
            elif keys_lower == "tab":
                ActionChains(self.driver).send_keys(Keys.TAB).perform()
                return "Pressed Tab"
            else:
                ActionChains(self.driver).send_keys(keys).perform()
                return f"Pressed '{keys}'"
                
        except Exception as e:
            return f"Key press failed: {str(e)}"

    def close_popups(self) -> str:
        """🔧 Basic popup closing"""
        try:
            ActionChains(self.driver).send_keys(Keys.ESCAPE).perform()
            time.sleep(0.5)
            return "Pressed ESC to close popups"
        except Exception as e:
            return f"Popup close failed: {str(e)}"

    def wait_for_element(self, selector: str, timeout: int = 10) -> str:
        """🔧 Basic element wait"""
        try:
            wait = WebDriverWait(self.driver, timeout)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            return f"Element '{selector}' found"
        except TimeoutException:
            return f"Element '{selector}' not found within {timeout}s"
        except Exception as e:
            return f"Wait failed: {str(e)}"

    def wait_for_page_load(self, timeout: int = 30) -> str:
        """🔧 Basic page load wait"""
        try:
            wait = WebDriverWait(self.driver, timeout)
            wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
            return "Page loaded completely"
        except TimeoutException:
            return f"Page load timeout after {timeout}s"
        except Exception as e:
            return f"Page load wait failed: {str(e)}"

    def close(self) -> str:
        """🔧 Basic session closing"""
        try:
            if self.is_active:
                if DEPENDENCIES_AVAILABLE and 'helium' in globals():
                    helium.kill_browser()
                else:
                    self.driver.quit()
                    
                self.driver = None
                self.is_active = False
                
                return "Browser session closed successfully"
            return "No active session"
            
        except Exception as e:
            return f"Error closing session: {str(e)}"

    def cleanup(self):
        """🔧 Basic cleanup"""
        try:
            # Clean up downloaded files except important ones
            if os.path.exists(self.download_dir):
                for file in os.listdir(self.download_dir):
                    if not file.endswith(('.json', '.png')):  # Preserve logs and screenshots
                        try:
                            os.remove(os.path.join(self.download_dir, file))
                        except:
                            pass
            
            self.downloaded_files = []
            self.screenshots = []
            print(f"🧹 Browser session cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
