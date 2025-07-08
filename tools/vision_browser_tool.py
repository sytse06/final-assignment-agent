import os
import time
import tempfile
import shutil
import json
from typing import Optional, Dict
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

class BrowserSession:
    def __init__(self, download_dir: Optional[str] = None):
        self.driver = None
        self.is_active = False
        self.download_dir = download_dir or tempfile.mkdtemp(prefix="browser_")
        self.downloaded_files = []
        
    def initialize(self, headless: bool = False):
        if self.is_active:
            return "Session already active"
            
        try:
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--force-device-scale-factor=1")
            chrome_options.add_argument("--window-size=1000,1350")
            chrome_options.add_argument("--disable-pdf-viewer")
            chrome_options.add_argument("--disable-web-security")
            
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
            
            self.driver = helium.start_chrome(headless=headless, options=chrome_options)
            self.is_active = True
            os.makedirs(self.download_dir, exist_ok=True)
            return "Session initialized"
            
        except Exception as e:
            return f"Initialization failed: {str(e)}"
    
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
        except Exception:
            pass

_browser_session = BrowserSession()

class VisionWebBrowserTool(Tool):
    name = "vision_web_browser"
    description = """Navigate web pages, detect content types, download files, and extract content URLs. Handles PDFs, images, tables, and interactive content through browser automation."""
    inputs = {
        "task_description": {
            "type": "string",
            "description": "Description of what content to acquire from the webpage.",
        },
        "url": {
            "type": "string", 
            "description": "Target URL to navigate to.",
            "nullable": True,
        },
        "action_type": {
            "type": "string",
            "description": "Action to perform: 'discover_content', 'acquire_pdf', 'click_download', 'extract_url', 'see_image', 'close_browser'",
        },
        "target_content": {
            "type": "string",
            "description": "Specific content to look for on the page.",
            "nullable": True,
        },
        "element_text": {
            "type": "string", 
            "description": "Text of element to click for click_download action.",
            "nullable": True,
        },
        "download_timeout": {
            "type": "integer",
            "description": "Seconds to wait for downloads. Default 30.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._session = None
        self._initialized = False
    
    def setup(self):
        self._session = BrowserSession()
        self._initialized = True
    
    def forward(
        self, 
        task_description: str,
        action_type: str,
        url: Optional[str] = None,
        target_content: Optional[str] = None,
        element_text: Optional[str] = None,
        download_timeout: Optional[int] = None
    ) -> str:
        
        if not DEPENDENCIES_AVAILABLE:
            return "Dependencies not available. Install: pip install helium selenium"
        
        timeout = download_timeout or 30
        
        try:
            if not _browser_session.is_active:
                init_result = _browser_session.initialize()
                if "failed" in init_result:
                    return init_result
            
            if action_type == "discover_content":
                result = self._discover_content(url, task_description, target_content)
                
            elif action_type == "acquire_pdf":
                result = self._acquire_pdf(url, target_content)
                
            elif action_type == "click_download":
                result = self._click_download(element_text, timeout)
                
            elif action_type == "see_image":
                result = self._see_image(url, target_content)
                
            elif action_type == "extract_url":
                result = self._extract_url(url, target_content)
                
            elif action_type == "close_browser":
                result = ContentResult(
                    content_type="session",
                    handoff_instructions=_browser_session.close(),
                    success=True
                )
                
            else:
                result = ContentResult(
                    content_type="error",
                    success=False,
                    error_message=f"Unknown action_type: {action_type}"
                )
            
            return self._format_result(result)
            
        except Exception as e:
            return f"Browser error: {str(e)}"
    
    def _discover_content(self, url: str, task_description: str, target_content: Optional[str]) -> ContentResult:
        if not url:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="URL required"
            )
        
        helium.go_to(url)
        time.sleep(3)
        
        content_type = _browser_session.detect_content_type()
        
        if content_type == "direct_pdf":
            return self._handle_direct_pdf(url, target_content)
        elif content_type == "embedded_pdf":
            return self._handle_embedded_pdf(target_content)
        else:
            return self._handle_web_content(url, target_content)
    
    def _acquire_pdf(self, url: str, target_content: Optional[str]) -> ContentResult:
        if not url:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="URL required"
            )
        
        helium.go_to(url)
        time.sleep(3)
        
        content_type = _browser_session.detect_content_type()
        
        if content_type == "direct_pdf":
            return self._handle_direct_pdf(url, target_content)
        else:
            return self._handle_embedded_pdf(target_content)
    
    def _click_download(self, element_text: str, timeout: int) -> ContentResult:
        if not element_text:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="element_text required"
            )
        
        try:
            helium.click(element_text)
            downloaded_file = _browser_session.wait_for_download(timeout)
            
            if downloaded_file:
                file_ext = Path(downloaded_file).suffix.lower()
                content_type = "pdf" if file_ext == ".pdf" else "file"
                
                return ContentResult(
                    content_type=content_type,
                    file_path=downloaded_file,
                    handoff_instructions=f"File downloaded to {downloaded_file}. Use content_retriever_tool to process.",
                    success=True
                )
            else:
                return ContentResult(
                    content_type="error",
                    success=False,
                    error_message=f"Download timeout after {timeout} seconds"
                )
                
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Click failed: {str(e)}"
            )
    
    def _see_image(self, url: str, target_content: Optional[str]) -> ContentResult:
        if not url:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="URL required"
            )
        
        try:
            helium.go_to(url)
            time.sleep(3)
            
            # Take screenshot for visual analysis
            driver = helium.get_driver()
            png_bytes = driver.get_screenshot_as_png()
            
            # Save screenshot to temporary file
            screenshot_path = os.path.join(_browser_session.download_dir, "screenshot.png")
            with open(screenshot_path, 'wb') as f:
                f.write(png_bytes)
            
            # Look for images on the page
            images = driver.find_elements(By.TAG_NAME, "img")
            image_info = []
            
            for img in images[:5]:  # Limit to first 5 images
                src = img.get_attribute('src')
                alt = img.get_attribute('alt') or ""
                if src:
                    image_info.append(f"Image: {src} (alt: {alt})")
            
            image_list = "\n".join(image_info) if image_info else "No images found"
            
            return ContentResult(
                content_type="image",
                file_path=screenshot_path,
                url=url,
                content_text=image_list,
                handoff_instructions=f"Screenshot saved to {screenshot_path}. Images on page: {len(images)}. Use image analysis tools for visual content about: {target_content or 'visual elements'}",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Image capture failed: {str(e)}"
            )
    
    def _extract_url(self, url: str, target_content: Optional[str]) -> ContentResult:
        if not url:
            return ContentResult(
                content_type="error",
                success=False,
                error_message="URL required"
            )
        
        helium.go_to(url)
        time.sleep(2)
        
        pdf_url = _browser_session.extract_pdf_url()
        
        if pdf_url:
            return ContentResult(
                content_type="pdf",
                url=pdf_url,
                handoff_instructions=f"PDF URL found: {pdf_url}. Use content_retriever_tool to process.",
                success=True
            )
        else:
            return ContentResult(
                content_type="web_content",
                url=url,
                success=False,
                error_message="No PDF URLs found"
            )
    
    def _handle_direct_pdf(self, url: str, target_content: Optional[str]) -> ContentResult:
        try:
            current_url = helium.get_driver().current_url
            downloaded_file = _browser_session.download_url_directly(current_url)
            
            if downloaded_file:
                return ContentResult(
                    content_type="pdf",
                    file_path=downloaded_file,
                    url=current_url,
                    handoff_instructions=f"PDF downloaded to {downloaded_file}. Use content_retriever_tool to extract text and search for: {target_content or 'relevant content'}",
                    success=True
                )
            else:
                return ContentResult(
                    content_type="pdf",
                    url=current_url,
                    handoff_instructions=f"PDF URL: {current_url}. Use content_retriever_tool to process this URL and search for: {target_content or 'relevant content'}",
                    success=True
                )
                
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Direct PDF handling failed: {str(e)}"
            )
    
    def _handle_embedded_pdf(self, target_content: Optional[str]) -> ContentResult:
        try:
            driver = helium.get_driver()
            pdf_elements = driver.find_elements(By.XPATH, "//a[contains(@href, '.pdf') or contains(text(), 'PDF') or contains(text(), 'Download')]")
            
            if pdf_elements:
                pdf_elements[0].click()
                time.sleep(2)
                
                current_url = driver.current_url
                if current_url.endswith('.pdf'):
                    downloaded_file = _browser_session.download_url_directly(current_url)
                    
                    if downloaded_file:
                        return ContentResult(
                            content_type="pdf",
                            file_path=downloaded_file,
                            url=current_url,
                            handoff_instructions=f"PDF downloaded to {downloaded_file}. Use content_retriever_tool to extract text and search for: {target_content or 'relevant content'}",
                            success=True
                        )
                    else:
                        return ContentResult(
                            content_type="pdf",
                            url=current_url,
                            handoff_instructions=f"PDF URL: {current_url}. Use content_retriever_tool to process this URL and search for: {target_content or 'relevant content'}",
                            success=True
                        )
                else:
                    downloaded_file = _browser_session.wait_for_download(30)
                    
                    if downloaded_file:
                        return ContentResult(
                            content_type="pdf",
                            file_path=downloaded_file,
                            handoff_instructions=f"PDF downloaded to {downloaded_file}. Use content_retriever_tool to extract text and search for: {target_content or 'relevant content'}",
                            success=True
                        )
            
            pdf_url = _browser_session.extract_pdf_url()
            if pdf_url:
                return ContentResult(
                    content_type="pdf",
                    url=pdf_url,
                    handoff_instructions=f"PDF URL found: {pdf_url}. Use content_retriever_tool to process this URL and search for: {target_content or 'relevant content'}",
                    success=True
                )
            
            return ContentResult(
                content_type="web_content",
                success=False,
                error_message="No PDF content found"
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Embedded PDF handling failed: {str(e)}"
            )
    
    def _handle_web_content(self, url: str, target_content: Optional[str]) -> ContentResult:
        try:
            driver = helium.get_driver()
            page_text = driver.find_element(By.TAG_NAME, "body").text
            
            if len(page_text) > 5000:
                page_text = page_text[:5000] + "..."
            
            return ContentResult(
                content_type="web_content",
                content_text=page_text,
                url=url,
                handoff_instructions=f"Web page content extracted. Search for: {target_content or 'relevant content'}",
                success=True
            )
            
        except Exception as e:
            return ContentResult(
                content_type="error",
                success=False,
                error_message=f"Web content handling failed: {str(e)}"
            )
    
    def _format_result(self, result: ContentResult) -> str:
        result_dict = {
            "success": result.success,
            "content_type": result.content_type,
            "handoff_instructions": result.handoff_instructions,
        }
        
        if result.file_path:
            result_dict["file_path"] = result.file_path
        
        if result.url:
            result_dict["url"] = result.url
        
        if result.content_text:
            text = result.content_text[:500] + "..." if len(result.content_text) > 500 else result.content_text
            result_dict["content_text"] = text
        
        if result.metadata:
            result_dict["metadata"] = result.metadata
        
        if result.error_message:
            result_dict["error_message"] = result.error_message
        
        result_dict["files_downloaded"] = len(_browser_session.downloaded_files)
        
        return json.dumps(result_dict, indent=2)

def cleanup_browser_session():
    global _browser_session
    _browser_session.close()
    _browser_session.cleanup()
    return "Browser session cleaned up"