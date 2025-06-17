# tools/get_attachment_tool.py
# GAIA file attachment retrieval tool

from smolagents import Tool
import requests
from urllib.parse import urljoin
import base64
import tempfile
import os
from typing import Optional
from pathlib import Path


class GetAttachmentTool(Tool):
    """
    Retrieves GAIA task attachments from the evaluation API.
    
    Essential for HF Spaces deployment and GAIA benchmark evaluation.
    Supports multiple output formats for flexible file processing.
    """
    
    name = "get_attachment"
    description = """Retrieves the attachment file for the current GAIA task in specified format.
    
Use this tool to access files (Excel, PDF, images, audio, etc.) that are part of GAIA questions.
The tool can return the file in different formats depending on your processing needs.

Returns empty string if no file is attached to the current task."""

    inputs = {
        "fmt": {
            "type": "string", 
            "description": "Format to retrieve attachment. Options: 'URL', 'DATA_URL', 'LOCAL_FILE_PATH', 'TEXT'.",
            "nullable": True,
            "default": "URL"
        }
    }
    output_type = "string"

    def __init__(
        self, 
        agent_evaluation_api: Optional[str] = None,
        task_id: Optional[str] = None,
        hf_cache_fallback: bool = True,  
        **kwargs
    ):
        """
        Initialize GetAttachmentTool.
        
        Args:
            agent_evaluation_api: Base URL for evaluation API 
            task_id: Current task ID (can be set later)
        """
        self.agent_evaluation_api = (
            agent_evaluation_api or 
            "https://agents-course-unit4-scoring.hf.space/"
        )
        self.task_id = task_id
        super().__init__(**kwargs)

    def attachment_for(self, task_id: Optional[str]):
        """Set the task ID for file retrieval"""
        self.task_id = task_id
        if task_id:
            print(f"🔗 GetAttachmentTool configured for task: {task_id}")

    def forward(self, fmt: str = "URL") -> str:
        """
        Retrieve attachment in specified format.
        
        Args:
            fmt: Format type - URL, DATA_URL, LOCAL_FILE_PATH, or TEXT
            
        Returns:
            File content in requested format, or empty string if no file
        """
        # This happens when the tool is called incorrectly by the agent
        if fmt and len(fmt) > 10 and '-' in fmt:
            # This looks like a task_id, not a format
            print(f"⚠️  Detected task_id passed as format: {fmt}")
            print(f"🔧 Auto-correcting: setting task_id and using default format 'URL'")
            self.task_id = fmt
            fmt = "LOCAL_FILE_PATH"
        
        fmt = fmt.upper().strip()
        
        # Validate format
        valid_formats = ["URL", "DATA_URL", "LOCAL_FILE_PATH", "TEXT"]
        if fmt not in valid_formats:
            return f"Error: Invalid format '{fmt}'. Use: {', '.join(valid_formats)}"

        # Check if task_id is available
        if not self.task_id:
            print("⚠️  No task_id set - cannot retrieve attachment")
            return ""

        try:
            # Construct file URL
            file_url = urljoin(self.agent_evaluation_api, f"files/{self.task_id}")
            
            # For URL format, return the URL directly
            if fmt == "URL":
                print(f"📎 Returning file URL: {file_url}")
                return file_url

            # For other formats, fetch the file
            print(f"📥 Fetching file from: {file_url}")
            response = requests.get(
                file_url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=30  # Add timeout for reliability
            )
            
            # Handle client errors (4xx) - likely no file attached
            if 400 <= response.status_code < 500:
                print(f"ℹ️  No file attachment found (HTTP {response.status_code})")
            
                # Try HF cache fallback for LOCAL_FILE_PATH requests
                if fmt == "LOCAL_FILE_PATH":
                    print(f"🔍 Trying HF cache fallback...")
                    hf_cache_path = self._try_hf_cache_fallback()
                    if hf_cache_path:
                        return hf_cache_path
            
                return ""
            
            # Raise for other HTTP errors
            response.raise_for_status()
            
            # Get MIME type
            mime_type = response.headers.get("content-type", "application/octet-stream")
            print(f"📄 File type detected: {mime_type}")
            
            # Handle different output formats
            if fmt == "TEXT":
                if mime_type.startswith("text/"):
                    content = response.text
                    print(f"📝 Text content retrieved: {len(content)} characters")
                    return content
                else:
                    return f"Error: File type '{mime_type}' cannot be retrieved as TEXT. Use 'URL' or 'LOCAL_FILE_PATH' instead."
            
            elif fmt == "DATA_URL":
                encoded_content = base64.b64encode(response.content).decode('utf-8')
                data_url = f"data:{mime_type};base64,{encoded_content}"
                print(f"🔗 Data URL created: {len(data_url)} characters")
                return data_url
            
            elif fmt == "LOCAL_FILE_PATH":
                # Create temporary file with appropriate extension
                file_extension = self._get_file_extension(mime_type)
                with tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix=file_extension,
                    prefix=f"gaia_task_{self.task_id}_"
                ) as tmp_file:
                    tmp_file.write(response.content)
                    file_path = tmp_file.name
                
                print(f"💾 File saved locally: {file_path}")
                return file_path
            
            else:
                return f"Error: Unsupported format: {fmt}"
                
        except requests.exceptions.Timeout:
            # NEW: Add HF cache fallback for timeouts too
            if fmt == "LOCAL_FILE_PATH":
                print(f"⏰ API timeout, trying HF cache fallback...")
                hf_cache_path = self._try_hf_cache_fallback()
                if hf_cache_path:
                    return hf_cache_path
            
            return "Error: File download timed out. Try again or use 'URL' format."
            
        except requests.exceptions.RequestException as e:
            # NEW: Add HF cache fallback for request errors
            if fmt == "LOCAL_FILE_PATH":
                print(f"🌐 API error, trying HF cache fallback...")
                hf_cache_path = self._try_hf_cache_fallback()
                if hf_cache_path:
                    return hf_cache_path
            
            print(f"❌ Request error: {e}")
            return f"Error: Failed to retrieve file - {str(e)}"
            
        except Exception as e:
            # NEW: Add HF cache fallback for any other errors
            if fmt == "LOCAL_FILE_PATH":
                print(f"💥 Unexpected error, trying HF cache fallback...")
                hf_cache_path = self._try_hf_cache_fallback()
                if hf_cache_path:
                    return hf_cache_path
            
            print(f"❌ Unexpected error: {e}")
            return f"Error: Unexpected error retrieving attachment - {str(e)}"

    def _get_file_extension(self, mime_type: str) -> str:
        """Get appropriate file extension based on MIME type"""
        mime_to_ext = {
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "application/vnd.ms-excel": ".xls", 
            "text/csv": ".csv",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "audio/mpeg": ".mp3",
            "audio/mp4": ".m4a",
            "audio/wav": ".wav",
            "video/quicktime": ".mov",
            "video/mp4": ".mp4",
            "application/json": ".json",
            "application/xml": ".xml",
            "text/xml": ".xml",
            "application/zip": ".zip",
            "text/plain": ".txt",
            "text/python": ".py"
        }
        
        return mime_to_ext.get(mime_type, ".bin")

    def __repr__(self):
        return f"GetAttachmentTool(api='{self.agent_evaluation_api}', task_id='{self.task_id}')"

    def _try_hf_cache_fallback(self) -> Optional[str]:
        """
        NEW: Try to find file in HF cache when API fails.
        
        This method attempts to locate the file using common HF cache patterns.
        """
        if not self.task_id:
            return None
        
        # Check if we have a direct file path hint (set by agent_testing.py)
        if hasattr(self, '_hf_cache_hint') and self._hf_cache_hint:
            from pathlib import Path
            hint_path = Path(self._hf_cache_hint)
            if hint_path.exists():
                print(f"✅ Using HF cache hint: {hint_path}")
                return str(hint_path)
            else:
                print(f"⚠️  HF cache hint file not found: {hint_path}")
        
        # Try to discover HF cache path
        import os
        from pathlib import Path
        
        # Common HF cache locations
        cache_bases = [
            os.path.expanduser("~/.cache/huggingface/datasets/downloads"),
            os.path.expanduser("~/.cache/huggingface/hub/datasets"),
            "/tmp/huggingface_cache/datasets",
        ]
        
        print(f"🔍 Searching HF cache for task: {self.task_id}")
        
        for cache_base in cache_bases:
            cache_path = Path(cache_base)
            if cache_path.exists():
                print(f"   Checking: {cache_path}")
                
                # Look for files containing the task_id
                try:
                    for file_path in cache_path.rglob("*"):
                        if file_path.is_file():
                            # Check if filename contains task_id
                            if self.task_id in file_path.name:
                                print(f"✅ Found HF cache file: {file_path}")
                                return str(file_path)
                            
                            # Also check if file content might match (for renamed files)
                            # This is a heuristic - could be improved
                            if file_path.stat().st_size > 1000:  # Only check reasonably sized files
                                try:
                                    # Quick check if this might be our file
                                    with open(file_path, 'rb') as f:
                                        first_bytes = f.read(100)
                                    
                                    # Check for common file signatures that match expected types
                                    if self._looks_like_expected_file_type(first_bytes):
                                        print(f"🎯 Potential match found: {file_path}")
                                        return str(file_path)
                                        
                                except (PermissionError, OSError):
                                    continue  # Skip files we can't read
                                    
                except (PermissionError, OSError):
                    print(f"   ❌ Cannot access {cache_path}")
                    continue
        
        print(f"❌ Could not find file in HF cache for task: {self.task_id}")
        return None

    def _looks_like_expected_file_type(self, file_bytes: bytes) -> bool:
        """
        NEW: Quick heuristic to check if file bytes match expected file types.
        
        This is a simple heuristic - could be made more sophisticated.
        """
        if not file_bytes:
            return False
        
        # Common file signatures
        signatures = {
            b'PK\x03\x04': 'xlsx/zip',  # Excel files are ZIP-based
            b'%PDF': 'pdf',
            b'\x89PNG': 'png',
            b'\xff\xd8\xff': 'jpeg',
            b'ID3': 'mp3',
        }
        
        for signature, file_type in signatures.items():
            if file_bytes.startswith(signature):
                return True
        
        # Also accept CSV files (text-based)
        try:
            text_content = file_bytes.decode('utf-8')
            if ',' in text_content and '\n' in text_content:
                return True  # Likely CSV
        except UnicodeDecodeError:
            pass
        
        return False

    def set_hf_cache_hint(self, file_path: str):
        """
        NEW: Allow agent_testing.py to provide HF cache path hint.
        
        This method can be called by agent_testing.py to provide the exact
        HF cache path when it's known from gaia_dataset_utils.py.
        """
        self._hf_cache_hint = file_path
        if file_path:
            print(f"💡 HF cache hint set: {file_path[:60]}...")

    # Convenience function for testing
    def test_get_attachment_tool(task_id: str = None):
        """Test the GetAttachmentTool functionality"""
        tool = GetAttachmentTool()
        
        if task_id:
            tool.attachment_for(task_id)
            print(f"Testing with task_id: {task_id}")
            
            # Test URL format
            url_result = tool.forward("URL")
            print(f"URL result: {url_result}")
            
            # Test if file exists
            if url_result and not url_result.startswith("Error"):
                try:
                    response = requests.head(url_result, timeout=10)
                    if response.status_code == 200:
                        print(f"✅ File exists and accessible")
                        print(f"Content-Type: {response.headers.get('content-type')}")
                        print(f"Content-Length: {response.headers.get('content-length', 'Unknown')}")
                    else:
                        print(f"⚠️  File check returned: {response.status_code}")
                except Exception as e:
                    print(f"❌ Error checking file: {e}")
        else:
            print("ℹ️  No task_id provided - tool ready but no file to retrieve")
        
        return tool


if __name__ == "__main__":
    # Basic test
    print("🧪 Testing GetAttachmentTool...")
    test_tool = test_get_attachment_tool()
    print("✅ GetAttachmentTool loaded successfully")