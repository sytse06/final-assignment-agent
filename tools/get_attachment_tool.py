# tools/get_attachment_tool.py
# GAIA file attachment retrieval tool

from smolagents import Tool
import requests
from urllib.parse import urljoin
import base64
import tempfile
import os
from typing import Optional


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
            "default": "URL",
        }
    }
    output_type = "string"

    def __init__(
        self, 
        agent_evaluation_api: Optional[str] = None,
        task_id: Optional[str] = None,
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
            return "Error: File download timed out. Try again or use 'URL' format."
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            return f"Error: Failed to retrieve file - {str(e)}"
        except Exception as e:
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