# tools/get_attachment_tool.py
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
    GAIA attachment tool to access file attachments of questions. Uses url or local file paths
    in metadata.json.
    """
    
    name = "get_attachment"
    description = """Retrieves the attachment file for the current GAIA task.

Format options:
- 'URL': Returns download URL
- 'LOCAL_FILE_PATH': Downloads and returns local file path  
- 'DATA_URL': Returns base64 data URL
- 'TEXT': Returns raw text (text files only)

For data analysis, use 'URL' or 'LOCAL_FILE_PATH' to get the file path, then process with Python.

Returns empty string if no file is attached to the current task."""

    inputs = {
        "fmt": {
            "type": "string", 
            "description": "Format to retrieve attachment. Options: 'URL', 'DATA_URL', 'LOCAL_FILE_PATH', 'TEXT', 'CONTENT'.",
            "nullable": True,
            "default": "LOCAL_FILE_PATH"
        }
    }
    output_type = "string"
    
    def __init__(self, agent_evaluation_api: str = None, task_id: str = None, **kwargs):
        self.agent_evaluation_api = (
            agent_evaluation_api or 
            "https://agents-course-unit4-scoring.hf.space/"
        )
        self.task_id = task_id
        super().__init__(**kwargs)

    def attachment_for(self, task_id: str):
        """Configure tool for specific task"""
        self.task_id = task_id
        if task_id:
            print(f"🔗 GetAttachmentTool configured for task: {task_id}")

    def forward(self, fmt: str = "LOCAL_FILE_PATH") -> str:
        """Retrieve attachment using hybrid approach"""
        try:
            # Handle task_id passed as format (error correction)
            if fmt and len(fmt) > 10 and '-' in fmt:
                print(f"⚠️  Detected task_id passed as format: {fmt}")
                print(f"🔧 Auto-correcting: setting task_id and using LOCAL_FILE_PATH format")
                self.task_id = fmt
                fmt = "LOCAL_FILE_PATH"
            
            fmt = fmt.upper().strip()
            valid_formats = ["URL", "DATA_URL", "LOCAL_FILE_PATH", "TEXT"]
            
            if fmt not in valid_formats:
                return f"Error: Invalid format '{fmt}'. Use: {', '.join(valid_formats)}"

            if not self.task_id:
                return "Error: No task_id set"

            # TRY LOCAL CACHE FIRST (development)
            local_result = self._try_local_cache(fmt)
            if local_result and not local_result.startswith("Error:"):
                return local_result
            
            # FALLBACK TO API (deployment)
            return self._try_api_access(fmt)
            
        except Exception as e:
            error_msg = f"Error retrieving attachment: {str(e)}"
            print(f"❌ GetAttachmentTool error: {error_msg}")
            return error_msg

    def _try_local_cache(self, fmt: str) -> str:
        """Try to find file in local HF cache"""
        try:
            # Try to get file info from GAIA metadata
            file_info = self._get_local_file_info()
            if not file_info:
                return "Error: File not found in local cache"
            
            file_path = file_info.get('file_path')
            if not file_path or not os.path.exists(file_path):
                return "Error: Local file path not found"
            
            print(f"✅ Found local file: {file_path}")
            
            # Process based on format
            if fmt == "LOCAL_FILE_PATH":
                return file_path
            elif fmt == "TEXT":
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except UnicodeDecodeError:
                    return f"Error: File cannot be read as text. Binary file at: {file_path}"
            elif fmt == "DATA_URL":
                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    mime_type = self._guess_mime_type(file_path)
                    encoded_content = base64.b64encode(file_content).decode('utf-8')
                    return f"data:{mime_type};base64,{encoded_content}"
                except Exception as e:
                    return f"Error creating data URL: {str(e)}"
            
            return file_path  # Default fallback
            
        except Exception as e:
            return f"Error accessing local cache: {str(e)}"

    def _try_api_access(self, fmt: str) -> str:
        """Fallback to API access (original reference pattern)"""
        try:
            file_url = urljoin(self.agent_evaluation_api, f"files/{self.task_id}")
            
            if fmt == "URL":
                return file_url

            print(f"📥 Trying API access: {file_url}")
            response = requests.get(file_url, timeout=30)
            
            if 400 <= response.status_code < 500:
                return "Error: No file attachment found via API"

            response.raise_for_status()
            mime = response.headers.get("content-type", "text/plain")
            
            if fmt == "TEXT":
                if mime.startswith("text/"):
                    return response.text
                else:
                    return f"Error: Content type {mime} cannot be read as TEXT"
                    
            elif fmt == "DATA_URL":
                encoded = base64.b64encode(response.content).decode('utf-8')
                return f"data:{mime};base64,{encoded}"
                
            elif fmt == "LOCAL_FILE_PATH":
                # Download and save temporarily
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(response.content)
                    return tmp_file.name
            
            return f"Error: Unsupported format: {fmt}"
            
        except Exception as e:
            return f"Error accessing API: {str(e)}"

    def _get_local_file_info(self) -> dict:
        """Get file info from local GAIA metadata"""
        try:
            # Try to load metadata.json
            metadata_paths = [
                "./tests/gaia_data/metadata.json",
                "../tests/gaia_data/metadata.json",
                "./gaia_data/metadata.json",
                "./metadata.json"
            ]
            
            for metadata_path in metadata_paths:
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Handle different metadata structures
                    questions = []
                    if isinstance(metadata, dict) and 'validation' in metadata:
                        questions = metadata['validation']
                    elif isinstance(metadata, list):
                        questions = metadata
                    
                    # Find our task
                    for question in questions:
                        if question.get('task_id') == self.task_id:
                            return {
                                'file_name': question.get('file_name'),
                                'file_path': question.get('file_path'),
                                'task_id': self.task_id
                            }
            
            return None
            
        except Exception as e:
            print(f"❌ Error reading local metadata: {e}")
            return None

    def _guess_mime_type(self, file_path: str) -> str:
        """Guess MIME type from file extension"""
        ext_to_mime = {
            ".pdf": "application/pdf",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".csv": "text/csv",
            ".txt": "text/plain",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        
        ext = os.path.splitext(file_path)[1].lower()
        return ext_to_mime.get(ext, "application/octet-stream")

# QUICK TEST
def test_simple_tool():
    """Test the simplified GetAttachmentTool"""
    tool = GetAttachmentTool()
    tool.attachment_for("9318445f-fe6a-4e1b-acbf-c68228c9906a")
    
    result = tool.forward(fmt="LOCAL_FILE_PATH")
    print(f"Simple tool result: {result}")
    print(f"Result type: {type(result)}")
    print(f"Has .strip(): {hasattr(result, 'strip')}")
    
    return result