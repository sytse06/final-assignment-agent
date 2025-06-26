from smolagents import Tool
import requests
from urllib.parse import urljoin
import base64
import tempfile
import os
import shutil


class GetAttachmentTool(Tool):
    name = "get_attachment"
    description = """Retrieves attachment for current task in specified format."""
    inputs = {
        "fmt": {
            "type": "string",
            "description": "Format to retrieve attachment. Options are: URL (preferred), DATA_URL, LOCAL_FILE_PATH, TEXT. URL returns the URL of the file, DATA_URL returns a base64 encoded data URL, LOCAL_FILE_PATH returns a local file path to the downloaded file, and TEXT returns the content of the file as text.",
            "nullable": True,
            "default": "LOCAL_FILE_PATH",
        }
    }
    output_type = "string"

    def __init__(
        self,
        agent_evaluation_api: str | None = None,
        task_id: str | None = None,
        **kwargs,
    ):
        self.agent_evaluation_api = (
            agent_evaluation_api
            if agent_evaluation_api is not None
            else "https://agents-course-unit4-scoring.hf.space/"
        )
        self.task_id = task_id
        self._state_file_path = None
        self._file_name = None
        super().__init__(**kwargs)

    def attachment_for(self, task_id: str | None):
        self.task_id = task_id
        
    def configure_from_state(self, state_file_path: str, file_name: str = ""):
        """Give tool access to attachments via file path and recover file type from name"""
        self._state_file_path = state_file_path
        self._file_name = file_name
        print(f"✅ GetAttachmentTool configured with path and name")

    def forward(self, fmt: str = "URL") -> str:
        fmt = fmt.upper()
        assert fmt in ["URL", "DATA_URL", "LOCAL_FILE_PATH", "TEXT"]

        if not self.task_id:
            return ""

        # Use state file path and file name extension to access files
        if fmt == "LOCAL_FILE_PATH" and self._state_file_path:
            import os
            import shutil
            
            original_path = self._state_file_path
            
            # Find extension in file_names
            if self._file_name:
                file_extension = os.path.splitext(self._file_name)[1]
                original_has_extension = bool(os.path.splitext(original_path)[1])
                
                # If original path lacks extension but file_name has one
                if not original_has_extension and file_extension:
                    # Create new path with extension
                    new_path = original_path + file_extension
                    
                    # Copy file with proper extension if not already exists
                    if not os.path.exists(new_path) and os.path.exists(original_path):
                        try:
                            shutil.copy2(original_path, new_path)
                            print(f"🔧 Copied file with extension: {new_path}")
                            return new_path
                        except Exception as e:
                            print(f"⚠️ Could not copy file with extension: {e}")
                            print(f"🎯 Returning original path: {original_path}")
                            return original_path
                    elif os.path.exists(new_path):
                        print(f"🎯 Using existing file with extension: {new_path}")
                        return new_path
                
                # If original already has extension, use it directly
                if original_has_extension:
                    print(f"🎯 Original path has extension: {original_path}")
                    return original_path
            
            # Fallback: return original path (no file_name available or no extension needed)
            print(f"🎯 Returning state file path: {original_path}")
            return original_path

        # Original logic for other cases
        file_url = urljoin(self.agent_evaluation_api, f"files/{self.task_id}")
        if fmt == "URL":
            return file_url

        response = requests.get(
            file_url,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        if 400 <= response.status_code < 500:
            return ""

        response.raise_for_status()
        mime = response.headers.get("content-type", "text/plain")
        if fmt == "TEXT":
            if mime.startswith("text/"):
                return response.text
            else:
                raise ValueError(
                    f"Content of file type {mime} cannot be retrieved as TEXT."
                )
        elif fmt == "DATA_URL":
            return f"data:{mime};base64,{base64.b64encode(response.content).decode('utf-8')}"
        elif fmt == "LOCAL_FILE_PATH":
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(response.content)
                return tmp_file.name
        else:
            raise ValueError(
                f"Unsupported format: {fmt}. Supported formats are URL, DATA_URL, LOCAL_FILE_PATH, and TEXT."
            )
