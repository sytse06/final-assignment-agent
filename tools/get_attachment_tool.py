from smolagents import Tool
import requests
from urllib.parse import urljoin
import base64
import tempfile


class GetAttachmentTool(Tool):
    name = "get_attachment"
    description = """Retrieves attachment for current task in specified format."""
    inputs = {
        "fmt": {
            "type": "string",
            "description": "Format to retrieve attachment. Options are: URL (preferred), DATA_URL, LOCAL_FILE_PATH, TEXT. URL returns the URL of the file, DATA_URL returns a base64 encoded data URL, LOCAL_FILE_PATH returns a local file path to the downloaded file, and TEXT returns the content of the file as text.",
            "nullable": True,
            "default": "URL",
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
        self.local_file_path = None
        super().__init__(**kwargs)

    def attachment_for(self, task_id: str | None):
        self.task_id = task_id
        
    def configure_from_state(self, task_id: str = None, file_path: str = None):
        """Configure from state"""
        self.task_id = task_id
        self.local_file_path = file_path

    def forward(self, fmt: str = "URL") -> str:
        fmt = fmt.upper()
        assert fmt in ["URL", "DATA_URL", "LOCAL_FILE_PATH", "TEXT"]

        if not self.task_id:
            return ""

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
