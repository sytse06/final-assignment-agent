from smolagents import Tool
from googleapiclient.discovery import build
import os


class GoogleSearchTool(Tool):
    name = "web_search"
    description = """Performs a google web search for query then returns top search results in markdown format."""
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform search.",
        },
    }
    output_type = "string"

    skip_forward_signature_validation = True

    def __init__(
        self,
        api_key: str | None = None,
        search_engine_id: str | None = None,
        num_results: int = 10,
        **kwargs,
    ):
        api_key = api_key if api_key is not None else os.getenv("GOOGLE_SEARCH_API_KEY")
        if not api_key:
            raise ValueError(
                "Please set the GOOGLE_SEARCH_API_KEY environment variable."
            )
        search_engine_id = (
            search_engine_id
            if search_engine_id is not None
            else os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        )
        if not search_engine_id:
            raise ValueError(
                "Please set the GOOGLE_SEARCH_ENGINE_ID environment variable."
            )

        self.cse = build("customsearch", "v1", developerKey=api_key).cse()
        self.cx = search_engine_id
        self.num = num_results
        super().__init__(**kwargs)

    def _collect_params(self) -> dict:
        return {}

    def forward(self, query: str, *args, **kwargs) -> str:
        params = {
            "q": query,
            "cx": self.cx,
            "fields": "items(title,link,snippet)",
            "num": self.num,
        }

        params = params | self._collect_params(*args, **kwargs)

        response = self.cse.list(**params).execute()
        if "items" not in response:
            return "No results found."

        result = "\n\n".join(
            [
                f"[{item['title']}]({item['link']})\n{item['snippet']}"
                for item in response["items"]
            ]
        )
        return result


class GoogleSiteSearchTool(GoogleSearchTool):
    name = "site_search"
    description = """Performs a google search within the website for query then returns top search results in markdown format."""
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform search.",
        },
        "site": {
            "type": "string",
            "description": "The domain of the site on which to search.",
        },
    }

    def _collect_params(self, site: str) -> dict:
        return {
            "siteSearch": site,
            "siteSearchFilter": "i",
        }
