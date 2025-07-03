from smolagents import Tool
from openai import OpenAI


class ClassifierTool(Tool):
    name = "open_classifier"
    description = """Classifies given items into given categories from perspective of specific knowledge area."""
    inputs = {
        "knowledge_area": {
            "type": "string",
            "description": "The knowledge area that should be used for classification.",
        },
        "environment": {  # context make models too verbose
            "type": "string",
            "description": "Couple words that describe environment or location in which items should be classified in case of plural meaning or if only part of item relevant for classification.",
        },
        "categories": {
            "type": "string",
            "description": "Comma separated list of categories to distribute objects.",
        },
        "items": {
            "type": "string",
            "description": "Comma separated list of items to be classified. Please include adjectives if available.",
        },
    }
    output_type = "string"

    def __init__(
        self,
        client: OpenAI | None = None,
        model_id: str = "gpt-4.1-mini",
        **kwargs,
    ):
        self.client = client or OpenAI()
        self.model_id = model_id

        super().__init__(**kwargs)

    def forward(
        self, knowledge_area: str, environment: str, categories: str, items: str
    ) -> str:
        response = self.client.responses.create(
            model=self.model_id,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self._prompt(
                                knowledge_area=knowledge_area,
                                context=environment,
                                categories=categories,
                                items=items,
                            ),
                        },
                    ],
                }
            ],
        )
        answer = response.output_text
        return answer

    def _prompt(
        self, knowledge_area: str, context: str, categories: str, items: str
    ) -> str:
        return f"""\
You are {knowledge_area} classifier located in {context} context.
I will provide you a list of items and a list of categories and context in which items should be considered.

Your task is to classify the items into the categories.
Use context to determine the meaning of the items and decide if you need to classify entire item or only part of it.

Do not miss any item and do not add any item to the list of categories.
Use highest probability category for each item.
You can add category "Other" if you are not sure about the classification.

Use only considerations from from the {knowledge_area} perspective.
Explain your reasoning from {knowledge_area} perspective in {context} context and then provide final answer.
Important: Do not allow {context} influence your judgment for classification.

ITEMS: {items}
CATEGORIES: {categories}

Now provide your reasoning and finalize it with the classification in the following format:
Category 1: items list
Category 2: items list
Other (if needed): items list
"""