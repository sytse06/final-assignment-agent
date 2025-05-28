from smolagents import Tool, tool
from openai import OpenAI
import shutil


@tool
def chess_engine_locator() -> str | None:
    """
    Get the path to the chess engine binary. Can be used with chess.engine.SimpleEngine.popen_uci function from chess.engine Python module.
    Returns:
        str: Path to the chess engine.
    """
    path = shutil.which("stockfish")
    return path if path else None


class ImageToChessBoardFENTool(Tool):
    name = "image_to_chess_board_fen"
    description = """Convert a chessboard image to board part of the FEN."""
    inputs = {
        "image_url": {
            "type": "string",
            "description": "Public URL of the image (preferred) or base64 encoded image in data URL format.",
        }
    }
    output_type = "string"

    def __init__(self, client: OpenAI | None = None, **kwargs):
        self.client = client if client is not None else OpenAI()
        super().__init__(**kwargs)

    def attachment_for(self, task_id: str | None):
        self.task_id = task_id

    def forward(self, image_url: str) -> str:
        """
        Convert a chessboard image to board part of the FEN.
        Args:
            image_url (str): Public URL of the image (preferred) or base64 encoded image in data URL format.
        Returns:
            str: Board part of the FEN.
        """
        client = self.client

        response = client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Describe the position of the pieces on the chessboard from the image. Please, nothing else but description.",
                        },
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
        )

        response = client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Describe the position of the pieces on the chessboard from the image. Please, nothing else but description.",
                        },
                    ],
                }
            ]
            + response.output
            + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": """\
          Write down all positions with known pieces.
          Use a standard one-letter code to name pieces.

          It is important to use the correct case for piece code. Use upper case for white and lower case for black.
          It is important to include information about all the mentioned positions.

          Describe each position in a new line.
          Follow format: <piece><position> (piece first, than position, no spaces)
          Return nothing but lines with positions.
          """,
                        },
                    ],
                }
            ],
        )
        board_pos = response.output_text

        pos_dict = {}
        for pos_str in board_pos.splitlines():
            pos_str = pos_str.strip()
            if len(pos_str) != 3:
                continue
            piece = pos_str[0]
            pos = pos_str[1:3]
            pos_dict[pos] = piece

        board_fen = ""
        for rank in range(8, 0, -1):
            empty = 0
            for file_c in range(ord("a"), ord("h") + 1):
                file = chr(file_c)
                square = file + str(rank)
                if square in pos_dict:
                    if empty > 0:
                        board_fen += str(empty)
                        empty = 0
                    board_fen += pos_dict[square]
                else:
                    empty += 1
            if empty > 0:
                board_fen += str(empty)
            if rank != 1:
                board_fen += "/"

        return board_fen
