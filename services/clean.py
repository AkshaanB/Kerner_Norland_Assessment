import re

def strip_think(text: str) -> str:
    """Clean the output by removing <think> tags and their content.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text with <think> tags and their content removed.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
