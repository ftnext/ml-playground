from promptflow.core import tool


@tool
def fetch_text_content_from_url(url: str) -> str:
    return """<ここに決め打ちのHTML>"""
