# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "MainContentExtractor",
#     "promptflow",
# ]
# ///
import httpx
from main_content_extractor import MainContentExtractor
from promptflow.core import tool


@tool
def fetch_text_content_from_url(url: str) -> str:
    response = httpx.get(url)
    response.raise_for_status()

    response.encoding = "utf-8"
    return MainContentExtractor.extract(response.text)


if __name__ == "__main__":
    import sys

    print(fetch_text_content_from_url(sys.argv[1]))
