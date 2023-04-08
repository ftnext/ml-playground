import csv
from typing import TypedDict

import jsonlines


class StackOverflowRawData(TypedDict):
    extracted_tags: str
    original_tags: str
    text: str


class StackOverflowData(TypedDict):
    text: str
    tags: list[str]


if __name__ == "__main__":
    data: list[StackOverflowData] = []
    with open("data/so_data.csv") as f:
        reader = csv.DictReader(f)
        for d in reader:  # type: StackOverflowRawData
            assert d["extracted_tags"] and d["text"]  # dropnaでは変わっていない
            data.append(
                dict(text=d["text"], tags=d["extracted_tags"].split(","))
            )

    with jsonlines.open("data/processed_so_data.jsonl", mode="w") as writer:
        writer.write_all(data)
