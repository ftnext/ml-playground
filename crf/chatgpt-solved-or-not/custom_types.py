from typing import TypedDict


class Conll03Example(TypedDict):
    id: str
    tokens: list[str]
    ner_tags: list[int]


class Example(Conll03Example):
    prompt: str


class Prediction(Example):
    response: str
