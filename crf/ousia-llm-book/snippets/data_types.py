from typing import TypedDict


class Entity(TypedDict):
    name: str
    span: list[int]
    type: str
