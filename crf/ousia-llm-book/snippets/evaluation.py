from collections.abc import Iterable
from typing import TypedDict

from seqeval.metrics import classification_report


class Entity(TypedDict):
    name: str
    span: list[int]
    type: str


def create_character_labels(
    text: str, entities: Iterable[Entity]
) -> list[str]:
    """
    >>> entities = [{"name": "大谷翔平", "span": [0, 4], "type": "人名"}]
    >>> create_character_labels("大谷翔平は", entities)
    ['B-人名', 'I-人名', 'I-人名', 'I-人名', 'O']
    """
    labels = ["O"] * len(text)
    for entity in entities:
        entity_span, entity_type = entity["span"], entity["type"]
        # spanの開始の文字がB
        labels[entity_span[0]] = f"B-{entity_type}"
        # 開始の文字以外はI
        for i in range(entity_span[0] + 1, entity_span[1]):
            labels[i] = f"I-{entity_type}"

    return labels


class Result(TypedDict):
    text: str
    entities: list[Entity]
    pred_entities: list[Entity]


def convert_results_to_labels(
    results: Iterable[Result],
) -> tuple[list[list[str]], list[list[str]]]:
    """
    >>> results = [
    ...   {
    ...     "text": "大谷翔平は岩手県水沢市出身",
    ...     "entities": [
    ...       {"name": "大谷翔平", "span": [0, 4], "type": "人名"},
    ...       {"name": "岩手県水沢市", "span": [5, 11], "type": "地名"},
    ...     ],
    ...     "pred_entities": [
    ...       {"name": "大谷翔平", "span": [0, 4], "type": "人名"},
    ...       {"name": "岩手県", "span": [5, 8], "type": "地名"},
    ...       {"name": "水沢市", "span": [8, 11], "type": "施設名"},
    ...     ],
    ...   }
    ... ]
    >>> true_labels, pred_labels = convert_results_to_labels(results)
    >>> true_labels
    [['B-人名', 'I-人名', 'I-人名', 'I-人名', 'O', 'B-地名', 'I-地名', 'I-地名', 'I-地名', 'I-地名', 'I-地名', 'O', 'O']]
    >>> pred_labels
    [['B-人名', 'I-人名', 'I-人名', 'I-人名', 'O', 'B-地名', 'I-地名', 'I-地名', 'B-施設名', 'I-施設名', 'I-施設名', 'O', 'O']]
    """
    true_labels, pred_labels = [], []
    for result in results:
        true_labels.append(
            create_character_labels(result["text"], result["entities"])
        )
        pred_labels.append(
            create_character_labels(result["text"], result["pred_entities"])
        )

    return true_labels, pred_labels


if __name__ == "__main__":
    results = [
        {
            "text": "大谷翔平は岩手県水沢市出身",
            "entities": [
                {"name": "大谷翔平", "span": [0, 4], "type": "人名"},
                {"name": "岩手県水沢市", "span": [5, 11], "type": "地名"},
            ],
            "pred_entities": [
                {"name": "大谷翔平", "span": [0, 4], "type": "人名"},
                {"name": "岩手県", "span": [5, 8], "type": "地名"},
                {"name": "水沢市", "span": [8, 11], "type": "施設名"},
            ],
        }
    ]

    true_labels, pred_labels = convert_results_to_labels(results)
    print(classification_report(true_labels, pred_labels))
