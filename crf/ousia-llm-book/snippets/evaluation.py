from collections.abc import Iterable
from typing import TypedDict

from seqeval.metrics import classification_report
from seqeval.metrics.sequence_labeling import precision_recall_fscore_support

from data_types import Entity


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
    return convert_true_labels(results), convert_pred_labels(results)


def convert_true_labels(results: Iterable[Result]) -> list[list[str]]:
    return [
        create_character_labels(result["text"], result["entities"])
        for result in results
    ]


def convert_pred_labels(results: Iterable[Result]) -> list[list[str]]:
    return [
        create_character_labels(result["text"], result["pred_entities"])
        for result in results
    ]


class Scores(TypedDict):
    precision: float
    recall: float
    f1_score: float


def compute_scores(
    true_labels: list[list[str]], pred_labels: list[list[str]], average: str
) -> Scores:
    precision, recall, fscore, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average=average
    )
    return {"precision": precision, "recall": recall, "f1_score": fscore}


if __name__ == "__main__":
    results = [
        Result(
            text="大谷翔平は岩手県水沢市出身",
            entities=[
                {"name": "大谷翔平", "span": [0, 4], "type": "人名"},
                {"name": "岩手県水沢市", "span": [5, 11], "type": "地名"},
            ],
            pred_entities=[
                {"name": "大谷翔平", "span": [0, 4], "type": "人名"},
                {"name": "岩手県", "span": [5, 8], "type": "地名"},
                {"name": "水沢市", "span": [8, 11], "type": "施設名"},
            ],
        )
    ]

    true_labels = convert_true_labels(results)
    pred_labels = convert_pred_labels(results)
    print(classification_report(true_labels, pred_labels))

    print(compute_scores(true_labels, pred_labels, "micro"))
