import argparse
import json
from pathlib import Path

import jsonlines
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from constants import index2tag
from custom_types import ParsedPrediction


def load_labels(
    predictions_path: Path,
) -> tuple[list[list[str]], list[list[str]]]:
    predictions = load_predictions(predictions_path)
    y_true = [[index2tag[idx] for idx in p["ner_tags"]] for p in predictions]
    y_pred = [
        [index2tag[idx] for idx in p["predicted_tags"]] for p in predictions
    ]
    return y_true, y_pred


def load_predictions(predictions_path: Path) -> list[ParsedPrediction]:
    with jsonlines.open(predictions_path) as reader:
        return list(reader)


def evaluate(
    y_true: list[list[str]], y_pred: list[list[str]]
) -> dict[str, float]:
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_jsonl", type=Path)
    args = parser.parse_args()

    y_true, y_pred = load_labels(args.predictions_jsonl)
    print(classification_report(y_true, y_pred))
    print(json.dumps(evaluate(y_true, y_pred)))
