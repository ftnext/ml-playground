from collections.abc import Iterable
from pprint import pprint

from datasets import load_dataset

from data_types import Entity


def create_label2id(
    entities_collection: Iterable[list[Entity]],
) -> dict[str, int]:
    label2id = {"O": 0}
    entity_types = {
        entity["type"]
        for entities in entities_collection
        for entity in entities
    }
    for i, entity_type in enumerate(sorted(entity_types)):
        label2id[f"B-{entity_type}"] = i * 2 + 1
        label2id[f"I-{entity_type}"] = i * 2 + 2
    return label2id


if __name__ == "__main__":
    dataset = load_dataset("llm-book/ner-wikipedia-dataset")

    label2id = create_label2id(dataset["train"]["entities"])
    id2label = {v: k for k, v in label2id.items()}
    pprint(id2label)
