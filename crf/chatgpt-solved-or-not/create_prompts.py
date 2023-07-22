import argparse
import json
from collections.abc import Generator
from typing import TypedDict

import jsonlines
from datasets import load_dataset
from langchain.prompts import HumanMessagePromptTemplate


class Conll03Example(TypedDict):
    id: str
    tokens: list[str]
    ner_tags: list[int]


class Example(Conll03Example):
    prompt: str


def load_test_set() -> Generator[Conll03Example, None, None]:
    conll03 = load_dataset("conll2003")
    for example in conll03["test"]:
        yield Conll03Example(
            id=example["id"],
            tokens=example["tokens"],
            ner_tags=example["ner_tags"],
        )


TYPE_VERBOSE_NAMES = ["Organization", "Person", "Location", "Miscellaneous"]


def build_prompt_template() -> str:
    instruction = """\
Given the list of entity types {}, read the given sentence and find out all words/phrases that indicate the above types of named entities.
Answer in the format ["entity_type", "entity_name"] without any explanation. If no entity exists, then just answer "[]".
""".format(
        json.dumps(TYPE_VERBOSE_NAMES)
    )

    template = (
        instruction
        + """\
Sentence: {sentence}
Answer:\
"""
    )
    return HumanMessagePromptTemplate.from_template(template)


def create_prompts() -> Generator[Example, None, None]:
    prompt_template = build_prompt_template()
    examples = load_test_set()
    for example in examples:
        sentence = " ".join(example["tokens"])
        yield Example(
            **example,
            prompt=prompt_template.format(sentence=sentence).content,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_jsonl")
    args = parser.parse_args()

    with jsonlines.open(args.output_jsonl, "w") as writer:
        writer.write_all(create_prompts())
