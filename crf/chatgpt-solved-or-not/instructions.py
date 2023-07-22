# ref: https://github.com/FreedomIntelligence/Evaluation-of-ChatGPT-on-Information-Extraction/blob/bda6894cffd041ff629ba1c2c9473e757ba217eb/1_NER/ner_test_with_api.py#L18-L37
import json
from typing import Literal

TYPE_VERBOSE_NAMES = ["Organization", "Person", "Location", "Miscellaneous"]

answer_instruction = """\
Answer in the format ["entity_type", "entity_name"] without any explanation. If no entity exists, then just answer "[]".
"""

instruction1 = """\
Considering {} types of named entities including {} and {}, recognize all named entities in the given sentence.
""".format(
    len(TYPE_VERBOSE_NAMES),
    ", ".join(TYPE_VERBOSE_NAMES[:-1]),
    TYPE_VERBOSE_NAMES[-1],
)

instruction2 = """\
Given the list of entity types {}, read the given sentence and find out all words/phrases that indicate the above types of named entities.
""".format(
    json.dumps(TYPE_VERBOSE_NAMES)
)

instruction3 = """\
Read the given sentence carefully, identify all named entities of type {} or {}.
""".format(
    ", ".join(TYPE_VERBOSE_NAMES[:-1]), TYPE_VERBOSE_NAMES[-1]
)

instruction4 = """\
Analyze the given sentence and extract all word spans that refer to specific named entities of type {} or {}.
""".format(
    ", ".join(TYPE_VERBOSE_NAMES[:-1]), TYPE_VERBOSE_NAMES[-1]
)

instruction5 = """\
What named entities are mentioned in the given sentence? Only return named entities of type {} or {}.
""".format(
    ", ".join(TYPE_VERBOSE_NAMES[:-1]), TYPE_VERBOSE_NAMES[-1]
)


def get_instruction(number: Literal[1, 2, 3, 4, 5]) -> str:
    instructions = {
        1: instruction1,
        2: instruction2,
        3: instruction3,
        4: instruction4,
        5: instruction5,
    }
    return instructions[number] + answer_instruction
