import argparse
import ast
import re
from pathlib import Path

import jsonlines

from constants import ner_tags
from custom_types import Prediction

verbose2short = {
    "Person": "PER",
    "Organization": "ORG",
    "Location": "LOC",
    "Miscellaneous": "MISC",
}


def parse_response(response: str | None) -> list[list[str]]:
    r"""
    Returns [["Entity type", "Named entity"], ["Entity type", "Named entity"], ...]

    >>> parse_response('["Location", "JAPAN"]')
    [['Location', 'JAPAN']]
    >>> parse_response('["Location", "Syria"], ["Organization", "opening meeting"]')
    [['Location', 'Syria'], ['Organization', 'opening meeting']]
    >>> parse_response('["Location", "Syria"], ["[]"]')  # ["[]"]は前処理で[""]となるが、要素2個のリストではないので除かれる
    [['Location', 'Syria']]
    >>> parse_response('["Location", "JAPAN"]\n["Location", "CHINA"]')
    [['Location', 'JAPAN'], ['Location', 'CHINA']]
    >>> parse_response('["Location", "JAPAN"]\n["Location", "CHINA"]\n[]')
    [['Location', 'JAPAN'], ['Location', 'CHINA']]
    >>> parse_response('[Location, "AL-AIN"], [Miscellaneous, "1996-12-06"]')
    [['Location', 'AL-AIN'], ['Miscellaneous', '1996-12-06']]
    >>> parse_response('["Location", "Syria"] (no Person entity is mentioned)')
    Traceback (most recent call last):
      ...
    ValueError: '["Location", "Syria"] (no Person entity is mentioned)'
    >>> parse_response("[]")
    []
    >>> parse_response('[] (no entity for "favourites")')
    Traceback (most recent call last):
      ...
    ValueError: '[] (no entity for "favourites")'
    >>> parse_response('[["Person", "Takuya Takagi"], ["Location", "group C"]]')
    [['Person', 'Takuya Takagi'], ['Location', 'group C']]
    >>> parse_response(None)  # APIエラーのときはNoneにしている
    []
    >>> parse_response("[] ")
    []
    >>> parse_response("[[], [], [], []]")
    []
    """
    if response is None:
        return []
    if response.strip() == "[]":
        return []

    if "\n" in response:
        chunks = response.rstrip().split("\n")
        results = []
        for chunk in chunks:
            results.extend(parse_response(chunk))
        return results

    # Entity typeにダブルクォートが付いていない場合は付ける
    response = re.sub(
        r"\[(Location|Miscellaneous|Person|Organization),",
        r'["\1",',
        response,
    )
    try:
        literal = ast.literal_eval(response)
    except SyntaxError:  # ChatGPTが指示に従わず、後処理で拾えない場合
        raise ValueError(repr(response))

    if isinstance(literal, list):
        if isinstance(literal[0], list):  # ネストしたリストで返すときがある
            return list(filter(None, literal))
        return [literal]  # ネストしていないリストの場合
    if isinstance(literal, tuple):  # カンマを含んでいる場合
        return list(filter(lambda list_: len(list_) == 2, literal))


def get_index(tokens: list[str], word: str) -> int:
    """
    >>> get_index(["their", "opening", "meeting", "against", "Syria"], "Syria")
    4
    >>> get_index(["their", "opening", "meeting", "against", "Syria"], "opening meeting")
    1
    """
    if len(word.split()) == 1:
        return tokens.index(word)
    else:  # len(word.split()) >= 2:
        return tokens.index(word.split()[0])


def is_in_tokens(word: str, tokens: list[str]) -> bool:
    if len(word.split()) == 1:
        return word in set(tokens)
    else:
        return set(word.split()) <= set(tokens)


def as_iob2_format(
    tokens: list[str], recognized_entities: list[list[str]]
) -> list[str]:
    """
    >>> as_iob2_format(["SOCCER", "-", "JAPAN", "GET"], [["Location", "JAPAN"]])
    ['O', 'O', 'B-LOC', 'O']
    >>> as_iob2_format(["JAPAN", "WIN", "CHINA"], [["Location", "JAPAN"], ["Location", "CHINA"]])
    ['B-LOC', 'O', 'B-LOC']
    >>> as_iob2_format(["AL-AIN", "1996-12-06"], [["Location", "AL-AIN"], ["Miscellaneous", "1996-12-06"]])
    ['B-LOC', 'B-MISC']
    >>> as_iob2_format(["Nadim", "Ladki"], [["Person", "Nadim Ladki"]])
    ['B-PER', 'I-PER']
    >>> as_iob2_format(["United", "Arab", "Emirates", "1996-12-06"], [["Location", "United Arab Emirates"], ["Miscellaneous", "1996-12-06"]])
    ['B-LOC', 'I-LOC', 'I-LOC', 'B-MISC']
    >>> as_iob2_format(["their", "opening", "meeting", "against", "Syria"], [["Location", "Syria"], ["Organization", "opening meeting"]])
    ['O', 'B-ORG', 'I-ORG', 'O', 'B-LOC']
    >>> as_iob2_format(["Friday"], [["Date", "Friday"]])
    ['O']
    >>> as_iob2_format(["JAPAN"], [["Location", "Japan"]])
    ['O']
    """
    # ChatGPTが指定した4つの固有表現タイプ以外も入れてくるので除く（例：Date）
    only_four_types = filter(
        lambda e: e[0] in verbose2short, recognized_entities
    )
    # ChatGPTがtokenのママ返さないときは除く（例：JAPANなのにJapanに変換して返す）
    only_same_tokens = filter(
        lambda e: is_in_tokens(e[1], tokens), only_four_types
    )
    recognized_entities = sorted(
        only_same_tokens, key=lambda e: get_index(tokens, e[1])
    )

    iob2_tags = []
    next_token_index = -1
    recognition_index = 0
    for index, token in enumerate(tokens):
        if index < next_token_index:
            continue
        if recognition_index >= len(recognized_entities):
            iob2_tags.append("O")
            continue

        recognized_entity = recognized_entities[recognition_index][1]
        recognized_entity_type = recognized_entities[recognition_index][0]
        entity_word_length = len(recognized_entity.split(" "))
        if entity_word_length == 1:  # single word entity
            if token != recognized_entity:
                iob2_tags.append("O")
                continue
            # Case: token == recognized_entity
            iob2_tags.append(f"B-{verbose2short[recognized_entity_type]}")
            recognition_index += 1
            continue
        else:  # multiple word entity
            if (
                " ".join(tokens[index : index + entity_word_length])
                != recognized_entity
            ):
                iob2_tags.append("O")
                continue
            # Case: " ".join(tokens[index: index+entity_word_length]) == recognized_entity
            iob2_tags.append(f"B-{verbose2short[recognized_entity_type]}")
            for _ in range(entity_word_length - 1):
                iob2_tags.append(f"I-{verbose2short[recognized_entity_type]}")
            next_token_index = index + entity_word_length
            recognition_index += 1
    return iob2_tags


def post_process(prediction: Prediction):
    try:
        parsed_responses = parse_response(prediction["response"])
    except ValueError:
        return {"predicted_tags": [ner_tags["O"]] * len(prediction["tokens"])}
    else:
        return {
            "predicted_tags": [
                ner_tags[tag]
                for tag in as_iob2_format(
                    prediction["tokens"], parsed_responses
                )
            ]
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    args = parser.parse_args()

    with jsonlines.open(args.input_jsonl) as reader, jsonlines.open(
        args.output_jsonl, "w"
    ) as writer:
        writer.write_all(obj | post_process(obj) for obj in reader)
