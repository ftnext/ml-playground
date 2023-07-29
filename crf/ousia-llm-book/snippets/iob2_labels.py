from collections.abc import Iterable
from typing import TypedDict

from spacy_alignments import get_alignments
from transformers import AutoTokenizer, PreTrainedTokenizer

model_name = "cl-tohoku/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(text: str, tokenizer: PreTrainedTokenizer) -> list[str]:
    """
    >>> tokenize("大谷翔平は岩手県水沢市出身", tokenizer)
    ['[CLS]', '大谷', '翔', '##平', 'は', '岩手', '県', '水沢', '市', '出身', '[SEP]']
    """
    return tokenizer.convert_ids_to_tokens(tokenizer.encode(text))


def get_char_to_token_alignments(
    text: str, tokens: list[str]
) -> list[list[int]]:
    assert text == "".join(t.lstrip("#") for t in tokens[1:-1])

    characters = list(text)
    # [[1], [1], [1], [2], [2]] のように、何文字目が何番目のトークンかを表す
    char_to_token_indices, _ = get_alignments(characters, tokens)
    return char_to_token_indices


class Entity(TypedDict):
    name: str
    span: list[int]
    type: str


def output_labels(
    text: str, tokens: list[str], entities: Iterable[Entity]
) -> list[str]:
    """
    >>> text = "大谷翔平は岩手県水沢市出身"
    >>> tokens = ["[CLS]", "大谷", "翔", "##平", "は", "岩手", "県", "水沢", "市", "出身", "[SEP]"]
    >>> entities = [
    ...   {"name": "大谷翔平", "span": [0, 4], "type": "人名"},
    ...   {"name": "岩手県水沢市", "span": [5, 11], "type": "地名"},
    ... ]
    >>> output_labels(text, tokens, entities)
    ['-', 'B-人名', 'I-人名', 'I-人名', 'O', 'B-地名', 'I-地名', 'I-地名', 'I-地名', 'O', '-']
    """
    char_to_token_indices = get_char_to_token_alignments(text, tokens)

    labels = ["O"] * len(tokens)
    for entity in entities:
        entity_span, entity_type = entity["span"], entity["type"]
        start_token_id: int = char_to_token_indices[entity_span[0]][0]
        end_token_id: int = char_to_token_indices[entity_span[1] - 1][0]

        labels[start_token_id] = f"B-{entity_type}"
        for idx in range(start_token_id + 1, end_token_id + 1):
            labels[idx] = f"I-{entity_type}"

    # 特殊トークンにはラベルを設定しない
    labels[0] = "-"
    labels[-1] = "-"

    return labels
