from collections.abc import Iterable
from difflib import Differ
from pprint import pformat

from spacy_alignments import get_alignments
from transformers import AutoTokenizer, PreTrainedTokenizer

from data_types import Entity

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
    # [UNK]が入ってくると検証できない
    if "[UNK]" not in set(tokens):
        text_without_space = text.replace(" ", "")
        joined_tokens = "".join(t.removeprefix("##") for t in tokens[1:-1])
        assert text_without_space == joined_tokens, pformat(
            diff(text_without_space, joined_tokens)
        )

    characters = list(text)
    # [[1], [1], [1], [2], [2]] のように、何文字目が何番目のトークンかを表す
    char_to_token_indices, _ = get_alignments(characters, tokens)
    return char_to_token_indices


def diff(first: str, second: str) -> list[str]:
    d = Differ()
    return list(d.compare(first, second))


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

    >>> entities = [{"name": "矢吹怗一", "span": [0, 4], "type": "人名"}]
    >>> output_labels("矢吹怗一監督", ["[CLS]", "矢", "##吹", "[UNK]", "監督", "[SEP]"], entities)
    ['-', 'O', 'O', 'O', 'O', '-']
    >>> entities = [{"name": "ああいいうう", "span": [0, 6], "type": "法人名"}]
    >>> output_labels("ああいいうう", ["[CLS]", "ああ", "[UNK]", "うう", "[SEP]"], entities)
    ['-', 'B-法人名', 'I-法人名', 'I-法人名', '-']
    """
    char_to_token_indices = get_char_to_token_alignments(text, tokens)

    labels = ["O"] * len(tokens)
    for entity in entities:
        entity_span, entity_type = entity["span"], entity["type"]
        start_token_indices = char_to_token_indices[entity_span[0]]
        end_token_indices = char_to_token_indices[entity_span[1] - 1]
        # "[UNK]"があるとき、リストが空（start_token_indices[0]がIndexError）
        if len(start_token_indices) == 0 or len(end_token_indices) == 0:
            continue
        start_token_id: int = char_to_token_indices[entity_span[0]][0]
        end_token_id: int = char_to_token_indices[entity_span[1] - 1][0]

        labels[start_token_id] = f"B-{entity_type}"
        for idx in range(start_token_id + 1, end_token_id + 1):
            labels[idx] = f"I-{entity_type}"

    # 特殊トークンにはラベルを設定しない
    labels[0] = "-"
    labels[-1] = "-"

    return labels
