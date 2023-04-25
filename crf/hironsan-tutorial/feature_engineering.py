from dataclasses import dataclass
from typing import Literal, TypedDict

from corpus import MorphInfo, Sentence


def is_hiragana(ch: str) -> bool:
    return 0x3040 <= ord(ch) <= 0x309F


def is_katakana(ch: str) -> bool:
    return 0x30A0 <= ord(ch) <= 0x30FF


def get_character_type(ch: str) -> str:
    if ch.isspace():
        return "ZSPACE"
    elif ch.isdigit():
        return "ZDIGIT"
    elif ch.islower():
        return "ZLLET"
    elif ch.isupper():
        return "ZULET"
    elif is_hiragana(ch):
        return "HIRAG"
    elif is_katakana(ch):
        return "KATAK"
    else:
        return "OTHER"


def get_character_types(string: str) -> str:
    """stringがどんな文字種から構成されているかを返す

    >>> get_character_types("2005")
    'ZDIGIT'
    """
    character_types = map(get_character_type, string)
    return "-".join(sorted(set(character_types)))


def extract_pos_with_subtype(morph: MorphInfo) -> str:
    """
    >>> extract_pos_with_subtype(["2005", "名詞", "数", "*", "*", "*", "*", "*", "B-DAT"])
    '名詞-数'
    >>> extract_pos_with_subtype(["年", "名詞", "接尾", "助数詞", "*", "*", "*", "年", "ネン", "ネン", "I-DAT"])
    '名詞-接尾-助数詞'
    """
    idx = morph.index("*")
    return "-".join(morph[1:idx])


class Feature(TypedDict):
    word: str
    type: str
    postag: str


@dataclass(frozen=True)
class FeatureFactory:
    morph: MorphInfo

    def create(self) -> Feature:
        word = self.morph[0]
        return Feature(
            word=word,
            type=get_character_types(word),
            postag=extract_pos_with_subtype(self.morph),
        )


def create_nth_before_feature(
    sentence: Sentence, i: int, n: Literal[1, 2]
) -> list[str]:
    if i >= n:
        sentence_nth_before = sentence[i - n]
        nth_before_factory = FeatureFactory(sentence_nth_before)
        nth_before_feature = nth_before_factory.create()
        return [
            f"-{n}:word={nth_before_feature['word']}",
            f"-{n}:type={nth_before_feature['type']}",
            f"-{n}:postag={nth_before_feature['postag']}",
            f"-{n}:iobtag={sentence_nth_before[-1]}",
        ]
    else:
        return ["BOS"]


def create_nth_after_feature(
    sentence: Sentence, i: int, n: Literal[1, 2]
) -> list[str]:
    if i < len(sentence) - n:
        sentence_nth_after = sentence[i + n]
        nth_after_factory = FeatureFactory(sentence_nth_after)
        nth_after_feature = nth_after_factory.create()
        return [
            f"+{n}:word={nth_after_feature['word']}",
            f"+{n}:type={nth_after_feature['type']}",
            f"+{n}:postag={nth_after_feature['postag']}",
        ]
    else:
        return ["EOS"]


def word2features(sentence: Sentence, i: int) -> list[str]:
    sentence_i = sentence[i]
    ith_factory = FeatureFactory(sentence_i)
    ith_feature = ith_factory.create()
    features = [
        "bias",
        f"word={ith_feature['word']}",
        f"type={ith_feature['type']}",
        f"postag={ith_feature['postag']}",
    ]

    features.extend(create_nth_before_feature(sentence, i, 2))
    features.extend(create_nth_before_feature(sentence, i, 1))

    features.extend(create_nth_after_feature(sentence, i, 1))
    features.extend(create_nth_after_feature(sentence, i, 2))

    return features


def sent2features(sentence: Sentence) -> list[list[str]]:
    return [word2features(sentence, i) for i in range(len(sentence))]


def sent2labels(sentence: Sentence) -> list[str]:
    return [morph[-1] for morph in sentence]


def create_X_y(sentences: list[Sentence]):
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    return X, y
