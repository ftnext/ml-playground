import codecs
import csv
from typing import Literal

MorphInfo = list[str]
Sentence = list[MorphInfo]


class CorpusReader:
    def __init__(self, path) -> None:
        with codecs.open(path, encoding="utf-8") as f:
            all_morph_info = csv.reader(f, "excel-tab")
            sentence: Sentence = []
            sentences: list[Sentence] = []
            for morph_info in all_morph_info:  # type: MorphInfo
                if morph_info == []:  # 空行は文と文の間の区切り
                    sentences.append(sentence)
                    sentence = []
                    continue
                sentence.append(morph_info)
        train_num = int(len(sentences) * 0.9)
        self.__train_sents = sentences[:train_num]
        self.__test_sents = sentences[train_num:]

    def iob_sents(self, name: Literal["train", "test"]) -> list[Sentence]:
        if name == "train":
            return self.__train_sents
        if name == "test":
            return self.__test_sents
        raise ValueError
