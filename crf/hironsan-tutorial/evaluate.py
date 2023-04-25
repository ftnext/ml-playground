from itertools import chain

import pycrfsuite
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from corpus import CorpusReader, Sentence
from feature_engineering import create_X_y


def sent2tokens(sentence: Sentence) -> list[str]:
    return [morph[0] for morph in sentence]


def bio_classification_report(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {"O"}
    tagset = sorted(tagset, key=lambda tag: tag.split("-", 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )


if __name__ == "__main__":
    c = CorpusReader("data/hironsan.txt")
    test_sents = c.iob_sents("test")
    X_test, y_test = create_X_y(test_sents)

    tagger = pycrfsuite.Tagger()
    tagger.open("model.crfsuite")

    example_sent = test_sents[0]
    print(" ".join(sent2tokens(example_sent)))
    print("Predicted:", " ".join(tagger.tag(X_test[0])))
    print("Correct:  ", " ".join(y_test[0]))

    y_pred = [tagger.tag(xseq) for xseq in X_test]
    print(bio_classification_report(y_test, y_pred))
