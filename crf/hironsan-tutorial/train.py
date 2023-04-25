import pycrfsuite

from corpus import CorpusReader
from feature_engineering import create_X_y

if __name__ == "__main__":
    c = CorpusReader("data/hironsan.txt")
    train_sents = c.iob_sents("train")
    X_train, y_train = create_X_y(train_sents)

    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params(
        {
            "c1": 1.0,
            "c2": 1e-3,
            "max_iterations": 50,
            "feature.possible_transitions": True,
        }
    )
    trainer.train("model.crfsuite")
