from sklearn.datasets import load_iris
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(f"labeled data: {len(y)}")

    rskf = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=100, random_state=20220925
    )

    classifier = DecisionTreeClassifier(random_state=1)
    scores = cross_val_score(classifier, X, y, cv=rskf, n_jobs=1, verbose=1)
    print(f"# of CV iterations: {len(scores)}")
    print(f"ACC: {scores.mean() * 100:.2f} +/- {scores.std() * 100:.2f}")
