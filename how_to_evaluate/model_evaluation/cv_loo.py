from sklearn.datasets import load_iris
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(f"labeled data: {len(y)}")

    loo = LeaveOneOut()

    classifier = DecisionTreeClassifier(random_state=1)
    scores = cross_val_score(classifier, X, y, cv=loo, n_jobs=1, verbose=1)
    print(f"# of CV iterations: {len(scores)}")
    print(f"ACC: {scores.mean() * 100:.2f}")
