from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier


def custom_cv_train_val(X, y):
    idx_train, idx_val = train_test_split(
        range(X.shape[0]),
        test_size=0.2,
        shuffle=True,
        stratify=y,
        random_state=42,
    )
    yield idx_train, idx_val


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )
    print(f"train: {len(y_train)}, test: {len(y_test)}")

    dtree = DecisionTreeClassifier(random_state=1)
    param_grid = {
        "max_depth": list(range(1, 10)) + [None],
        "criterion": ["gini", "entropy"],
    }
    custom_cv = custom_cv_train_val(X_train, y_train)
    classifier = GridSearchCV(
        dtree,
        param_grid,
        scoring="accuracy",
        n_jobs=1,
        cv=custom_cv,
        verbose=1,
    )
    classifier.fit(X_train, y_train)

    print(f"best parameters: {classifier.best_params_}")
    score = classifier.score(X_test, y_test)
    print(f"ACC: {score * 100:.2f}")

    # TODO あとは全データで訓練し、モデルを保存する
