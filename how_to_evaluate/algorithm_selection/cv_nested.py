import argparse

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ALGORITHMS = ("DTree", "SVM")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")
    _ = subparsers.add_parser("search")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("algorithm", choices=ALGORITHMS)
    args = parser.parse_args()

    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )
    print(f"train: {len(y_train)}, test: {len(y_test)}")

    dtree = DecisionTreeClassifier(random_state=1)
    svc = SVC(random_state=1)

    param_dtree = {
        "max_depth": list(range(1, 10)) + [None],
        "criterion": ["gini", "entropy"],
    }
    param_svc = {
        "kernel": ["rbf"],
        "C": np.power(10.0, np.arange(-4, 4)),
        "gamma": np.power(10.0, np.arange(-5, 0)),
    }

    gridcvs = {}
    inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for param_grid, estimator, name in zip(
        (param_dtree, param_svc), (dtree, svc), ALGORITHMS
    ):
        gcv = GridSearchCV(
            estimator,
            param_grid,
            scoring="accuracy",
            n_jobs=1,
            cv=inner_cv,
            verbose=1,
        )
        gridcvs[name] = gcv

    if args.mode == "search":
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for name, gs_est in sorted(gridcvs.items()):
            nested_scores = cross_val_score(
                gs_est, X_train, y_train, cv=outer_cv, verbose=1
            )
            print(f"# of CV iterations: {len(nested_scores)}")
            print(
                f"{name} | outer ACC: {nested_scores.mean() * 100:.2f}% "
                f"+/- {nested_scores.std() * 100:.2f}"
            )

    if args.mode == "evaluate":
        best_algo = gridcvs[args.algorithm]
        best_algo.fit(X_train, y_train)

        print(f"ACC (average over CV test folds): {best_algo.best_score_}")
        print(f"Best parameters: {best_algo.best_params_}")
        print(
            f"Training accuracy: {best_algo.score(X_train, y_train) * 100:.2f}"
        )
        print(f"Test accuracy: {best_algo.score(X_test, y_test) * 100:.2f}")
