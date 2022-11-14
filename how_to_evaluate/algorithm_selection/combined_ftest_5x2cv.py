"""$ python combined_ftest_5x2cv.py
model1: Logistic regression vs. model2: Decision tree
Logistic regression accuracy: 97.37%
Decision tree accuracy: 92.11%
F statistic: 0.918
significance level = 0.05
p value: 0.577

model1: Logistic regression vs. model2: Decision tree (simple ver.)
Logistic regression accuracy: 97.37%
Decision tree (simple ver.) accuracy: 65.79%
F statistic: 208.360
significance level = 0.05
p value: 0.000
"""
# ref: https://rasbt.github.io/mlxtend/user_guide/evaluate/combined_ftest_5x2cv/#example-1-5x2cv-combined-f-test
from mlxtend.evaluate import combined_ftest_5x2cv
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42

if __name__ == "__main__":
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        shuffle=True,
        stratify=y,
        random_state=123,
    )

    clf1 = LogisticRegression(
        random_state=RANDOM_SEED, solver="liblinear", multi_class="ovr"
    )
    clf2 = DecisionTreeClassifier(random_state=RANDOM_SEED)

    score1 = clf1.fit(X_train, y_train).score(X_test, y_test)
    score2 = clf2.fit(X_train, y_train).score(X_test, y_test)
    print("model1: Logistic regression vs. model2: Decision tree")
    print(f"Logistic regression accuracy: {score1 * 100:.2f}%")
    print(f"Decision tree accuracy: {score2 * 100:.2f}%")

    f, p = combined_ftest_5x2cv(clf1, clf2, X=X, y=y, random_seed=RANDOM_SEED)
    print(f"F statistic: {f:.3f}")
    print("significance level = 0.05")
    print(f"p value: {p:.3f}")

    print()

    clf2 = DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=1)
    score2 = clf2.fit(X_train, y_train).score(X_test, y_test)

    print(
        "model1: Logistic regression vs. model2: Decision tree (simple ver.)"
    )
    print(f"Logistic regression accuracy: {score1 * 100:.2f}%")
    print(f"Decision tree (simple ver.) accuracy: {score2 * 100:.2f}%")

    f, p = combined_ftest_5x2cv(clf1, clf2, X=X, y=y, random_seed=RANDOM_SEED)
    print(f"F statistic: {f:.3f}")
    print("significance level = 0.05")
    print(f"p value: {p:.3f}")
