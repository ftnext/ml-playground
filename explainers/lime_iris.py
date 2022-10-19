import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

if __name__ == "__main__":
    iris = load_iris()
    train, test, labels_train, labels_test = train_test_split(
        iris.data,
        iris.target,
        train_size=0.8,
        shuffle=True,
        stratify=iris.target,
        random_state=RANDOM_SEED,
    )
    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_SEED)
    rf.fit(train, labels_train)
    accuracy = accuracy_score(labels_test, rf.predict(test))
    print(accuracy)

    explainer = LimeTabularExplainer(
        train,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        discretize_continuous=True,
        random_state=RANDOM_SEED,
    )

    i = np.random.randint(0, test.shape[0])
    prediction = rf.predict(test[i].reshape(1, -1))
    label_predicted = iris.target_names[prediction[0]]
    print(i)
    for name, value in zip(iris.feature_names, test[i]):
        print(name, "=", value)
    print("Predicted as", label_predicted)
    print()

    explanation = explainer.explain_instance(
        test[i], rf.predict_proba, num_features=2, top_labels=1
    )
    map_explanation = explanation.as_map()
    for label_index, importances in map_explanation.items():
        print(iris.target_names[label_index])
        for feature_index, score in importances:
            print(iris.feature_names[feature_index], score)
