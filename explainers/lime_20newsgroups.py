from lime.lime_text import LimeTextExplainer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

if __name__ == "__main__":
    newsgroups_train = fetch_20newsgroups(subset="train")
    newsgroups_test = fetch_20newsgroups(subset="test")
    class_names = newsgroups_train.target_names
    for i, name in enumerate(class_names):
        print(i, name)
    print()

    vectorizer = TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(newsgroups_train.data)
    test_vectors = vectorizer.transform(newsgroups_test.data)
    print("train:", train_vectors.shape)
    print("test:", test_vectors.shape)
    print()

    nb = MultinomialNB(alpha=0.1)
    nb.fit(train_vectors, newsgroups_train.target)

    preds = nb.predict(test_vectors)
    f1 = f1_score(newsgroups_test.target, preds, average="weighted")
    print("F1 score:", f1)
    print()

    c = make_pipeline(vectorizer, nb)
    explainer = LimeTextExplainer(class_names=class_names, random_state=42)

    idx = 1340
    text = newsgroups_test.data[idx]
    print("-" * 60)
    print(text)
    print("-" * 60)
    print()
    true_class_idx = newsgroups_test.target[idx]
    print("True:", true_class_idx, class_names[true_class_idx])
    pred = nb.predict(test_vectors[idx])
    pred_class_idx = pred.reshape(1, -1)[0, 0]
    print("Predicted:", pred_class_idx, class_names[pred_class_idx])
    print()

    exp = explainer.explain_instance(
        text, c.predict_proba, num_features=6, labels=[0, 15]
    )
    print("Explanation:")
    print("-" * 40)
    raw_exp = exp.as_map()
    for label_idx, pairs in raw_exp.items():
        readable_exp = exp.domain_mapper.map_exp_ids(pairs)
        print("label", label_idx, class_names[label_idx])
        for pair in readable_exp:
            print(pair)
        print("-" * 40)
