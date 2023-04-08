import jsonlines
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing import text

VOCAB_SIZE = 400


def build_model(vocab_size, num_tags):
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Dense(50, input_shape=(vocab_size,), activation="relu")
    )
    model.add(tf.keras.layers.Dense(25, activation="relu"))
    model.add(tf.keras.layers.Dense(num_tags, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    with jsonlines.open("data/processed_so_data.jsonl") as reader:
        texts, tags = [], []
        for d in shuffle(list(reader), random_state=22):
            texts.append(d["text"])
            tags.append(d["tags"])

    tag_encoder = MultiLabelBinarizer()
    tag_encoded = tag_encoder.fit_transform(tags)
    num_tags = len(tag_encoder.classes_)
    print(f"{tag_encoder.classes_=}")

    train_size = int(len(texts) * 0.8)
    print(f"Train size: {train_size}")
    print(f"Test size: {len(texts) - train_size}")

    train_tags = tag_encoded[:train_size]
    test_tags = tag_encoded[train_size:]
    train_qs = texts[:train_size]
    test_qs = texts[train_size:]

    tokenizer = text.Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(train_qs)

    body_train = tokenizer.texts_to_matrix(train_qs)  # 0/1並び
    body_test = tokenizer.texts_to_matrix(test_qs)

    model = build_model(VOCAB_SIZE, num_tags)
    model.summary()
    model.fit(
        body_train, train_tags, epochs=3, batch_size=128, validation_split=0.1
    )
    print(
        f"Eval loss/accuracy:{model.evaluate(body_test, test_tags, batch_size=128)}"
    )

    predictions = model.predict(body_test[:3])
    for q_idx, probabilities in enumerate(predictions):
        print(test_qs[q_idx])
        for idx, tag_prob in enumerate(probabilities):
            if tag_prob > 0.7:
                print(tag_encoder.classes_[idx], round(tag_prob * 100, 2), "%")
        print()
