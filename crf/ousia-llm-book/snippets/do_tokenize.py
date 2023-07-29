from datasets import load_dataset
from transformers import AutoTokenizer

if __name__ == "__main__":
    dataset = load_dataset("llm-book/ner-wikipedia-dataset")

    model_name = "cl-tohoku/bert-base-japanese-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokens = tokenizer.tokenize(dataset["train"][0]["text"])
    assert tokens == [
        "さくら",
        "学院",
        "、",
        "C",
        "##ia",
        "##o",
        "Sm",
        "##ile",
        "##s",
        "の",
        "メンバー",
        "。",
    ]
