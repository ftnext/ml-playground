from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("llm-book/ner-wikipedia-dataset")
    print(dataset)
