from datasets import load_dataset
from spacy_alignments import get_alignments
from transformers import AutoTokenizer

if __name__ == "__main__":
    dataset = load_dataset("llm-book/ner-wikipedia-dataset")

    model_name = "cl-tohoku/bert-base-japanese-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = "さくら学院"
    characters = list(text)
    assert characters == ["さ", "く", "ら", "学", "院"]
    # tokenizeメソッドでない理由は、[CLS]や[SEP]などの特殊トークンを含めたtokensを得るため
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    assert tokens == ["[CLS]", "さくら", "学院", "[SEP]"]

    char_to_token_indices, token_to_char_indices = get_alignments(
        characters, tokens
    )

    # さ/く/らでtokens[1]、学/院でtokens[2]
    assert char_to_token_indices == [[1], [1], [1], [2], [2]]

    # [CLS]と[SEP]を構成する文字はなし
    # tokens[1]（さくら）は characters[0: 2+1]
    # tokens[2]（学院）は characters[3: 4+1]
    assert token_to_char_indices == [[], [0, 1, 2], [3, 4], []]
