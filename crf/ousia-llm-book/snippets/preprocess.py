from typing import TypedDict, cast

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from data_types import Entity
from iob2_labels import output_labels, tokenize

model_name = "cl-tohoku/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(model_name)


class Example(TypedDict):
    curid: str
    text: str
    entities: list[Entity]


class EncodedExample(TypedDict):
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    special_tokens_mask: torch.Tensor
    attention_mask: torch.Tensor


def preprocess_data(
    data: Example, tokenizer: PreTrainedTokenizer, label2id: dict[str, int]
) -> BatchEncoding:
    inputs: EncodedExample = tokenizer(
        data["text"], return_tensors="pt", return_special_tokens_mask=True
    )
    # torch.Tensorのsizeが[1, トークン長]となっているので、squeezeしてsize [トークン長] とする
    flatten_inputs = {
        k: cast(torch.Tensor, v).squeeze(0) for k, v in inputs.items()
    }

    tokens = tokenize(data["text"], tokenizer)
    string_labels = output_labels(data["text"], tokens, data["entities"])
    assert len(string_labels) == flatten_inputs["input_ids"].size(0)

    # string_labelsには[CLS]と[SEP]に対応する-があり、これはlabel2idに含まれない
    tensor_labels = torch.tensor(
        [label2id.get(label, 0) for label in string_labels]
    )
    tensor_labels[torch.where(flatten_inputs["special_tokens_mask"])] = -100
    flatten_inputs["labels"] = tensor_labels
    return flatten_inputs


if __name__ == "__main__":
    from label_ids import create_label2id

    dataset = load_dataset("llm-book/ner-wikipedia-dataset")
    label2id = create_label2id(dataset["train"]["entities"])

    train_dataset = dataset["train"].map(
        preprocess_data,
        fn_kwargs={"tokenizer": tokenizer, "label2id": label2id},
        remove_columns=dataset["train"].column_names,
    )
    validation_dataset = dataset["validation"].map(
        preprocess_data,
        fn_kwargs={"tokenizer": tokenizer, "label2id": label2id},
        remove_columns=dataset["validation"].column_names,
    )
