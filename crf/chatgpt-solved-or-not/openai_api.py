import argparse
import asyncio
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import TypedDict

import jsonlines
import openai
from more_itertools import chunked
from tqdm import tqdm

from custom_types import Example, Prediction


def load_examples(input_path: Path) -> list[Example]:
    with jsonlines.open(input_path) as reader:
        return list(reader)


def prompts_generator(
    examples: Iterable[Example],
) -> Generator[str, None, None]:
    for example in examples:
        yield example["prompt"]


class OpenAIResponse(TypedDict):
    response: str


async def call_api(
    propmts: Iterable[str],
    chunk_size: int,
    model: str = "gpt-3.5-turbo-0301",
    temperature: float = 0.0,
) -> list[OpenAIResponse]:
    responses: list[OpenAIResponse] = []
    for chunk in chunked(propmts, chunk_size):
        coroutines = [
            openai.ChatCompletion.acreate(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            for prompt in chunk
        ]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                responses.append({"response": None})
            else:
                responses.append(
                    {
                        "response": result["choices"][0]["message"][
                            "content"
                        ].strip()
                    }
                )
    return responses


def response_added_example_generator(
    examples: Iterable[Example], responses: Iterable[OpenAIResponse]
) -> Generator[Prediction, None, None]:
    for example, response in zip(examples, responses):
        yield example | response


async def main(input_path: Path, output_path: Path, chunk_size: int):
    examples = load_examples(input_path)
    responses = await call_api(tqdm(prompts_generator(examples)), chunk_size)

    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(response_added_example_generator(examples, responses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--chunk_size", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(main(args.input_jsonl, args.output_jsonl, args.chunk_size))
