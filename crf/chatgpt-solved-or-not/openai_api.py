import argparse
import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, Generator, Iterable
from pathlib import Path
from typing import TypedDict

import backoff
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
) -> AsyncGenerator[OpenAIResponse, None, None]:
    for chunk in chunked(propmts, chunk_size):
        coroutines = [
            _single_call(prompt, model, temperature) for prompt in chunk
        ]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                print(repr(result))
                yield {"response": None}
            else:
                yield {
                    "response": result["choices"][0]["message"][
                        "content"
                    ].strip()
                }


@backoff.on_exception(
    backoff.expo,
    (
        openai.error.RateLimitError,
        openai.error.APIConnectionError,
        openai.error.APIError,
        openai.error.ServiceUnavailableError,
    ),
    max_tries=3,
)
async def _single_call(
    prompt: str, model: str = "gpt-3.5-turbo-0301", temperature: float = 0.0
):
    return await openai.ChatCompletion.acreate(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )


async def response_added_example_generator(
    examples: Iterable[Example], responses: AsyncIterable[OpenAIResponse]
) -> AsyncGenerator[Prediction, None, None]:
    examples_generator = (e for e in examples)
    async for response in responses:
        example = next(examples_generator)
        yield example | response


async def main(input_path: Path, output_path: Path, chunk_size: int):
    examples = load_examples(input_path)
    responses = call_api(tqdm(prompts_generator(examples)), chunk_size)
    with jsonlines.open(output_path, "w") as writer:
        async for example in response_added_example_generator(
            examples, responses
        ):
            writer.write(example)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--chunk_size", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(main(args.input_jsonl, args.output_jsonl, args.chunk_size))
