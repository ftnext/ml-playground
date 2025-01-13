# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "browser-use",
#     "jsonlines",
#     "langchain-openai",
#     "pydantic",
# ]
# ///
import os

os.environ["ANONYMIZED_TELEMETRY"] = "false"

import asyncio

import jsonlines
from browser_use import Agent, Controller
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

controller = Controller()


class Article(BaseModel):
    title: str
    url: str


class Articles(BaseModel):
    articles: list[Article]

    def __iter__(self):
        return iter(self.articles)


@controller.action("Save articles", param_model=Articles)
def save_models(params: Articles):
    with jsonlines.open("articles.jsonl", "w") as f:
        f.write_all(model.model_dump() for model in params)


async def main(newsletter_url: str):
    agent = Agent(
        task=f"Python Newsletter {newsletter_url!r} からArticleのタイトルとURLを抜き出し、ファイルに保存する",
        # task=f"Open {newsletter_url!r} (Python Newsletter) and extract articles, save them.",  # Can't see HTML?
        llm=ChatOpenAI(model="gpt-4o"),
        controller=controller,
    )
    return await agent.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("newsletter_url")
    args = parser.parse_args()

    print(asyncio.run(main(args.newsletter_url)))
