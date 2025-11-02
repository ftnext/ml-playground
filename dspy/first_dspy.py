# /// script
# requires-python = ">=3.11,<=3.13"
# dependencies = [
#     "datasets==4.3.0",
#     "dspy==3.0.3",
#     "marimo",
#     "mlflow==3.5.1",
# ]
# ///

import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    ref: https://dspy.ai/tutorials/gepa_aime/
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import random
    from typing import Literal

    import dspy
    import mlflow
    from datasets import load_dataset
    return Literal, dspy, load_dataset, mlflow, random


@app.cell
def _(mo):
    mo.md(r"""
    MLflow integration

    ```shell
    $ uvx --python 3.13 mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns.db
    [MLflow] Security middleware enabled with default settings (localhost-only). To allow connections from other hosts, use --host 0.0.0.0 and configure --allowed-hosts and --cors-allowed-origins.
    ```
    """)
    return


@app.cell
def _(mlflow):
    mlflow.set_tracking_uri("http://0.0.0.0:5000")
    mlflow.set_experiment("Default")

    mlflow.dspy.autolog(
        log_compiles=True,
        log_evals=True,
        log_traces=True,
    )
    return


@app.cell
def _(dspy):
    lm = dspy.LM("openai/gpt-4.1-mini", temperature=1, max_tokens=32000)
    dspy.configure(lm=lm)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Loading the AIME dataset
    """)
    return


@app.cell
def _(dspy, load_dataset, random):
    def init_dataset():
        train_split = [
            dspy.Example(
                {
                    "problem": x["problem"],
                    "solution": x["solution"],
                    "answer": x["answer"],
                }
            ).with_inputs("problem")
            for x in load_dataset("AI-MO/aimo-validation-aime")["train"]
        ]
        random.Random(0).shuffle(train_split)
        total_num = len(train_split)
        train_set = train_split[: int(0.5 * total_num)]
        val_set = train_split[int(0.5 * total_num):]

        test_split = [
            dspy.Example(
                {
                    "problem": x["problem"],
                    "answer": x["answer"],
                }
            ).with_inputs("problem")
            for x in load_dataset("MathArena/aime_2025")["train"]
        ]
        test_set = test_split * 5

        return train_set, val_set, test_set
    return (init_dataset,)


@app.cell
def _(init_dataset):
    train_set, val_set, test_set = init_dataset()

    len(train_set), len(val_set), len(test_set)
    return test_set, train_set


@app.cell
def _(train_set):
    train_set[0]["problem"]
    return


@app.cell
def _(train_set):
    train_set[0]["answer"]
    return


@app.cell
def _(train_set):
    train_set[0]["solution"][:200]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A simple dspy.ChainOfThought
    """)
    return


@app.cell
def _(dspy):
    class GenerateResponse(dspy.Signature):
        """Solve the problem and provide the answer in the correct format."""
        problem = dspy.InputField()
        answer = dspy.OutputField()
    return (GenerateResponse,)


@app.cell
def _(GenerateResponse, dspy):
    program = dspy.ChainOfThought(GenerateResponse)
    return (program,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Defining the evaluation metric
    """)
    return


@app.cell
def _(Literal):
    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None) -> Literal[0, 1]:
        correct_answer = int(example["answer"])
        try:
            llm_answer = int(prediction.answer)
        except ValueError:
            return 0
        return int(correct_answer == llm_answer)
    return (metric,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Evaluating unoptimized Chain Of Thought
    """)
    return


@app.cell
def _(dspy, metric, test_set):
    evaluate = dspy.Evaluate(
        devset=test_set,
        metric=metric,
        num_threads=32,
        display_table=True,
        display_progress=True,
    )
    return (evaluate,)


@app.cell
def _(evaluate, program):
    evaluate(program)
    return


if __name__ == "__main__":
    app.run()
