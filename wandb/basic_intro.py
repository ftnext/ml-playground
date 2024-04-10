import random
import wandb

random.seed(42)
total_runs = 3
for run in range(total_runs):
    with wandb.init(
        project="basic-intro",
        name=f"experiment_{run}",
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        },
    ) as run:

        epochs = 10
        offset = random.random() / 5
        for epoch in range(2, epochs):
            loss = 2**-epoch + random.random() / epoch + offset
            acc = 1 - loss
            run.log({"acc": acc, "loss": loss})
