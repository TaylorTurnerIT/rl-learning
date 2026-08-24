from __future__ import annotations

import argparse
import json
from pathlib import Path

from dodge.control import ControlInputError


def plot_training_history(history_path: Path, output: Path) -> None:
    if not history_path.is_file():
        raise ControlInputError(f"training history does not exist: {history_path}")
    try:
        history = json.loads(history_path.read_text())
        epochs = history["epochs"]
        if not isinstance(epochs, list) or not epochs:
            raise ValueError
        epoch_numbers = [int(entry["epoch"]) for entry in epochs]
        training_losses = [float(entry["training_loss"]) for entry in epochs]
        validation_losses = [float(entry["validation_loss"]) for entry in epochs]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ControlInputError("training history is invalid") from error

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = pyplot.subplots(figsize=(8, 5))
    axis.plot(epoch_numbers, training_losses, label="training loss")
    axis.plot(epoch_numbers, validation_losses, label="validation loss")
    axis.set(
        xlabel="epoch", ylabel="cross-entropy loss", title="Dodge behavior cloning"
    )
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output)
    pyplot.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-bc-plot")
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    output = arguments.output or arguments.history.with_suffix(".png")
    try:
        plot_training_history(arguments.history, output)
    except (ControlInputError, OSError) as error:
        print(f"dodge-bc-plot: {error}")
        return 1
    print(json.dumps({"history": str(arguments.history), "plot": str(output)}))
    return 0
