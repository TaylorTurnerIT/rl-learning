import sys
from pathlib import Path

import marimo

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

__generated_with = "0.20.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from dodge.imitation.data import load_demonstrations, split_demonstrations
    from dodge.imitation.train import (
        save_training_history,
        save_training_result,
        train_behavior_cloning,
    )

    return (
        Path,
        load_demonstrations,
        mo,
        save_training_history,
        save_training_result,
        split_demonstrations,
        train_behavior_cloning,
    )


@app.cell
def _(mo):
    dataset_path = mo.ui.text(
        value="dodge-dataset.sqlite3", label="SQLite snapshot path"
    )
    output_path = mo.ui.text(
        value="history/dodge/models/behavior-cloning.pt", label="Model output path"
    )
    epochs = mo.ui.number(value=50, start=1, step=1, label="Epochs")
    batch_size = mo.ui.number(value=128, start=1, step=1, label="Batch size")
    validation_seed_count = mo.ui.number(
        value=10, start=1, step=1, label="Validation seeds"
    )
    train = mo.ui.run_button(label="Train")
    mo.vstack(
        [
            mo.md(
                "# Dodge behavior cloning\n\n"
                "Export the live collector first with `just dodge-dataset-export` "
                "and upload that snapshot here. Training uses a GPU when one is "
                "attached, otherwise it runs on CPU."
            ),
            dataset_path,
            output_path,
            mo.hstack([epochs, batch_size, validation_seed_count]),
            train,
        ]
    )
    return batch_size, dataset_path, epochs, output_path, train, validation_seed_count


@app.cell
def _(
    Path,
    batch_size,
    dataset_path,
    epochs,
    load_demonstrations,
    mo,
    output_path,
    save_training_history,
    save_training_result,
    split_demonstrations,
    train,
    train_behavior_cloning,
    validation_seed_count,
):
    mo.stop(not train.value)
    split = split_demonstrations(
        load_demonstrations(Path(dataset_path.value)), int(validation_seed_count.value)
    )
    result = train_behavior_cloning(
        split.training,
        epochs=int(epochs.value),
        batch_size=int(batch_size.value),
        device="auto",
        validation_demonstrations=split.validation,
    )
    output = Path(output_path.value)
    save_training_result(result, output)
    history = output.with_suffix(".metrics.json")
    save_training_history(result, history, split.validation_seeds)
    return history, output, result


@app.cell
def _(history, mo, output, result):
    mo.stop(result is None, mo.md("Set paths, then press **Train**."))
    mo.md(
        f"Trained {result.examples:,} decisions on `{result.device}`. "
        f"Final training loss: `{result.final_loss:.4f}`. "
        f"Final validation loss: `{result.final_validation_loss:.4f}`. "
        f"Artifact: `{output}`. History: `{history}`."
    )


if __name__ == "__main__":
    app.run()
