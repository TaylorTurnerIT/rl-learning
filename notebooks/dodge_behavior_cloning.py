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
    import torch

    from dodge.imitation.data import load_demonstrations
    from dodge.imitation.train import save_training_result, train_behavior_cloning

    return (
        Path,
        load_demonstrations,
        mo,
        save_training_result,
        torch,
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
    train = mo.ui.run_button(label="Train on GPU")
    mo.vstack(
        [
            mo.md(
                "# Dodge behavior cloning\n\n"
                "Export the live collector first with `just dodge-dataset-export` "
                "and upload that snapshot here. Attach a GPU before training."
            ),
            dataset_path,
            output_path,
            mo.hstack([epochs, batch_size]),
            train,
        ]
    )
    return batch_size, dataset_path, epochs, output_path, train


@app.cell
def _(
    Path,
    batch_size,
    dataset_path,
    epochs,
    load_demonstrations,
    mo,
    output_path,
    save_training_result,
    torch,
    train,
    train_behavior_cloning,
):
    mo.stop(not train.value)
    mo.stop(
        not torch.cuda.is_available(),
        mo.md("Attach a CUDA GPU before running Dodge training."),
    )
    result = train_behavior_cloning(
        load_demonstrations(Path(dataset_path.value)),
        epochs=int(epochs.value),
        batch_size=int(batch_size.value),
        device="cuda",
    )
    output = Path(output_path.value)
    save_training_result(result, output)
    return output, result


@app.cell
def _(mo, output, result):
    mo.stop(
        result is None, mo.md("Set paths, attach a GPU, then press **Train on GPU**.")
    )
    mo.md(
        f"Trained {result.examples:,} decisions on `{result.device}`. "
        f"Final loss: `{result.final_loss:.4f}`. Artifact: `{output}`."
    )


if __name__ == "__main__":
    app.run()
