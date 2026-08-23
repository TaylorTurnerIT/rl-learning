from __future__ import annotations

import ast
from pathlib import Path


def test_v62_marimo_cells_have_no_conditional_returns() -> None:
    notebook = Path("notebooks/dodge_behavior_cloning.py")
    module = ast.parse(notebook.read_text())
    cells = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Attribute) and decorator.attr == "cell"
            for decorator in node.decorator_list
        )
    ]

    assert cells
    for cell in cells:
        conditional_returns = [
            node
            for node in ast.walk(cell)
            if isinstance(node, ast.Return) and node not in cell.body[-1:]
        ]
        assert not conditional_returns
