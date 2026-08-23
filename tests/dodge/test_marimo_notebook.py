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


def test_v63_notebook_adds_repository_src_to_its_import_path() -> None:
    notebook = Path("notebooks/dodge_behavior_cloning.py")
    module = ast.parse(notebook.read_text())
    assignments = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PROJECT_SRC"
            for target in node.targets
        )
    ]

    assert len(assignments) == 1
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "insert"
        for node in ast.walk(module)
    )
