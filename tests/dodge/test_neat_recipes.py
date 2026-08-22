from __future__ import annotations

from dodge.control import PROJECT_ROOT


def test_v14_neat_recipes_use_the_project_uv_environment() -> None:
    devenv = (PROJECT_ROOT / "devenv.nix").read_text(encoding="utf-8")

    assert "exec uv run python -m dodge.neat.train" in devenv
    assert "exec uv run python -m dodge.neat.replay" in devenv
