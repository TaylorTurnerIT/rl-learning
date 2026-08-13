# RL Learning

A basic Python project initialized with [uv](https://github.com/astral-sh/uv) and [devenv](https://devenv.sh/).

## Features
- **devenv**: Reproducible development environment powered by Nix.
- **uv**: Fast Python package installer and virtual environment manager.
- **Hatchling**: Modern PEP 621 compliant build backend.
- **Pytest & Ruff**: Pre-configured testing and linting tools.

## Getting Started

### Using devenv

To enter the development environment shell:

```bash
devenv shell
```

Or if using `direnv`:

```bash
direnv allow
```

### Development Commands

Inside `devenv shell` or via `devenv shell <script>`:

- Run application: `devenv shell app-run` or `uv run rl-learning`
- Run tests: `devenv shell app-test` or `uv run pytest`
- Format code: `devenv shell app-fmt` or `uv run ruff format .`
- Lint code: `devenv shell app-check` or `uv run ruff check .`
