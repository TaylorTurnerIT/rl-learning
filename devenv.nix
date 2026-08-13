{ pkgs, lib, config, inputs, ... }:

{
  # System packages available in environment
  packages = [
    pkgs.git
    pkgs.ruff
  ];

  # Language configuration: Enable Python and uv with automatic sync
  languages.python = {
    enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  # Environment scripts for common tasks
  scripts = {
    app-run.exec = "uv run rl-learning";
    app-test.exec = "uv run pytest";
    app-fmt.exec = "ruff format .";
    app-check.exec = "ruff check .";
  };

  # Shell hook executed when entering devenv shell
  enterShell = ''
    echo "=========================================="
    echo "🚀 Python + uv devenv environment active!"
    echo "Python: $(python --version 2>/dev/null || echo 'N/A')"
    echo "uv:     $(uv --version 2>/dev/null || echo 'N/A')"
    echo "=========================================="
  '';
}
