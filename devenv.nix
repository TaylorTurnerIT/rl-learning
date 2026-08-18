{ lib, pkgs, ... }:

{
  # System packages available in environment
  packages = [
    pkgs.git
    pkgs.ruff
    pkgs.SDL2
    pkgs.xdotool
    pkgs.zlib
  ];

  # Provide Python and uv; uv owns the conventional project-local .venv.
  languages.python = {
    enable = true;
    uv = {
      enable = true;
    };
  };

  # Use uv's conventional, editor-discoverable project environment.
  env.UV_PROJECT_ENVIRONMENT = lib.mkForce ".venv";

  # Environment scripts for common tasks
  scripts = {
    app-run.exec = "uv run rl-learning";
    app-test.exec = "uv run pytest";
    app-fmt.exec = "ruff format .";
    app-check.exec = "ruff check .";
    dodge-run.exec = ''
      export LD_LIBRARY_PATH=${lib.makeLibraryPath [ pkgs.SDL2 ]}:"$LD_LIBRARY_PATH"
      exec ./src/dodge/runtime/pemsa ./src/dodge/game/dodge.p8 --no-splash --no-fullscreen
    '';
    dodge-control.exec = ''
      export LD_LIBRARY_PATH=${lib.makeLibraryPath [ pkgs.SDL2 ]}:"$LD_LIBRARY_PATH"
      exec python ./src/dodge/control.py "$@"
    '';
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
