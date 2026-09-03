{ lib, pkgs, ... }:

let
  x11Runtime = with pkgs; [
    SDL2
    libX11
    libXext
    libXcursor
    libXrandr
    libXi
    libXinerama
    libXfixes
    libXtst
    libxcb
    libxkbcommon
    libglvnd
    mesa
  ];
  x11LibraryPath = lib.makeLibraryPath x11Runtime;
in

{
  # System packages available in environment
  packages = [
    pkgs.git
    pkgs.rustc
    pkgs.cargo
    pkgs.rustfmt
    pkgs.clippy
    pkgs.rust-analyzer
    pkgs.rustup
    pkgs.cargo-nextest
    pkgs.maturin
    pkgs.bacon
    pkgs.curl
    pkgs.ruff
    pkgs.SDL2
    pkgs.xdotool
    pkgs.xorg.xorgserver
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
    app-test.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      uv run pytest "$@"
    '';
    app-fmt.exec = "ruff format .";
    app-check.exec = "ruff check .";
    dodge-run.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      exec ./src/dodge/runtime/pemsa ./src/dodge/game/dodge.p8 --no-splash --no-fullscreen
    '';
    dodge-control.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      exec python ./src/dodge/control.py "$@"
    '';
    dodge-headless.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec python -m dodge.headless "$@"
    '';
    dodge-native-oracle.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run dodge-native-oracle "$@"
    '';
    dodge-native-probe.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run dodge-native-probe "$@"
    '';
    dodge-native-extract-assets.exec = ''
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run dodge-native-extract-assets "$@"
    '';
    dodge-native-p2-report.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run dodge-native-p2-report "$@"
    '';
    dodge-replay.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec python -m dodge.history "$@"
    '';
    dodge-replay-run.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec python -m dodge.history replay-run "$@"
    '';
    dodge-replay-latest.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec python -m dodge.history replay-latest "$@"
    '';
    dodge-neat-train.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run python -m dodge.neat.train "$@"
    '';
    dodge-ppo-train.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run dodge-ppo-train "$@"
    '';
    dodge-ng-manifest.exec = ''
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run dodge-ng-manifest "$@"
    '';
    dodge-ng-train.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run dodge-ng-train "$@"
    '';
    dodge-ng-report.exec = ''
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run dodge-ng-report "$@"
    '';
    dodge-neat-replay.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run python -m dodge.neat.replay "$@"
    '';
    dodge-neat-replay-latest.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run python -m dodge.neat.replay replay-latest "$@"
    '';
    dodge-native-viewer.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export LIBGL_DRIVERS_PATH=${pkgs.mesa}/lib/dri:"$LIBGL_DRIVERS_PATH"
      exec cargo run --manifest-path native/Cargo.toml -p dodge-viewer -- "$@"
    '';
    dodge-native-benchmark.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run --extra native python scripts/benchmark_dodge_native_batch.py "$@"
    '';
    dodge-native-fuzz.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run --extra native python scripts/native_differential_fuzz.py "$@"
    '';
    dodge-native-provenance.exec = ''
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run --extra native python scripts/record_native_provenance.py "$@"
    '';
    dodge-native-regression-bench.exec = ''
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run --extra native python scripts/check_native_benchmark.py "$@"
    '';
    dodge-native-visual-compare.exec = ''
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run --extra native python scripts/visual_compare_native.py "$@"
    '';
    dodge-native-ga-visual-compare.exec = ''
      export LD_LIBRARY_PATH=${x11LibraryPath}:"$LD_LIBRARY_PATH"
      export PYTHONPATH="$PWD/src:''${PYTHONPATH:-}"
      exec uv run --extra native python scripts/native_ga_differential.py "$@"
    '';
  };

  # Shell hook executed when entering devenv shell
  enterShell = ''
    export PATH=${pkgs.curl}/bin:"$PATH"
    if [ "''${DODGE_HEADLESS:-0}" != "1" ]; then
      echo "=========================================="
      echo "🚀 Python + uv devenv environment active!"
      echo "Python: $(python --version 2>/dev/null || echo 'N/A')"
      echo "uv:     $(uv --version 2>/dev/null || echo 'N/A')"
      echo "=========================================="
    fi
  '';
}
