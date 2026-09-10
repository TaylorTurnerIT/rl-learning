#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/home/dev/Projects/rl-learning"
JOB_ID="colab-1b3bd06231"
JOB_DIR="$PROJECT_ROOT/history/dodge/ng/colab-jobs/$JOB_ID"
AUTOMATION_DIR="$JOB_DIR/automation"
STATUS_OUTPUT="$AUTOMATION_DIR/last-status.txt"
MONITOR_LOG="$AUTOMATION_DIR/monitor.log"
HANDOFF="$AUTOMATION_DIR/READY_FOR_CODEX.md"
JUST="/run/current-system/sw/bin/just"
RUN_DIRECTORY=$(jq -r '.run_directory // empty' "$JOB_DIR/job.json")

mkdir -p "$AUTOMATION_DIR"
cd "$PROJECT_ROOT"

# systemd user services do not inherit the devenv shell's native library path.
# colabctl's ZeroMQ extension needs libstdc++ when the monitor runs unattended.
gcc_lib=$(gcc -print-file-name=libstdc++.so.6)
if [[ "$gcc_lib" == */libstdc++.so.6 ]]; then
    gcc_lib_dir=$(dirname "$(readlink -f "$gcc_lib")")
    export LD_LIBRARY_PATH="${gcc_lib_dir}:${LD_LIBRARY_PATH:-}"
fi
export NIX_LD="${NIX_LD:-/run/current-system/sw/share/nix-ld/lib/ld.so}"

timestamp() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

local_state=$(jq -r '.state // empty' "$JOB_DIR/status.json" 2>/dev/null || true)
if [[ "$local_state" =~ ^(succeeded|failed|cancelled)$ ]] && [[ -e "$HANDOFF" ]]; then
    printf '%s state=%s handoff=complete\n' "$(timestamp)" "$local_state" >>"$MONITOR_LOG"
    exit 0
fi

recover_session() {
    local session_name endpoint recovery_output
    session_name=$(jq -r '.colab.session_name // empty' "$JOB_DIR/job.json")
    endpoint=$(python3 - "$session_name" <<'PY'
import json
import sys
from pathlib import Path

name = sys.argv[1]
history = Path("/home/dev/.config/colab-cli/history") / f"{name}.jsonl"
endpoint = ""
if history.is_file():
    for line in history.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event_type") == "keep_alive_started":
            value = event.get("endpoint")
            if isinstance(value, str):
                endpoint = value
print(endpoint)
PY
)
    if [[ -z "$session_name" || -z "$endpoint" ]]; then
        return 1
    fi
    recovery_output="$AUTOMATION_DIR/session-recovery-$(date -u +%s).log"
    devenv -q shell -- uv run --extra colab python - "$session_name" "$endpoint" >"$recovery_output" 2>&1 <<'PY'
import logging
import os
import sys

logging.disable(logging.CRITICAL)
from colab_cli.auth import AuthProvider
from colab_cli.common import state
from colab_cli.commands.session import spawn_keep_alive
from colab_cli.state import SessionState, StateStore

name, endpoint = sys.argv[1:3]
state.auth_provider = AuthProvider.ADC
assignment = next(
    (item for item in state.client.list_assignments() if item.endpoint == endpoint),
    None,
)
if assignment is None:
    raise SystemExit("target assignment is no longer advertised")
proxy = assignment.runtime_proxy_info
store = StateStore()
existing = store.get(name)
if existing is not None and existing.endpoint == endpoint and existing.keep_alive_pid:
    try:
        os.kill(existing.keep_alive_pid, 0)
    except OSError:
        pass
    else:
        print(f"session {name} already has a live keep-alive daemon")
        raise SystemExit(0)
session = SessionState(
    name=name,
    token=proxy.token,
    url=proxy.url,
    endpoint=assignment.endpoint,
    variant="GPU",
    accelerator="T4",
)
store.add(session)
pid = spawn_keep_alive(
    assignment.endpoint,
    name,
    auth_provider=AuthProvider.ADC,
    config_path=None,
)
session.keep_alive_pid = pid
store.add(session)
print(f"restored {name} on {assignment.endpoint}; keep_alive_pid={pid}")
PY
}

if ! "$JUST" dodge-ng-colab status "$JOB_ID" >"$STATUS_OUTPUT" 2>&1; then
    printf '%s status=command-error; attempting-session-recovery\n' "$(timestamp)" >>"$MONITOR_LOG"
    if recover_session; then
        printf '%s session=recovered; retrying-status\n' "$(timestamp)" >>"$MONITOR_LOG"
        if ! "$JUST" dodge-ng-colab status "$JOB_ID" >"$STATUS_OUTPUT" 2>&1; then
            printf '%s status=command-error-after-recovery\n' "$(timestamp)" >>"$MONITOR_LOG"
            exit 1
        fi
    else
        printf '%s session-recovery=unavailable\n' "$(timestamp)" >>"$MONITOR_LOG"
        exit 1
    fi
fi

state=$(jq -r '.state // "unknown"' "$JOB_DIR/status.json")
accelerator=$(jq -r '.accelerator // "unknown"' "$JOB_DIR/status.json")
printf '%s state=%s accelerator=%s\n' "$(timestamp)" "$state" "$accelerator" >>"$MONITOR_LOG"

case "$state" in
    succeeded|failed|cancelled)
        if [[ -e "$HANDOFF" ]]; then
            exit 0
        fi
        watch_output="$AUTOMATION_DIR/watch-$(date -u +%s).log"
        "$JUST" dodge-ng-colab watch "$JOB_ID" --poll-seconds 1 >"$watch_output" 2>&1 || true
        if grep -Eq '"retrieved"[[:space:]]*:[[:space:]]*true' "$watch_output"; then
            report_output="$AUTOMATION_DIR/report-$(date -u +%s).log"
            report_state="not-run"
            if [[ -f "$RUN_DIRECTORY/hpo.json" ]]; then
                if devenv -q shell -- uv run --extra native --extra colab \
                    python -m dodge.ng.hpo_report "$RUN_DIRECTORY" \
                    --manifest context/kits/dodge-ng/ng-v1.json \
                    --job-dir "$JOB_DIR" >"$report_output" 2>&1; then
                    report_state="generated"
                    printf '%s report=generated\n' "$(timestamp)" >>"$MONITOR_LOG"
                else
                    report_state="failed; see $report_output"
                    printf '%s report=failed; see %s\n' "$(timestamp)" "$report_output" >>"$MONITOR_LOG"
                fi
            else
                report_state="missing-hpo-json"
                printf '%s report=skipped; hpo.json-missing\n' "$(timestamp)" >>"$MONITOR_LOG"
            fi
            {
                printf '# Colab HPO handoff\n\n'
                printf -- '- Job: `%s`\n' "$JOB_ID"
                printf -- '- Terminal state: `%s`\n' "$state"
                printf -- '- GPU: `%s`\n' "$accelerator"
                printf -- '- Artifacts retrieved and runtime released.\n'
                printf -- '- HPO report: `%s`\n' "$report_state"
                printf -- '- Next: inspect `hpo.json`, `HPO_PERFORMANCE_REPORT.md`, metrics, and checkpoints; then diagnose and iterate.\n'
            } >"$HANDOFF"
        else
            printf '%s retrieval=pending\n' "$(timestamp)" >>"$MONITOR_LOG"
        fi
        ;;
esac
