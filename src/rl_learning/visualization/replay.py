import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from rl_learning.agent.agent import Agent


def replay_history(history: list[list[Agent]]) -> None:
    """Serve the recorded epochs to a minimal browser-based replay."""
    if not history:
        return

    replay_json = json.dumps(_history_payload(history)).encode()
    page = Path(__file__).with_name("index.html").read_bytes()

    class ReplayHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self._send(page, "text/html; charset=utf-8")
            elif self.path == "/replay.json":
                self._send(replay_json, "application/json")
            else:
                self.send_error(404)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), ReplayHandler)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Replay available at {url} (Ctrl+C stops the replay server)")
    webbrowser.open_new_tab(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _history_payload(history: list[list[Agent]]) -> dict[str, list[dict[str, object]]]:
    return {"epochs": [_epoch_payload(agents) for agents in history]}


def _epoch_payload(agents: list[Agent]) -> dict[str, object]:
    game = agents[0].game
    return {
        "game": {
            "x_size": game.x_size,
            "y_size": game.y_size,
            "pit": [game.pit_pos.x, game.pit_pos.y],
            "wumpus": [game.wumpus_pos.x, game.wumpus_pos.y],
            "win": [game.win_pos.x, game.win_pos.y],
        },
        "agents": [
            {
                "start": [agent.starting_pos.x, agent.starting_pos.y],
                "actions": [action.value for action in agent.brain.actions],
            }
            for agent in agents
        ],
    }
