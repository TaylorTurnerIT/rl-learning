from __future__ import annotations

from pathlib import Path

from dodge.neat.visual import (
    _node_label,
    network_visualization_html,
    write_network_visualization,
)


class Connection:
    def __init__(self, weight: float, enabled: bool = True) -> None:
        self.weight = weight
        self.enabled = enabled


class Node:
    def __init__(self, bias: float) -> None:
        self.bias = bias


class Genome:
    nodes = {0: Node(0.25), 1: Node(-0.5), 4: Node(0.0)}
    connections = {
        (-1, 0): Connection(2.5),
        (-2, 1): Connection(-1.25),
        (-1, 4): Connection(0.2, enabled=False),
    }


class Config:
    class genome_config:
        input_keys = (-1, -2)
        output_keys = (0, 1)


def test_network_visualization_is_self_contained_and_interactive(
    tmp_path: Path,
) -> None:
    path = write_network_visualization(
        Genome(), Config(), tmp_path, enemy_slots=1, aoe_slots=1
    )
    html = path.read_text(encoding="utf-8")

    assert path.name == "network.html"
    assert "const graph =" in html
    assert '"label":"player.x"' in html
    assert '"label":"neutral"' in html
    assert '"weight":2.5' in html
    assert '"weight":-1.25' in html
    assert "edge-limit" in html
    assert "renderEdges" in html
    assert '"totalConnections":3' in html


def test_network_visualization_html_contains_every_enabled_edge() -> None:
    html = network_visualization_html(Genome(), Config(), enemy_slots=1, aoe_slots=1)

    assert '"source":"node:-1","target":"node:0"' in html
    assert '"source":"node:-2","target":"node:1"' in html
    assert '"source":"node:-1","target":"node:4"' not in html


def test_v27_visual_labels_time_to_intersection_for_v2_projection() -> None:
    input_keys = tuple(-index for index in range(1, 222))

    assert _node_label(-7, input_keys, (0,), 16, 8) == "enemy 1.time_to_intersection"
