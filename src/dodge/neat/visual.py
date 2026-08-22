# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

from dodge.neat.bridge import Direction

ACTIONS: tuple[Direction, ...] = (
    "neutral",
    "left",
    "right",
    "up",
    "down",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
)


def write_network_visualization(
    genome: object,
    config: object,
    directory: Path,
    *,
    enemy_slots: int,
    aoe_slots: int,
) -> Path:
    path = directory / "network.html"
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(
        network_visualization_html(
            genome,
            config,
            enemy_slots=enemy_slots,
            aoe_slots=aoe_slots,
        ),
        encoding="utf-8",
    )
    return path


def network_visualization_html(
    genome: object,
    config: object,
    *,
    enemy_slots: int = 16,
    aoe_slots: int = 8,
) -> str:
    graph = _graph_data(
        genome,
        config,
        enemy_slots=enemy_slots,
        aoe_slots=aoe_slots,
    )
    encoded_graph = json.dumps(graph, separators=(",", ":")).replace("</", "<\\/")
    return _HTML.replace("__NETWORK_JSON__", encoded_graph)


def _graph_data(
    genome: object,
    config: object,
    *,
    enemy_slots: int,
    aoe_slots: int,
) -> dict[str, object]:
    genome_config = getattr(config, "genome_config", None)
    nodes = getattr(genome, "nodes", {})
    connections = getattr(genome, "connections", {})
    input_keys = tuple(getattr(genome_config, "input_keys", ()))
    output_keys = tuple(getattr(genome_config, "output_keys", ()))
    if not isinstance(nodes, dict) or not isinstance(connections, dict):
        raise ValueError("genome does not expose NEAT nodes and connections")

    output_set = set(output_keys)
    input_set = set(input_keys)
    hidden_keys = sorted(set(nodes).difference(output_set), key=str)
    all_nodes = [
        *(
            _node_data(
                key,
                "input",
                input_keys,
                output_keys,
                nodes,
                enemy_slots,
                aoe_slots,
            )
            for key in input_keys
        ),
        *(
            _node_data(
                key,
                "hidden",
                input_keys,
                output_keys,
                nodes,
                enemy_slots,
                aoe_slots,
            )
            for key in hidden_keys
        ),
        *(
            _node_data(
                key,
                "output",
                input_keys,
                output_keys,
                nodes,
                enemy_slots,
                aoe_slots,
            )
            for key in output_keys
        ),
    ]
    enabled_edges = [
        {
            "source": _node_id(source),
            "target": _node_id(target),
            "weight": float(getattr(connection, "weight", 0.0)),
        }
        for (source, target), connection in connections.items()
        if getattr(connection, "enabled", False)
        and source in input_set.union(nodes)
        and target in output_set.union(nodes)
    ]
    enabled_edges.sort(key=lambda edge: abs(float(edge["weight"])), reverse=True)
    return {
        "nodes": all_nodes,
        "edges": enabled_edges,
        "totalConnections": len(connections),
    }


def _node_data(
    key: object,
    layer: str,
    input_keys: tuple[object, ...],
    output_keys: tuple[object, ...],
    nodes: dict[object, object],
    enemy_slots: int,
    aoe_slots: int,
) -> dict[str, object]:
    node = nodes.get(key)
    return {
        "id": _node_id(key),
        "label": _node_label(
            key,
            input_keys,
            output_keys,
            enemy_slots,
            aoe_slots,
        ),
        "layer": layer,
        "bias": float(getattr(node, "bias", 0.0)) if node is not None else None,
    }


def _node_id(key: object) -> str:
    return f"node:{key}"


def _node_label(
    key: object,
    input_keys: tuple[object, ...],
    output_keys: tuple[object, ...],
    enemy_slots: int,
    aoe_slots: int,
) -> str:
    if key in output_keys:
        index = output_keys.index(key)
        return ACTIONS[index] if index < len(ACTIONS) else f"output{index}"
    if key not in input_keys:
        return f"hidden {key}"
    index = input_keys.index(key)
    player_features = ("player.x", "player.y", "player.vx", "player.vy", "player.size")
    entity_features = ("present", "dx", "dy", "vx", "vy", "width", "height", "stage")
    if index < len(player_features):
        return player_features[index]
    entity_index = index - len(player_features)
    enemy_feature_count = enemy_slots * len(entity_features)
    if entity_index < enemy_feature_count:
        slot, feature = divmod(entity_index, len(entity_features))
        return f"enemy {slot + 1}.{entity_features[feature]}"
    aoe_index = entity_index - enemy_feature_count
    if aoe_index < aoe_slots * len(entity_features):
        slot, feature = divmod(aoe_index, len(entity_features))
        return f"aoe {slot + 1}.{entity_features[feature]}"
    return f"input {index}"


_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dodge NEAT network</title>
<style>
  :root { color-scheme: dark; font-family: system-ui, sans-serif; }
  body { margin: 0; background: #10131a; color: #e5e7eb; }
  header { position: sticky; top: 0; z-index: 2; padding: 16px 24px;
    background: #171b26; border-bottom: 1px solid #30384b; }
  h1 { margin: 0 0 8px; font-size: 18px; }
  p { margin: 0; color: #aab5cc; font-size: 14px; }
  label { display: block; margin-top: 12px; font-size: 14px; }
  input { width: min(680px, 80vw); vertical-align: middle; }
  #count { display: inline-block; min-width: 90px; margin-left: 8px; }
  main { overflow: auto; padding: 20px; }
  svg { display: block; min-width: 1100px; background: #0c1018; border: 1px solid #30384b; }
  .edge { fill: none; stroke-linecap: round; }
  .node rect { stroke: #d5d9e4; stroke-width: 1; rx: 4; }
  .node text { fill: #f8fafc; font-size: 12px; dominant-baseline: middle; }
  .input rect { fill: #1d4ed8; } .hidden rect { fill: #7e22ce; }
  .output rect { fill: #15803d; }
  .layer-label { fill: #aab5cc; font-size: 18px; font-weight: 650; }
</style>
</head>
<body>
<header>
  <h1>Best Dodge NEAT network</h1>
  <p id="stats"></p>
  <label>Visible strongest enabled connections:
    <input id="edge-limit" type="range" min="0" step="1">
    <span id="count"></span>
  </label>
</header>
<main><svg id="network" aria-label="NEAT network diagram"></svg></main>
<script>
const graph = __NETWORK_JSON__;
const svg = document.querySelector('#network');
const ns = 'http://www.w3.org/2000/svg';
const limit = document.querySelector('#edge-limit');
const count = document.querySelector('#count');
const byLayer = {input: [], hidden: [], output: []};
for (const node of graph.nodes) byLayer[node.layer].push(node);
const layers = ['input', 'hidden', 'output'];
const positions = new Map();
const width = 1500;
const maxNodes = Math.max(1, ...layers.map(layer => (byLayer[layer] || []).length));
const height = Math.max(760, 110 + maxNodes * 24);
const layerX = { input: 100, hidden: 750, output: 1400 };
svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
svg.setAttribute('width', width);
svg.setAttribute('height', height);
for (const layer of layers) {
  const nodes = byLayer[layer] || [];
  const spacing = (height - 100) / Math.max(1, nodes.length);
  nodes.forEach((node, index) => positions.set(node.id, {
    x: layerX[layer], y: 70 + spacing * (index + .5), node
  }));
  const label = document.createElementNS(ns, 'text');
  label.setAttribute('class', 'layer-label');
  label.setAttribute('x', layerX[layer]); label.setAttribute('y', 32);
  label.setAttribute('text-anchor', 'middle'); label.textContent = layer.toUpperCase();
  svg.append(label);
}
const edgeLayer = document.createElementNS(ns, 'g');
const nodeLayer = document.createElementNS(ns, 'g');
svg.append(edgeLayer, nodeLayer);
for (const {x, y, node} of positions.values()) {
  const group = document.createElementNS(ns, 'g');
  group.setAttribute('class', `node ${node.layer}`);
  const rect = document.createElementNS(ns, 'rect');
  rect.setAttribute('x', x - 52); rect.setAttribute('y', y - 9);
  rect.setAttribute('width', 104); rect.setAttribute('height', 18);
  const text = document.createElementNS(ns, 'text');
  text.setAttribute('x', x); text.setAttribute('y', y + 1);
  text.setAttribute('text-anchor', 'middle'); text.textContent = node.label;
  const title = document.createElementNS(ns, 'title');
  title.textContent = node.bias === null ? node.label : `${node.label}; bias ${node.bias.toFixed(3)}`;
  group.append(rect, text, title); nodeLayer.append(group);
}
limit.max = graph.edges.length;
limit.value = Math.min(80, graph.edges.length);
document.querySelector('#stats').textContent =
  `${graph.nodes.length} nodes; ${graph.edges.length}/${graph.totalConnections} enabled connections. ` +
  'Blue: positive weight. Red: negative weight. Hover a node for its bias.';
function renderEdges() {
  const visible = graph.edges.slice(0, Number(limit.value));
  edgeLayer.replaceChildren();
  const maxWeight = Math.max(1, ...visible.map(edge => Math.abs(edge.weight)));
  for (const edge of visible) {
    const source = positions.get(edge.source); const target = positions.get(edge.target);
    if (!source || !target) continue;
    const line = document.createElementNS(ns, 'path');
    const midpoint = (source.x + target.x) / 2;
    line.setAttribute('d', `M ${source.x + 52} ${source.y} C ${midpoint} ${source.y}, ${midpoint} ${target.y}, ${target.x - 52} ${target.y}`);
    line.setAttribute('class', 'edge');
    line.setAttribute('stroke', edge.weight >= 0 ? '#38bdf8' : '#fb7185');
    line.setAttribute('stroke-width', (0.4 + 2.6 * Math.abs(edge.weight) / maxWeight).toFixed(2));
    line.setAttribute('opacity', '0.75');
    const title = document.createElementNS(ns, 'title'); title.textContent = `weight ${edge.weight.toFixed(4)}`;
    line.append(title); edgeLayer.append(line);
  }
  count.textContent = `${visible.length} / ${graph.edges.length}`;
}
limit.addEventListener('input', renderEdges);
renderEdges();
</script>
</body>
</html>
"""
