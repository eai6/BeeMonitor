"""
Drawflow node-graph ⇄ execution-model converters (Phase 2 — the DAG canvas).

The canvas stores a raw Drawflow export on ``Pipeline.graph``; the engine runs
``Pipeline.steps``. These two functions bridge them:

* ``graph_to_steps`` — Drawflow export → ``steps[]`` (with ``inputs`` from edges).
* ``build_initial_steps`` — ``steps[]`` → node/edge descriptors so the canvas can
  render a pipeline that was built before the canvas (or a seeded template) on
  first open.

Named ports live in ``registry.get_input_ports``; Drawflow numbers them input_1,
input_2, … in that order.
"""

from .registry import get_block, get_input_ports, num_output_ports


def _num(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def graph_to_steps(graph):
    """Convert a Drawflow export dict into an execution ``steps`` list."""
    nodes = (((graph or {}).get("drawflow") or {}).get("Home") or {}).get("data") or {}
    items = sorted(nodes.items(), key=lambda kv: _num(kv[0]))

    # Drawflow node-id (int-ish) -> our stable step id.
    idmap = {}
    for nid, node in items:
        data = node.get("data") or {}
        idmap[str(nid)] = data.get("step_id") or f"n{nid}"

    steps = []
    for nid, node in items:
        data = node.get("data") or {}
        block_type = data.get("block_type")
        if not block_type or not get_block(block_type):
            continue
        ports = get_input_ports(block_type)
        node_inputs = node.get("inputs") or {}
        inputs = {}
        for i, port in enumerate(ports):
            conns = (node_inputs.get(f"input_{i + 1}") or {}).get("connections") or []
            if conns:
                src = str(conns[0].get("node"))
                if src in idmap:
                    inputs[port["name"]] = idmap[src]
        step = {
            "id": idmap[str(nid)],
            "block_type": block_type,
            "config": data.get("config") or {},
        }
        if inputs:
            step["inputs"] = inputs
        steps.append(step)
    return steps


def _saved_positions(pipeline):
    """Map step_id -> (pos_x, pos_y) from a previously-saved Drawflow graph."""
    nodes = (((pipeline.graph or {}).get("drawflow") or {}).get("Home") or {}).get("data") or {}
    pos = {}
    for node in nodes.values():
        sid = (node.get("data") or {}).get("step_id")
        if sid:
            pos[sid] = (node.get("pos_x", 0), node.get("pos_y", 0))
    return pos


def build_initial_steps(pipeline):
    """Describe a pipeline's steps as canvas nodes+edges for first render.

    Each item: {id, block_type, config, num_in, num_out, edges:[{from, port_index}],
    pos_x, pos_y}. The canvas is always rebuilt from ``steps`` (the source of truth,
    always carrying current config); the saved graph only supplies node positions.
    Edges come from each step's ``inputs``; a legacy step with no ``inputs`` links to
    the previous step (the old linear default) so pre-canvas pipelines still show.
    """
    steps = pipeline.steps or []
    ids = {s.get("id") for s in steps}
    positions = _saved_positions(pipeline)
    out = []
    for i, step in enumerate(steps):
        block_type = step.get("block_type")
        ports = get_input_ports(block_type)
        step_inputs = step.get("inputs") or {}
        edges = []
        for k, port in enumerate(ports):
            src = step_inputs.get(port["name"])
            if not src and not step_inputs and k == 0 and i > 0:
                src = steps[i - 1].get("id")
            if src in ids:
                edges.append({"from": src, "port_index": k + 1})
        sid = step.get("id")
        px, py = positions.get(sid, (60 + i * 250, 80 + (i % 4) * 160))
        out.append({
            "id": sid,
            "block_type": block_type,
            "config": step.get("config") or {},
            "num_in": len(ports),
            "num_out": num_output_ports(block_type),
            "edges": edges,
            "pos_x": px,
            "pos_y": py,
        })
    return out
