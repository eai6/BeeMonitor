"""Drawflow canvas ⇄ execution-steps conversion.

``Pipeline.steps`` is the source of truth and the canvas is rebuilt from it, so
the round-trip has to be lossless for legacy pipelines too — an edge dropped here
is a pipeline that silently opens disconnected.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.pipelines.graph import build_initial_steps, graph_to_steps
from apps.pipelines.models import Pipeline

User = get_user_model()


def _graph(nodes):
    """Wrap {node_id: (step_id, block_type, config, {port_index: src_node_id})}."""
    data = {}
    for nid, (step_id, block_type, config, edges) in nodes.items():
        data[str(nid)] = {
            "id": nid,
            "data": {"step_id": step_id, "block_type": block_type, "config": config},
            "inputs": {
                f"input_{port}": {"connections": [{"node": str(src), "input": "output_1"}]}
                for port, src in edges.items()
            },
            "outputs": {},
            "pos_x": 10 * nid,
            "pos_y": 20 * nid,
        }
    return {"drawflow": {"Home": {"data": data}}}


class GraphToStepsTests(TestCase):
    def test_named_ports_land_on_the_inputs_map(self):
        graph = _graph({
            1: ("v", "input.video", {}, {}),
            2: ("d", "detect.objects", {"reference_source": "none"}, {1: 1}),
            3: ("m", "track.mot", {"tracker": "beetrack"}, {1: 2}),
            4: ("f", "analyze.foraging_trips", {}, {1: 3}),
        })

        steps = graph_to_steps(graph)

        self.assertEqual([s["id"] for s in steps], ["v", "d", "m", "f"])
        self.assertEqual(steps[1]["inputs"], {"video": "v"})
        self.assertEqual(steps[2]["inputs"], {"detections": "d"})
        self.assertEqual(steps[3]["inputs"], {"tracks": "m"})

    def test_legacy_multi_port_node_keeps_both_edges(self):
        graph = _graph({
            1: ("v", "input.video", {}, {}),
            2: ("r", "roi.nest_layout", {"source": "device"}, {1: 1}),
            3: ("t", "track.bee", {}, {1: 1, 2: 2}),
        })

        steps = graph_to_steps(graph)

        self.assertEqual(steps[2]["inputs"], {"video": "v", "rois": "r"})

    def test_unknown_block_types_are_dropped(self):
        graph = _graph({
            1: ("v", "input.video", {}, {}),
            2: ("x", "nope.nope", {}, {1: 1}),
        })

        self.assertEqual([s["block_type"] for s in graph_to_steps(graph)], ["input.video"])


class BuildInitialStepsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")

    def _pipeline(self, steps, graph=None):
        return Pipeline.objects.create(
            user=self.user, title="P", steps=steps, graph=graph or {},
        )

    def test_module_pipeline_round_trips(self):
        steps = [
            {"id": "v", "block_type": "input.video", "config": {}},
            {"id": "d", "block_type": "detect.objects", "config": {},
             "inputs": {"video": "v"}},
            {"id": "m", "block_type": "track.mot", "config": {},
             "inputs": {"detections": "d"}},
        ]
        nodes = build_initial_steps(self._pipeline(steps))

        self.assertEqual(nodes[1]["edges"], [{"from": "v", "port_index": 1}])
        self.assertEqual(nodes[2]["edges"], [{"from": "d", "port_index": 1}])

    def test_legacy_rois_port_still_produces_an_edge(self):
        """Regression guard: renaming a port on a pre-existing block would make
        every saved pipeline open with that edge missing, silently."""
        steps = [
            {"id": "v", "block_type": "input.video", "config": {}},
            {"id": "r", "block_type": "roi.nest_layout", "config": {},
             "inputs": {"in": "v"}},
            {"id": "t", "block_type": "track.bee", "config": {},
             "inputs": {"video": "v", "rois": "r"}},
        ]
        nodes = build_initial_steps(self._pipeline(steps))

        self.assertEqual(nodes[2]["edges"], [
            {"from": "v", "port_index": 1},
            {"from": "r", "port_index": 2},
        ])

    def test_pre_canvas_pipeline_falls_back_to_the_linear_chain(self):
        steps = [
            {"id": "v", "block_type": "input.video", "config": {}},
            {"id": "t", "block_type": "track.bee", "config": {}},
        ]
        nodes = build_initial_steps(self._pipeline(steps))

        self.assertEqual(nodes[1]["edges"], [{"from": "v", "port_index": 1}])

    def test_saved_positions_are_reused(self):
        steps = [{"id": "v", "block_type": "input.video", "config": {}}]
        graph = _graph({1: ("v", "input.video", {}, {})})
        graph["drawflow"]["Home"]["data"]["1"]["pos_x"] = 321
        graph["drawflow"]["Home"]["data"]["1"]["pos_y"] = 654

        nodes = build_initial_steps(self._pipeline(steps, graph))

        self.assertEqual((nodes[0]["pos_x"], nodes[0]["pos_y"]), (321, 654))

    def test_full_round_trip_preserves_edges(self):
        steps = [
            {"id": "v", "block_type": "input.video", "config": {}},
            {"id": "d", "block_type": "detect.objects", "config": {},
             "inputs": {"video": "v"}},
            {"id": "m", "block_type": "track.mot", "config": {},
             "inputs": {"detections": "d"}},
            {"id": "g", "block_type": "analyze.visitation", "config": {},
             "inputs": {"tracks": "m"}},
        ]
        nodes = build_initial_steps(self._pipeline(steps))
        rebuilt = graph_to_steps(_graph({
            i + 1: (n["id"], n["block_type"], n["config"],
                    {e["port_index"]: [x["id"] for x in nodes].index(e["from"]) + 1
                     for e in n["edges"]})
            for i, n in enumerate(nodes)
        }))

        self.assertEqual([s["inputs"] for s in rebuilt[1:]],
                         [{"video": "v"}, {"detections": "d"}, {"tracks": "m"}])
