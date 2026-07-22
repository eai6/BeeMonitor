"""Registry invariants + the hidden-block mechanism.

The registry is the single enumeration of node types — the editor renders from
its JSON, the engine dispatches on it, and the canvas rebuilds saved pipelines
through it. So the contract worth pinning is: the palette shows only the module
blocks, while *every* block (including the legacy ones) stays resolvable, or
pipelines saved before the module refactor stop opening.
"""

from django.test import SimpleTestCase

from apps.pipelines.registry import (
    ARTIFACT_TYPES, BLOCK_REGISTRY, CATEGORY_META, _LEGACY_BLOCKS,
    accepted_types, get_block, get_categories, get_input_ports,
    num_output_ports, serialize_blocks, validate_steps,
)

MODULE_STEPS = [
    {"id": "v", "block_type": "input.video", "config": {}},
    {"id": "d", "block_type": "detect.objects",
     "config": {"reference_source": "device_layout"}, "inputs": {"video": "v"}},
    {"id": "m", "block_type": "track.mot", "config": {"tracker": "beetrack"},
     "inputs": {"detections": "d"}},
    {"id": "f", "block_type": "analyze.foraging_trips",
     "config": {"event_confidence": 0.6}, "inputs": {"tracks": "m"}},
]

LEGACY_STEPS = [
    {"id": "v", "block_type": "input.video", "config": {}},
    {"id": "r", "block_type": "roi.nest_layout", "config": {"source": "device"},
     "inputs": {"in": "v"}},
    {"id": "t", "block_type": "track.bee", "config": {"confidence": 0.4},
     "inputs": {"video": "v", "rois": "r"}},
    {"id": "f", "block_type": "analyze.foraging_trips", "config": {},
     "inputs": {"tracks": "t"}},
    {"id": "o", "block_type": "output.table", "config": {}, "inputs": {"in": "f"}},
]


class BlockSchemaTests(SimpleTestCase):
    def test_every_block_declares_the_required_keys(self):
        required = ("display_name", "description", "category", "icon",
                    "input_type", "output_type", "backend", "config_fields")
        for block_type, block in BLOCK_REGISTRY.items():
            for key in required:
                self.assertIn(key, block, f"{block_type} is missing '{key}'")

    def test_every_declared_type_is_a_known_artifact_type(self):
        for block_type, block in BLOCK_REGISTRY.items():
            for value in (block["input_type"], block["output_type"]):
                self.assertIn(value, ARTIFACT_TYPES, f"{block_type}: {value}")
            for value in accepted_types(block_type):
                self.assertIn(value, ARTIFACT_TYPES, f"{block_type} accepts: {value}")

    def test_every_category_has_metadata(self):
        for block_type, block in BLOCK_REGISTRY.items():
            self.assertIn(block["category"], CATEGORY_META, block_type)

    def test_show_if_always_names_a_sibling_field(self):
        for block_type, block in BLOCK_REGISTRY.items():
            names = {f["name"] for f in block.get("config_fields", [])}
            for field in block.get("config_fields", []):
                show_if = field.get("show_if")
                if show_if:
                    self.assertIn(show_if["field"], names,
                                  f"{block_type}.{field['name']} show_if dangles")

    def test_backend_is_local_or_gpu(self):
        for block_type, block in BLOCK_REGISTRY.items():
            self.assertIn(block["backend"], ("local", "gpu"), block_type)

    def test_ports_agree_with_the_declared_types(self):
        for block_type, block in BLOCK_REGISTRY.items():
            ports = get_input_ports(block_type)
            if block["input_type"] == "none":
                self.assertEqual(ports, [], block_type)
            else:
                self.assertIn(ports[0]["type"], accepted_types(block_type), block_type)
            expected_out = 0 if block["output_type"] == "none" else 1
            self.assertEqual(num_output_ports(block_type), expected_out, block_type)


class HiddenBlockTests(SimpleTestCase):
    def test_palette_is_the_three_modules_plus_input_and_identity(self):
        palette = {b["type"] for c in get_categories() for b in c["blocks"]}
        self.assertEqual(palette, {
            "input.video",
            "detect.objects",
            "track.mot",
            "analyze.foraging_trips", "analyze.visitation",
            "analyze.interaction", "analyze.detection_count",
            "identify.marker",
        })

    def test_no_legacy_block_reaches_the_palette(self):
        palette = {b["type"] for c in get_categories() for b in c["blocks"]}
        self.assertEqual(palette & set(_LEGACY_BLOCKS), set())

    def test_categories_that_went_empty_are_dropped(self):
        slugs = {c["slug"] for c in get_categories()}
        # roi/filter/output are entirely legacy now — no empty accordions.
        self.assertEqual(slugs & {"roi", "filter", "output"}, set())

    def test_include_hidden_brings_the_legacy_blocks_back(self):
        palette = {b["type"] for c in get_categories(include_hidden=True)
                   for b in c["blocks"]}
        self.assertTrue(set(_LEGACY_BLOCKS) <= palette)

    def test_legacy_blocks_stay_resolvable_and_serialized(self):
        """The canvas needs them, or a saved pipeline renders as blank nodes."""
        blocks = serialize_blocks()
        for block_type in _LEGACY_BLOCKS:
            self.assertIsNotNone(get_block(block_type), block_type)
            self.assertIn(block_type, blocks, block_type)
            self.assertTrue(blocks[block_type]["hidden"], block_type)

    def test_module_blocks_are_not_flagged_hidden(self):
        blocks = serialize_blocks()
        for block_type in ("detect.objects", "track.mot", "analyze.interaction",
                           "analyze.detection_count"):
            self.assertFalse(blocks[block_type]["hidden"], block_type)


class ValidateStepsTests(SimpleTestCase):
    def test_module_pipeline_is_clean(self):
        self.assertEqual(validate_steps(MODULE_STEPS), [])

    def test_legacy_pipeline_still_validates(self):
        """The regression that would break every pre-refactor pipeline."""
        self.assertEqual(validate_steps(LEGACY_STEPS), [])

    def test_detector_straight_into_an_analyzer_is_allowed(self):
        steps = [
            {"id": "v", "block_type": "input.video", "config": {}},
            {"id": "d", "block_type": "detect.objects", "config": {},
             "inputs": {"video": "v"}},
            {"id": "c", "block_type": "analyze.detection_count",
             "config": {"metric": "total"}, "inputs": {"detections": "d"}},
        ]
        self.assertEqual(validate_steps(steps), [])

    def test_tracks_into_a_detections_first_analyzer_is_allowed(self):
        steps = MODULE_STEPS[:3] + [
            {"id": "c", "block_type": "analyze.detection_count",
             "config": {"metric": "total"}, "inputs": {"detections": "m"}},
        ]
        self.assertEqual(validate_steps(steps), [])

    def test_type_mismatch_is_rejected(self):
        steps = [
            {"id": "v", "block_type": "input.video", "config": {}},
            {"id": "g", "block_type": "analyze.visitation", "config": {},
             "inputs": {"tracks": "v"}},
        ]
        errors = validate_steps(steps)
        self.assertEqual(len(errors), 1)
        self.assertIn("tracks or detections", errors[0])

    def test_unknown_block_is_rejected(self):
        errors = validate_steps([{"id": "x", "block_type": "nope.nope", "config": {}}])
        self.assertEqual(len(errors), 1)
        self.assertIn("unknown block", errors[0])

    def test_missing_required_config_is_rejected(self):
        steps = MODULE_STEPS[:3] + [
            {"id": "c", "block_type": "analyze.detection_count",
             "config": {"metric": ""}, "inputs": {"detections": "m"}},
        ]
        self.assertTrue(any("Metric" in e for e in validate_steps(steps)))

    def test_input_step_first_is_not_flagged(self):
        self.assertEqual(validate_steps([MODULE_STEPS[0]]), [])
