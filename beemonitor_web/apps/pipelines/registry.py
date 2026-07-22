"""
Block registry for the BeeMonitor visual pipeline builder (Phase 1).

Ported from the pan-APA Workshop builder and re-themed for ecological video/image
analysis. Each block declares its display metadata, its typed input/output *ports*
(the artifact type that flows on an edge), the backend that runs it, and the config
fields the UI renders.

The block schema is intentionally the same shape as the Workshop's ``BLOCK_REGISTRY``
so the editor UI + engine dispatch carry over unchanged. Two additions for BeeMonitor:

* a richer **artifact-type** vocabulary (Video / ROI / Detections / Tracks / Events /
  Observations / Table) alongside the scalar types, and
* a ``backend`` value of ``"gpu"`` for steps that run as async SageMaker ``analysis.Job``s
  (vs ``"local"`` steps that run inline in the request/advance cycle).

See ``memory/23_pipeline_builder_port_design.md`` for the design.
"""

# ── Artifact types (edge "port" types) ────────────────────────────────────────
# Ordered roughly by how data flows through an ecological pipeline. ``any`` matches
# every type; ``none`` means the port is absent (source/sink blocks).
ARTIFACT_TYPES = {
    "none":         "Nothing",
    "video":        "Video",
    "frames":       "Image set / frames",
    "roi":          "Region(s) of interest",
    "detections":   "Per-frame detections",
    "tracks":       "Tracks (trajectories)",
    "events":       "Events (foraging trips / interactions)",
    "observations": "Observations (taxa + counts)",
    "table":        "Table (rows for export / plot)",
    # scalar types carried over from the Workshop builder
    "text":         "Text",
    "image":        "Image",
    "data":         "Structured data",
    "any":          "Any",
}


def types_compatible(upstream_out, downstream_in):
    """Return True if an edge from ``upstream_out`` into ``downstream_in`` is legal.

    ``any`` on either side matches everything; otherwise the types must be equal.
    ``none`` only matches ``none``/``any`` (a source feeding a sink is nonsensical
    but harmless). This is the enforcement the Workshop builder lacked.
    """
    if upstream_out == "any" or downstream_in == "any":
        return True
    return upstream_out == downstream_in


def accepted_types(block_type):
    """Every artifact type a block's primary input will take.

    Most blocks accept exactly their ``input_type``. Analyzers declare an
    ``accepts`` list instead, because the same node reads either tracks (from the
    MOT module) or raw detections (straight from the Detector) — the branch the
    design doc calls for. ``input_type`` stays scalar so the editor's type chip
    (``t-<type>``) and port labels keep rendering one canonical type.
    """
    block = BLOCK_REGISTRY.get(block_type, {})
    return block.get("accepts") or [block.get("input_type", "none")]


BLOCK_REGISTRY = {
    # ── Input ─────────────────────────────────────────────────────────────────
    "input.video": {
        "display_name": "Video Input",
        "description": "The video the pipeline runs on. The specific clip is chosen "
                       "at run time (Processing → Run on videos), not here — a "
                       "pipeline is a reusable template.",
        "category": "input",
        "icon": "🎞️",
        "input_type": "none",
        "output_type": "video",
        "backend": "local",
        # No config — the video is injected per run (engine.steps_with_video).
        "config_fields": [],
    },
    "input.image_set": {
        "display_name": "Image Set",
        "description": "A set of images / frames (e.g. motion crops) to analyse.",
        "category": "input",
        "icon": "🖼️",
        "input_type": "none",
        "output_type": "frames",
        "backend": "local",
        "config_fields": [
            {
                "name": "source",
                "label": "Source",
                "field_type": "select",
                "required": True,
                "default": "device_crops",
                "choices": [
                    {"value": "device_crops", "label": "Device motion crops"},
                    {"value": "upload", "label": "Uploaded images"},
                ],
            },
        ],
    },

    # ── Region of interest ────────────────────────────────────────────────────
    "roi.nest_layout": {
        "display_name": "Use Bee-Hotel ROI + Nest Tubes",
        "description": "Use the device's saved ROI and nest-tube layout drawn in the ROI editor.",
        "category": "roi",
        "icon": "📐",
        "input_type": "video",
        "output_type": "roi",
        "backend": "local",
        "config_fields": [
            {
                "name": "source",
                "label": "Layout source",
                "field_type": "select",
                "required": True,
                "default": "device",
                "choices": [
                    {"value": "device", "label": "The video's device layout"},
                ],
            },
        ],
    },
    "roi.draw": {
        "display_name": "Draw Region(s)",
        "description": "Draw one or more regions of interest (e.g. a flower, a nest entrance).",
        "category": "roi",
        "icon": "✏️",
        "input_type": "video",
        "output_type": "roi",
        "backend": "local",
        "config_fields": [
            {
                "name": "regions",
                "label": "Regions (JSON)",
                "field_type": "textarea",
                "required": False,
                "default": "[]",
                "choices": None,
            },
        ],
    },

    # ── Detect ────────────────────────────────────────────────────────────────
    "detect.nest": {
        "display_name": "Detect Nest / Hotel",
        "description": "Detect nest holes / bee-hotel tubes and cluster them into a layout.",
        "category": "detect",
        "icon": "🏨",
        "input_type": "video",
        "output_type": "roi",
        "backend": "gpu",
        "config_fields": [
            {
                "name": "confidence",
                "label": "Confidence Threshold",
                "field_type": "number",
                "required": False,
                "default": 0.5,
                "choices": None,
            },
            {
                "name": "nest_model",
                "label": "Model",
                "field_type": "nest_model",  # UI renders the custom-model picker (populated at render)
                "required": False,
                "default": "",
                "choices": None,
            },
        ],
    },
    "detect.bee": {
        "display_name": "Detect Objects",
        "description": "Detect objects in each frame. YOLO = fast, bee/wasp/nest "
                       "classes. SAM 3 = open-vocabulary, text-prompt any organism "
                       "(much slower). Emits per-frame detections.",
        "category": "detect",
        "icon": "🐝",
        "input_type": "video",
        "output_type": "detections",
        "backend": "gpu",
        "config_fields": [
            {
                "name": "confidence",
                "label": "Confidence Threshold",
                "field_type": "number",
                "required": False,
                "default": 0.4,
                "choices": None,
            },
            {
                # YOLO = fast, trained classes. SAM 3 = open-vocabulary text prompt.
                "name": "detector",
                "label": "Detector",
                "field_type": "select",
                "required": False,
                "default": "yolo",
                "choices": [
                    {"value": "yolo", "label": "YOLO (fast)"},
                    {"value": "sam3", "label": "SAM 3 (text prompt, slow)"},
                ],
            },
            {
                # Grounding prompt used when detector == sam3. Only shown for SAM 3.
                "name": "text_prompt",
                "label": "SAM 3 prompt",
                "field_type": "text",
                "required": False,
                "default": "bee",
                "choices": None,
                "show_if": {"field": "detector", "value": "sam3"},
            },
            {
                # Only shown when Detector = YOLO.
                "name": "bee_model",
                "label": "Model (YOLO)",
                "field_type": "bee_model",  # UI renders the custom-model picker (populated at render)
                "required": False,
                "default": "",
                "choices": None,
                "show_if": {"field": "detector", "value": "yolo"},
            },
        ],
    },

    # ── Module 1 — Detector ───────────────────────────────────────────────────
    # Finds the organisms to track/count AND the reference objects (bee hotel,
    # nest tubes, a flower) that the analyzers measure activity against. The
    # reference used to be separate roi.* nodes; folding it in here is what makes
    # the three-module abstraction hold.
    "detect.objects": {
        "display_name": "Detector — Objects & Reference",
        "description": "Find the insects to track or count, and the reference "
                       "object activity is measured against (bee hotel / nest "
                       "tubes / a drawn region). YOLO is fast with trained "
                       "classes; SAM 3 takes any text prompt but is much slower. "
                       "Detection and tracking run in one GPU pass, so adding a "
                       "MOT module downstream costs nothing extra.",
        "category": "detect",
        "icon": "🎯",
        "input_type": "video",
        "output_type": "detections",
        "backend": "gpu",
        "config_fields": [
            {
                "name": "model_family",
                "label": "Detector",
                "field_type": "select",
                "required": False,
                "default": "yolo",
                "choices": [
                    {"value": "yolo", "label": "YOLO (fast)"},
                    {"value": "sam3", "label": "SAM 3 (text prompt, slow)"},
                ],
            },
            {
                # Becomes the detection's label -> the tracking CSV's `taxon`.
                "name": "text_prompt",
                "label": "SAM 3 prompt",
                "field_type": "text",
                "required": False,
                "default": "bee",
                "choices": None,
                "show_if": {"field": "model_family", "value": "sam3"},
            },
            {
                "name": "object_model",
                "label": "Object model (YOLO)",
                "field_type": "bee_model",  # custom-model picker, populated at render
                "required": False,
                "default": "",
                "choices": None,
                "show_if": {"field": "model_family", "value": "yolo"},
            },
            {
                "name": "confidence",
                "label": "Confidence Threshold",
                "field_type": "number",
                "required": False,
                "default": 0.4,
                "choices": None,
            },
            {
                # Replaces the old roi.nest_layout / roi.draw / detect.nest nodes.
                "name": "reference_source",
                "label": "Reference object",
                "field_type": "select",
                "required": False,
                "default": "device_layout",
                "choices": [
                    {"value": "device_layout", "label": "The device's saved hotel + nest tubes"},
                    {"value": "detect", "label": "Detect the nest / hotel in the video"},
                    {"value": "drawn", "label": "Region(s) I draw"},
                    {"value": "none", "label": "None"},
                ],
            },
            {
                "name": "regions",
                "label": "Regions (JSON)",
                "field_type": "textarea",
                "required": False,
                "default": "[]",
                "choices": None,
                "show_if": {"field": "reference_source", "value": "drawn"},
            },
            {
                "name": "reference_model",
                "label": "Nest / hotel model",
                "field_type": "nest_model",
                "required": False,
                "default": "",
                "choices": None,
                "show_if": {"field": "reference_source", "value": "detect"},
            },
            {
                # The only way to reach the cheap nest-only GPU path. Explicit,
                # because that path produces NO detections or tracks at all —
                # inferring it from the graph would silently yield empty results.
                "name": "run_scope",
                "label": "Run scope",
                "field_type": "select",
                "required": False,
                "default": "full",
                "choices": [
                    {"value": "full", "label": "Objects + reference"},
                    {"value": "reference_only", "label": "Reference only (fast, no objects)"},
                ],
            },
            {
                # Rendering the overlay roughly doubles runtime; off by default.
                "name": "annotated_video",
                "label": "Annotated video",
                "field_type": "select",
                "required": False,
                "default": "off",
                "choices": [
                    {"value": "off", "label": "Off (faster)"},
                    {"value": "on", "label": "On (render overlay video)"},
                ],
            },
        ],
    },

    # ── Module 2 — MOT ────────────────────────────────────────────────────────
    "track.mot": {
        "display_name": "MOT — Track Objects",
        "description": "Multi-object tracking: associate per-frame detections "
                       "into trajectories. Runs inside the Detector's GPU pass — "
                       "this module picks the algorithm and hands tracks to the "
                       "analyzers.",
        "category": "track",
        "icon": "🛰️",
        "input_type": "detections",
        "output_type": "tracks",
        "backend": "local",
        "config_fields": [
            {
                "name": "tracker",
                "label": "Tracking algorithm",
                "field_type": "select",
                "required": False,
                "default": "beetrack",
                "choices": [
                    {"value": "beetrack", "label": "BeeTrack (default)"},
                ],
            },
        ],
    },

    # ── Track (legacy) ────────────────────────────────────────────────────────
    "track.bee": {
        "display_name": "Track Objects (MOT)",
        "description": "Multi-object tracking — turn per-frame detections into trajectories. "
                       "Detect with YOLO (fast) or SAM 3 (type any prompt: bee, hoverfly, beetle).",
        "category": "track",
        "icon": "🛰️",
        "input_type": "video",
        "output_type": "tracks",
        "backend": "gpu",
        "config_fields": [
            {
                "name": "confidence",
                "label": "Detection Confidence",
                "field_type": "number",
                "required": False,
                "default": 0.4,
                "choices": None,
            },
            {
                # YOLO = fast, bee/wasp/nest classes. SAM 3 = open-vocabulary,
                # text-prompt any organism; much slower (per-frame transformer).
                "name": "detector",
                "label": "Detector",
                "field_type": "select",
                "required": False,
                "default": "yolo",
                "choices": [
                    {"value": "yolo", "label": "YOLO (fast)"},
                    {"value": "sam3", "label": "SAM 3 (text prompt, slow)"},
                ],
            },
            {
                # Grounding prompt used when detector == sam3 (e.g. "bee",
                # "hoverfly", "beetle"). Becomes the track's taxon label.
                # Only shown when Detector = SAM 3.
                "name": "text_prompt",
                "label": "SAM 3 prompt",
                "field_type": "text",
                "required": False,
                "default": "bee",
                "choices": None,
                "show_if": {"field": "detector", "value": "sam3"},
            },
            {
                # Only shown when Detector = YOLO.
                "name": "bee_model",
                "label": "Model (YOLO)",
                "field_type": "bee_model",  # UI renders the custom-model picker (populated at render)
                "required": False,
                "default": "",
                "choices": None,
                "show_if": {"field": "detector", "value": "yolo"},
            },
            {
                # Rendering the annotated video roughly doubles runtime and can
                # time out long clips; off by default. CSVs/trips don't need it.
                "name": "annotated_video",
                "label": "Annotated video",
                "field_type": "select",
                "required": False,
                "default": "off",
                "choices": [
                    {"value": "off", "label": "Off (faster)"},
                    {"value": "on", "label": "On (render overlay video)"},
                ],
            },
        ],
    },

    # ── Analyze ───────────────────────────────────────────────────────────────
    "analyze.foraging_trips": {
        "display_name": "Foraging Trips",
        "description": "Derive foraging-trip events (Exit→Entry) from tracks + the "
                       "nest/hotel layout. Event Confidence tunes the Entry/Exit "
                       "classifier applied during tracking.",
        "category": "analyze",
        "icon": "🌻",
        "input_type": "tracks",
        "accepts": ["tracks", "detections"],
        "output_type": "events",
        "backend": "local",
        "config_fields": [
            {
                # Event-classifier cutoff for Entry/Exit events. 0.6 = best F1;
                # lower (0.3-0.4) keeps more real events at some noise cost. Read
                # by the upstream Track step (events are computed during tracking).
                "name": "event_confidence",
                "label": "Event Confidence",
                "field_type": "number",
                "required": False,
                "default": 0.6,
                "choices": None,
            },
        ],
    },
    "analyze.visitation": {
        "display_name": "Visitation Count",
        "description": "Count unique tracks visiting the reference object (a "
                       "flower, a nest tube, anything the Detector found), plus "
                       "how long each stayed.",
        "category": "analyze",
        "icon": "🔢",
        "input_type": "tracks",
        "accepts": ["tracks", "detections"],
        "output_type": "table",
        "backend": "local",
        "config_fields": [],
    },
    "analyze.interaction": {
        "display_name": "Interactions",
        "description": "Proximity interactions — insect-to-insect, and "
                       "insect-to-reference (e.g. a bee at a nest tube) — with "
                       "their durations.",
        "category": "analyze",
        "icon": "🤝",
        "input_type": "tracks",
        "accepts": ["tracks", "detections"],
        "output_type": "table",
        "backend": "local",
        "config_fields": [
            {
                "name": "interaction_type",
                "label": "Interaction type",
                "field_type": "select",
                "required": False,
                "default": "all",
                "choices": [
                    {"value": "all", "label": "All"},
                    {"value": "organism_organism", "label": "Insect ↔ insect"},
                    {"value": "organism_reference", "label": "Insect ↔ reference"},
                ],
            },
        ],
    },
    "analyze.detection_count": {
        "display_name": "Detection Count",
        "description": "Count detections rather than trips or visits — total, "
                       "per frame, or binned over time. Wire it straight to the "
                       "Detector when you only need 'how much was there', not "
                       "who went where.",
        "category": "analyze",
        "icon": "#️⃣",
        "input_type": "detections",
        "accepts": ["detections", "tracks"],
        "output_type": "table",
        "backend": "local",
        "config_fields": [
            {
                "name": "metric",
                "label": "Metric",
                "field_type": "select",
                "required": True,
                "default": "total",
                "choices": [
                    {"value": "total", "label": "Totals"},
                    {"value": "per_frame", "label": "Per frame"},
                    {"value": "over_time", "label": "Over time"},
                ],
            },
            {
                "name": "bin_seconds",
                "label": "Bin size (seconds)",
                "field_type": "number",
                "required": False,
                "default": 5,
                "choices": None,
                "show_if": {"field": "metric", "value": "over_time"},
            },
        ],
    },
    "analyze.colony_activity": {
        "display_name": "Colony Activity",
        "description": "Occupancy / motion-over-time series from tracks (optionally within an ROI).",
        "category": "analyze",
        "icon": "🏗️",
        "input_type": "tracks",
        "output_type": "table",
        "backend": "local",
        "config_fields": [
            {
                "name": "metric",
                "label": "Metric",
                "field_type": "select",
                "required": True,
                "default": "occupancy",
                "choices": [
                    {"value": "occupancy", "label": "Occupancy over time"},
                    {"value": "motion", "label": "Detections over time"},
                ],
            },
        ],
    },

    # ── Identify ──────────────────────────────────────────────────────────────
    "identify.marker": {
        "display_name": "Read Bee Marker (QR / Colour)",
        "description": "Read which individual each track is, from its paint "
                       "mark. Decoded from the per-track crops the tracking run "
                       "already saved, voting across a track's crops — so it "
                       "also works on videos you analysed earlier. Printed tag "
                       "(ArUco / QR) decoding is not implemented yet.",
        "category": "identify",
        "icon": "🏷️",
        "input_type": "tracks",
        "accepts": ["tracks", "detections"],
        "output_type": "table",
        "backend": "local",
        "config_fields": [
            {
                "name": "marker_type",
                "label": "Marker Type",
                "field_type": "select",
                "required": True,
                "default": "auto",
                "choices": [
                    {"value": "auto", "label": "Auto (try every decoder)"},
                    {"value": "color", "label": "Colour paint mark"},
                    # No decoder behind these two yet — labelled so choosing one
                    # is an informed choice, not a silent no-op.
                    {"value": "number", "label": "Number / ArUco tag (not yet available)"},
                    {"value": "qr", "label": "QR / data-matrix tag (not yet available)"},
                ],
            },
        ],
    },

    # ── Filter ────────────────────────────────────────────────────────────────
    "filter.roi": {
        "display_name": "Filter by ROI",
        "description": "Keep only tracks/detections inside the region(s) of interest.",
        "category": "filter",
        "icon": "🧲",
        "input_type": "tracks",
        "output_type": "tracks",
        "backend": "local",
        "config_fields": [],
    },
    "filter.confidence": {
        "display_name": "Filter by Confidence",
        "description": "Drop detections/tracks below a confidence threshold.",
        "category": "filter",
        "icon": "🎚️",
        "input_type": "any",
        "output_type": "any",
        "backend": "local",
        "config_fields": [
            {
                "name": "min_confidence",
                "label": "Minimum Confidence",
                "field_type": "number",
                "required": True,
                "default": 0.5,
                "choices": None,
            },
        ],
    },
    "filter.taxon": {
        "display_name": "Filter by Taxon",
        "description": "Keep only observations of a given taxon / rank.",
        "category": "filter",
        "icon": "🧬",
        "input_type": "observations",
        "output_type": "observations",
        "backend": "local",
        "config_fields": [
            {
                "name": "taxon",
                "label": "Taxon name",
                "field_type": "text",
                "required": False,
                "default": "",
                "choices": None,
            },
        ],
    },
    "filter.time": {
        "display_name": "Filter by Time Window",
        "description": "Restrict to a time-of-day / date window.",
        "category": "filter",
        "icon": "⏱️",
        "input_type": "any",
        "output_type": "any",
        "backend": "local",
        "config_fields": [
            {
                "name": "start",
                "label": "Start (HH:MM or ISO)",
                "field_type": "text",
                "required": False,
                "default": "",
                "choices": None,
            },
            {
                "name": "end",
                "label": "End (HH:MM or ISO)",
                "field_type": "text",
                "required": False,
                "default": "",
                "choices": None,
            },
        ],
    },

    # ── Output ────────────────────────────────────────────────────────────────
    "output.table": {
        "display_name": "Table / CSV",
        "description": "Render the result as a table and offer a CSV download.",
        "category": "output",
        "icon": "📄",
        "input_type": "any",
        "output_type": "none",
        "backend": "local",
        "config_fields": [],
    },
    "output.chart": {
        "display_name": "Chart",
        "description": "Visualise data as a bar, line, or pie chart.",
        "category": "output",
        "icon": "📈",
        "input_type": "table",
        "output_type": "none",
        "backend": "local",
        "status": "beta",  # passes the table through; no rendered chart yet
        "config_fields": [
            {
                "name": "chart_type",
                "label": "Chart Type",
                "field_type": "select",
                "required": True,
                "default": "bar",
                "choices": [
                    {"value": "bar", "label": "Bar Chart"},
                    {"value": "line", "label": "Line Chart"},
                    {"value": "pie", "label": "Pie Chart"},
                ],
            },
        ],
    },
    "output.summary": {
        "display_name": "Ecological Summary",
        "description": "Generate a natural-language summary of the results.",
        "category": "output",
        "icon": "📝",
        "input_type": "any",
        "output_type": "none",
        "backend": "local",
        "status": "beta",  # passes upstream through; no generated summary yet
        "config_fields": [],
    },
    "output.dataset": {
        "display_name": "Export Dataset",
        "description": "Export crops + labels as a training dataset (YOLO / classification).",
        "category": "output",
        "icon": "📦",
        "input_type": "any",
        "output_type": "none",
        "backend": "local",
        "status": "beta",  # passes upstream through; no dataset written yet
        "config_fields": [
            {
                "name": "format",
                "label": "Format",
                "field_type": "select",
                "required": True,
                "default": "yolo",
                "choices": [
                    {"value": "yolo", "label": "YOLO detection"},
                    {"value": "classification", "label": "Classification (folders)"},
                ],
            },
        ],
    },
}


# ── Legacy blocks (superseded by the three-module refactor, 2026-07) ──────────
# The builder collapsed to Video → Detector → MOT → Analyzer (+ Identity), which
# absorbed these: the roi.* nodes became the Detector's ``reference_source``,
# detect.bee/track.bee became detect.objects + track.mot, and the filter/output
# nodes were never more than pass-throughs (see the executors).
#
# They stay in BLOCK_REGISTRY on purpose. ``get_block`` — which the engine, the
# executors and the canvas rebuild all go through — keeps resolving them, so every
# pipeline saved before the refactor still validates, opens with its edges intact
# and runs. They are simply absent from the palette, so no *new* pipeline can be
# built on them. Do not delete these entries without a data migration.
_LEGACY_BLOCKS = (
    "input.image_set",
    "roi.nest_layout", "roi.draw",
    "detect.nest", "detect.bee", "track.bee",
    "analyze.colony_activity",
    "filter.roi", "filter.confidence", "filter.taxon", "filter.time",
    "output.table", "output.chart", "output.summary", "output.dataset",
)

for _legacy_type in _LEGACY_BLOCKS:
    BLOCK_REGISTRY[_legacy_type]["hidden"] = True


CATEGORY_META = {
    "input":    {"name": "Input",           "slug": "input",    "icon": "📥", "order": 0},
    "roi":      {"name": "Region",          "slug": "roi",      "icon": "📐", "order": 1},
    "detect":   {"name": "1 · Detect",      "slug": "detect",   "icon": "🎯", "order": 2},
    "track":    {"name": "2 · Track (MOT)", "slug": "track",    "icon": "🛰️", "order": 3},
    "analyze":  {"name": "3 · Analyze",     "slug": "analyze",  "icon": "📊", "order": 4},
    "identify": {"name": "Identity",        "slug": "identify", "icon": "🔬", "order": 5},
    "filter":   {"name": "Filter",    "slug": "filter",   "icon": "🧲", "order": 6},
    "output":   {"name": "Output",    "slug": "output",   "icon": "📤", "order": 7},
}


def get_categories(include_hidden=False):
    """Return an ordered list of categories, each with its blocks (for the palette).

    Blocks marked ``hidden`` are legacy: they still execute and still render on the
    canvas (``serialize_blocks``/``get_block`` keep returning them, so pipelines
    built before the module refactor open and run unchanged), but they are not
    offered for new work. A category whose blocks are all hidden is dropped
    entirely rather than rendering an empty accordion.
    """
    cats = {}
    for block_type, block in BLOCK_REGISTRY.items():
        if block.get("hidden") and not include_hidden:
            continue
        cat_key = block["category"]
        if cat_key not in cats:
            meta = CATEGORY_META.get(
                cat_key, {"name": cat_key.title(), "slug": cat_key, "icon": "📦", "order": 99}
            )
            cats[cat_key] = {**meta, "blocks": []}
        cats[cat_key]["blocks"].append({
            "type": block_type,
            "name": block["display_name"],
            "icon": block["icon"],
            "description": block["description"],
            **block,
        })
    return sorted(
        (c for c in cats.values() if c["blocks"]),
        key=lambda c: c.get("order", 99),
    )


def get_block(block_type):
    """Return a block definition or None."""
    return BLOCK_REGISTRY.get(block_type)


# ── Named ports (Phase 2 — the DAG canvas) ────────────────────────────────────
# Most blocks have a single input port named after their input_type; a few take
# several typed inputs (e.g. a video AND its ROI). Port *order* is stable and maps
# to Drawflow's input_1, input_2, … The executors that read specific ports
# (foraging/visitation/colony read ``tracks``) rely on these names.
_MULTI_INPUT_PORTS = {
    "detect.objects":          [{"name": "video", "type": "video"}],
    "track.mot":               [{"name": "detections", "type": "detections"}],
    "analyze.interaction":     [{"name": "tracks", "type": "tracks",
                                 "accepts": ["tracks", "detections"]}],
    "analyze.detection_count": [{"name": "detections", "type": "detections",
                                 "accepts": ["detections", "tracks"]}],
    "identify.marker":         [{"name": "tracks", "type": "tracks",
                                 "accepts": ["tracks", "detections"]}],
    "detect.bee":              [{"name": "video", "type": "video"}],
    "detect.nest":             [{"name": "video", "type": "video"}],
    "track.bee":               [{"name": "video", "type": "video"},
                                {"name": "rois", "type": "roi", "optional": True}],
    "analyze.foraging_trips":  [{"name": "tracks", "type": "tracks"},
                                {"name": "rois", "type": "roi", "optional": True}],
    "analyze.visitation":      [{"name": "tracks", "type": "tracks"},
                                {"name": "rois", "type": "roi", "optional": True}],
    "analyze.colony_activity": [{"name": "tracks", "type": "tracks"},
                                {"name": "rois", "type": "roi", "optional": True}],
}


def get_input_ports(block_type):
    """Return the ordered list of input ports [{name, type, optional?}] for a block."""
    if block_type in _MULTI_INPUT_PORTS:
        return _MULTI_INPUT_PORTS[block_type]
    block = BLOCK_REGISTRY.get(block_type, {})
    in_type = block.get("input_type", "none")
    if in_type == "none":
        return []
    return [{"name": "in", "type": in_type}]


def num_output_ports(block_type):
    """1 if the block emits an artifact, else 0."""
    block = BLOCK_REGISTRY.get(block_type, {})
    return 0 if block.get("output_type", "none") == "none" else 1


def serialize_blocks():
    """A JSON-safe dict of every block for the canvas palette + node rendering."""
    out = {}
    for block_type, block in BLOCK_REGISTRY.items():
        out[block_type] = {
            "type": block_type,
            "display_name": block["display_name"],
            "description": block["description"],
            "category": block["category"],
            "icon": block["icon"],
            "input_type": block["input_type"],
            "output_type": block["output_type"],
            "accepts": accepted_types(block_type),
            "input_ports": get_input_ports(block_type),
            "num_out": num_output_ports(block_type),
            "config_fields": block.get("config_fields", []),
            # Legacy block: still renders + runs on the canvas, but is not in the
            # palette. The editor tags these nodes so it's clear why.
            "hidden": bool(block.get("hidden")),
            # "beta" = the block runs but doesn't yet produce a full deliverable
            # (surfaced as a palette badge so "valid graph" != "graph that does
            # everything" is visible).
            "status": block.get("status", "ready"),
        }
    return out


def get_blocks_by_category(category, include_hidden=True):
    """Return all blocks in a given category (legacy blocks included by default)."""
    return {
        bt: block for bt, block in BLOCK_REGISTRY.items()
        if block["category"] == category
        and (include_hidden or not block.get("hidden"))
    }


def validate_steps(steps):
    """Validate a list of pipeline steps; return a list of human-readable errors.

    Enforces (a) known block types, (b) required config present, and (c) port-type
    compatibility along each declared edge. A step's upstream is either the id(s) in
    its ``inputs`` map or, absent that, the previous step (linear default).
    """
    errors = []
    by_id = {s.get("id"): s for s in steps if s.get("id")}

    for i, step in enumerate(steps):
        block_type = step.get("block_type", "")
        block = BLOCK_REGISTRY.get(block_type)
        if not block:
            errors.append(f"Step {i + 1}: unknown block '{block_type}'.")
            continue

        # Required config fields
        config = step.get("config", {})
        for field in block.get("config_fields", []):
            if field.get("required") and config.get(field["name"], "") in ("", None):
                errors.append(
                    f"Step {i + 1} ({block['display_name']}): '{field['label']}' is required."
                )

        # Port-type compatibility
        in_type = block.get("input_type", "none")
        if in_type == "none":
            continue
        # A block may accept several artifact types (analyzers take tracks OR
        # detections); in_type is only the canonical one used for display.
        in_types = accepted_types(block_type)

        def _accepts(up_out):
            return any(types_compatible(up_out, t) for t in in_types)

        upstreams = step.get("inputs")
        if upstreams:
            # Multi-input: a step may pull several ports (e.g. a video AND its ROI).
            # Require every referenced step to exist and at least one port to satisfy
            # the block's declared input type.
            matched = False
            for port, up_id in upstreams.items():
                up = by_id.get(up_id)
                if not up:
                    errors.append(f"Step {i + 1}: input '{port}' points to a missing step.")
                    continue
                up_block = BLOCK_REGISTRY.get(up.get("block_type", ""), {})
                if _accepts(up_block.get("output_type", "none")):
                    matched = True
            if upstreams and not matched:
                errors.append(
                    f"Step {i + 1} ({block['display_name']}): needs a "
                    f"'{' or '.join(in_types)}' input but none of its connected "
                    f"steps produce one."
                )
        elif i == 0:
            errors.append(
                f"Step {i + 1} ({block['display_name']}): needs an input but is first."
            )
        else:
            prev = steps[i - 1]
            prev_block = BLOCK_REGISTRY.get(prev.get("block_type", ""), {})
            prev_out = prev_block.get("output_type", "none")
            if not _accepts(prev_out):
                errors.append(
                    f"Step {i + 1} ({block['display_name']}): expects "
                    f"'{' or '.join(in_types)}' but the previous step produces "
                    f"'{prev_out}'."
                )
    return errors
