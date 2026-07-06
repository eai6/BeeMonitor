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


BLOCK_REGISTRY = {
    # ── Input ─────────────────────────────────────────────────────────────────
    "input.video": {
        "display_name": "Video Input",
        "description": "Pick an uploaded clip or a device recording to feed the pipeline.",
        "category": "input",
        "icon": "🎞️",
        "input_type": "none",
        "output_type": "video",
        "backend": "local",
        "config_fields": [
            {
                "name": "video_id",
                "label": "Video",
                "field_type": "video",   # UI renders the video picker (populated at render)
                "required": True,
                "default": "",
                "choices": None,
            },
        ],
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
        "display_name": "Detect Bees (YOLO)",
        "description": "Detect bees in each frame with the bee YOLO model.",
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
                "name": "bee_model",
                "label": "Model",
                "field_type": "bee_model",  # UI renders the custom-model picker (populated at render)
                "required": False,
                "default": "",
                "choices": None,
            },
        ],
    },

    # ── Track ─────────────────────────────────────────────────────────────────
    "track.bee": {
        "display_name": "Track Bees (MOT)",
        "description": "Multi-object tracking — turn per-frame detections into trajectories.",
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
                # Event-classifier cutoff for Entry/Exit events. 0.6 = best F1;
                # lower (0.3-0.4) keeps more real events at some noise cost.
                "name": "ml_threshold",
                "label": "Event Confidence",
                "field_type": "number",
                "required": False,
                "default": 0.6,
                "choices": None,
            },
            {
                "name": "bee_model",
                "label": "Model",
                "field_type": "bee_model",  # UI renders the custom-model picker (populated at render)
                "required": False,
                "default": "",
                "choices": None,
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
        "description": "Derive foraging-trip events from tracks + the nest/hotel layout.",
        "category": "analyze",
        "icon": "🌻",
        "input_type": "tracks",
        "output_type": "events",
        "backend": "local",
        "config_fields": [],
    },
    "analyze.visitation": {
        "display_name": "Visitation Count",
        "description": "Count visits (and dwell time) of tracks entering a region of interest.",
        "category": "analyze",
        "icon": "🔢",
        "input_type": "tracks",
        "output_type": "table",
        "backend": "local",
        "config_fields": [],
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
    "identify.taxon": {
        "display_name": "Identify Species (BioCLIP)",
        "description": "Assign taxonomic IDs to crops / tracked individuals with BioCLIP.",
        "category": "identify",
        "icon": "🔬",
        "input_type": "tracks",
        "output_type": "observations",
        "backend": "local",
        "config_fields": [
            {
                "name": "region_prior",
                "label": "Use location prior",
                "field_type": "select",
                "required": False,
                "default": "on",
                "choices": [
                    {"value": "on", "label": "On (constrain to region taxa)"},
                    {"value": "off", "label": "Off (Tree-of-Life)"},
                ],
            },
        ],
    },
    "identify.marker": {
        "display_name": "Read Bee Marker (QR / Colour)",
        "description": "Read per-track individual IDs (colour / QR / number tags) from tracking.",
        "category": "identify",
        "icon": "🏷️",
        "input_type": "tracks",
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
                    {"value": "auto", "label": "Auto (any tag the tracker read)"},
                    {"value": "color", "label": "Colour tag"},
                    {"value": "qr", "label": "QR / data-matrix tag"},
                    {"value": "number", "label": "Number tag"},
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


CATEGORY_META = {
    "input":    {"name": "Input",     "slug": "input",    "icon": "📥", "order": 0},
    "roi":      {"name": "Region",    "slug": "roi",      "icon": "📐", "order": 1},
    "detect":   {"name": "Detect",    "slug": "detect",   "icon": "🎯", "order": 2},
    "track":    {"name": "Track",     "slug": "track",    "icon": "🛰️", "order": 3},
    "analyze":  {"name": "Analyze",   "slug": "analyze",  "icon": "📊", "order": 4},
    "identify": {"name": "Identify",  "slug": "identify", "icon": "🔬", "order": 5},
    "filter":   {"name": "Filter",    "slug": "filter",   "icon": "🧲", "order": 6},
    "output":   {"name": "Output",    "slug": "output",   "icon": "📤", "order": 7},
}


def get_categories():
    """Return an ordered list of categories, each with its blocks (for the palette)."""
    cats = {}
    for block_type, block in BLOCK_REGISTRY.items():
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
    return sorted(cats.values(), key=lambda c: c.get("order", 99))


def get_block(block_type):
    """Return a block definition or None."""
    return BLOCK_REGISTRY.get(block_type)


# ── Named ports (Phase 2 — the DAG canvas) ────────────────────────────────────
# Most blocks have a single input port named after their input_type; a few take
# several typed inputs (e.g. a video AND its ROI). Port *order* is stable and maps
# to Drawflow's input_1, input_2, … The executors that read specific ports
# (foraging/visitation/colony read ``tracks``) rely on these names.
_MULTI_INPUT_PORTS = {
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
            "input_ports": get_input_ports(block_type),
            "num_out": num_output_ports(block_type),
            "config_fields": block.get("config_fields", []),
        }
    return out


def get_blocks_by_category(category):
    """Return all blocks in a given category."""
    return {
        bt: block for bt, block in BLOCK_REGISTRY.items()
        if block["category"] == category
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
                if types_compatible(up_block.get("output_type", "none"), in_type):
                    matched = True
            if upstreams and not matched:
                errors.append(
                    f"Step {i + 1} ({block['display_name']}): needs a '{in_type}' input "
                    f"but none of its connected steps produce one."
                )
        elif i == 0:
            errors.append(
                f"Step {i + 1} ({block['display_name']}): needs an input but is first."
            )
        else:
            prev = steps[i - 1]
            prev_block = BLOCK_REGISTRY.get(prev.get("block_type", ""), {})
            prev_out = prev_block.get("output_type", "none")
            if not types_compatible(prev_out, in_type):
                errors.append(
                    f"Step {i + 1} ({block['display_name']}): expects '{in_type}' but "
                    f"the previous step produces '{prev_out}'."
                )
    return errors
