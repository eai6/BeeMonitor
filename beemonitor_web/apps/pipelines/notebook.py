"""
Export a pipeline to a runnable, explainable Google Colab notebook (Phase 3b).

The notebook is the **education** artifact: for each block it emits a markdown cell
explaining what the step does (plus its configured parameters) and a code cell — real
inline logic for the pandas-based analyze/output steps, and clearly-scaffolded model
calls (YOLO / tracking / BioCLIP) for the compute-heavy steps so a learner can see and
run the whole pipeline outside the platform.

Produces an nbformat v4 dict; the view serialises it as a downloadable ``.ipynb``.
Design: ``memory/23_pipeline_builder_port_design.md`` (Phase 3c → notebook export).
"""

from .registry import get_block


# ── nbformat cell helpers ─────────────────────────────────────────────────────

def _md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source}


def _config_md(step):
    cfg = step.get("config") or {}
    if not cfg:
        return ""
    lines = "\n".join(f"- `{k}` = `{v}`" for k, v in cfg.items() if v not in ("", None))
    return f"\n\n**Parameters**\n{lines}" if lines else ""


# ── Package resolution ────────────────────────────────────────────────────────

def _pip_packages(steps):
    pkgs = ["pandas", "numpy", "matplotlib"]
    types = {s.get("block_type", "") for s in steps}
    if any(t.startswith("detect.") or t.startswith("track.") for t in types):
        pkgs += ["ultralytics", "opencv-python-headless"]
    if "identify.taxon" in types:
        pkgs += ["pybioclip"]
    # de-dup, keep order
    seen, out = set(), []
    for p in pkgs:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


# ── Per-block code generators ─────────────────────────────────────────────────
# Each returns the body of a code cell (a string). Keep them short and honest:
# real logic where it's a few lines, a guided scaffold where a model is involved.

def _gen_input_video(step):
    return ("# Point this at your video (upload via the Files panel, or mount Drive).\n"
            "VIDEO_PATH = 'my_clip.mp4'  # TODO: set to your uploaded file\n"
            "print('Video:', VIDEO_PATH)")


def _gen_input_image_set(step):
    return ("import glob\n"
            "IMAGES = sorted(glob.glob('images/*.jpg'))  # TODO: your image folder\n"
            "print(len(IMAGES), 'images')")


def _gen_roi(step):
    return ("# Regions of interest, normalised 0..1 as [x1, y1, x2, y2].\n"
            "# (In BeeMonitor these come from the device's saved ROI / nest tubes.)\n"
            "ROIS = [[0.1, 0.2, 0.9, 0.8]]  # TODO: your region(s)")


def _gen_detect_bee(step):
    conf = (step.get("config") or {}).get("confidence", 0.4)
    return ("from ultralytics import YOLO\n"
            "bee_model = YOLO('bee_yolo.pt')  # TODO: your trained bee model\n"
            f"results = bee_model.predict(VIDEO_PATH, conf={conf}, stream=True)\n"
            "# Each result carries boxes; collect them into a detections table.\n"
            "rows = []\n"
            "for i, r in enumerate(results):\n"
            "    for b in r.boxes.xyxy.tolist():\n"
            "        x1, y1, x2, y2 = b\n"
            "        rows.append({'frame': i, 'cx': (x1+x2)/2, 'cy': (y1+y2)/2})\n"
            "import pandas as pd; detections = pd.DataFrame(rows); detections.head()")


def _gen_detect_nest(step):
    return ("from ultralytics import YOLO\n"
            "nest_model = YOLO('nest_detection.pt')  # BeeMonitor nest/hotel model\n"
            "# Detect nest holes on a reference frame, then cluster into a tube layout.\n"
            "# See src/beemonitor/detection/nest_detector.py for the clustering logic.")


def _gen_track_bee(step):
    return ("# Multi-object tracking turns per-frame detections into trajectories.\n"
            "# BeeMonitor runs this on GPU (SageMaker); to reproduce locally use\n"
            "# ultralytics' built-in tracker:\n"
            "from ultralytics import YOLO\n"
            "model = YOLO('bee_yolo.pt')\n"
            "rows = []\n"
            "for i, r in enumerate(model.track(VIDEO_PATH, persist=True, stream=True)):\n"
            "    if r.boxes.id is None: continue\n"
            "    for box, tid in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist()):\n"
            "        x1, y1, x2, y2 = box\n"
            "        rows.append({'frame': i, 'track_id': int(tid), 'cx': (x1+x2)/2, 'cy': (y1+y2)/2})\n"
            "import pandas as pd; tracks = pd.DataFrame(rows); tracks.head()")


def _gen_foraging(step):
    return ("# A foraging trip = a bee exits the nest and later re-enters.\n"
            "# BeeMonitor derives these from entry/exit events at each nest tube.\n"
            "# See cloud/wrapper/foraging.py — pairs each Exit with the next Entry.\n"
            "print('foraging trips derived from tracks + nest ROI')")


def _gen_visitation(step):
    return ("# Count visits: a track that enters an ROI is a visit; dwell = frames/fps.\n"
            "def in_box(x, y, boxes):\n"
            "    return any(bx1 <= x <= bx2 and by1 <= y <= by2 for bx1,by1,bx2,by2 in boxes)\n"
            "W = tracks[['cx','cy']].abs().max().max() or 1  # normalise if pixel coords\n"
            "vis = []\n"
            "for tid, g in tracks.sort_values('frame').groupby('track_id'):\n"
            "    inside = [in_box(cx/W, cy/W, ROIS) for cx, cy in zip(g.cx, g.cy)]\n"
            "    if any(inside): vis.append({'track': tid, 'frames_in': sum(inside)})\n"
            "visitation = pd.DataFrame(vis); visitation")


def _gen_colony(step):
    metric = (step.get("config") or {}).get("metric", "occupancy")
    return (f"# Colony activity: {metric} over time, binned per ~5s.\n"
            "FPS = 30; BIN = 5 * FPS\n"
            "tracks['bin'] = (tracks['frame'] // BIN).astype(int)\n"
            "series = tracks.groupby('bin')['track_id'].nunique()  # occupancy\n"
            "series.plot(kind='line', xlabel='time bin', ylabel='bees'); import matplotlib.pyplot as plt; plt.show()")


def _gen_taxon(step):
    return ("# Species ID with BioCLIP (open Tree-of-Life model).\n"
            "from bioclip import TreeOfLifeClassifier\n"
            "clf = TreeOfLifeClassifier()\n"
            "# Classify a crop of each tracked individual:\n"
            "# preds = clf.predict('crop.jpg'); print(preds[:3])")


def _gen_marker(step):
    return ("# Per-track individual ID from colour / QR / number tags.\n"
            "# BeeMonitor's tracker writes bee_id / bee_id_method / bee_id_confidence.\n"
            "if 'bee_id' in tracks.columns:\n"
            "    ids = tracks.dropna(subset=['bee_id']).groupby('track_id')['bee_id'].agg(lambda s: s.mode().iloc[0])\n"
            "    print(ids)\n"
            "else:\n"
            "    print('Enable individual identification to populate bee_id.')")


def _gen_filter(step):
    return ("# Filter step — keep rows matching your criterion (edit as needed).\n"
            "# e.g. filtered = detections[detections['confidence'] >= 0.5]")


def _gen_output_table(step):
    return "# Show / export the result table.\ntry:\n    display(tracks.head())\nexcept NameError:\n    pass"


def _gen_output_chart(step):
    return ("import matplotlib.pyplot as plt\n"
            "# Plot your result table (bar/line/pie as configured).\n"
            "# result.plot(kind='bar'); plt.show()")


def _gen_output_summary(step):
    return "print('Ecological summary — describe your findings here or via an LLM call.')"


def _gen_output_dataset(step):
    return ("# Export crops + labels as a training dataset (YOLO / classification).\n"
            "# See the BeeMonitor 'annotations' app for the YOLO export format.")


BLOCK_CELLS = {
    "input.video": _gen_input_video,
    "input.image_set": _gen_input_image_set,
    "roi.nest_layout": _gen_roi,
    "roi.draw": _gen_roi,
    "detect.bee": _gen_detect_bee,
    "detect.nest": _gen_detect_nest,
    "track.bee": _gen_track_bee,
    "analyze.foraging_trips": _gen_foraging,
    "analyze.visitation": _gen_visitation,
    "analyze.colony_activity": _gen_colony,
    "identify.taxon": _gen_taxon,
    "identify.marker": _gen_marker,
    "filter.roi": _gen_filter,
    "filter.confidence": _gen_filter,
    "filter.taxon": _gen_filter,
    "filter.time": _gen_filter,
    "output.table": _gen_output_table,
    "output.chart": _gen_output_chart,
    "output.summary": _gen_output_summary,
    "output.dataset": _gen_output_dataset,
}


def generate_notebook(pipeline):
    """Build an nbformat v4 notebook dict from a Pipeline."""
    steps = pipeline.steps or []
    cells = [
        _md(f"# {pipeline.title}\n\n{pipeline.description or ''}\n\n"
            f"_Generated from a BeeMonitor pipeline — {len(steps)} step(s)._\n\n"
            "This notebook explains each step and gives runnable starting code. "
            "Model steps (detection / tracking / BioCLIP) point at the relevant "
            "open tools; the analysis steps run as-is on a tracks table."),
        _md("## Setup"),
        _code("!pip -q install " + " ".join(_pip_packages(steps))),
        _code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt"),
    ]

    for i, step in enumerate(steps, start=1):
        block = get_block(step.get("block_type", "")) or {}
        title = block.get("display_name", step.get("block_type", "Step"))
        icon = block.get("icon", "🔧")
        desc = block.get("description", "")
        cells.append(_md(
            f"## Step {i}: {icon} {title}\n\n{desc}"
            f"\n\n_{block.get('input_type','')} → {block.get('output_type','')}_"
            f"{_config_md(step)}"
        ))
        gen = BLOCK_CELLS.get(step.get("block_type", ""))
        cells.append(_code(gen(step) if gen else f"# {title}: see BeeMonitor docs."))

    cells.append(_md("---\n*Built with BeeMonitor. Edit freely — this is your lesson to run.*"))
    return _notebook(cells)


def _notebook(cells):
    """Wrap a list of cells into an nbformat v4 notebook dict."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def generate_api_notebook(pipeline, base_url):
    """A Colab that runs the pipeline for REAL via the public API — no SDK, no
    install (Colab already has ``requests``). It creates the pipeline from these
    steps, runs it on live BeeMonitor endpoints, and pulls per-step outputs.

    ``base_url`` is the deployed site root (e.g. https://beemonitor.edwardamoah.com).
    """
    import json as _json

    steps = pipeline.steps or []
    steps_lit = _json.dumps(steps, indent=2)
    api = base_url.rstrip("/") + "/api/v1"

    setup = (
        "import requests, time\n"
        "from getpass import getpass\n\n"
        f'BASE = "{api}"\n'
        'API_KEY = getpass("BeeMonitor API key (Developer page → Create API Key): ")\n'
        'H = {"Authorization": f"Bearer {API_KEY}"}\n\n'
        "def api(method, path, **kw):\n"
        '    r = requests.request(method, f"{BASE}/{path}", headers=H, **kw)\n'
        "    if not r.ok:\n"
        '        raise RuntimeError(f"{r.status_code}: {r.text[:400]}")\n'
        "    return r.json() if r.content else {}\n\n"
        'print("blocks available:", len(api("get", "pipelines/blocks/")["blocks"]))'
    )

    upload = (
        "# OPTION A — upload a local video (skip if you already have a video id):\n"
        "# import os\n"
        "# path = 'clip.mp4'\n"
        "# init = api('post', 'pipelines/uploads/initiate',\n"
        "#            json={'filename': os.path.basename(path), 'size_bytes': os.path.getsize(path)})\n"
        "# requests.put(init['upload_url'], data=open(path,'rb'),\n"
        "#              headers={'Content-Type': 'video/mp4'}).raise_for_status()\n"
        "# done = api('post', 'pipelines/uploads/complete',\n"
        "#            json={'storage_key': init['storage_key'], 'file_size_bytes': os.path.getsize(path),\n"
        "#                  'title': path, 'site_name': 'My Site'})\n"
        "# VIDEO_IDS = [done['video_id']]\n\n"
        "# OPTION B — use existing video id(s) from your account:\n"
        "VIDEO_IDS = []  # <- put your video id(s) here"
    )

    create = (
        f"steps = {steps_lit}\n\n"
        f"pipeline = api('post', 'pipelines/', json={{'title': {pipeline.title!r}, 'steps': steps}})\n"
        "print('pipeline', pipeline['id'], '| warnings:', pipeline.get('warnings'))"
    )

    run = (
        "resp = api('post', f\"pipelines/{pipeline['id']}/run/\", json={'video_ids': VIDEO_IDS})\n"
        "run_ids = [r['run_id'] for r in resp['runs']]\n"
        "print('started', len(run_ids), 'run(s)')\n\n"
        "for rid in run_ids:\n"
        "    while True:\n"
        "        run = api('get', f'pipeline-runs/{rid}/')\n"
        "        if run['status'] in ('completed', 'failed'):\n"
        "            break\n"
        "        time.sleep(8)\n"
        "    print(rid, '->', run['status'])"
    )

    out_steps = [s for s in steps if s.get("block_type", "").split(".")[0]
                 in ("analyze", "identify", "output")]
    outputs_code_lines = [
        "run = api('get', f'pipeline-runs/{run_ids[0]}/')  # first video",
        "import pandas as pd",
        "",
    ]
    for s in out_steps:
        sid = s.get("id")
        outputs_code_lines.append(f"# {s.get('block_type')}")
        outputs_code_lines.append(f"out = run['outputs'].get({sid!r}, {{}})")
        outputs_code_lines.append("display(pd.DataFrame(out['rows']) if out.get('rows') else out)")
    outputs_code = "\n".join(outputs_code_lines)

    cells = [
        _md(f"# {pipeline.title} — run on BeeMonitor (live API)\n\n"
            f"{pipeline.description or ''}\n\n"
            "This notebook runs the pipeline on **real BeeMonitor endpoints** via the public "
            "API — no install, no SDK, just `requests`. You need an **API key**: create one on "
            "the **Developer** page of your BeeMonitor account (it's shown once)."),
        _md("## 1. Connect"),
        _code(setup),
        _md("## 2. Pick your video(s)"),
        _code(upload),
        _md("## 3. Create the pipeline\nThese are your saved steps — edit freely."),
        _code(create),
        _md("## 4. Run on real endpoints + wait"),
        _code(run),
    ]
    if out_steps:
        cells.append(_md("## 5. Results per step"))
        cells.append(_code(outputs_code))
    cells.append(_md("---\n*Runs on live BeeMonitor GPU + analytics; bills your account credits.*"))
    return _notebook(cells)
