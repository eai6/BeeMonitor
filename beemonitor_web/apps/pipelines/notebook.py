"""
Export a pipeline to a **live** Google Colab notebook that runs it for real via the
public API (raw ``requests`` — no SDK, no install). The notebook creates the pipeline
from its steps, runs it on live BeeMonitor endpoints, polls, and shows per-step
outputs. Core IP stays server-side; the notebook only makes HTTP calls.

Design: ``memory/24_pipeline_api_design.md`` (P3).
"""

import json as _json


def _md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source}


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
    install (Colab already has ``requests``).

    ``base_url`` is the deployed site root (e.g. https://beemonitor.edwardamoah.com).
    """
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
    outputs_lines = [
        "run = api('get', f'pipeline-runs/{run_ids[0]}/')  # first video",
        "import pandas as pd",
        "",
    ]
    for s in out_steps:
        sid = s.get("id")
        outputs_lines.append(f"# {s.get('block_type')}")
        outputs_lines.append(f"out = run['outputs'].get({sid!r}, {{}})")
        outputs_lines.append("display(pd.DataFrame(out['rows']) if out.get('rows') else out)")

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
        cells.append(_code("\n".join(outputs_lines)))
    cells.append(_md("---\n*Runs on live BeeMonitor GPU + analytics; bills your account credits.*"))
    return _notebook(cells)


def create_colab_gist(notebook_dict, filename, token, description=""):
    """Push a notebook to a public GitHub gist and return its Colab URL.

    Colab opens notebooks from GitHub/gist/Drive (not arbitrary URLs), so a gist is
    the no-Drive way to open a generated notebook in the browser. The notebook holds
    no secrets (the API key is entered at runtime), so a public gist is fine. Raises
    on failure; the caller falls back to a download.
    """
    import urllib.request

    payload = _json.dumps({
        "description": description or "BeeMonitor pipeline",
        "public": True,
        "files": {filename: {"content": _json.dumps(notebook_dict, indent=1)}},
    }).encode()
    req = urllib.request.Request(
        "https://api.github.com/gists", data=payload, method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "BeeMonitor",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        gist = _json.loads(r.read())
    return f"https://colab.research.google.com/gist/{gist['id']}"
