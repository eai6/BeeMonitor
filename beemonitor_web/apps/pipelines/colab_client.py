"""
BeeMonitor pipeline API — single-cell Colab client (P1).

Paste this whole file into one Colab cell (only needs `requests`; `pandas` optional
for step outputs). No package install required — publishing `pip install beemonitor`
is P3. See memory/24_pipeline_api_design.md.

    bm = BeeMonitor("bmk_...")                       # key from the Developer page
    print([b for b in bm.blocks()["blocks"]])        # available blocks
    p = bm.from_template("Foraging trips")           # or bm.create_pipeline(title, steps)
    runs = bm.run(p["id"], video_ids=[123])          # runs on REAL SageMaker endpoints
    run = runs[0]; run.wait()                         # poll until done
    df = run.step_output("f")                         # a step's rows as a DataFrame
    run.summary()                                     # per-step status + outputs
"""

import time

import requests

DEFAULT_BASE_URL = "https://beemonitor.edwardamoah.com"


class Run:
    def __init__(self, client, run_id):
        self.client = client
        self.id = run_id
        self._data = None

    def refresh(self):
        self._data = self.client._get(f"pipeline-runs/{self.id}/")
        return self._data

    @property
    def status(self):
        return (self._data or self.refresh()).get("status")

    def wait(self, timeout=1800, interval=8):
        """Poll until the run reaches completed/failed (or timeout)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.refresh()
            if self._data.get("status") in ("completed", "failed"):
                return self._data
            time.sleep(interval)
        raise TimeoutError(f"Run {self.id} did not finish within {timeout}s")

    def outputs(self):
        return (self._data or self.refresh()).get("outputs", {})

    def step_output(self, step_id, as_df=True):
        """A step's output. Tabular steps -> a pandas DataFrame (if available)."""
        out = self.outputs().get(step_id, {})
        rows = out.get("rows")
        if as_df and rows:
            try:
                import pandas as pd
                return pd.DataFrame(rows)
            except ImportError:
                pass
        return out

    def summary(self):
        d = self._data or self.refresh()
        print(f"Run {self.id} — {d.get('status')}")
        for step in d.get("steps", []):
            sid = step.get("id")
            st = (d.get("step_status") or {}).get(sid, "?")
            out = (d.get("outputs") or {}).get(sid, {})
            note = out.get("note") or out.get("error") or ""
            print(f"  [{st:>7}] {step.get('block_type'):26} {note}")
        return d


class BeeMonitor:
    def __init__(self, api_key, base_url=DEFAULT_BASE_URL):
        self.base = base_url.rstrip("/") + "/api/v1/"
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {api_key}"

    # ── low-level ──
    def _url(self, path):
        return self.base + path.lstrip("/")

    def _get(self, path, **params):
        r = self.s.get(self._url(path), params=params or None)
        r.raise_for_status()
        return r.json()

    def _post(self, path, json=None):
        r = self.s.post(self._url(path), json=json or {})
        if not r.ok:
            raise requests.HTTPError(f"{r.status_code}: {r.text[:400]}")
        return r.json() if r.content else {}

    def _put(self, path, json=None):
        r = self.s.put(self._url(path), json=json or {})
        r.raise_for_status()
        return r.json()

    def _delete(self, path):
        r = self.s.delete(self._url(path))
        r.raise_for_status()

    # ── uploads (P2) ──
    def upload_video(self, path, title=None, site=None, device_id=None, content_type="video/mp4"):
        """Upload a local video file straight to S3 (presign -> PUT -> confirm).

        Returns the created video dict ({video_id, title, site_name, ...}); pass
        video_id into run(). Not all videos come from devices — set `site` to group
        the clip under a location.
        """
        import os
        filename = os.path.basename(path)
        size = os.path.getsize(path)
        init = self._post("pipelines/uploads/initiate",
                          {"filename": filename, "size_bytes": size, "content_type": content_type})
        with open(path, "rb") as f:
            put = requests.put(init["upload_url"], data=f, headers={"Content-Type": content_type})
        if not put.ok:
            raise requests.HTTPError(f"S3 PUT {put.status_code}: {put.text[:300]}")
        body = {"storage_key": init["storage_key"], "file_size_bytes": size,
                "title": title or filename}
        if site:
            body["site_name"] = site
        if device_id:
            body["device_id"] = device_id
        return self._post("pipelines/uploads/complete", body)

    # ── pipelines ──
    def blocks(self):
        return self._get("pipelines/blocks/")

    def pipelines(self):
        return self._get("pipelines/")

    def create_pipeline(self, title, steps):
        return self._post("pipelines/", {"title": title, "steps": steps})

    def get_pipeline(self, pipeline_id):
        return self._get(f"pipelines/{pipeline_id}/")

    def update_pipeline(self, pipeline_id, **fields):
        return self._put(f"pipelines/{pipeline_id}/", fields)

    def delete_pipeline(self, pipeline_id):
        self._delete(f"pipelines/{pipeline_id}/")

    def validate(self, steps):
        return self._post("pipelines/validate/", {"steps": steps})

    def clone(self, pipeline_id):
        return self._post(f"pipelines/{pipeline_id}/clone/")

    def from_template(self, title):
        """Clone the template named `title` into your pipelines."""
        for t in self.pipelines().get("templates", []):
            if t["title"] == title:
                return self.clone(t["id"])
        raise ValueError(f"No template named {title!r}")

    # ── runs ──
    def run(self, pipeline_id, video_ids):
        resp = self._post(f"pipelines/{pipeline_id}/run/", {"video_ids": list(video_ids)})
        return [Run(self, r["run_id"]) for r in resp.get("runs", [])]

    def runs(self):
        return self._get("pipeline-runs/").get("runs", [])

    def run_by_id(self, run_id):
        return Run(self, run_id)
