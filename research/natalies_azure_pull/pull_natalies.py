#!/usr/bin/env python3
"""Pull all analyzed 'natalies' site data from Azure blob (beemonitorstore/processed),
build aggregated_events_natalies.csv and foraging_trips_natalies.csv.

- Scans every 1/modal_*/job_metadata.json to find jobs whose source_video is a
  'natalies' video (site is only recorded inside the metadata, not the path).
- Downloads each natalies events.csv into ./events_raw/<video_name>_events.csv
- Aggregates events using the SAME schema as cloud_scripts/extract_data.py
- Builds foraging trips using the user's canonical Exit->Entry greedy pairing.
"""
import os, re, sys, json, concurrent.futures as cf
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from azure.storage.blob import BlobServiceClient

ACCOUNT = "beemonitorstore"
CONTAINER = "processed"
SITE = "natalies"
HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "events_raw"
RESEARCH = HERE.parent / "research_data"
RAW_DIR.mkdir(parents=True, exist_ok=True)

KEY = os.environ["AZ_KEY"]
svc = BlobServiceClient(account_url=f"https://{ACCOUNT}.blob.core.windows.net", credential=KEY)
cc = svc.get_container_client(CONTAINER)

def log(*a): print(*a, flush=True)

# ---------------------------------------------------------------
# 1. List all job_metadata.json blobs
# ---------------------------------------------------------------
log("Listing job_metadata.json blobs under 1/modal_ ...")
meta_blobs = [b.name for b in cc.list_blobs(name_starts_with="1/modal_")
              if b.name.endswith("/job_metadata.json")]
log(f"  found {len(meta_blobs)} job metadata files")

# ---------------------------------------------------------------
# 2. Download + parse each metadata concurrently, filter natalies
# ---------------------------------------------------------------
def fetch_meta(name):
    try:
        data = cc.download_blob(name).readall()
        m = json.loads(data)
        src = m.get("source_video", "") or ""
        # presence of the events_csv key reliably indicates the job produced output
        return (name, os.path.basename(src), m.get("files_uploaded", {}).get("events_csv"))
    except Exception:
        return (name, None, None)

log("Fetching metadata (concurrent)...")
natalies = []  # (job_prefix, video_basename, events_csv_path_or_None)
with cf.ThreadPoolExecutor(max_workers=32) as ex:
    for i, (name, base, events_path) in enumerate(ex.map(fetch_meta, meta_blobs), 1):
        if i % 500 == 0:
            log(f"  parsed {i}/{len(meta_blobs)}")
        if base and base.startswith(SITE + "_"):
            natalies.append((name.rsplit("/", 1)[0], base, events_path))

log(f"\nnatalies jobs (modal runs): {len(natalies)}")

# Group jobs by video_name; a video may have >1 job (re-runs). Prefer a job that
# actually produced an events.csv. Videos where ALL jobs lack output = never analyzed.
def video_name(base): return re.sub(r"\.mp4$", "", base)
jobs_by_video = {}
for job_prefix, base, ev in natalies:
    jobs_by_video.setdefault(video_name(base), []).append((job_prefix, ev))

by_video = {}            # video_name -> (job_prefix, events_csv_path)  [only ones WITH output]
no_output = []           # video_names where no job produced events.csv
multi = 0
for vn, jobs in jobs_by_video.items():
    if len(jobs) > 1:
        multi += 1
    with_ev = [(jp, ev) for jp, ev in jobs if ev]
    if with_ev:
        by_video[vn] = with_ev[0]   # prefer a job that produced output
    else:
        no_output.append(vn)

log(f"unique natalies videos:      {len(jobs_by_video)}")
log(f"  videos WITH analysis output: {len(by_video)}")
log(f"  videos with NO output (all jobs failed/empty-folder): {len(no_output)}")
log(f"  videos that had >1 modal job (re-runs): {multi}")
# persist the no-output list so collaborators know which videos lack analysis
(HERE / "natalies_videos_without_output.txt").write_text(
    "\n".join(sorted(no_output)) + ("\n" if no_output else ""))

# ---------------------------------------------------------------
# 3. Download events.csv for each unique natalies video
# ---------------------------------------------------------------
def dl_events(item):
    vn, (job_prefix, ev) = item
    out = RAW_DIR / f"{vn}_events.csv"
    try:
        data = cc.download_blob(ev).readall()
        out.write_bytes(data)
        return (vn, out, None)
    except Exception as e:
        return (vn, None, str(e))

log("\nDownloading events.csv files...")
ok, failed = [], []
with cf.ThreadPoolExecutor(max_workers=32) as ex:
    for i, (vn, out, err) in enumerate(ex.map(dl_events, by_video.items()), 1):
        if i % 200 == 0:
            log(f"  downloaded {i}/{len(by_video)}")
        (failed if err else ok).append((vn, err))
log(f"  downloaded {len(ok)} events files; failures: {len(failed)}")
for vn, err in failed[:20]:
    log(f"    FAIL {vn}: {err}")

# ---------------------------------------------------------------
# 4. Aggregate events (schema identical to extract_data.py)
# ---------------------------------------------------------------
FNAME_RE = re.compile(r'(\w+)_(\d{4})-(\d{2})-(\d{2})_(\d{2})_(\d{2})_(\d{2})')

def parse_video_filename(filename):
    m = FNAME_RE.search(filename)
    if not m: return None
    site, y, mo, d, h, mi, s = m.groups()
    try:
        dt = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s))
    except ValueError:
        return None
    return {'site': site, 'date': dt.date(), 'time': dt.time(), 'datetime': dt,
            'hour': int(h), 'video_name': f"{site}_{y}-{mo}-{d}_{h}_{mi}_{s}"}

def load_events_csv(fp: Path):
    df = pd.read_csv(fp)
    if df.empty: return None
    vi = parse_video_filename(fp.name)
    if vi is None:
        log(f"  WARN unparseable filename {fp.name}"); return None
    df['site'] = vi['site']
    df['video_date'] = vi['date']
    df['video_time'] = vi['time']
    df['video_datetime'] = vi['datetime']
    df['video_hour'] = vi['hour']
    df['video_name'] = vi['video_name']
    df['source_file'] = fp.name
    if 'frame_number' in df.columns:
        df['seconds_in_video'] = df['frame_number'] / 30
        df['event_datetime'] = df.apply(
            lambda r: vi['datetime'] + timedelta(seconds=r['seconds_in_video']), axis=1)
    else:
        df['event_datetime'] = vi['datetime']
    return df

log("\nAggregating events...")
frames = []
empty = 0
for fp in sorted(RAW_DIR.glob("*_events.csv")):
    d = load_events_csv(fp)
    if d is not None and len(d) > 0:
        frames.append(d)
    else:
        empty += 1
combined = pd.concat(frames, ignore_index=True)
combined = combined.sort_values('event_datetime').reset_index(drop=True)

# match column order of existing aggregated_events.csv
COLS = ['action','nest','frame_number','track_id','ml_confidence','timestamp',
        'site','video_date','video_time','video_datetime','video_hour',
        'video_name','source_file','seconds_in_video','event_datetime']
combined = combined[COLS]
agg_out = RESEARCH / "aggregated_events_natalies.csv"
combined.to_csv(agg_out, index=False)
log(f"  events files with data: {len(frames)}, empty: {empty}")
log(f"  total events: {len(combined):,}")
log(f"  date range: {combined['event_datetime'].min()} -> {combined['event_datetime'].max()}")
log(f"  wrote {agg_out}")

# ---------------------------------------------------------------
# 5. Foraging trips -- user's canonical greedy Exit->Entry pairing
# ---------------------------------------------------------------
log("\nBuilding foraging trips (canonical pairing)...")
events = combined.copy()
events['event_datetime'] = pd.to_datetime(events['event_datetime'])
events = events.sort_values(['nest', 'event_datetime'])

trips = []
for nest, grp in events.groupby('nest'):
    grp = grp.sort_values('event_datetime').reset_index(drop=True)
    i = 0
    while i < len(grp) - 1:
        if grp.loc[i, 'action'] == 'Exit':
            for j in range(i + 1, len(grp)):
                if grp.loc[j, 'action'] == 'Entry':
                    exit_time = grp.loc[i, 'event_datetime']
                    entry_time = grp.loc[j, 'event_datetime']
                    duration = (entry_time - exit_time).total_seconds() / 60.0
                    trips.append({'nest': nest, 'exit_time': exit_time,
                                  'entry_time': entry_time, 'duration_min': duration})
                    i = j + 1
                    break
            else:
                i += 1
        else:
            i += 1

trips_df = pd.DataFrame(trips)
trips_out = RESEARCH / "foraging_trips_natalies.csv"
trips_df.to_csv(trips_out, index=False)
log(f"  trips: {len(trips_df):,}")
log(f"  wrote {trips_out}")
log("\nDONE.")
