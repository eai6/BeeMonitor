#!/usr/bin/env python3
"""Investigate why ~1322 natalies jobs returned BlobNotFound for events.csv."""
import os, json, concurrent.futures as cf
from azure.storage.blob import BlobServiceClient

ACCOUNT, CONTAINER = "beemonitorstore", "processed"
svc = BlobServiceClient(account_url=f"https://{ACCOUNT}.blob.core.windows.net",
                        credential=os.environ["AZ_KEY"])
cc = svc.get_container_client(CONTAINER)

# Re-scan metadata to get (job_prefix, source_video, metadata events_csv) for natalies
metas = [b.name for b in cc.list_blobs(name_starts_with="1/modal_")
         if b.name.endswith("/job_metadata.json")]

def fetch(name):
    m = json.loads(cc.download_blob(name).readall())
    return (name.rsplit("/",1)[0], os.path.basename(m.get("source_video","") or ""),
            m.get("files_uploaded", {}))

rows = []
with cf.ThreadPoolExecutor(max_workers=32) as ex:
    rows = list(ex.map(fetch, metas))
nat = [(jp, base, fu) for jp, base, fu in rows if base.startswith("natalies_")]

# For natalies jobs, list actual blobs present in each folder and classify
def folder_files(jp):
    return jp, sorted(n.name.split("/")[-1] for n in cc.list_blobs(name_starts_with=jp+"/"))

# sample the ones missing events.csv: check folder contents
print(f"natalies jobs: {len(nat)}")

# Check: does metadata even list events_csv?
no_meta_events = [jp for jp,b,fu in nat if not fu.get("events_csv")]
print(f"jobs whose metadata has NO events_csv key: {len(no_meta_events)}")

# Sample 12 jobs and show their actual folder contents
import itertools
sample = nat[:0]
# pick a spread: every Nth
step = max(1, len(nat)//15)
sample = nat[::step][:15]
print("\n=== sample job folder contents ===")
with cf.ThreadPoolExecutor(max_workers=16) as ex:
    res = list(ex.map(folder_files, [jp for jp,_,_ in sample]))
meta_by_jp = {jp:(b,fu) for jp,b,fu in nat}
for jp, files in res:
    b, fu = meta_by_jp[jp]
    print(f"{jp}  [{b}]")
    print(f"    files: {files}")
    print(f"    meta.events_csv: {fu.get('events_csv')}")

# Aggregate: how many natalies folders actually CONTAIN an events.csv blob?
print("\n=== scanning ALL natalies folders for events.csv presence (this lists blobs) ===")
def has_events(jp):
    names = [n.name for n in cc.list_blobs(name_starts_with=jp+"/")]
    base = [x.split("/")[-1] for x in names]
    return ("events.csv" in base, len(names), base)
with cf.ThreadPoolExecutor(max_workers=32) as ex:
    pres = list(ex.map(has_events, [jp for jp,_,_ in nat]))
with_ev = sum(1 for ok,_,_ in pres if ok)
empty_folder = sum(1 for ok,n,_ in pres if n==0)
print(f"folders containing events.csv: {with_ev}/{len(nat)}")
print(f"completely empty folders: {empty_folder}")
# what file sets do the no-events folders have?
from collections import Counter
no_ev_sets = Counter()
for (ok,n,base),(jp,b,fu) in zip(pres, nat):
    if not ok:
        no_ev_sets[tuple(sorted(set(base)))] += 1
print("\nfile-sets of folders WITHOUT events.csv (count: fileset):")
for fs, c in no_ev_sets.most_common(15):
    print(f"  {c:4d}: {list(fs)}")
