#!/usr/bin/env python3
"""Split natalies foraging trips into Release 1 / Release 2, consistent with the
mendels analysis: apply trips_clean filter (0.5 < duration_min < 30 AND exit hour
in 6..20), then split by release window.

Release windows (from nesting_data.csv, identical for both sites):
  Release 1: 2024-04-16  ->  2024-05-16   (install -> collection)
  Release 2: 2024-05-16  ->  2024-06-07
"""
from pathlib import Path
import pandas as pd

RESEARCH = Path(__file__).resolve().parent.parent / "research_data"

t = pd.read_csv(RESEARCH / "foraging_trips_natalies.csv")
t["exit_time"] = pd.to_datetime(t["exit_time"])
t["entry_time"] = pd.to_datetime(t["entry_time"])

n_raw = len(t)

# --- trips_clean: same filter used for mendels ---
clean = t[
    (t["duration_min"] > 0.5)
    & (t["duration_min"] < 30)
    & (t["exit_time"].dt.hour.between(6, 20))
].copy()

# --- release windows ---
r1 = clean[(clean["exit_time"] >= "2024-04-16") & (clean["exit_time"] < "2024-05-16")].copy()
r2 = clean[(clean["exit_time"] >= "2024-05-16") & (clean["exit_time"] < "2024-06-08")].copy()

r1.to_csv(RESEARCH / "foraging_trips_natalies_release1.csv", index=False)
r2.to_csv(RESEARCH / "foraging_trips_natalies_release2.csv", index=False)

print(f"raw trips:                 {n_raw}")
print(f"after trips_clean filter:  {len(clean)}  "
      f"(removed {n_raw - len(clean)}: dur<=0.5 or >=30, or exit hour outside 6-20)")
print(f"  Release 1 [04-16,05-16): {len(r1)} trips, {r1['nest'].nunique()} nests")
print(f"  Release 2 [05-16,06-07]: {len(r2)} trips, {r2['nest'].nunique()} nests")
print(f"  outside both windows:    {len(clean) - len(r1) - len(r2)}")
print(f"\nwrote foraging_trips_natalies_release1.csv")
print(f"wrote foraging_trips_natalies_release2.csv")
