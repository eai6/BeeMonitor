from __future__ import annotations

import argparse
import logging
import tempfile
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests

# PRISM web service (docs: prism.oregonstate.edu/documents/PRISM_downloads_web_service.pdf).
# Since 2025 a grid comes as a zip holding a Cloud Optimized GeoTIFF (.tif);
# older downloads held a .bil. Both are read below.
URL = "https://services.nacse.org/prism/data/get/us/4km/{var}/{day:%Y%m%d}"
# PRISM blocks a file after two downloads in 24 h: cached zips are never fetched again.
PAUSE_S = 1.0
COLUMNS = {"ppt": "ppt_mm", "tmean": "tmean_c", "tmin": "tmin_c", "tmax": "tmax_c"}

log = logging.getLogger("prism")


def download(var: str, day: date, cache: Path) -> Path | None:
    """The day's grid zip, from the cache or the web service; None if it can't be had."""
    path = cache / f"prism_{var}_us_25m_{day:%Y%m%d}.zip"
    if path.exists():
        return path
    url = URL.format(var=var, day=day)
    for attempt in (1, 2):
        try:
            time.sleep(PAUSE_S)
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            # Over the limit, PRISM answers 200 with a text message, not a zip.
            if not r.content.startswith(b"PK"):
                raise ValueError(r.text.strip()[:200])
            tmp = path.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(path)
            return path
        except (requests.RequestException, ValueError) as e:
            log.warning("%s %s attempt %d failed: %s", var, day, attempt, e)
    return None


def extract_point(zip_path: Path, lon: float, lat: float) -> float:
    """The grid's value at (lon, lat), NaN for nodata. PRISM grids are NAD83 lon/lat."""
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        raster = next((p for ext in ("*.tif", "*.bil") for p in Path(tmp).glob(ext)), None)
        if raster is None:
            raise ValueError(f"no .tif or .bil in {zip_path.name}")
        with rasterio.open(raster) as src:
            value = float(next(src.sample([(lon, lat)]))[0])
            nodata = src.nodata if src.nodata is not None else -9999
    return np.nan if value == nodata or value <= -9999 else value


def main() -> None:
    ap = argparse.ArgumentParser(description="PRISM 4 km daily values at one point, as a CSV.")
    ap.add_argument("--lat", type=float, default=40.7934)
    ap.add_argument("--lon", type=float, default=-77.8600)
    ap.add_argument("--start", type=date.fromisoformat, default=date(2024, 4, 15))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2024, 6, 8))
    ap.add_argument("--vars", nargs="+", default=["ppt", "tmean"], choices=sorted(COLUMNS))
    ap.add_argument("--out", default="prism_state_college_2024.csv")
    ap.add_argument("--cache", type=Path, default=Path("prism_cache"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.cache.mkdir(parents=True, exist_ok=True)

    # A PRISM "day" is the 24 h ending 12:00 GMT (07:00 EST) on that date, so
    # rain on the afternoon of the 1st is in the 2nd's ppt.
    days = [args.start + timedelta(n) for n in range((args.end - args.start).days + 1)]
    rows, failed = [], []
    for day in days:
        row = {"date": day.isoformat()}
        for var in args.vars:
            path = download(var, day, args.cache)
            value = np.nan
            if path is not None:
                try:
                    value = extract_point(path, args.lon, args.lat)
                except Exception as e:  # a damaged zip: drop it so the next run fetches it
                    log.warning("%s %s unreadable (%s); removed from the cache", var, day, e)
                    path.unlink(missing_ok=True)
            if np.isnan(value):
                failed.append(f"{day} {var}")
            row[COLUMNS[var]] = round(value, 2)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv(args.out, index=False)
    print(f"{len(df)} days -> {args.out}")
    if "ppt_mm" in df:
        print(f"total ppt: {df['ppt_mm'].sum():.1f} mm")
    if "tmean_c" in df:
        print(f"mean tmean: {df['tmean_c'].mean():.1f} °C")
    print("missing: " + (", ".join(failed) if failed else "none"))


if __name__ == "__main__":
    main()
