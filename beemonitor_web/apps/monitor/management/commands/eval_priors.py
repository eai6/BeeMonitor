"""Score location-constrained vs unconstrained BioCLIP on a labeled crop set.

The Phase 2 ship/no-ship gate (memory/15 §11.5): does constraining BioCLIP to a
region's taxa actually improve accuracy? Each labeled crop is run through the
endpoint twice — unconstrained (full Tree of Life) and constrained (the location
prior) — and we report top-1 species/genus accuracy and the lift. This isolates
the constraint's effect, so it does NOT apply the production confidence fallback.

Manifest (CSV with a header row). Either reference existing frames:

    frame_id,truth
    123,Bombus impatiens

…or external images (``image`` is a local path or an S3 key in raw-videos):

    image,truth,lat,lon,month
    /data/crops/a.jpg,Bombus impatiens,40.80,-77.86,6
    users/1/devices/2/activity_frames/2026/06/x.jpg,Apis mellifera,40.80,-77.86,6

  python manage.py eval_priors --manifest labels.csv [--limit N]
"""

import csv

from django.core.management.base import BaseCommand, CommandError

from apps.monitor import pipeline, priors


# --- pure helpers (unit-tested) --------------------------------------------

def genus_of(binomial: str) -> str:
    """Genus = first token of a scientific binomial ('Bombus impatiens' -> 'Bombus')."""
    parts = (binomial or "").strip().split()
    return parts[0] if parts else ""


def top_species_genus(preds) -> tuple:
    """(species, genus) of the top-1 prediction, ('','') if none."""
    if not preds:
        return "", ""
    ranks = preds[0].get("ranks", {}) or {}
    return (ranks.get("species") or "").strip(), (ranks.get("genus") or "").strip()


def read_manifest(path: str) -> list:
    """Parse the CSV manifest into normalized row dicts."""
    rows = []
    with open(path, newline="") as fh:
        for raw in csv.DictReader(fh):
            truth = (raw.get("truth") or "").strip()
            if not truth:
                continue
            rows.append({
                "frame_id": (raw.get("frame_id") or "").strip() or None,
                "image": (raw.get("image") or "").strip() or None,
                "truth": truth,
                "lat": _f(raw.get("lat")),
                "lon": _f(raw.get("lon")),
                "month": _i(raw.get("month")),
            })
    return rows


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class Command(BaseCommand):
    help = "Score location-constrained vs unconstrained BioCLIP on a labeled set."

    def add_arguments(self, parser):
        parser.add_argument("--manifest", required=True, help="CSV label file.")
        parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = all).")

    def handle(self, *args, **opts):
        if not pipeline.enabled():
            raise CommandError("SAGEMAKER_BIOCLIP_ENDPOINT_NAME is not set — no endpoint to call.")
        rows = read_manifest(opts["manifest"])
        if opts["limit"]:
            rows = rows[: opts["limit"]]
        if not rows:
            raise CommandError("No usable rows in the manifest.")

        tally = {"unconstrained": {"species": 0, "genus": 0},
                 "constrained": {"species": 0, "genus": 0}}
        n = no_prior = 0
        for row in rows:
            jpeg, lat, lon, month = self._resolve(row)
            if jpeg is None:
                self.stderr.write(self.style.WARNING(f"  skipped (no crop): {row}"))
                continue
            truth_sp = row["truth"]
            truth_gn = genus_of(truth_sp)

            cands = priors.region_taxa(lat, lon, month)
            if not cands:
                no_prior += 1
            unconstrained = pipeline._invoke_bioclip(jpeg, None)
            constrained = pipeline._invoke_bioclip(jpeg, cands) if cands else unconstrained

            for mode, preds in (("unconstrained", unconstrained), ("constrained", constrained)):
                sp, gn = top_species_genus(preds)
                if sp and sp.lower() == truth_sp.lower():
                    tally[mode]["species"] += 1
                if gn and gn.lower() == truth_gn.lower():
                    tally[mode]["genus"] += 1
            n += 1
            self.stdout.write(
                f"  truth={truth_sp!r}  unc={top_species_genus(unconstrained)[0]!r}  "
                f"con={top_species_genus(constrained)[0]!r}")

        self._report(tally, n, no_prior)

    def _resolve(self, row):
        """Return (jpeg_bytes, lat, lon, month) for a manifest row, or (None, …)."""
        if row["frame_id"]:
            from apps.monitor.models import ActivityFrame
            fr = (ActivityFrame.objects.select_related("activity")
                  .filter(pk=row["frame_id"]).first())
            if fr is None:
                return None, None, None, None
            act = fr.activity
            month = row["month"] or (act.started_at.month if act.started_at else None)
            try:
                return (pipeline._read_crop_bytes(fr.storage_key),
                        row["lat"] if row["lat"] is not None else act.lat,
                        row["lon"] if row["lon"] is not None else act.lon, month)
            except Exception as e:  # noqa: BLE001
                self.stderr.write(self.style.WARNING(f"  read failed {fr.storage_key}: {e}"))
                return None, None, None, None
        image = row["image"]
        if not image:
            return None, None, None, None
        import os
        try:
            if os.path.exists(image):
                with open(image, "rb") as fh:
                    return fh.read(), row["lat"], row["lon"], row["month"]
            return pipeline._read_crop_bytes(image), row["lat"], row["lon"], row["month"]
        except Exception as e:  # noqa: BLE001
            self.stderr.write(self.style.WARNING(f"  read failed {image}: {e}"))
            return None, None, None, None

    def _report(self, tally, n, no_prior):
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(f"Eval over {n} crop(s):"))
        self.stdout.write(f"{'rank':<10}{'unconstrained':>15}{'constrained':>14}{'lift':>10}")
        for rank in ("species", "genus"):
            u = 100.0 * tally["unconstrained"][rank] / n if n else 0.0
            c = 100.0 * tally["constrained"][rank] / n if n else 0.0
            self.stdout.write(f"{rank:<10}{u:>14.1f}%{c:>13.1f}%{c - u:>+8.1f}pp")
        if no_prior:
            self.stdout.write(self.style.WARNING(
                f"note: {no_prior}/{n} crop(s) had no location prior (no GPS / API miss) "
                "— constrained == unconstrained for those."))
