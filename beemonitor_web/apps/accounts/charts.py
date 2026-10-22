"""Credits over time as stacked bars, drawn by the server with no JS.

Used by the account page (per day, by kind of work) and the platform
dashboard (per week, by kind or by tier). The template is
templates/components/stacked_bars.html.
"""

import math
from datetime import timedelta

from django.db.models import Sum
from django.db.models.functions import TruncDay, TruncWeek
from django.utils import timezone

# Series colours: distinct in lightness as well as hue.
COLORS = ["#1d4ed8", "#fb923c", "#93c5fd", "#7c2d12", "#9ca3af", "#0f766e"]


def credits_by_period(events, periods=12, unit="week", split="kind"):
    """Credits in each of the last ``periods`` days or weeks, split by ``kind``
    of work or by the user's ``tier``. Returns (rows, series keys)."""
    step = timedelta(weeks=1) if unit == "week" else timedelta(days=1)
    start = timezone.localdate() - step * (periods - 1)
    if unit == "week":
        start -= timedelta(days=start.weekday())
    trunc = TruncWeek if unit == "week" else TruncDay
    key = "kind" if split == "kind" else "user__profile__tier"
    rows = (events.filter(created_at__date__gte=start).annotate(p=trunc("created_at"))
            .values("p", key).annotate(cr=Sum("credits")).order_by("p"))
    series = sorted({r[key] or "" for r in rows})
    by_period = {}
    for r in rows:
        by_period.setdefault(r["p"].date(), {})[r[key] or ""] = float(r["cr"] or 0)
    out = []
    for i in range(periods):
        day = start + step * i
        vals = by_period.get(day, {})
        out.append({"start": day, "values": [vals.get(s, 0.0) for s in series], "total": sum(vals.values())})
    return out, series


def nice_max(peak):
    """The top of the y-axis: the peak rounded up to 1, 2, 2.5 or 5 × 10^n."""
    if peak <= 0:
        return 1
    base = 10 ** math.floor(math.log10(peak))
    for step in (1, 2, 2.5, 5, 10):
        if peak <= step * base:
            return step * base
    return 10 * base


def stacked(rows, labels, x_title, height=160, label_every=2):
    """What stacked_bars.html draws: bars as pixel heights, a credits y-axis in
    fifths of a round top, and a date under every ``label_every``-th bar."""
    peak = max([r["total"] for r in rows] + [0])
    top = nice_max(peak)
    colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
    return {
        "series": [{"label": label, "color": c} for label, c in zip(labels, colors)],
        "peak": peak,
        "x_title": x_title,
        # Top to bottom, as the axis is drawn; ``y`` is the pixel offset from the top.
        "ticks": [{"v": f"{top * f:g}", "y": round(height * (1 - f))} for f in (1, 0.8, 0.6, 0.4, 0.2, 0)],
        "bars": [{"start": r["start"], "total": r["total"], "label": i % label_every == 0,
                  "parts": [{"h": round(v / top * height), "color": colors[j]}
                            for j, v in enumerate(r["values"]) if v]}
                 for i, r in enumerate(rows)],
        "height": height,
    }
