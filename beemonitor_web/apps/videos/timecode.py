"""Times in a clip as people write them: 1:33, 1:02:05 (memory/53)."""


def fmt_clock(seconds):
    """93 -> '1:33'; 3725 -> '1:02:05'."""
    seconds = int(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parse_clock(text):
    """'1:33' / '93' / '1:02:05' -> seconds. ValueError when it isn't a time."""
    parts = str(text or "0").strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"“{text}” isn't a time. Use minutes:seconds, like 2:30.")
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + int(p)
    return seconds


def part_label(config, video):
    """'Analysed 0:00–5:00 of 34:32' for a run on part of a clip, else ''."""
    start, end = (config or {}).get("start_seconds"), (config or {}).get("end_seconds")
    if start in (None, "") and end in (None, ""):
        return ""
    duration = float(getattr(video, "duration_seconds", 0) or 0)
    stop = float(end) if end not in (None, "") else duration
    of = f" of {fmt_clock(duration)}" if duration else ""
    return f"Analysed {fmt_clock(float(start or 0))}–{fmt_clock(stop) if stop else 'the end'}{of}"
