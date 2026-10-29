"""How the overlay is drawn — one spec for the player and the rendered video
(memory/61).

The browser player (``static/js/track_overlay.js``) and the GPU renderer
(``sagemaker_backend/inference.py`` ``_annotate_video``) are two renderers of
the same picture. Neither keeps its own colours, sizes or timings: both read
``STYLE`` from the payload they are given, so a change here changes both.

The rules both renderers follow, beyond the numbers in ``STYLE``:

- **Track colour**: HSL(id × ``hue_step`` mod 360, ``sat``%, ``light``%).
- **Box**: ``box_width`` px stroke in the track colour. A lost (predicted) row
  is dashed (``dash``) at ``lost_alpha``. Label above the box (below when it
  would leave the frame): a filled chip in the track colour with dark text,
  "id", then " · species" (Species layer), then " · lost N.N s" (seconds since
  the track's last detection), then " · in R · N.N s" while an in-reference
  interaction is active (Interactions layer; R = the reference's number).
- **Trail**: the track's centres over the last ``trail_seconds``; a gap over
  ``trail_gap_seconds`` breaks the line.
- **ROI**: dashed outline in ``roi_rgba``. **Reference object**: dashed outline in
  ``reference_rgba`` with its number in a small dark chip at the top-left
  corner, shrunk to fit, left off if it can't fit.
- **Event** (Events layer): a chip "EXIT 3" / "ENTER 3" / "STAY 3" (+ " · at
  edge") in the action's colour, under the insect's box (or at the reference
  when the insect is not in the frame), and the reference re-stroked solid in
  that colour (``glow_width``). Shown for ``event_seconds`` from an exit, for
  ``event_seconds`` before an enter, and for the whole of a stay.
- **Interaction** (Interactions layer), while the spell lasts: in a reference →
  the reference re-stroked solid in the insect's colour (``in_reference_width``);
  near → a line between the two insects' centres (``near_rgba``,
  ``near_width``) with a chip "near N.N bl" at its middle.
- **Clock** (rendered video only, Clock layer): top-left chip, clip time
  "mm:ss.s" then the timestamp when the clip's start is known.
"""

STYLE = {
    "v": 1,
    "hue_step": 137.508,
    "sat": 85,
    "light": 58,
    "box_width": 2,
    "dash": [5, 4],
    "lost_alpha": 0.7,
    "label_font_px": 12,
    "label_height": 16,
    "label_pad": 4,
    "label_text_rgba": [17, 24, 39, 1.0],
    "trail_seconds": 2.0,
    "trail_gap_seconds": 1.0,
    "trail_width": 1.5,
    "trail_alpha": 0.6,
    "region_width": 1,
    "roi_rgba": [74, 222, 128, 0.8],
    "reference_rgba": [255, 255, 255, 0.55],
    "region_label_px": 11,
    "event_seconds": 1.5,
    "event_rgba": {
        "exit": [3, 105, 161, 1.0],
        "enter": [180, 83, 9, 1.0],
        "stay": [109, 40, 217, 1.0],
    },
    "glow_width": 3,
    "chip_font_px": 12,
    "chip_text_rgba": [255, 255, 255, 1.0],
    "in_reference_width": 2.5,
    "near_rgba": [253, 230, 138, 1.0],
    "near_width": 2,
}

# Layers a viewer can turn on and off, and their defaults in the player.
LAYERS = ("boxes", "species", "trails", "lost", "regions", "events", "interactions", "clock")
DEFAULT_LAYERS = {"boxes": True, "species": True, "trails": True, "lost": True,
                  "regions": False, "events": True, "interactions": True, "clock": False}


def track_rgb(track_id):
    """The track colour as (r, g, b) 0–255 — the HSL rule above, for Python
    renderers and tests."""
    import colorsys

    hue = (float(track_id) * STYLE["hue_step"]) % 360
    r, g, b = colorsys.hls_to_rgb(hue / 360, STYLE["light"] / 100, STYLE["sat"] / 100)
    return round(r * 255), round(g * 255), round(b * 255)


def layers_from(request_values):
    """The layers a render was asked for. From the render dialog (it sends
    ``layers``) a layer is on only when ticked — a form leaves unticked boxes
    out. Without that marker, anything not named keeps its default."""
    from_form = "layers" in request_values
    out = dict(DEFAULT_LAYERS)
    for name in LAYERS:
        if name in request_values:
            out[name] = str(request_values.get(name)).lower() in ("1", "true", "on", "yes")
        elif from_form:
            out[name] = False
    return out
