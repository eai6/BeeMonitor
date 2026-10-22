"""Dict lookup by key — Django templates cannot index a dict by a variable."""

from django import template

register = template.Library()


@register.filter
def get(mapping, key):
    """``mapping|get:key`` with the key coerced to str (pks arrive as UUID/int)."""
    if not mapping:
        return {}
    return mapping.get(str(key), {})


@register.filter
def gpu_hours(seconds):
    """Seconds of GPU time as hours, the unit users see (``6712.5`` -> ``1.86``)."""
    try:
        hours = float(seconds) / 3600
    except (TypeError, ValueError):
        return "--"
    return f"{hours:.2f}" if hours < 10 else f"{hours:.1f}"
