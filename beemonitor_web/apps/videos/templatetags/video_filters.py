import re

from django import template

register = template.Library()

_NATALIES_RE = re.compile(r"natalies?", re.IGNORECASE)


@register.filter
def sanitize_name(value):
    """Replace personal site names with generic identifiers for display."""
    if not value:
        return value
    return _NATALIES_RE.sub("SiteA", str(value))
