"""Dict lookup by key — Django templates cannot index a dict by a variable."""

from django import template

register = template.Library()


@register.filter
def get(mapping, key):
    """``mapping|get:key`` with the key coerced to str (pks arrive as UUID/int)."""
    if not mapping:
        return {}
    return mapping.get(str(key), {})
