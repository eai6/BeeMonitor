"""Small display helpers for the batch results page."""

from django import template

from apps.pipelines import failures

register = template.Library()


@register.filter
def rsub(value, arg):
    """``arg - value`` — the remainder of a percentage bar."""
    try:
        return max(0, float(arg) - float(value))
    except (TypeError, ValueError):
        return 0


@register.filter
def duration_min(seconds):
    """Seconds as a human duration; a batch's GPU time is minutes, not 2461.0."""
    try:
        seconds = float(seconds or 0)
    except (TypeError, ValueError):
        return "—"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


@register.filter
def failure_title(message):
    """The short cause for an error message — what a row shows instead of a
    500-character platform string."""
    return failures.classify(message)["title"]
