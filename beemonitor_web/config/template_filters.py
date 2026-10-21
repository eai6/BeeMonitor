"""Template filters every template gets without ``{% load %}``.

``js`` is for a JSON string the view already built, written into a <script>:

    var points = {{ map_points|js }};

``|safe`` there let any user-controlled string in the data (a unit's name, an
annotation class) close the script with ``</script>`` and run its own. ``js``
escapes <, > and & as JSON \\u escapes, the way Django's ``json_script`` does,
so the value parses the same and can't leave the script.
"""

from django import template
from django.utils.html import _json_script_escapes
from django.utils.safestring import mark_safe

register = template.Library()


@register.filter
def js(value):
    return mark_safe(str(value).translate(_json_script_escapes))
