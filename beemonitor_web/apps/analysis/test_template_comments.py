"""No multi-line {# #} in any template.

Django's {# #} comment is SINGLE-LINE only. Spread one over two lines and it is
not parsed as a comment at all — the text renders verbatim to the user. It
passes `manage.py check`, passes every view test that only asserts what SHOULD
be on the page, and looks correct in the editor.

It shipped three times in one evening: a note about the clip id appeared inside
the video player's header, and two more sat on the job detail page and the
analysis config panel. Hence a test that reads the templates themselves.
"""

import re
from pathlib import Path

from django.conf import settings
from django.test import TestCase

# A {# that never meets a #} on the same line.
UNCLOSED = re.compile(r"\{#(?![^\n]*#\})")


def template_files():
    roots = [Path(d) for d in settings.TEMPLATES[0]["DIRS"]]
    roots.append(Path(settings.BASE_DIR) / "apps")
    for root in roots:
        if root.exists():
            yield from root.rglob("*.html")


class TemplateCommentTests(TestCase):
    def test_no_comment_tag_spans_more_than_one_line(self):
        offenders = []
        for path in template_files():
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                if UNCLOSED.search(line):
                    offenders.append(f"{path}:{lineno}")

        self.assertEqual(offenders, [], (
            "Multi-line {# #} renders as visible text. Use "
            "{% comment %}...{% endcomment %} instead:\n  "
            + "\n  ".join(offenders)))

    def test_the_check_would_catch_a_regression(self):
        """The detector, not the templates — a guard that cannot fail is worse
        than no guard."""
        self.assertTrue(UNCLOSED.search("{# starts here"))
        self.assertTrue(UNCLOSED.search("  {# wraps onto"))
        self.assertFalse(UNCLOSED.search("{# fine on one line #}"))
        self.assertFalse(UNCLOSED.search("{% comment %} fine {% endcomment %}"))
        self.assertFalse(UNCLOSED.search("<p>nothing to see</p>"))
