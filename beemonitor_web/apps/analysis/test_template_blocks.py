"""Markup after {% endblock %} is silently discarded.

A child template's content outside a block is not an error and not a warning —
Django's inheritance simply drops it. A <script> that lands there renders
nothing, so the page looks right and every button on it is dead. That is
exactly what happened to the batch page's clip viewer.
"""

import re
from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase

BLOCK = re.compile(r"{%\s*(block|endblock)\b[^%]*%}")
EXTENDS = re.compile(r"{%\s*extends\b")
# Content that would actually render if it were inside a block.
RENDERS = re.compile(r"<[a-zA-Z]|{{")


COMMENT = re.compile(r"{%\s*comment\b.*?{%\s*endcomment\s*%}", re.S)
SHORT_COMMENT = re.compile(r"{#.*?#}", re.S)


def _blank_comments(text):
    """Replace comment bodies with blank lines, keeping line numbers intact."""
    def _blank(match):
        return "\n" * match.group(0).count("\n")

    return SHORT_COMMENT.sub(_blank, COMMENT.sub(_blank, text))


def template_files():
    roots = [Path(settings.BASE_DIR)]
    for root in roots:
        for path in root.rglob("*.html"):
            if "node_modules" not in path.parts and ".venv" not in path.parts:
                yield path


class TemplateBlockTests(SimpleTestCase):
    def test_no_child_template_renders_markup_outside_a_block(self):
        offenders = []
        for path in template_files():
            text = path.read_text(errors="replace")
            if not EXTENDS.search(text):
                continue                      # a base template renders its own body

            # Blank out comment bodies first: a {% comment %} explaining the
            # page often contains example markup, and that renders nothing.
            text = _blank_comments(text)

            depth, tail = 0, []
            for line_no, line in enumerate(text.splitlines(), start=1):
                opens = len(re.findall(r"{%\s*block\b", line))
                closes = len(re.findall(r"{%\s*endblock\b", line))
                # A line that both opens and closes ({% block title %}x{% endblock %})
                # never leaves the block, so judge the line at its starting depth.
                if depth == 0 and not opens and RENDERS.search(line) \
                        and not line.strip().startswith("{%"):
                    tail.append(line_no)
                depth += opens - closes
            if tail:
                offenders.append(f"{path.relative_to(settings.BASE_DIR)}:{tail[0]}")

        self.assertEqual(offenders, [], (
            "Markup outside {% block %} in a child template is dropped silently — "
            "move it inside the block it belongs to:\n  " + "\n  ".join(offenders)))
