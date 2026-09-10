"""A local re-import shadows the module-level name for the WHOLE function.

`ExportProjectView.get` imported `io` forty lines below its own `io.BytesIO()`
call. Python binds `io` as a local for the entire function body, so the earlier
line raised UnboundLocalError and dataset export failed for everyone, every
time — with no test touching it.

The pattern is easy to reintroduce (adding a convenience import inside a `try`)
and produces a runtime error nowhere near the line that caused it.
"""

import ast
from pathlib import Path

from django.conf import settings
from django.test import SimpleTestCase


def _module_level_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names |= {a.asname or a.name for a in node.names}
    return names


def offenders(path):
    tree = ast.parse(path.read_text(errors="replace"))
    top = _module_level_names(tree)
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = {}
        for inner in ast.walk(fn):
            if inner is fn or not isinstance(inner, (ast.Import, ast.ImportFrom)):
                continue
            for a in inner.names:
                nm = a.asname or (a.name.split(".")[0]
                                  if isinstance(inner, ast.Import) else a.name)
                if nm in top:
                    local.setdefault(nm, inner.lineno)
        for nm, imported_at in local.items():
            for node in ast.walk(fn):
                if (isinstance(node, ast.Name) and node.id == nm
                        and isinstance(node.ctx, ast.Load)
                        and node.lineno < imported_at):
                    yield (f"{path.relative_to(settings.BASE_DIR)}:{node.lineno} "
                           f"{fn.name}() uses {nm!r} before re-importing it at "
                           f"line {imported_at}")
                    break


class ShadowedImportTests(SimpleTestCase):
    def test_no_function_uses_a_name_it_later_re_imports(self):
        found = []
        for path in Path(settings.BASE_DIR, "apps").rglob("*.py"):
            if "migrations" in path.parts:
                continue
            try:
                found += list(offenders(path))
            except SyntaxError:
                continue

        self.assertEqual(found, [], (
            "A local re-import makes the name local for the whole function, so "
            "an earlier use raises UnboundLocalError at runtime:\n  "
            + "\n  ".join(found)))
