"""Django apps package.

This file exists so ``apps`` is a regular package rather than a namespace one.
Without it, ``manage.py test`` discovered only the 104 tests in the root
``tests.py``: unittest's discovery calls ``os.path.abspath(module.__file__)``,
and a namespace package's ``__file__`` is None, so it silently skipped
everything under ``apps/`` (and errored outright on an explicit
``manage.py test apps.<label>``). With it, discovery finds 394.
"""
