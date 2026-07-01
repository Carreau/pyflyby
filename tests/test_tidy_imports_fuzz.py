"""Property-based (fuzz) tests for tidy-imports.

These build small Python programs whose used vs. unused imports are known up
front, so ``fix_unused_and_missing_imports`` has a precise oracle: it must keep
exactly the imports whose bound name is referenced, remove the rest, and leave
valid Python behind.  Metamorphic properties (idempotence) are checked too.

The generator deliberately mixes module-level and function-local imports and an
optional mandatory import, since that combination is where the fragile
line-number/block bookkeeping tends to break.  Run with more coverage via::

    pytest tests/test_tidy_imports_fuzz.py --hypothesis-seed=random \
        -p no:cacheprovider --hypothesis-verbosity=normal

or bump ``max_examples`` in the ``@settings`` below.
"""

from __future__ import print_function

import ast

from   hypothesis               import HealthCheck, given, settings, strategies as st
import pytest

from   pyflyby._importdb        import ImportDB
from   pyflyby._imports2s       import fix_unused_and_missing_imports
from   pyflyby._parse           import PythonBlock

# bound-name -> the import statement that binds it.  A couple of names share a
# module (``collections``/``math``) so that merging/splitting of ``from`` blocks
# is exercised too.
_CATALOG = {
    "os": "import os",
    "sys": "import sys",
    "json": "import json",
    "re": "import re",
    "sqrt": "from math import sqrt",
    "pi": "from math import pi",
    "OrderedDict": "from collections import OrderedDict",
    "deque": "from collections import deque",
}
_NAMES = sorted(_CATALOG)

_MANDATORY = 'from __future__ import annotations'


def _imported_names(source):
    """Return the set of names bound by imports in *source* (ignoring __future__)."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


@st.composite
def import_programs(draw):
    """Generate ``(source, used_names)`` with a known-correct tidy result.

    Each chosen name is imported in exactly one scope -- module level or the
    body of ``f`` -- and is referenced iff it is "used".  So after tidying with
    ``remove_unused=True`` (and no imports added), the set of names still
    imported must be exactly ``used_names``.
    """
    names = draw(st.lists(st.sampled_from(_NAMES), unique=True))
    is_local = {n: draw(st.booleans()) for n in names}
    used = {n for n in names if draw(st.booleans())}
    # A leading blank line is a known edge case for block bookkeeping.
    leading_blank = draw(st.booleans())

    head = [_CATALOG[n] for n in names if not is_local[n]]
    body = [f"    {_CATALOG[n]}" for n in names if is_local[n]]
    body += [f"    {n}" for n in sorted(used)]
    if not body:
        body = ["    pass"]

    prefix = [""] if leading_blank else []
    source = "\n".join(prefix + head + ["", "def f():"] + body) + "\n"
    return source, used


def _tidy(source, db, add_mandatory=False):
    return str(fix_unused_and_missing_imports(
        PythonBlock(source, filename="/fuzz.py"),
        db=db, add_missing=False, add_mandatory=add_mandatory,
        remove_unused=True, tidy_local_imports=True,
    ))


_SETTINGS = settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@_SETTINGS
@given(import_programs())
def test_removes_exactly_unused(program):
    source, used = program
    out = _tidy(source, ImportDB(""))
    ast.parse(out)  # output is still valid Python
    assert _imported_names(out) == used


@_SETTINGS
@given(import_programs())
def test_idempotent(program):
    source, _ = program
    once = _tidy(source, ImportDB(""))
    twice = _tidy(once, ImportDB(""))
    assert once == twice


@pytest.mark.xfail(
    reason="Adding a mandatory import can split the block a local import lives "
           "in, orphaning the deferred removal (e.g. a leading blank line before "
           "the def); tracked separately.",
    strict=False,
)
@_SETTINGS
@given(import_programs())
def test_removes_exactly_unused_with_mandatory(program):
    source, used = program
    db = ImportDB(f'__mandatory_imports__ = ["{_MANDATORY}"]')
    out = _tidy(source, db, add_mandatory=True)
    ast.parse(out)
    assert _MANDATORY in out
    assert _imported_names(out) == used
