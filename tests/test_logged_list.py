# pyflyby/test_logged_list.py

# License for THIS FILE ONLY: CC0 Public Domain Dedication
# http://creativecommons.org/publicdomain/zero/1.0/

# Tests for pyflyby._py.LoggedList, the list substituted for sys.argv.  It must
# behave exactly like a list, while additionally tracking which items were
# never accessed.  Most tests therefore compare it against a plain list.

import copy
import inspect
import operator
# Safe: the only data unpickled here is what this test just pickled itself.
import pickle

import pytest

from   pyflyby._py              import LoggedList


def _raw(ll):
    # Contents without going through the overrides, so that inspecting a
    # LoggedList in a test is not itself an access.
    return list.copy(ll)


def _check_aligned(ll):
    # The core invariant: ``_unaccessed`` mirrors the items position for
    # position, each entry being the item itself or the accessed sentinel.  A
    # mutation that updates one but not the other breaks this, and the unused
    # arguments are then misreported.
    items = _raw(ll)
    assert len(ll._unaccessed) == len(items), (
        "desynced: %d items, %d tracking slots" % (len(items), len(ll._unaccessed)))
    for i, (item, tracked) in enumerate(zip(items, ll._unaccessed)):
        assert tracked is LoggedList._ACCESSED or tracked == item, (
            "misaligned at %d: item %r, tracking %r" % (i, item, tracked))


# ---------------------------------------------------------------------------
# Interface: what LoggedList inherits, and what its overrides look like.
# ---------------------------------------------------------------------------

# Left to ``list`` because they neither mutate nor need to mark anything.
# Dunders are listed too: filtering them out by name would blind this to
# exactly the methods most likely to read the raw array (__add__, __mul__).
_TRACKING_NEUTRAL = {
    "count",                                       # reports a tally only
    "__len__",                                     # a count, not a read
    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",  # see below
    "__hash__",                                    # None, as on list
    "__reduce_ex__", "__getstate__",               # defer to __reduce__
    "__class__", "__class_getitem__", "__init_subclass__", "__new__",
    "__subclasshook__", "__dir__", "__format__", "__getattribute__",
    "__setattr__", "__delattr__", "__sizeof__",
}


def test_inherits_only_tracking_neutral_methods():
    # Anything not overridden operates on the underlying storage directly,
    # silently skipping the tracking.  This is the tripwire for a method (e.g.
    # one added by a future Python) slipping through untracked.
    inherited = sorted(name for name in dir(list)
                       if name not in vars(LoggedList)
                       and name not in _TRACKING_NEUTRAL)
    assert inherited == [], (
        "inherited rather than overridden, so untracked: %s" % (inherited,))


def _unannotated(sig):
    # LoggedList's methods are annotated; the list methods they mirror are C
    # builtins that cannot be, so types are the one part not comparable.
    return sig.replace(
        parameters=[p.replace(annotation=p.empty) for p in sig.parameters.values()],
        return_annotation=inspect.Signature.empty)


def test_method_signatures_match_list():
    # The overrides must be indistinguishable from what they replace: parameter
    # names and positional-only markers decide which calls are legal
    # (``list.append`` takes no ``object=`` keyword), and a drifted default
    # would silently change behavior.
    mismatched = {}
    for name in sorted(n for n in dir(list)
                       if not n.startswith("_") and n in vars(LoggedList)):
        actual = _unannotated(inspect.signature(getattr(LoggedList, name)))
        expected = _unannotated(inspect.signature(getattr(list, name)))
        if actual != expected:
            mismatched[name] = "%s != %s" % (actual, expected)
    assert mismatched == {}


# ---------------------------------------------------------------------------
# Behaves like a list: same return value, same resulting contents, same
# exceptions -- with the tracking invariant intact afterwards.
# ---------------------------------------------------------------------------

def _iadd(x):
    x += [3, 4]      # real +=, must return self rather than None
    return x


def _imul(x):
    x *= 3
    return x


_OPERATIONS = {
    "append":            ([1, 2, 3],       lambda x: x.append(4)),
    "clear":             ([1, 2, 3],       lambda x: x.clear()),
    "copy":              ([1, 2, 3],       lambda x: x.copy()),
    "count":             ([1, 2, 2],       lambda x: x.count(2)),
    "extend-list":       ([1, 2],          lambda x: x.extend([3, 4])),
    "extend-iter":       ([1, 2],          lambda x: x.extend(iter([3, 4]))),
    "index":             (["a", "b", "a"], lambda x: x.index("b")),
    "index-start-stop":  (["a", "b", "a"], lambda x: x.index("a", 1, 3)),
    "insert":            ([1, 2, 3],       lambda x: x.insert(1, 99)),
    "insert-negative":   ([1, 2, 3],       lambda x: x.insert(-9, 99)),
    "pop":               ([1, 2, 3],       lambda x: x.pop()),
    "pop-index":         ([1, 2, 3],       lambda x: x.pop(0)),
    "remove":            ([1, 2, 3, 2],    lambda x: x.remove(2)),
    "reverse":           ([1, 2, 3],       lambda x: x.reverse()),
    "sort":              ([3, 1, 2],       lambda x: x.sort()),
    "sort-key-reverse":  (["bbb", "a"],    lambda x: x.sort(key=len, reverse=True)),
    "getitem":           ([1, 2, 3],       lambda x: x[2]),
    "getitem-negative":  ([1, 2, 3],       lambda x: x[-1]),
    "getitem-slice":     ([1, 2, 3, 4],    lambda x: x[1:3]),
    "getitem-slice-step": ([1, 2, 3, 4],   lambda x: x[::-1]),
    "setitem":           ([1, 2, 3],       lambda x: x.__setitem__(1, 99)),
    "setitem-slice":     ([1, 2, 3, 4],    lambda x: x.__setitem__(slice(1, 3), [8, 9])),
    "setitem-slice-grow": ([1, 2, 3],      lambda x: x.__setitem__(slice(1, 2), iter("ab"))),
    "setitem-slice-empty": ([1, 2, 3],     lambda x: x.__setitem__(slice(1, 2), iter([]))),
    "setitem-slice-step": ([1, 2, 3],      lambda x: x.__setitem__(slice(None, None, 2), iter("ab"))),
    "delitem":           ([1, 2, 3],       lambda x: x.__delitem__(1)),
    "delitem-slice":     ([1, 2, 3, 4],    lambda x: x.__delitem__(slice(1, 3))),
    "len":               ([1, 2, 3],       lambda x: len(x)),
    "iter":              ([1, 2, 3],       lambda x: list(iter(x))),
    "reversed":          ([1, 2, 3],       lambda x: list(reversed(x))),
    "contains":          ([1, 2, 3],       lambda x: (2 in x, 9 in x)),
    "add":               ([1, 2],          lambda x: x + [3]),
    "mul":               ([1, 2],          lambda x: x * 3),
    "rmul":              ([1, 2],          lambda x: 3 * x),
    "iadd":              ([1, 2],          _iadd),
    "imul":              ([1, 2],          _imul),
    "imul-zero":         ([1, 2],          lambda x: x.__imul__(0)),
    "repr":              ([1, 2, 3],       lambda x: repr(x)),
    "str":               ([1, 2, 3],       lambda x: str(x)),
    "self-extend":       ([1, 2],          lambda x: x.extend(x)),
    "self-slice-assign": ([1, 2],          lambda x: x.__setitem__(slice(None), x)),
}


@pytest.mark.parametrize("seed, op", list(_OPERATIONS.values()),
                         ids=list(_OPERATIONS.keys()))
def test_operation_matches_list(seed, op):
    ref, ll = list(seed), LoggedList(seed)
    assert op(ref) == op(ll)                 # same return value...
    assert _raw(ll) == ref                   # ...and same contents
    _check_aligned(ll)


_RAISING = {
    "index-missing":    ([1, 2],    lambda x: x.index(9),          ValueError),
    "remove-missing":   ([1, 2],    lambda x: x.remove(9),         ValueError),
    "remove-empty":     ([],        lambda x: x.remove(9),         ValueError),
    "pop-empty":        ([],        lambda x: x.pop(),             IndexError),
    "pop-out-of-range": ([1, 2],    lambda x: x.pop(5),            IndexError),
    "getitem-oob":      ([1, 2],    lambda x: x[5],                IndexError),
    "setitem-oob":      ([1, 2],    lambda x: x.__setitem__(5, 0), IndexError),
    "delitem-oob":      ([1, 2],    lambda x: x.__delitem__(5),    IndexError),
    "getitem-str-idx":  ([1, 2],    lambda x: x["a"],              TypeError),
    "setitem-ext-size": ([1, 2, 3], lambda x: x.__setitem__(slice(None, None, 2), [1]),
                                                                   ValueError),
}


@pytest.mark.parametrize("seed, op, exc_type", list(_RAISING.values()),
                         ids=list(_RAISING.keys()))
def test_operation_raises_like_list(seed, op, exc_type):
    # The message matters as much as the type: ``remove`` once reported
    # "list.index(x): x not in list" because it delegated to index.
    with pytest.raises(exc_type) as ref_exc:
        op(list(seed))
    ll = LoggedList(seed)
    with pytest.raises(exc_type) as exc:
        op(ll)
    assert str(exc.value) == str(ref_exc.value)
    _check_aligned(ll)                       # a failure must not half-apply


_COMPARISONS = [operator.eq, operator.ne, operator.lt,
                operator.le, operator.gt, operator.ge]


@pytest.mark.parametrize("left, right", [
    ([1, 2, 3], [1, 2, 3]),      # equal
    ([1, 2, 3], [1, 2, 4]),      # differs in the last element
    ([1, 2, 4], [1, 2, 3]),      # ...and the other way around
    ([1, 2],    [1, 2, 3]),      # proper prefix (shorter sorts first)
    ([1, 2, 3], [1, 2]),         # ...and the other way around
    ([2],       [1, 2, 3]),      # first element decides, whatever the length
    ([],        [1]),            # empty vs non-empty
    (["a", "b"], ["a", "c"]),    # non-numeric elements
])
def test_comparisons_match_list(left, right):
    # Unequal operands matter most: they exercise the orderings that comparing
    # equal lists barely touches.  Plain lists supply the expected answers.
    for op in _COMPARISONS:
        expected = op(left, right)
        assert op(LoggedList(left), right) == expected
        assert op(LoggedList(left), LoggedList(right)) == expected
        # A plain list on the left goes through the reflected operator.
        assert op(left, LoggedList(right)) == expected


def test_unhashable_like_list():
    for x in ([1, 2, 3], LoggedList([1, 2, 3])):
        with pytest.raises(TypeError):
            hash(x)


def test_copy_returns_plain_list():
    c = LoggedList([1, 2, 3]).copy()
    assert isinstance(c, list) and not isinstance(c, LoggedList)


# ---------------------------------------------------------------------------
# Access tracking, which a plain list has none of.
# ---------------------------------------------------------------------------

def _sort_mutating_key(x):
    def key(v):
        x.append(v)
        return v
    return x.sort(key=key)


# (seed, operation, the items expected to remain unaccessed afterwards)
_TRACKING = {
    "getitem":       (["a", "b", "c"], lambda x: x[1],            ["a", "c"]),
    "getitem-slice": (["a", "b", "c"], lambda x: x[1:2],          ["a", "c"]),
    "setitem":       (["a", "b", "c"], lambda x: x.__setitem__(1, "x"), ["a", "c"]),
    "setitem-neg":   (["a", "b", "c"], lambda x: x.__setitem__(-1, "z"), ["a", "b"]),
    # Assigning counts as an access, for slices as for a single index.
    "setitem-slice": (["a", "b", "c"], lambda x: x.__setitem__(slice(1, 2), iter("xy")),
                                                                  ["a", "c"]),
    "index":         (["a", "b", "c"], lambda x: x.index("b"),    ["a", "c"]),
    "iter":          (["a", "b"],      lambda x: list(x),         []),
    "reversed":      (["a", "b"],      lambda x: list(reversed(x)), []),
    "repr":          (["a", "b"],      lambda x: repr(x),         []),
    "copy":          (["a", "b"],      lambda x: x.copy(),        []),
    "clear":         (["a", "b"],      lambda x: x.clear(),       []),
    # ``+`` and ``*`` read every element; without marking, a script doing
    # ``args = sys.argv + extra`` would be told it never used its arguments.
    "add":           (["a", "b"],      lambda x: x + ["z"],       []),
    "mul":           (["a", "b"],      lambda x: x * 2,           []),
    "rmul":          (["a", "b"],      lambda x: 2 * x,           []),
    # Neither counting, measuring nor comparing is "using" an argument.
    "len":           (["a", "b"],      lambda x: len(x),          ["a", "b"]),
    "count":         (["a", "b"],      lambda x: x.count("a"),    ["a", "b"]),
    "compare":       (["a", "b"],      lambda x: x != ["a", "c"], ["a", "b"]),
    "sort":          ([3, 1, 2],       lambda x: x.sort(),        [1, 2, 3]),
    "append":        (["a"],           lambda x: x.append("b"),   ["a"]),
}


@pytest.mark.parametrize("seed, op, expected", list(_TRACKING.values()),
                         ids=list(_TRACKING.keys()))
def test_operation_marks_expected_items_accessed(seed, op, expected):
    ll = LoggedList(seed)
    op(ll)
    assert ll.unaccessed == expected
    _check_aligned(ll)


def test_slice_assignment_takes_any_iterable():
    # The new length must come from the value -- which the assignment itself
    # would consume, so it has to be materialized first -- not from the slice.
    for value, expected in [(iter(["x"]),      [1, "x", 3]),
                            (iter(["a", "b"]), [1, "a", "b", 3]),
                            (iter([]),         [1, 3])]:
        ll = LoggedList([1, 2, 3])
        ll[1:2] = value
        assert _raw(ll) == expected
        _check_aligned(ll)


def test_sort_with_mutating_key_raises_like_list():
    # list.sort refuses to sort a list its own key mutated; silently discarding
    # the mutation (as a blind write-back would) loses data.
    def message(seq):
        with pytest.raises(ValueError) as exc:
            _sort_mutating_key(seq)
        return str(exc.value)
    assert message(LoggedList([3, 1, 2])) == message([3, 1, 2])


# ---------------------------------------------------------------------------
# copy/pickle: list subclasses are rebuilt without running __init__, so the
# tracking state has to be carried explicitly.
# ---------------------------------------------------------------------------

_CLONES = {
    "copy":     copy.copy,
    "deepcopy": copy.deepcopy,
    "pickle":   lambda x: pickle.loads(pickle.dumps(x)),
}


class _SubList(LoggedList):
    def __init__(self, items, extra=None):
        super().__init__(items)
        self.extra = extra


@pytest.mark.parametrize("clone", list(_CLONES.values()), ids=list(_CLONES))
def test_round_trip_preserves_tracking(clone):
    ll = LoggedList(["a", "b", "c"])
    ll[1]                                    # access exactly one item
    c = clone(ll)
    assert isinstance(c, LoggedList)
    assert _raw(c) == ["a", "b", "c"]
    assert c.unaccessed == ["a", "c"]
    _check_aligned(c)
    c[0]                                     # clone tracks independently
    assert (c.unaccessed, ll.unaccessed) == (["c"], ["a", "c"])


@pytest.mark.parametrize("clone", list(_CLONES.values()), ids=list(_CLONES))
def test_round_trip_preserves_subclass_attributes(clone):
    # A plain list subclass keeps its instance __dict__; defining __reduce__
    # must not lose it.
    c = clone(_SubList(["a"], extra="hi"))
    assert isinstance(c, _SubList) and c.extra == "hi"
    _check_aligned(c)


@pytest.mark.parametrize("clone", [copy.deepcopy, _CLONES["pickle"]],
                         ids=["deepcopy", "pickle"])
def test_round_trip_self_referential(clone):
    # Only the deep clones have a cycle to resolve (a shallow copy keeps
    # pointing at the original, as it does for a plain list).  Passing the
    # items as reconstructor *arguments* would recurse forever, since the
    # object cannot be memoized until they are all built.
    ll = LoggedList(["a"])
    ll.append(ll)
    c = clone(ll)
    assert len(c) == 2 and c[1] is c
    _check_aligned(c)


@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy], ids=["copy", "deepcopy"])
def test_round_trip_with_changed_init_signature(clone):
    # Rebuilding via ``type(self)(items)`` would break this subclass.
    class Sub(LoggedList):
        def __init__(self, a, b):
            super().__init__([a, b])
    c = clone(Sub(1, 2))
    assert _raw(c) == [1, 2]
    _check_aligned(c)


def test_pickle_all_protocols():
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        ll = LoggedList(["a", "b", "c"])
        ll[1]
        c = pickle.loads(pickle.dumps(ll, proto))
        assert c.unaccessed == ["a", "c"], "protocol %d" % (proto,)
        _check_aligned(c)
