# pyflyby/test_logged_list.py

# License for THIS FILE ONLY: CC0 Public Domain Dedication
# http://creativecommons.org/publicdomain/zero/1.0/

# Tests for pyflyby._py.LoggedList, the list proxy substituted for sys.argv.
# Since sys.argv is a plain list, LoggedList must expose the same public API and
# behave identically to a list for every operation, while additionally tracking
# which items were never accessed.

from __future__ import print_function

import copy
import inspect
import operator
# Safe: the only data unpickled here is what this test just pickled itself.
import pickle
import random

import pytest

from   pyflyby._py              import LoggedList


def _raw(ll):
    # Read a LoggedList's contents without going through its overrides, so that
    # inspecting it in a test does not itself count as accessing the items.
    return list.copy(ll)


def _assert_tracking_aligned(ll):
    # The core invariant: ``_unaccessed`` mirrors the items position for
    # position, each entry being either the item itself (not yet accessed) or
    # the accessed sentinel.  Any mutation that updates the items but not the
    # tracking state -- or updates them in a different order -- breaks this,
    # which then misreports which arguments went unused.
    items = _raw(ll)
    assert len(ll._unaccessed) == len(items), (
        "tracking desynced: %d items but %d tracking slots"
        % (len(items), len(ll._unaccessed)))
    for i, (item, tracked) in enumerate(zip(items, ll._unaccessed)):
        assert tracked is LoggedList._ACCESSED or tracked == item, (
            "tracking misaligned at %d: item %r but tracking holds %r"
            % (i, item, tracked))


# Methods deliberately left to ``list``, each because it neither mutates the
# list nor needs to mark anything accessed, so inheriting it is exactly
# equivalent to forwarding to ``super()``.  Dunders are included deliberately:
# an earlier version of this test filtered them out by name, which made it
# blind to exactly the ones most likely to read the raw array.
_TRACKING_NEUTRAL = {
    "count",             # reads items but reports only a tally
    "__class__", "__class_getitem__", "__init_subclass__",
    "__new__", "__subclasshook__", "__dir__", "__format__",
    "__getattribute__", "__setattr__", "__delattr__", "__sizeof__",
    "__reduce_ex__",     # defers to __reduce__, which is overridden
    "__hash__",          # None on list; LoggedList is unhashable the same way
    "__len__",           # a count, not a read of any item
    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__",
                         # comparing is not "using" an argument; see
                         # test_logged_list_unequal_comparison_does_not_mark
    "__getstate__",      # unused: __reduce__ supplies the state
}


def test_logged_list_inherits_only_tracking_neutral_methods():
    # ``LoggedList`` is a real ``list`` subclass, so any method it does *not*
    # override operates on the underlying storage directly -- silently skipping
    # the access tracking and desynchronizing ``_unaccessed``.  A method may
    # therefore only be inherited if it is listed above as tracking-neutral;
    # this test is the tripwire for anything else (e.g. a list method added by
    # a future Python) quietly slipping through untracked.  Dunders count:
    # ``__add__`` and ``__mul__`` read every element, so inheriting those would
    # silently lose tracking.
    inherited = sorted(name for name in dir(list)
                       if name not in vars(LoggedList)
                       and name not in _TRACKING_NEUTRAL)
    assert inherited == [], (
        "LoggedList inherits these list methods rather than overriding them, "
        "so they bypass access tracking: %s" % (inherited,))


def _unannotated(sig):
    # Strip type annotations from a signature.  ``LoggedList``'s methods are
    # annotated; the ``list`` methods they mirror are C builtins and cannot
    # carry annotations at all, so the types are the one part of the signature
    # that cannot be compared against the reference.
    return sig.replace(
        parameters=[p.replace(annotation=p.empty)
                    for p in sig.parameters.values()],
        return_annotation=inspect.Signature.empty)


@pytest.mark.parametrize("name", sorted(
    name for name in dir(list)
    if not name.startswith("_") and name in vars(LoggedList)))
def test_logged_list_method_signature_matches_list(name):
    # ``LoggedList`` stands in for ``sys.argv``, so its overrides must be
    # indistinguishable from the ``list`` methods they replace -- identical
    # signatures, not merely compatible ones.  Parameter names and
    # positional-only markers are part of that: they are what ``help()``,
    # ``inspect.signature`` and editor completions report, and they decide
    # which calls are legal (``list.append`` takes no ``object=`` keyword, so
    # neither may ours).  Defaults count too; a drifting default (say
    # ``reverse=True``) would silently change behavior.
    actual = _unannotated(inspect.signature(getattr(LoggedList, name)))
    expected = _unannotated(inspect.signature(getattr(list, name)))
    assert actual == expected, (
        "LoggedList.%s%s does not match list.%s%s"
        % (name, actual, name, expected))


# ---------------------------------------------------------------------------
# LoggedList must behave like a plain ``list`` for every operation, since it
# is substituted for ``sys.argv``.  The bulk of these are the same shape --
# apply an operation to a real ``list`` and to a ``LoggedList`` seeded with the
# same data, then check the return value and resulting contents match -- so
# they are driven by a single parametrized harness.  The few tests that assert
# LoggedList's *access-tracking* behavior (which a plain list does not have)
# are kept separate below.
# ---------------------------------------------------------------------------

def _iadd(x):
    x += [3, 4]      # exercises real += (must return self, not None)
    return x


def _imul(x):
    x *= 3           # exercises real *= (must return self, not None)
    return x


# Each case is (seed_items, operation).  The operation is applied to both a
# list and a LoggedList; both its return value and the container's resulting
# contents must match.
_LIST_OPERATIONS = {
    "append":           ([1, 2, 3],          lambda x: x.append(4)),
    "clear":            ([1, 2, 3],          lambda x: x.clear()),
    "copy":             ([1, 2, 3],          lambda x: x.copy()),
    "count-present":    ([1, 2, 2, 3, 2],    lambda x: x.count(2)),
    "count-absent":     ([1, 2, 2, 3, 2],    lambda x: x.count(9)),
    "extend-list":      ([1, 2],             lambda x: x.extend([3, 4])),
    "extend-iter":      ([1, 2],             lambda x: x.extend(iter([3, 4]))),
    "index":            (["a", "b", "c", "b"], lambda x: x.index("b")),
    "index-start":      (["a", "b", "c", "b"], lambda x: x.index("b", 2)),
    "index-start-stop": (["a", "b", "c", "b"], lambda x: x.index("b", 1, 4)),
    "insert":           ([1, 2, 3],          lambda x: x.insert(1, 99)),
    "pop-default":      ([1, 2, 3],          lambda x: x.pop()),
    "pop-index":        ([1, 2, 3],          lambda x: x.pop(0)),
    "remove":           ([1, 2, 3, 2],       lambda x: x.remove(2)),
    "reverse":          ([1, 2, 3],          lambda x: x.reverse()),
    "sort":             ([3, 1, 2],          lambda x: x.sort()),
    "sort-key-reverse": (["bbb", "a", "cc"], lambda x: x.sort(key=len, reverse=True)),
    "getitem":          ([1, 2, 3, 4],       lambda x: x[2]),
    "getitem-negative": ([1, 2, 3, 4],       lambda x: x[-1]),
    "getitem-slice":    ([1, 2, 3, 4],       lambda x: x[1:3]),
    "getitem-slice-step": ([1, 2, 3, 4],     lambda x: x[::-1]),
    "setitem":          ([1, 2, 3],          lambda x: x.__setitem__(1, 99)),
    "setitem-slice":    ([1, 2, 3, 4],       lambda x: x.__setitem__(slice(1, 3), [8, 9, 10])),
    "setitem-slice-iterator": ([1, 2, 3],    lambda x: x.__setitem__(slice(1, 2), iter(["x"]))),
    "setitem-slice-iterator-grow": ([1, 2, 3], lambda x: x.__setitem__(slice(1, 2), iter(["a", "b", "c"]))),
    "setitem-slice-iterator-empty": ([1, 2, 3], lambda x: x.__setitem__(slice(1, 2), iter([]))),
    "setitem-slice-step-iterator": ([1, 2, 3], lambda x: x.__setitem__(slice(None, None, 2), iter(["a", "b"]))),
    "delitem":          ([1, 2, 3],          lambda x: x.__delitem__(1)),
    "delitem-slice":    ([1, 2, 3, 4],       lambda x: x.__delitem__(slice(1, 3))),
    "len":              ([1, 2, 3],          lambda x: len(x)),
    "iter":             ([1, 2, 3],          lambda x: list(iter(x))),
    "reversed":         ([1, 2, 3],          lambda x: list(reversed(x))),
    "contains-present": ([1, 2, 3],          lambda x: 2 in x),
    "contains-absent":  ([1, 2, 3],          lambda x: 9 in x),
    "add":              ([1, 2],             lambda x: x + [3, 4]),
    "mul":              ([1, 2],             lambda x: x * 3),
    "rmul":             ([1, 2],             lambda x: 3 * x),
    "iadd":             ([1, 2],             _iadd),
    "imul":             ([1, 2],             _imul),
    "repr":             ([1, 2, 3],          lambda x: repr(x)),
    "str":              ([1, 2, 3],          lambda x: str(x)),
}


@pytest.mark.parametrize(
    "seed, op",
    list(_LIST_OPERATIONS.values()),
    ids=list(_LIST_OPERATIONS.keys()),
)
def test_logged_list_operation_matches_list(seed, op):
    ref = list(seed)
    ll = LoggedList(seed)
    # Same return value (compared before/independently of any mutation)...
    assert op(ref) == op(ll)
    # ...and same resulting contents (compared without relying on __eq__).
    assert _raw(ll) == ref
    _assert_tracking_aligned(ll)


_RAISING_OPERATIONS = {
    "index-missing":   ([1, 2],    lambda x: x.index(9),            ValueError),
    "index-empty":     ([],        lambda x: x.index(9),            ValueError),
    "remove-missing":  ([1, 2],    lambda x: x.remove(9),           ValueError),
    "remove-empty":    ([],        lambda x: x.remove(9),           ValueError),
    "pop-empty":       ([],        lambda x: x.pop(),               IndexError),
    "pop-out-of-range": ([1, 2],   lambda x: x.pop(5),              IndexError),
    "getitem-oob":     ([1, 2],    lambda x: x[5],                  IndexError),
    "setitem-oob":     ([1, 2],    lambda x: x.__setitem__(5, 0),   IndexError),
    "delitem-oob":     ([1, 2],    lambda x: x.__delitem__(5),      IndexError),
    "setitem-ext-size": ([1, 2, 3], lambda x: x.__setitem__(slice(None, None, 2), [1]),
                                                                    ValueError),
    "getitem-str-idx": ([1, 2],    lambda x: x["a"],                TypeError),
}


@pytest.mark.parametrize(
    "seed, op, exc_type",
    list(_RAISING_OPERATIONS.values()),
    ids=list(_RAISING_OPERATIONS.keys()),
)
def test_logged_list_operation_raises_like_list(seed, op, exc_type):
    # The exception *message* matters, not just the type: ``remove`` used to
    # report "list.index(x): x not in list" because it delegated to index,
    # which a bare pytest.raises(ValueError) could not see.
    with pytest.raises(exc_type) as ref_exc:
        op(list(seed))
    ll = LoggedList(seed)
    with pytest.raises(exc_type) as exc:
        op(ll)
    assert str(exc.value) == str(ref_exc.value)
    # A failed operation must leave the tracking state untouched, not
    # half-applied.
    _assert_tracking_aligned(ll)


def test_logged_list_copy_returns_plain_list():
    c = LoggedList([1, 2, 3]).copy()
    assert isinstance(c, list) and not isinstance(c, LoggedList)


@pytest.mark.parametrize("clone", [
    copy.copy,
    copy.deepcopy,
    lambda x: pickle.loads(pickle.dumps(x)),
], ids=["copy", "deepcopy", "pickle"])
def test_logged_list_round_trips(clone):
    # ``list`` subclasses are reconstructed without running __init__, so the
    # tracking state has to be carried explicitly -- otherwise the clone comes
    # back with no ``_unaccessed`` at all, or with a desynced one.
    ll = LoggedList(["a", "b", "c"])
    ll[1]                                    # access exactly one item
    c = clone(ll)
    assert isinstance(c, LoggedList)
    assert _raw(c) == ["a", "b", "c"]
    assert c.unaccessed == ["a", "c"]
    # The clone tracks independently of the original.
    c[0]
    assert c.unaccessed == ["c"]
    assert ll.unaccessed == ["a", "c"]


@pytest.mark.parametrize("op, expected", [
    (operator.eq, True),
    (operator.ne, False),
    (operator.lt, False),
    (operator.le, True),
    (operator.gt, False),
    (operator.ge, True),
], ids=["eq", "ne", "lt", "le", "gt", "ge"])
def test_logged_list_comparisons_match_list(op, expected):
    # LoggedList must compare against both plain lists and other LoggedLists
    # exactly as a list would.
    assert op(LoggedList([1, 2, 3]), [1, 2, 3]) == expected
    assert op(LoggedList([1, 2, 3]), LoggedList([1, 2, 3])) == expected
    assert op([1, 2, 3], [1, 2, 3]) == expected


_COMPARISON_OPS = [
    operator.eq, operator.ne, operator.lt,
    operator.le, operator.gt, operator.ge,
]


@pytest.mark.parametrize("left, right", [
    ([1, 2, 3], [1, 2, 4]),      # differs in the last element
    ([1, 2, 4], [1, 2, 3]),      # ...and the other way around
    ([1, 2],    [1, 2, 3]),      # proper prefix (shorter sorts first)
    ([1, 2, 3], [1, 2]),         # ...and the other way around
    ([2],       [1, 2, 3]),      # first element decides, regardless of length
    ([],        [1]),            # empty vs non-empty
    (["a", "b"], ["a", "c"]),    # non-numeric elements
], ids=["last-lt", "last-gt", "prefix", "extends", "first-wins", "empty", "str"])
@pytest.mark.parametrize("op", _COMPARISON_OPS,
                         ids=[op.__name__ for op in _COMPARISON_OPS])
def test_logged_list_unequal_comparisons_match_list(op, left, right):
    # Comparisons between *unequal* operands: these exercise the orderings
    # ``total_ordering`` derives from __eq__/__lt__, which comparing equal
    # lists barely touches.  The expected result is whatever plain lists do.
    expected = op(left, right)
    assert op(LoggedList(left), right) == expected
    assert op(LoggedList(left), LoggedList(right)) == expected
    # ...including with a plain list on the left, which goes through the
    # reflected operator.
    assert op(left, LoggedList(right)) == expected


def test_logged_list_unequal_comparison_does_not_mark_accessed():
    # Comparing does not count as accessing the individual arguments.
    ll = LoggedList(["a", "b"])
    assert (ll != ["a", "c"]) is True
    assert ll.unaccessed == ["a", "b"]


def test_logged_list_unhashable_like_list():
    with pytest.raises(TypeError):
        hash([1, 2, 3])
    with pytest.raises(TypeError):
        hash(LoggedList([1, 2, 3]))


# --- access-tracking behavior specific to LoggedList (a list has none) ------

def test_logged_list_getitem_marks_accessed():
    ll = LoggedList(["a", "b", "c"])
    assert ll[1] == "b"
    assert ll.unaccessed == ["a", "c"]


def test_logged_list_slice_marks_accessed():
    ll = LoggedList(["a", "b", "c", "d"])
    assert ll[1:3] == ["b", "c"]
    assert ll.unaccessed == ["a", "d"]


@pytest.mark.parametrize("value, expected", [
    (iter(["x"]),           [1, "x", 3]),
    (iter(["a", "b", "c"]), [1, "a", "b", "c", 3]),
    (iter([]),              [1, 3]),
], ids=["same-size", "grow", "shrink"])
def test_logged_list_setitem_slice_accepts_iterator(value, expected):
    # A slice assignment accepts any iterable, and may change the length of the
    # list.  The length therefore has to come from the value (which must be
    # materialized, since the assignment itself would consume the iterator),
    # not from the slice.
    ll = LoggedList([1, 2, 3])
    ll[1:2] = value
    assert _raw(ll) == expected
    # The access-tracking bookkeeping must stay in sync with the items.
    assert len(ll._unaccessed) == len(_raw(ll))


def test_logged_list_setitem_extended_slice_iterator_size_mismatch():
    # An extended slice assignment of the wrong size raises, as for a list, and
    # must leave both the items and the tracking state untouched.
    ll = LoggedList([1, 2, 3])
    with pytest.raises(ValueError):
        ll[::2] = iter(["a"])
    assert _raw(ll) == [1, 2, 3]
    assert len(ll._unaccessed) == 3


def test_logged_list_setitem_slice_marks_assigned_accessed():
    ll = LoggedList(["a", "b", "c"])
    ll[1:2] = iter(["x", "y"])
    # The freshly-assigned positions count as accessed; the rest do not.
    assert ll.unaccessed == ["a", "c"]


def test_logged_list_setitem_marks_assigned_accessed():
    # A single-index assignment marks that position accessed, consistently with
    # slice assignment: neither the replaced value nor the newly-assigned one is
    # an unused argument.
    ll = LoggedList(["a", "b", "c"])
    ll[1] = "x"
    assert ll.unaccessed == ["a", "c"]


def test_logged_list_setitem_does_not_mark_others_accessed():
    ll = LoggedList(["a", "b", "c"])
    ll[0] = "x"
    ll[-1] = "z"
    assert _raw(ll) == ["x", "b", "z"]
    assert ll.unaccessed == ["b"]


def test_logged_list_iter_marks_all_accessed():
    ll = LoggedList(["a", "b", "c"])
    list(ll)
    assert ll.unaccessed == []


def test_logged_list_len_does_not_mark():
    ll = LoggedList(["a", "b"])
    assert len(ll) == 2
    assert ll.unaccessed == ["a", "b"]


def test_logged_list_repr_marks_all():
    ll = LoggedList(["a", "b"])
    repr(ll)
    assert ll.unaccessed == []


def test_logged_list_index_marks_accessed():
    ll = LoggedList(["--flag", "value", "other"])
    assert ll.index("value") == 1
    assert ll.unaccessed == ["--flag", "other"]


def test_logged_list_sort_preserves_access_tracking():
    ll = LoggedList([3, 1, 2])
    ll.sort()
    # Sorting does not count as accessing any element.
    assert ll.unaccessed == [1, 2, 3]


def test_logged_list_clear_marks_nothing_unaccessed():
    ll = LoggedList([1, 2, 3])
    ll.clear()
    assert ll.unaccessed == []


def test_logged_list_copy_marks_all_accessed():
    # A copy reads every element, like ``sys.argv[:]``.
    ll = LoggedList([1, 2, 3])
    ll.copy()
    assert ll.unaccessed == []


def test_logged_list_concat_and_repeat_mark_accessed():
    # ``+`` and ``*`` read every element, so they must mark everything
    # accessed -- otherwise ``args = sys.argv + extra`` is reported as an
    # unused argument even though the code plainly used it.
    for build in [lambda x: x + ["z"],
                  lambda x: x * 2,
                  lambda x: 2 * x]:
        ll = LoggedList(["a", "b"])
        build(ll)
        assert ll.unaccessed == [], "%s left items unaccessed" % (build,)


def test_logged_list_sort_with_mutating_key_raises_like_list():
    # ``list.sort`` refuses to sort a list its own key function mutated;
    # silently discarding the mutation (as a blind write-back would) loses
    # data.
    def check(seq):
        def key(v):
            seq.append(v)
            return v
        with pytest.raises(ValueError) as exc:
            seq.sort(key=key)
        return str(exc.value)
    assert check(LoggedList([3, 1, 2])) == check([3, 1, 2])


class _SubList(LoggedList):
    def __init__(self, items, extra=None):
        super().__init__(items)
        self.extra = extra


@pytest.mark.parametrize("clone", [
    copy.copy,
    copy.deepcopy,
    lambda x: pickle.loads(pickle.dumps(x)),
], ids=["copy", "deepcopy", "pickle"])
def test_logged_list_round_trip_preserves_subclass_attributes(clone):
    # A plain list subclass keeps its instance __dict__ across copy/pickle;
    # LoggedList must not lose it just because it defines __reduce__.
    c = clone(_SubList(["a", "b"], extra="hi"))
    assert isinstance(c, _SubList)
    assert c.extra == "hi"
    _assert_tracking_aligned(c)


@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy], ids=["copy", "deepcopy"])
def test_logged_list_round_trip_with_changed_init_signature(clone):
    # Reconstructing through ``type(self)(items)`` would break any subclass
    # whose __init__ takes something else.
    class Sub(LoggedList):
        def __init__(self, a, b):
            super().__init__([a, b])
    c = clone(Sub(1, 2))
    assert _raw(c) == [1, 2]
    _assert_tracking_aligned(c)


@pytest.mark.parametrize("clone", [
    copy.deepcopy,
    lambda x: pickle.loads(pickle.dumps(x)),
], ids=["deepcopy", "pickle"])
def test_logged_list_round_trip_self_referential(clone):
    # A list containing itself round-trips for a plain list; putting the items
    # in the reconstructor's *arguments* would recurse forever instead, since
    # the object cannot be memoized until they are all built.
    ll = LoggedList(["a"])
    ll.append(ll)
    c = clone(ll)
    assert len(c) == 2
    assert c[1] is c
    _assert_tracking_aligned(c)


@pytest.mark.parametrize("proto", range(pickle.HIGHEST_PROTOCOL + 1))
def test_logged_list_pickle_preserves_tracking(proto):
    ll = LoggedList(["a", "b", "c"])
    ll[1]                                    # access exactly one item
    c = pickle.loads(pickle.dumps(ll, proto))
    assert _raw(c) == ["a", "b", "c"]
    assert c.unaccessed == ["a", "c"]
    _assert_tracking_aligned(c)


# Applied to a list and a LoggedList alike; each returns a value to compare and
# leaves the container in a state to compare.  Kept separate from
# _LIST_OPERATIONS because these are driven in random sequences, not one-shot.
_FUZZ_OPERATIONS = [
    lambda x, v: x.append(v),
    lambda x, v: x.insert(v % 7 - 3, v),
    lambda x, v: x.extend([v, v + 1]),
    lambda x, v: x.extend(iter([v])),
    lambda x, v: x.pop() if x else None,
    lambda x, v: x.pop(v % max(len(x), 1)) if x else None,
    lambda x, v: x.remove(v) if v in x else None,
    lambda x, v: x.reverse(),
    lambda x, v: x.sort(),
    lambda x, v: x.sort(key=lambda i: -i, reverse=True),
    lambda x, v: x.clear() if v % 11 == 0 else None,
    lambda x, v: x.__setitem__(slice(v % 3, v % 3 + 2), [v, v + 1]),
    lambda x, v: x.__setitem__(slice(None, None, 2), [v] * len(x[::2])),
    lambda x, v: x.__delitem__(slice(v % 3, v % 3 + 1)),
    lambda x, v: x.__setitem__(v % len(x), v) if x else None,
    lambda x, v: x.__imul__(v % 3),
    lambda x, v: x.__iadd__([v]),
    lambda x, v: x[v % max(len(x), 1)] if x else None,
    lambda x, v: x[::-1],
    lambda x, v: v in x,
    lambda x, v: x.copy(),
]


@pytest.mark.parametrize("seed", range(40))
def test_logged_list_fuzz_matches_list(seed):
    # Differential fuzz: drive a list and a LoggedList through the same random
    # sequence of operations and require identical results, identical contents,
    # and an intact tracking invariant at every step.  This is the net that
    # catches desyncs the hand-written cases do not think to try.
    rand = random.Random(seed)
    ref = [1, 2, 3]
    ll = LoggedList([1, 2, 3])
    for step in range(60):
        op = rand.choice(_FUZZ_OPERATIONS)
        value = rand.randrange(-5, 15)
        try:
            expected = op(ref, value)
            error = None
        except Exception as e:
            expected, error = None, (type(e), str(e))
        if error is None:
            actual = op(ll, value)
            assert actual == expected, (
                "step %d: got %r, list gave %r" % (step, actual, expected))
        else:
            with pytest.raises(error[0]) as exc:
                op(ll, value)
            assert str(exc.value) == error[1], "step %d message differs" % (step,)
        assert _raw(ll) == ref, "step %d: contents diverged" % (step,)
        _assert_tracking_aligned(ll)
