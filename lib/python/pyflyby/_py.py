# pyflyby/_py.py
# Copyright (C) 2014, 2015 Karl Chen.
# License: MIT http://opensource.org/licenses/MIT

"""
The `py' program (part of the pyflyby project) is a command-line multitool for
running python code, with heuristic intention guessing and automatic
importing.

Invocation summary
==================
  py --file   filename.py arg1 arg2   Execute a file
  py          filename.py arg1 arg2
  py        < filename.py

  py --eval  'function(arg1, arg2)'   Evaluate/execute an expression/statement
  py         'function(arg1, arg2)'

  py --apply  function arg1 arg2      Call function(arg1, arg2)
  py          function arg1 arg2

  py --module modname arg1 arg2       Run a module
  py          modname arg1 arg2

  py --map    function arg1 arg2      Call function(arg1); function(arg2)

  py          function?               Get help for a function or module
  py          function??              Get source of a function or module

  py                                  Start IPython with autoimporter
  py nb                               Start IPython Notebook with autoimporter


Features
========

  * Heuristic action mode guessing: If none of --file, --eval, --apply,
    --module, or --map is specified, then guess what to do, choosing one of
    these actions:
      * Execute (run) a file
      * Evaluate concatenated arguments
      * Run a module
      * Call (apply) a function
      * Evaluate first argument

  * Automatic importing: All action modes (except run_module) automatically
    import as needed.

  * Heuristic argument evaluation: By default, `py --eval', `py --apply', and
    `py --map' guess whether the arguments should be interpreted as
    expressions or literal strings. A "--" by itself will designate subsequent
    args as strings.  A "-" by itself will be replaced by the contents of
    stdin as a string.

  * Merged eval/exec: Code is eval()ed or exec()ed as appropriate.

  * Result printing: By default, results are pretty-printed if not None.

  * Heuristic flags: "print" can be used as a function or a statement.

  * Matplotlib/pylab integration: show() is called if appropriate to block on
    plots.


Warning
=======
`py' is intended as an interactive tool.  When writing shell aliases for
interactive use, the `--safe' option can be useful.  When writing scripts,
it's better to avoid all heuristic guessing; use regular `python -c ...', or
better yet, a full-fledged python program (and run tidy-imports).


Options
=======

Global options valid before code argument:

  --args=string   Interpret all arguments as literal strings.
                  (The "--" argument also specifies remaining arguments to be
                  literal strings.)
  --args=eval     Evaluate all arguments as expressions.
  --args=auto     (Default) Heuristically guess whether to evaluate arguments
                  as literal strings or expressions.
  --output=silent Don't print the result of evaluation.
  --output=str    Print str(result).
  --output=repr   Print repr(result).
  --output=pprint Print pprint.pformat(result).
  --output=repr-if-not-none
                  Print repr(result), but only if result is not None.
  --output=pprint-if-not-none
                  Print pprint.pformat(result), but only if result is not None.
  --output=interactive
                  (Default) Print result.__interactive_display__() if defined,
                  else pprint if result is not None.
  --output=exit   Raise SystemExit(result).
  --safe          Equivalent to --args=strings and PYFLYBY_PATH=EMPTY.
  --quiet, -q     Log only error messages to stderr; omit info and warnings.

Pseudo-actions valid before, after, or without code argument:

  --version       Print pyflyby version or version of a module.
  --help, -h, ?   Print this help or help for a function or module.
  --source, ??    Print source code for a function or module.


Examples
========

  Find the ASCII value of the letter "j" (apply builtin function):
    $ py ord j
    [PYFLYBY] ord('j')
    106

  Decode a base64-encoded string (apply autoimported function):
    $ py b64decode aGVsbG8=
    [PYFLYBY] from base64 import b64decode
    [PYFLYBY] b64decode('aGVsbG8=', altchars=None)
    'hello'

  Find the day of the week of some date (apply function in module):
    $ py calendar.weekday 2014 7 18
    [PYFLYBY] import calendar
    [PYFLYBY] calendar.weekday(2014, 7, 18)
    4

  Using named arguments:
    $ py calendar.weekday --day=16 --month=7 --year=2014
    [PYFLYBY] import calendar
    [PYFLYBY] calendar.weekday(2014, 7, 16)
    2

  Using short named arguments:
    $ py calendar.weekday -m 7 -d 15 -y 2014
    [PYFLYBY] import calendar
    [PYFLYBY] calendar.weekday(2014, 7, 15)
    1

  Invert a matrix (evaluate expression, with autoimporting):
    $ py 'matrix("1 3 3; 1 4 3; 1 3 4").I'
    [PYFLYBY] from numpy import matrix
    [PYFLYBY] matrix("1 3 3; 1 4 3; 1 3 4").I
    matrix([[ 7., -3., -3.],
            [-1.,  1.,  0.],
            [-1.,  0.,  1.]])

  Plot cosine (evaluate expression, with autoimporting):
    $ py 'plot(cos(arange(30)))'
    [PYFLYBY] from numpy import arange
    [PYFLYBY] from numpy import cos
    [PYFLYBY] from matplotlib.pyplot import plot
    [PYFLYBY] plot(cos(arange(30)))
    <plot>

  Command-line calculator (multiple arguments):
    $ py 3 / 4
    0.75

  Command-line calculator (single arguments):
    $ py '(5+7j) ** 12'
    (65602966976-150532462080j)

  Rationalize a decimal (apply bound method)
    $ py 2.5.as_integer_ratio
    [PYFLYBY] 2.5.as_integer_ratio()
    (5, 2)

  Rationalize a decimal (apply unbound method)
    $ py float.as_integer_ratio 2.5
    [PYFLYBY] float.as_integer_ratio(2.5)
    (5, 2)

  Rationalize decimals (map/apply)
    $ py --map float.as_integer_ratio 2.5 3.5
    [PYFLYBY] float.as_integer_ratio(2.5)
    (5, 2)
    [PYFLYBY] float.as_integer_ratio(3.5)
    (7, 2)

  Square numbers (map lambda)
    $ py --map 'lambda x: x**2' 3 4 5
    [PYFLYBY] (lambda x: x**2)(3)
    9
    [PYFLYBY] (lambda x: x**2)(4)
    16
    [PYFLYBY] (lambda x: x**2)(5)
    25

  Find length of string (using "-" for stdin):
    $ echo hello | py len -
    [PYFLYBY] len('hello\n')
    6

  Run stdin as code:
    $ echo 'print sys.argv[1:]' | py - hello world
    [PYFLYBY] import sys
    ['hello', 'world']

  Run libc functions:
    $ py --quiet --output=none 'CDLL("libc.so.6").printf' %03d 7
    007

  Download web page:
    $ py --print 'requests.get(sys.argv[1]).text' http://example.com

  Get function help:
    $ py b64decode?
    [PYFLYBY] from base64 import b64decode
    Python signature:
      >> b64decode(s, altchars=None)
    Command-line signature:
      $ py b64decode s [altchars]
      $ py b64decode --s=... [--altchars=...]
    ...

  Get module help:
    $ py pandas?
    [PYFLYBY] import pandas
    Version:
      0.13.1
    Filename:
      /usr/local/lib/python2.7/site-packages/pandas/__init__.pyc
    Docstring:
      pandas - a powerful data analysis and manipulation library for Python
      ...

"""

from __future__ import (absolute_import, division, print_function,
                        with_statement)

usage = """
py --- command-line python multitool with automatic importing

$ py  filename.py arg1 arg2      Execute file
$ py  function arg1 arg2         Call function
$ py 'function(arg1, arg2)'      Evaluate code
$ py                             IPython
""".strip()

# TODO: --debug/--pdb => enter debugger before execution
# TODO: ipdb tracebacks
# TODO: default to entering debugger upon exception
# TODO: add --tidy-imports, etc
# TODO: --interactive/-i => ipython after execution
# TODO: note additional features in documentation feature list

# TODO: new --action="concat_eval eval apply" etc.  specifying multiple
# actions means try each of them in that order.  then --safe can exclude
# concat-eval, and users can customize which action modes are included.
# --apply would be equivalent to --action=apply.

# TODO: plug-in system.  'py foo' should attempt something that the
# user/vendor can add to the system.  leading candidate: use entry_point
# system (http://stackoverflow.com/a/774859).  other candidates: maybe use
# python namespace like pyflyby.vendor.foo or pyflyby.commands.foo or
# pyflyby.magics.foo, or maybe a config file.
# TODO: note additional features in documentation feature list

# TODO: somehow do the right thing with glob.glob vs glob, pprint.pprint vs
# pprint, etc.  Ideas:
#   - make --apply special case detect modules and take module.module
#   - enhance auto_import() to keep track of the context while finding missing imports
#   - enhance auto_import() to scan for calls after importing

# TODO: pipe help/source output (all output?) through $PYFLYBY_PAGER (default "less -FRX").

# TODO: unparse ast node for info logging
# https://hg.python.org/cpython/log/tip/Tools/parser/unparse.py

# TODO: run_module should detect if the module doesn't check __name__ and
# therefore is unlikely to be meaningful to use with run_module.

# TODO: make sure run_modules etc work correctly with modules under namespace
# packages.

# TODO: detect deeper ImportError, e.g. suppose user accesses module1; module1
# imports badmodule, which can't be imported successfully and raises
# ImportError; we should get that ImportError instead of trying other things
# or turning it into a string.  probably do this by changing the place where
# we import modules to first get the loader, then if import fails, raise a
# subclass of ImportError.

# TODO: provide a way to omit newline in output.  maybe --output=write.

import ast
from   contextlib               import contextmanager
import inspect
import os
import re
import sys
import types
from   types                    import FunctionType, MethodType

from   pyflyby._autoimp         import (auto_eval, auto_import,
                                        find_missing_imports)
from   pyflyby._cmdline         import print_version_and_exit, syntax
from   pyflyby._file            import Filename, UnsafeFilenameError
from   pyflyby._flags           import CompilerFlags
from   pyflyby._idents          import is_identifier
from   pyflyby._interactive     import (run_ipython_line_magic,
                                        start_ipython_with_autoimporter)
from   pyflyby._log             import logger
from   pyflyby._modules         import ModuleHandle
from   pyflyby._parse           import PythonBlock
from   pyflyby._util            import indent, prefixes


# Default compiler flags (feature flags) used for all user code.  We include
# "print_function" here, but we also use auto_flags=True, which means
# print_function may be flipped off if the code contains print statements.
FLAGS = CompilerFlags(["absolute_import", "with_statement", "division",
                       "print_function"])


def _get_argspec(arg, _recurse=False):
    from inspect import getargspec, ArgSpec
    if isinstance(arg, FunctionType):
        return getargspec(arg)
    elif isinstance(arg, MethodType):
        argspec = getargspec(arg)
        if arg.im_self is not None:
            # For bound methods, ignore the "self" argument.
            return ArgSpec(argspec.args[1:], *argspec[1:])
        return argspec
    elif isinstance(arg, type):
        if arg.__new__ is not object.__new__:
            argspec = _get_argspec(arg.__new__)
            return ArgSpec(argspec.args[1:], *argspec[1:])
        else:
            argspec = _get_argspec(arg.__init__)
            return ArgSpec(argspec.args[1:], *argspec[1:])
    elif isinstance(arg, types.ClassType):
        argspec = _get_argspec(arg.__init__)
        return ArgSpec(argspec.args[1:], *argspec[1:])
    elif _recurse and hasattr(arg, '__call__'):
        return _get_argspec(arg.__call__, _recurse=False)
    elif callable(arg):
        # Unknown, probably a built-in method.
        return ArgSpec((), "args", "kwargs", None)
    raise TypeError(
        "_get_argspec: unexpected %s" % (type(arg).__name__,))


def _requires_parens_as_function(function_name):
    """
    Returns whether the given string of a callable would require parentheses
    around it to call it.

      >>> _requires_parens_as_function("foo.bar[4]")
      False

      >>> _requires_parens_as_function("foo+bar")
      True

      >>> _requires_parens_as_function("(foo+bar)()")
      False

      >>> _requires_parens_as_function("(foo+bar)")
      False

      >>> _requires_parens_as_function("(foo)+(bar)")
      True

    TODO: this might be obsolete if we use unparse instead of keeping original
    user formatting (or alternatively, unparse should use something like this).

    @type function_name:
      C{str}
    @rtype:
      C{bool}
    """
    flags = int(FLAGS) + ast.PyCF_ONLY_AST
    try:
        node = compile(function_name, "<unknown>", "eval", flags)
    except SyntaxError:
        # Couldn't parse?  Just assume we do need parens for now.  Or should
        # we raise an exception here?
        return True
    body = node.body
    # Is it something that doesn't need parentheses?
    if isinstance(body, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
        return False
    # Does it already have parentheses?
    if function_name.startswith("(") and function_name.endswith(")"):
        # It has parentheses, superficially.  Make sure it's not something
        # like "(foo)+(bar)".
        try:
            tnode = compile(function_name[1:-1], "<unknown>", "eval", flags)
        except SyntaxError:
            return True
        if ast.dump(tnode) == ast.dump(node):
            return False
        else:
            return True
    return True


def _format_call_spec(function_name, argspec):
    callspec = inspect.formatargspec(*argspec)
    if _requires_parens_as_function(function_name):
        return "(%s)%s" % (function_name, callspec)
    else:
        return "%s%s" % (function_name, callspec)


# TODO: move to util module
try:
    from shlex import quote as shquote # python3.3+
except ImportError:
    # Backport of shlex.quote from python3.3
    _find_unsafe = re.compile(r'[^\w@%+=:,./-]').search
    def shquote(s):
        """Return a shell-escaped version of the string *s*."""
        if not s:
            return "''"
        if _find_unsafe(s) is None:
            return s
        # use single quotes, and put single quotes into double quotes
        # the string $'b is then quoted as '$'"'"'b'
        return "'" + s.replace("'", "'\"'\"'") + "'"


def _build_function_usage_string(function_name, argspec, prefix):
    usage = []
    # TODO: colorize
    usage.append("Python signature:")
    usage.append("  >"+">> " + _format_call_spec(function_name, argspec))
    usage.append("")
    usage.append("Command-line signature:")
    if not argspec.args and argspec.varargs and argspec.keywords:
        # We have no information about the arguments.  It's probably a
        # built-in where getargspec failed.
        usage.append("  $ %s%s ...\n" % (prefix, function_name))
        return "\n".join(usage)
    defaults = argspec.defaults or ()
    first_with_default = len(argspec.args) - len(defaults)
    # Show first alternative of command-line syntax.
    syntax1 = "  $ %s%s" % (prefix, shquote(function_name),)
    for i, arg in enumerate(argspec.args):
        if i >= first_with_default:
            syntax1 += " [%s" % (arg,)
        else:
            syntax1 += " %s" % (arg,)
    if argspec.varargs:
        syntax1 += " %s..." % argspec.varargs
    syntax1 += "]" * len(defaults)
    if argspec.keywords:
        syntax1 += " [--...]"
    usage.append(syntax1)
    # usage.append("or:")
    syntax2 = "  $ %s%s" % (prefix, shquote(function_name),)
    for i, arg in enumerate(argspec.args):
        if i >= first_with_default:
            syntax2 += " [--%s=...]" % (arg,)
        else:
            syntax2 += " --%s=..." % (arg,)
    if argspec.varargs:
        syntax2 += " %s..." % argspec.varargs
    if argspec.keywords:
        syntax2 += " [--...]"
    usage.append(syntax2)
    usage.append("")
    return "\n".join(usage)


def _parse_value_or_string(arg, namespace, arg_mode="auto"):
    """
    Heuristically choose to auto_eval() a string if appropriate, else return
    the argument as a string.

    Heuristic auto-evaluation:
      >>> ns = _Namespace()
      >>> _parse_value_or_string('5+2', ns)
      7

      >>> _parse_value_or_string('5j+2', ns)
      (2+5j)

      >>> _parse_value_or_string('base64.b64decode("SGFsbG93ZWVu")', ns)
      [PYFLYBY] import base64
      'Halloween'

    Returning an unparsable argument as a string:
      >>> _parse_value_or_string('5foo+2', ns)
      '5foo+2'

    Returning an undefined (and not auto-importable) argument as a string:
      >>> _parse_value_or_string('foo5+2', ns)
      'foo5+2'

    @type arg:
      C{str}
    @type namespace:
      L{_Namespace}
    @type arg_mode:
      C{str}
    @param arg_mode:
      If C{"string"}, then return C{arg} unchanged.  If C{"eval"}, then always
      evaluate C{arg}.  If C{"auto"}, then heuristically evaluate if
      appropriate.
    """
    if not isinstance(arg, str):
        raise TypeError("_parse_value_or_string(): expected str instead of %s"
                        % (type(arg).__name__,))
    if arg_mode == "string":
        return arg
    elif arg_mode == "eval":
        # Try to parse into an AST.  (We don't need auto_flags here because
        # it only affects print_function, which is only relevant for
        # mode="exec".)
        block = PythonBlock(arg, flags=FLAGS, auto_flags=False)
        if block.parsable_as_expression:
            return namespace.auto_eval(block)
        else:
            raise SyntaxError("syntax error: %s" % (arg,))
    elif arg_mode == "auto":
        # Try parsing as a Python expression.
        block = PythonBlock(arg, flags=FLAGS, auto_flags=False)
        if block.parsable_as_expression:
            try:
                return namespace.auto_eval(block)
            except UnimportableNameError:
                # If the string can't be auto-evaluated, then fallback to a
                # string.
                #
                # (We used to only do the fallback behavior when the entire
                # string was an identifier.  However, we should also fallback
                # for "foo/bar", "foo-bar", etc.)
                return arg
                # TODO: unit test that we don't get confused by NameError in
                # user code
        else:
            # Not a Python expression.  Return as a string.
            return arg
    elif arg_mode == "error":
        raise ValueError("Expected no arguments; got %r" % (arg,))
    else:
        raise ValueError("_parse_value_or_string(): invalid arg_mode=%r"
                         % (arg_mode,))


class ParseError(Exception):
    pass


class _ParseInterruptedWantHelp(Exception):
    pass


class _ParseInterruptedWantSource(Exception):
    pass


# TODO: create a new class Arg (AutoEvaluatedArg?) which tracks string form,
# ast form, unparsed form, evaluated form and defers the auto-evaluation until
# later.  This allows:
#   * avoiding repr() for "-" and "--"
#   * making args=auto work for eval_maybe_apply
#   * integrating function/function_name stuff


def _parse_auto_apply_args(argspec, commandline_args, namespace, arg_mode="auto"):
    """
    Parse command-line arguments heuristically.  Arguments that can be
    evaluated are evaluated; otherwise they are treated as strings.

    @returns:
      C{args}, C{kwargs}
    """
    # This is implemented manually instead of using optparse or argparse.  We
    # do so because neither supports dynamic keyword arguments well.  Optparse
    # doesn't support parsing known arguments only, and argparse doesn't
    # support turning off interspersed positional arguments.

    # Create a map from argname to default value.
    defaults = argspec.defaults or ()
    argname2default = {}
    for argname, default in zip(argspec.args[len(argspec.args)-len(defaults):],
                                defaults):
        argname2default[argname] = default
    # Create a map from prefix to arguments with that prefix.  E.g. {"foo":
    # ["foobar", "foobaz"]}
    prefix2argname = {}
    for argname in argspec.args:
        for prefix in prefixes(argname):
            prefix2argname.setdefault(prefix, []).append(argname)
    # Enumerate over input arguments.
    got_pos_args = []
    got_keyword_args = {}
    args = list(commandline_args)
    while args:
        arg = args.pop(0)
        if arg in ["--?", "-?", "?"]:
            raise _ParseInterruptedWantHelp
        elif arg in ["--??", "-??", "??"]:
            raise _ParseInterruptedWantSource
        elif arg.startswith("-"):
            if arg == "-":
                # Read from stdin and stuff into next argument as a string.
                data = sys.stdin.read()
                if arg_mode != "string":
                    # Wrap in repr() to eval later.
                    # TODO: avoid this ugly kludge here
                    data = repr(data)
                got_pos_args.append(data)
                continue
            elif arg == "--":
                # Treat remaining arguments as strings.
                # TODO: optimize this function to avoid the need for the repr.
                if arg_mode == "string":
                    got_pos_args.extend(args)
                else:
                    got_pos_args.extend([repr(x) for x in args])
                del args[:]
                continue
            elif arg.startswith("--"):
                argname = arg[2:]
            else:
                argname = arg[1:]
            argname, equalsign, value = argname.partition("=")
            argname = argname.replace("-", "_")
            if not is_identifier(argname):
                raise ParseError("Invalid option name %s" % (argname,))
            matched_argnames = prefix2argname.get(argname, [])
            if len(matched_argnames) == 1:
                argname, = matched_argnames
            elif len(matched_argnames) == 0:
                if equalsign == "":
                    if argname in ["help", "h"]:
                        raise _ParseInterruptedWantHelp
                    if argname in ["source"]:
                        raise _ParseInterruptedWantSource
                if not argspec.keywords:
                    raise ParseError("Unknown option name %s" % (argname,))
            elif len(matched_argnames) > 1:
                raise ParseError(
                    "Ambiguous %s: could mean one of: %s"
                    % (argname,
                       ", ".join("--%s"%s for s in matched_argnames)))
            else:
                raise AssertionError
            if not value:
                try:
                    value = args.pop(0)
                except IndexError:
                    raise ParseError("Missing argument to %s" % (arg,))
                if value.startswith("--"):
                    raise ParseError(
                        "Missing argument to %s.  "
                        "If you really want to use %r as the argument to %s, "
                        "then use %s=%s."
                        % (arg, value, arg, arg, value))
            got_keyword_args[argname] = value
        else:
            got_pos_args.append(arg)

    parsed_args = []
    parsed_kwargs = {}
    for i, argname in enumerate(argspec.args):
        if i < len(got_pos_args):
            if argname in got_keyword_args:
                raise ParseError(
                    "%s specified both as positional argument (%s) "
                    "and keyword argument (%s)"
                    % (argname, got_pos_args[i], got_keyword_args[argname]))
            unparsed_value = got_pos_args[i]
            parsed_value = _parse_value_or_string(unparsed_value, namespace, arg_mode)
        else:
            try:
                unparsed_value = got_keyword_args.pop(argname)
            except KeyError:
                try:
                    parsed_value = argname2default[argname]
                except KeyError:
                    raise ParseError(
                        "missing required argument %s" % (argname,))
            else:
                parsed_value = _parse_value_or_string(unparsed_value, namespace, arg_mode)
        parsed_args.append(parsed_value)

    if len(got_pos_args) > len(argspec.args):
        if argspec.varargs:
            for unparsed_value in got_pos_args[len(argspec.args):]:
                parsed_args.append(_parse_value_or_string(unparsed_value, namespace, arg_mode))
        else:
            max_nargs = len(argspec.args)
            if argspec.defaults:
                expected = "%d-%d" % (max_nargs-len(argspec.defaults),max_nargs)
            else:
                expected = "%d" % (max_nargs,)
            raise ParseError(
                "Too many positional arguments.  "
                "Expected %s positional argument(s): %s.  Got %d args: %s"
                % (expected, ", ".join(argspec.args),
                   len(got_pos_args), " ".join(got_pos_args)))

    for argname, unparsed_value in sorted(got_keyword_args.items()):
        try:
            parsed_kwargs[argname] = _parse_value_or_string(unparsed_value, namespace, arg_mode)
        except Exception as e:
            raise ParseError(
                "Error parsing value for --%s=%s: %s: %s"
                % (argname, unparsed_value, type(e).__name__, e))

    return parsed_args, parsed_kwargs


def _format_call(function_name, argspec, args, kwargs):
    # TODO: print original unparsed arg strings
    defaults = argspec.defaults or ()
    first_with_default = len(argspec.args) - len(defaults)
    argparts = []
    for i in range(max(len(args), len(argspec.args))):
        if i >= first_with_default and len(args) <= len(argspec.args):
            argparts.append("%s=%r" % (argspec.args[i], args[i]))
        else:
            argparts.append(repr(args[i]))
    for k, v in sorted(kwargs.items()):
        argparts.append("%s=%r" % (k, v))
    if _requires_parens_as_function(function_name):
        function_name = "(%s)" % (function_name,)
    r = "%s(%s)" % (function_name, ", ".join(argparts))
    return r


class UnimportableNameError(NameError):
    pass


class NotAFunctionError(Exception):
    pass


# TODO: replace with an Arg class.
def _interpret_function_name(function, function_name, namespace):
    """
    Interpret function/function_name.

    If C{function} is a string, then evaluate it (with auto importing) to get
    the function.  (In general, it can be any callable, not just a function.)

    If C{function_name} is not specified, then use C{function} if it was
    originally specified as a string, else C{function.__name__}.

    @param function:
      Function, or a string to evaluate.
    @param function_name:
      Name of function.
    @return:
      (function, function_name)
    """
    if callable(function):
        if function_name is None:
            function_name = function.__name__
        return function, function_name
    elif isinstance(function, basestring):
        function = function.strip()
        if not function:
            raise ValueError("No function specified")
        if function_name is None:
            function_name = function
        else:
            function_name = function_name.strip()
        try:
            function = namespace.auto_eval(function, mode="eval")
        except Exception as e:
            raise ValueError("Invalid function %s: %s: %s"
                             % (function, type(e).__name__, e))
        if not callable(function):
            raise NotAFunctionError("Not a function", function)
        return function, function_name
    else:
        raise TypeError(
            "Expected function or string; got a %s"
            % (type(function).__name__,))


def _get_help(obj, name, verbosity=1):
    """
    Construct a help string.

    @param object:
      Object to generate help for.
    @param name:
      Name of object.
    @rtype:
      C{str}
    """
    # TODO: colorize headers
    result = ""
    if callable(obj):
        argspec = _get_argspec(obj)
        prefix = os.path.basename(sys.orig_argv[0]) + " "
        result += _build_function_usage_string(name, argspec, prefix)
    if verbosity == 0:
        include_filename = False
        include_doc      = False
        include_source   = False
    elif verbosity == 1:
        include_filename = True
        include_doc      = True
        include_source   = False
    elif verbosity == 2:
        include_filename = True
        include_doc      = False
        include_source   = True
    else:
        raise ValueError("invalid verbosity=%r" % (verbosity,))
    try:
        version = obj.__version__
    except Exception:
        pass
    else:
        result += "\nVersion:\n  %s\n" % (version,)
    if include_filename:
        try:
            filename = inspect.getfile(obj)
        except Exception:
            pass
        else:
            result += "\nFilename:\n  %s\n" % (filename,)
    if include_source:
        try:
            source = inspect.getsource(obj)
        except Exception:
            source = ""
        if source:
            # TODO: colorize source
            result += "\nSource:\n%s\n" % (indent(source, "  "))
        else:
            source = "(Not available)"
            include_doc = True
    if include_doc:
        doc = (inspect.getdoc(obj) or "").strip() or "(No docstring)"
        result += "\nDocstring:\n%s" % (indent(doc, "  "))
    return result


def auto_apply(function, commandline_args, namespace,
               arg_mode=None, function_name=None):
    """
    Call C{function} on command-line arguments.  Arguments can be positional
    or keyword arguments like "--foo=bar".  Arguments are by default
    heuristically evaluated.

    @param function:
      Function, or a string to evaluate.
    @param function_name:
      Name of function (used for usage strings).
    @type commandline_args:
      C{list} of C{str}
    @param commandline_args:
      Arguments to C{function} as strings.
    @param arg_mode:
      How to interpret C{commandline_args}.  If C{"string"}, then treat them
      as literal strings.  If C{"eval"}, then evaluate all arguments as
      expressions.  If C{"auto"} (the default), then heuristically decide
      whether to treat as expressions or strings.
    """
    function, function_name = _interpret_function_name(
        function, function_name, namespace)
    arg_mode = _interpret_arg_mode(arg_mode, default="auto")
    # Parse command-line arguments.
    argspec = _get_argspec(function)
    try:
        args, kwargs = _parse_auto_apply_args(argspec, commandline_args,
                                              namespace, arg_mode=arg_mode)
    except _ParseInterruptedWantHelp:
        usage = _get_help(function, function_name, verbosity=1)
        print(usage)
        raise SystemExit(0)
    except _ParseInterruptedWantSource:
        usage = _get_help(function, function_name, verbosity=2)
        print(usage)
        raise SystemExit(0)
    except ParseError as e:
        # Failed parsing command-line arguments.  Print usage.
        logger.error(e)
        usage = _get_help(function, function_name,
                          verbosity=(1 if logger.info_enabled else 0))
        sys.stderr.write("\n" + usage)
        raise SystemExit(1)
    # Log what we're doing.
    logger.info("%s", _format_call(function_name, argspec, args, kwargs))

    # *** Call the function. ***
    try:
        result = function(*args, **kwargs)
    except Exception as e:
        raise
        # TODO: print traceback only for user portion of code
        # TODO: arrange for traceback to show user code instead of
        # "function(*args, **kwargs)"
        tb = sys.exc_info()[2]
        import traceback; traceback.print_tb(tb)
        raise SystemExit(1)
    return result


class LoggedList(object):
    """
    List that logs which members have not yet been accessed (nor removed).
    """

    _ACCESSED = object()

    def __init__(self, items):
        self._items = list(items)
        self._unaccessed = list(self._items)

    def append(self, x):
        self._unaccessed.append(self._ACCESSED)
        self._items.append(x)

    def count(self):
        return self._items.count()

    def extend(self, new_items):
        new_items = list(new_items)
        self._unaccessed.extend([self._ACCESSED] * len(new_items))
        self._items.extend(new_items)

    def index(self, x, *start_stop):
        index = self._items.index(x, *start_stop) # may raise ValueError
        self._unaccessed[index] = self._ACCESSED
        return index

    def insert(self, index, x):
        self._unaccessed.insert(index, self._ACCESSED)
        self._items.insert(index, x)

    def pop(self, index):
        self._unaccessed.pop(index)
        return self._items.pop(index)

    def remove(self, x):
        index = self._items.index(x)
        self.pop(index)

    def reverse(self):
        self._items.reverse()
        self._unaccessed.reverse()

    def sort(self):
        indexes = range(len(self._items))
        indexes.sort(key=self._items.__getitem__) # argsort
        self._items = [self._items[i] for i in indexes]
        self._unaccessed = [self._unaccessed[i] for i in indexes]

    def __add__(self, other):
        return self._items + other

    def __contains__(self, x):
        try:
            self.index(x)
            return True
        except ValueError:
            return False

    def __delitem__(self, x):
        del self._items[x]
        del self._unaccessed[x]

    def __cmp__(self, x):
        return cmp(self._items, x)

    def __getitem__(self, idx):
        result = self._items[idx]
        if isinstance(idx, slice):
            self._unaccessed[idx] = [self._ACCESSED]*len(result)
        else:
            self._unaccessed[idx] = self._ACCESSED
        return result

    def __hash__(self):
        raise TypeError("unhashable type: 'LoggedList'")

    def __iadd__(self, x):
        self.extend(x)

    def __imul__(self, n):
        self._items *= n
        self._unaccessed *= n

    def __iter__(self):
        # Todo: detect mutation while iterating.
        for i, x in enumerate(self._items):
            self._unaccessed[i] = self._ACCESSED
            yield x

    def __len__(self):
        return len(self._items)


    def __mul__(self, n):
        return self._items * n

    def __reduce__(self):
        return

    def __repr__(self):
        self._unaccessed[:] = [self._ACCESSED]*len(self._unaccessed)
        return repr(self._items)

    def __reversed__(self):
        # Todo: detect mutation while iterating.
        for i in reversed(range(len(self._items))):
            self._unaccessed[i] = self._ACCESSED
            yield self._items[i]

    def __rmul__(self, n):
        return self._items * n

    def __setitem__(self, idx, value):
        self._items[idx] = value
        if isinstance(idx, slice):
            self._unaccessed[idx] = [self._ACCESSED]*len(value)
        else:
            self._unaccessed[idx] = value

    def __str__(self):
        self._unaccessed[:] = [self._ACCESSED]*len(self._unaccessed)
        return str(self._items)

    @property
    def unaccessed(self):
        return [x for x in self._unaccessed if x is not self._ACCESSED]


@contextmanager
def SysArgvCtx(*args):
    """
    Context manager that:
      * Temporarily sets sys.argv = args.
      * At end of context, complains if any args were never accessed.
    """
    # There should always be at least one arg, since the first one is
    # the program name.
    if not args:
        raise ValueError("Missing args")
    nargs = len(args) - 1
    # Create a list proxy that will log accesses.
    argv = LoggedList(args)
    # Don't consider first argument to be interesting.
    argv[0]
    orig_argv = list(sys.argv)
    try:
        sys.argv = argv
        # Run context code.
        yield
        # Complain if there are unaccessed arguments.
        unaccessed = argv.unaccessed
        if not unaccessed:
            pass
        else:
            if nargs == 1:
                msg = ("the command-line argument was not used: %s"
                       % (unaccessed[0],))
            elif len(unaccessed) == nargs:
                msg = ("all %d command-line arguments were not used: %s"
                       % (len(unaccessed), " ".join(unaccessed)))
            else:
                msg = ("%d of %d command-line arguments were not used: %s"
                       % (len(unaccessed), nargs, " ".join(unaccessed)))
            logger.error(msg)
            raise SystemExit(1)
    finally:
        sys.argv = orig_argv


def _seems_like_filename(arg):
    """
    Return whether C{arg} seems like a filename.

      >>> _seems_like_filename("foo.py")
      True

      >>> _seems_like_filename("%foo.py")
      False

      >>> _seems_like_filename("foo(bar)")
      False

      >>> _seems_like_filename("/foo/bar/baz.quux-66047003")
      True

      >>> _seems_like_filename("../foo/bar-24084866")
      True

    @type arg:
      C{str}
    @rtype:
      C{bool}
    """
    try:
        filename = Filename(arg)
    except UnsafeFilenameError:
        # If the filename isn't a "safe" filename, then don't treat it as one,
        # and don't even check whether it exists.  This means that for an
        # argument like "foo(bar)" or "lambda x:x*y" we won't even check
        # existence.  This is both a performance optimization and a safety
        # valve to avoid unsafe filenames being created to intercept expressions.
        return False
    # If the argument "looks" like a filename and is unlikely to be a python
    # expression, then assume it is a filename.  We assume so regardless of
    # whether the file actually exists; if it turns out to not exist, we'll
    # complain later.
    if filename.ext == ".py":
        # TODO: .pyc, .pyo
        return True
    if arg.startswith("/") or arg.startswith("./") or arg.startswith("../"):
        return True
    # Otherwise, even if it doesn't obviously look like a filename, but it
    # does exist as a filename, then use it as one.
    if filename.exists:
        return True
    return False


def _interpret_arg_mode(arg, default="auto"):
    """
      >>> _interpret_arg_mode("Str")
      'string'
    """
    if arg is None:
        arg = default
    rarg = str(arg).strip().lower()
    if rarg in ["eval", "evaluate", "exprs", "expr", "expressions", "expression", "e"]:
        return "eval"
    elif rarg in ["strings", "string", "str", "strs", "literal", "literals", "s"]:
        return "string"
    elif rarg in ["auto", "automatic", "a"]:
        return "auto"
    elif rarg == "error":
        # Intentionally not documented to user
        return "error"
    else:
        raise ValueError(
            "Invalid arg_mode=%r; expected one of eval/string/auto"
            % (arg,))


def _interpret_output_mode(arg, default="interactive"):
    """
      >>> _interpret_output_mode('Repr_If_Not_None')
      'repr-if-not-none'
    """
    if arg is None:
        arg = default
    rarg = str(arg).strip().lower().replace("-", "").replace("_", "")
    if rarg in ["none", "no", "n", "silent"]:
        return "silent"
    elif rarg in ["interactive", "i"]:
        return "interactive"
    elif rarg in ["print", "p", "string", "str"]:
        return "str"
    elif rarg in ["repr", "r"]:
        return "repr"
    elif rarg in ["pprint", "pp"]:
        return "pprint"
    elif rarg in ["reprifnotnone", "reprunlessnone", "rn"]:
        return "repr-if-not-none"
    elif rarg in ["pprintifnotnone", "pprintunlessnone", "ppn"]:
        return "pprint-if-not-none"
    elif rarg in ["systemexit", "exit", "raise"]:
        return "exit"
    else:
        raise ValueError(
            "Invalid output=%r; expected one of "
            "silent/interactive/str/repr/pprint/repr-if-not-none/pprint-if-not-none/exit"
            % (arg,))


def print_result(result, output_mode):
    output_mode = _interpret_output_mode(output_mode)
    if output_mode == "silent":
        return
    if output_mode == "interactive":
        # TODO: support IPython output stuff (text/plain)
        try:
            idisp = result.__interactive_display__
        except Exception:
            output_mode = "pprint-if-not-none"
        else:
            result = idisp()
            output_mode = "result-if-not-none"
        # Fall through.
    if output_mode == "str":
        print(str(result))
    elif output_mode == "repr":
        print(repr(result))
    elif output_mode == "pprint":
        import pprint
        pprint.pprint(result) # or equivalently, print pprint.pformat(result)
    elif output_mode == "repr-if-not-none":
        if result is not None:
            print(repr(result))
    elif output_mode == "pprint-if-not-none":
        if result is not None:
            import pprint
            pprint.pprint(result)
    elif output_mode == "exit":
        # TODO: only raise at end after pre_exit
        raise SystemExit(result)
    else:
        raise AssertionError("unexpected output_mode=%r" % (output_mode,))


class _Namespace(object):

    def __init__(self):
        self.globals      = {"__name__": "__main__"}
        self.autoimported = {}

    def auto_import(self, arg):
        return auto_import(arg, [self.globals], autoimported=self.autoimported)

    def auto_eval(self, block, mode=None, info=False):
        """
        Evaluate C{block} with auto-importing.
        """
        # Equivalent to::
        #   auto_eval(arg, mode=mode, flags=FLAGS, globals=self.globals)
        # but better logging and error raising.
        block = PythonBlock(block, flags=FLAGS, auto_flags=True)
        if not self.auto_import(block):
            missing = find_missing_imports(block, [self.globals])
            mstr = ", ".join(map(repr, missing))
            if len(missing) == 1:
                msg = "name %s is not defined and not importable" % mstr
            elif len(missing) > 1:
                msg = "names %s are not defined and not importable" % mstr
            else:
                raise AssertionError
            raise UnimportableNameError(msg)
        if info:
            logger.info(block)
        return auto_eval(block, globals=self.globals, mode=mode)


class _PyMain(object):

    def __init__(self, args):
        self.main_args = args
        self.namespace = _Namespace()

    def exec_stdin(self, cmd_args):
        arg_mode = _interpret_arg_mode(self.arg_mode, default="string")
        output_mode = _interpret_output_mode(self.output_mode, default="silent")
        cmd_args = [_parse_value_or_string(a, self.namespace, arg_mode)
                    for a in cmd_args]
        with SysArgvCtx(*cmd_args):
            result = self.namespace.auto_eval(Filename.STDIN)
            print_result(result, output_mode)

    def eval(self, cmd, cmd_args):
        arg_mode = _interpret_arg_mode(self.arg_mode, default="string")
        output_mode = _interpret_output_mode(self.output_mode)
        cmd_args = [_parse_value_or_string(a, self.namespace, arg_mode)
                    for a in cmd_args]
        with SysArgvCtx("-c", *cmd_args):
            cmd = PythonBlock(cmd)
            result = self.namespace.auto_eval(cmd, info=True)
            print_result(result, output_mode)

    def execfile(self, filename, cmd_args):
        # TODO: pass filename to import db target_filename; unit test.
        # TODO: set __file__
        # TODO: support compiled (.pyc/.pyo) files
        arg_mode = _interpret_arg_mode(self.arg_mode, default="string")
        output_mode = _interpret_output_mode(self.output_mode)
        cmd_args = [_parse_value_or_string(a, self.namespace, arg_mode)
                    for a in cmd_args]
        if filename == "-":
            filename = Filename.STDIN
        else:
            filename = Filename(filename)
        if not filename.exists:
            raise Exception("No such file: %s" % filename)
        with SysArgvCtx(str(filename), *cmd_args):
            result = self.namespace.auto_eval(filename)
            print_result(result, output_mode)

    def apply(self, function_name, cmd_args):
        arg_mode = _interpret_arg_mode(self.arg_mode, default="auto")
        output_mode = _interpret_output_mode(self.output_mode)
        # Todo: what should we set argv to?
        result = auto_apply(function_name, cmd_args, self.namespace, arg_mode)
        print_result(result, output_mode)

    def _seems_like_module(self, arg):
        if not is_identifier(arg, dotted=True):
            # It's not a single (dotted) identifier.
            return False
        if not find_missing_imports(arg, [{}]):
            # It's off of a builtin, e.g. "str.upper"
            return False
        m = ModuleHandle(arg)
        if m.parent:
            # Auto-import the parent, which is necessary in order to get the
            # filename of the module.  C{ModuleHandle.filename} does this
            # automatically, but we do it explicitly here so that we log
            # the import of the parent module.
            if not self.namespace.auto_import(str(m.parent.name)):
                return False
        if not m.filename:
            return False
        return True

    def heuristic_cmd(self, cmd, cmd_args, function_name=None):
        output_mode = _interpret_output_mode(self.output_mode)
        # If the "command" is just a module name, then call run_module.  Make
        # sure we check that it's not a builtin.
        if self._seems_like_module(str(cmd)):
            self.heuristic_run_module(str(cmd), cmd_args)
            return
        # FIXME TODO heed arg_mode for non-apply case.  This is tricky to
        # implement; will require assigning some proxy class to sys.argv
        # that's more sophisticated than just logging.
        with SysArgvCtx("-c", *cmd_args):
            # Log the expression before we evaluate it, unless we're likely to
            # log another substantially similar line.  (We can only guess
            # heuristically whether it's interesting enough to log it.  And we
            # can't know whether the result will be callable until we evaluate
            # it.)
            info = not re.match("^[a-zA-Z0-9_.]+$", function_name)
            result = self.namespace.auto_eval(cmd, info=info)
            if callable(result):
                result = auto_apply(result, cmd_args, self.namespace,
                                    self.arg_mode, function_name)
                print_result(result, output_mode)
                sys.argv[:] # mark as accessed
            else:
                if not info:
                    # We guessed wrong earlier and didn't log yet; log now.
                    logger.info(cmd)
                print_result(result, output_mode)
                unaccessed = sys.argv.unaccessed
                if unaccessed:
                    logger.error(
                        "%s is not callable.  Unexpected argument(s): %s",
                        result, " ".join(unaccessed))
                sys.argv[:] # don't complain again

    def run_module(self, module, args):
        arg_mode = _interpret_arg_mode(self.arg_mode, default="string")
        if arg_mode != "string":
            raise NotImplementedError(
                "run_module only supports string arguments")
        module = ModuleHandle(module)
        logger.info("python -m %s", ' '.join([str(module.name)] + args))
        # Imitate 'python -m'.
        # TODO: include only the traceback below runpy.run_module
        #   os.execvp(sys.executable, [sys.argv[0], "-m", modname] + args)
        sys.argv = [str(module.filename)] + args
        import runpy
        runpy.run_module(str(module.name), run_name="__main__")

    def heuristic_run_module(self, module, args):
        module = ModuleHandle(module)
        # If the user ran 'py numpy --version', then print the numpy
        # version, i.e. same as 'py --version numpy'.  This has the
        # downside of shadowing a possible "--version" feature
        # implemented by the module itself.  However, this is probably
        # not a big deal, because (1) a full-featured program that
        # supports --version would normally have a driver script and
        # not rely on 'python -m foo'; (2) it would probably do
        # something similar anyway; (3) the user can do 'py -m foo
        # --version' if necessary.
        if len(args)==1 and args[0] in ["--version", "-version"]:
            self.print_version(module)
            return
        if len(args)==1 and args[0] in ["--help", "-help", "--h", "-h",
                                        "--?", "-?", "?"]:
            usage = _get_help(module.module, str(module.name), 1)
            print(usage)
            return
        if len(args)==1 and args[0] in ["--source", "-source",
                                        "--??", "-??", "??"]:
            usage = _get_help(module.module, str(module.name), 2)
            print(usage)
            return
        self.run_module(module, args)

    def print_version(self, arg):
        if not arg:
            print_version_and_exit()
            return
        if isinstance(arg, (ModuleHandle, types.ModuleType)):
            module = ModuleHandle(arg).module
        else:
            module = self.namespace.auto_eval(arg, mode="eval")
        if not isinstance(module, types.ModuleType):
            raise TypeError("print_version(): got a %s instead of a module"
                            % (type(module).__name__,))
        try:
            version = module.__version__
        except AttributeError:
            raise AttributeError(
                "Module %s does not have a __version__ attribute"
                % (module.__name__,))
        print(version)

    def print_help(self, objname, verbosity=1):
        objname = objname and objname.strip()
        if not objname:
            print(__doc__)
            return
        obj = self.namespace.auto_eval(objname, mode="eval")
        usage = _get_help(obj, objname, verbosity)
        print(usage)

    def _parse_global_opts(self):
        args = list(self.main_args)
        self.verbosity   = 1
        self.arg_mode    = None
        self.output_mode = None
        while args:
            arg = args[0]
            if not arg.startswith("-"):
                break
            if arg.startswith("--"):
                argname = arg[2:]
            else:
                argname = arg[1:]
            argname, equalsign, value = argname.partition("=")
            def popvalue():
                if equalsign:
                    return value
                else:
                    try:
                        return args.pop(0)
                    except IndexError:
                        raise ValueError("expected argument to %s" % arg)
            def novalue():
                if equalsign:
                    raise ValueError("unexpected argument %s" % arg)
            if argname in ["debug", "pdb", "ipdb", "dbg"]:
                # TODO: set a flag that enables entering debugger before
                # evaluating.  (by default we should enter debugger
                # upon Exception [TODO]).
                # TODO: if argument is an integer PID, then attach to that process.
                # TODO: note additional features in documentation feature list
                raise NotImplementedError("TODO: debug mode")
            if argname == "verbose":
                novalue()
                logger.set_level("DEBUG")
                del args[0]
                continue
            if argname in ["quiet", "q"]:
                novalue()
                logger.set_level("ERROR")
                del args[0]
                continue
            if argname in ["safe"]:
                del args[0]
                novalue()
                self.arg_mode = _interpret_arg_mode("string")
                # TODO: make this less hacky, something like
                #   self.import_db = ""
                os.environ["PYFLYBY_PATH"] = "EMPTY"
                os.environ["PYFLYBY_KNOWN_IMPORTS_PATH"] = ""
                os.environ["PYFLYBY_MANDATORY_IMPORTS_PATH"] = ""
                continue
            if argname in ["arguments", "argument", "args", "arg",
                           "arg_mode", "arg-mode", "argmode"]:
                # Interpret --args=eval|string|auto.
                # Note that if the user didn't specify --args, then we
                # intentionally leave C{opts.arg_mode} set to C{None} for now,
                # because the default varies per action.
                del args[0]
                self.arg_mode = _interpret_arg_mode(popvalue())
                continue
            if argname in ["output", "output_mode", "output-mode",
                           "out", "outmode", "out_mode", "out-mode", "o"]:
                del args[0]
                self.output_mode = _interpret_output_mode(popvalue())
                continue
            if argname in ["print", "pprint", "silent", "repr"]:
                del args[0]
                novalue()
                self.output_mode = _interpret_output_mode(argname)
                continue
            break
        self.args = args

    def run(self):
        # Parse global options.
        sys.orig_argv = list(sys.argv)
        self._parse_global_opts()
        # Get the import database.
        self._run_action()
        self._pre_exit()

    def _run_action(self):
        args = self.args
        if not args or args[0] == "-":
            if os.isatty(0):
                # The user specified no arguments (or only a "-") and stdin is a
                # tty.  Run ipython with autoimporter enabled, i.e. equivalent to
                # autoipython.  Note that we directly start IPython in the same
                # process, instead of using subprocess.call(['autoipython']),
                # because the latter messes with Control-C handling.
                start_ipython_with_autoimporter([])
                return
            else:
                # Emulate python args.
                cmd_args = args or [""]
                # Execute code from stdin, with auto importing.
                self.exec_stdin(cmd_args)
                return

        # Consider --action=arg, --action arg, -action=arg, -action arg,
        # %action arg.
        arg0 = args.pop(0)
        if not arg0.strip():
            raise ValueError("got empty string as first argument")
        if arg0.startswith("--"):
            action, equalsign, cmdarg = arg0[2:].partition("=")
        elif arg0.startswith("-"):
            action, equalsign, cmdarg = arg0[1:].partition("=")
        elif arg0.startswith("%"):
            action, equalsign, cmdarg = arg0[1:], None, None
        elif len(arg0) > 1 or arg0 == "?":
            action, equalsign, cmdarg = arg0, None, None
        else:
            action, equalsign, cmdarg = None, None, None
        def popcmdarg():
            if equalsign:
                return cmdarg
            else:
                try:
                    return args.pop(0)
                except IndexError:
                    raise ValueError("expected argument to %s" % arg0)
        def nocmdarg():
            if equalsign:
                raise ValueError("unexpected argument %s" % arg0)

        if action in ["eval", "c", "e"]:
            # Evaluate a command from the command-line, with auto importing.
            # Supports expressions as well as statements.
            # Note: Regular python supports "python -cfoo" as equivalent to
            # "python -c foo".  For now, we intentionally don't support that.
            cmd = popcmdarg()
            self.eval(cmd, args)
        elif action in ["file", "execfile", "execf", "runfile", "run", "f"]:
            # Execute a file named on the command-line, with auto importing.
            cmd = popcmdarg()
            self.execfile(cmd, args)
        elif action in ["apply", "call"]:
            # Call a function named on the command-line, with auto importing and
            # auto evaluation of arguments.
            cmd = popcmdarg()
            self.apply(cmd, args)
        elif action in ["map"]:
            # Call function on each argument.
            # TODO: instead of making this a standalone mode, change this to a
            # flag that can eval/apply/exec/etc.  Set "_" to each argument.
            # E.g. py --map 'print _' obj1 obj2
            #      py --map _.foo obj1 obj2
            #      py --map '_**2' 3 4 5
            # when using heuristic mode, "lock in" the action mode on the
            # first argument.
            cmd = popcmdarg()
            if args and args[0] == '--':
                for arg in args[1:]:
                    self.apply(cmd, ['--', arg])
            else:
                for arg in args:
                    self.apply(cmd, [arg])
        elif action in ["xargs"]:
            # TODO: read lines from stdin and map.  default arg_mode=string
            raise NotImplementedError("TODO: xargs")
        elif action in ["module", "m", "runmodule", "run_module", "run-module"]:
            # Exactly like `python -m'.  Intentionally does NOT do auto
            # importing within the module, because modules should not be
            # sloppy; they should instead be tidied to have the correct
            # imports.
            modname = popcmdarg()
            self.run_module(modname, args)
        elif arg0.startswith("-m"):
            # Support "py -mbase64" in addition to "py -m base64".
            modname = arg0[2:]
            self.run_module(modname, args)
        elif action in ["ipython", "ip"]:
            # Start IPython.
            start_ipython_with_autoimporter(args)
        elif action in ["notebook", "nb"]:
            # Start IPython notebook.
            nocmdarg()
            start_ipython_with_autoimporter(["notebook"] + args)
        elif action in ["kernel"]:
            # Start IPython kernel.
            nocmdarg()
            start_ipython_with_autoimporter(["kernel"] + args)
        elif action in ["qtconsole", "qt"]:
            # Start IPython qtconsole.
            nocmdarg()
            start_ipython_with_autoimporter(["qtconsole"] + args)
        elif action in ["console"]:
            # Start IPython console (with new kernel).
            nocmdarg()
            start_ipython_with_autoimporter(["console"] + args)
        elif action in ["existing"]:
            # Start IPython console (with existing kernel).
            if equalsign:
                args.insert(0, cmdarg)
            start_ipython_with_autoimporter(["console", "--existing"] + args)
        elif action in ["nbconvert"]:
            # Start IPython nbconvert.  (autoimporter is irrelevant.)
            if equalsign:
                args.insert(0, cmdarg)
            start_ipython_with_autoimporter(["nbconvert"] + args)
        elif action in ["timeit"]:
            # TODO: make --timeit and --time flags which work with any mode
            # and heuristic, instead of only eval.
            # TODO: fallback if IPython isn't available.  above todo probably
            # requires not using IPython anyway.
            nocmdarg()
            run_ipython_line_magic("%timeit " + ' '.join(args))
        elif action in ["time"]:
            # TODO: make --timeit and --time flags which work with any mode
            # and heuristic, instead of only eval.
            # TODO: fallback if IPython isn't available.  above todo probably
            # requires not using IPython anyway.
            nocmdarg()
            run_ipython_line_magic("%time " + ' '.join(args))
        elif action in ["version"]:
            if equalsign:
                args.insert(0, cmdarg)
            self.print_version(args[0] if args else None)
        elif action in ["help", "h", "?"]:
            if equalsign:
                args.insert(0, cmdarg)
            self.print_help(args[0] if args else None, verbosity=1)
        elif action in ["pinfo"]:
            self.print_help(popcmdarg(), verbosity=1)
        elif action in ["source", "pinfo2", "??"]:
            self.print_help(popcmdarg(), verbosity=2)

        elif arg0.startswith("-"):
            # Unknown argument.
            msg = "Unknown option %s" % (arg0,)
            if arg0.startswith("-c"):
                msg += "; do you mean -c %s?" % (arg0[2:])
            syntax(msg, usage=usage)

        elif arg0.startswith("??"):
            # TODO: check number of args
            self.print_help(arg0[2:], verbosity=2)

        elif arg0.endswith("??"):
            # TODO: check number of args
            self.print_help(arg0[:-2], verbosity=2)

        elif arg0.startswith("?"):
            # TODO: check number of args
            self.print_help(arg0[1:], verbosity=1)

        elif arg0.endswith("?"):
            # TODO: check number of args
            self.print_help(arg0[:-1], verbosity=1)

        elif arg0.startswith("%"):
            run_ipython_line_magic(' '.join([arg0]+args))

        # Heuristically choose the behavior automatically based on what the
        # argument looks like.
        elif _seems_like_filename(arg0):
            # Implied --execfile.
            self.execfile(arg0, args)
        else:
            # Implied --eval.
            # When given multiple arguments, first see if the args can be
            # concatenated and parsed as a single python program/expression.
            # But don't try this if any arguments look like options, empty
            # string or whitespace, etc.
            if (args and
                self.arg_mode != "string" and
                not any(re.match("\s*$|-[a-zA-Z-]", a) for a in args)):
                cmd = PythonBlock(" ".join([arg0]+args),
                                  flags=FLAGS, auto_flags=True)
                if cmd.parsable:
                    # TODO: check for failure to auto import everything and
                    # fall back to not joining, e.g. 'py print foo' vs 'py
                    # --apply print foo' => auto args.
                    self.arg_mode = "error"
                    self.eval(cmd, [])
                    return
                # else fall through
            # Heuristic based on first arg: try run_module, apply, or exec/eval.
            cmd = PythonBlock(arg0, flags=FLAGS, auto_flags=True)
            if not cmd.parsable:
                logger.error(
                    "Could not interpret as filename or expression: %s",
                    arg0)
                syntax(usage=usage)
            self.heuristic_cmd(cmd, args, function_name=arg0)

    def _pre_exit(self):
        self._matplotlib_show()

    def _matplotlib_show(self):
        """
        If matplotlib.pyplot (pylab) is loaded, then call the show() function.
        This will cause the program to block until all figures are closed.
        Without this, a command like 'py plot(...)' would exit immediately.
        """
        try:
            pyplot = sys.modules["matplotlib.pyplot"]
        except KeyError:
            return
        pyplot.show()


def py_main(args=None):
    if args is None:
        args = sys.argv[1:]
    _PyMain(args).run()