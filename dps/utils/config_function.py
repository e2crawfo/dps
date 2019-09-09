# Stole this from sacred, which has an MIT license.
import ast
import inspect
import io
import re
from tokenize import tokenize, TokenError, COMMENT
import copy
from inspect import Parameter
from collections import OrderedDict
import numpy as np


ARG_TYPES = [
    Parameter.POSITIONAL_ONLY,
    Parameter.POSITIONAL_OR_KEYWORD,
    Parameter.KEYWORD_ONLY,
]
POSARG_TYPES = [Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD]
PYTHON_IDENTIFIER = re.compile("^[a-zA-Z_][_a-zA-Z0-9]*$")


def assert_is_valid_key(key):
    """
    Raise KeyError if a given config key violates any requirements.
    The requirements are the following and can be individually deactivated
    in ``sacred.SETTINGS.CONFIG_KEYS``:
      * ENFORCE_STRING (default: False):
        make sure all keys are string.
      * ENFORCE_VALID_PYTHON_IDENTIFIER (default: False):
        make sure all keys are valid python identifiers.
    Parameters
    ----------
    key:
      The key that should be checked
    Raises
    ------
    KeyError:
      if the key violates any requirements
    """
    if not isinstance(key, str):
        raise KeyError(
            'Invalid key "{}". Config-keys have to be strings, '
            "but was {}".format(key, type(key))
        )

    if isinstance(key, str) and not PYTHON_IDENTIFIER.match(key):
        raise KeyError('Key "{}" is not a valid python identifier'.format(key))

    if isinstance(key, str) and "=" in key:
        raise KeyError(
            'Invalid key "{}". Config keys may not contain an'
            'equals sign ("=").'.format("=")
        )


def get_argspec(f):
    sig = inspect.signature(f)
    args = [n for n, p in sig.parameters.items() if p.kind in ARG_TYPES]
    pos_args = [
        n
        for n, p in sig.parameters.items()
        if p.kind in POSARG_TYPES and p.default == inspect._empty
    ]
    varargs = [
        n for n, p in sig.parameters.items() if p.kind == Parameter.VAR_POSITIONAL
    ]
    # only use first vararg  (how on earth would you have more anyways?)
    vararg_name = varargs[0] if varargs else None

    varkws = [n for n, p in sig.parameters.items() if p.kind == Parameter.VAR_KEYWORD]
    # only use first varkw  (how on earth would you have more anyways?)
    kw_wildcard_name = varkws[0] if varkws else None
    kwargs = OrderedDict(
        [
            (n, p.default)
            for n, p in sig.parameters.items()
            if p.default != inspect._empty
        ]
    )

    return args, vararg_name, kw_wildcard_name, pos_args, kwargs


def dogmatize(obj):
    if isinstance(obj, dict):
        return DogmaticDict({key: dogmatize(val) for key, val in obj.items()})
    elif isinstance(obj, list):
        return DogmaticList([dogmatize(value) for value in obj])
    elif isinstance(obj, tuple):
        return tuple(dogmatize(value) for value in obj)
    else:
        return obj


def normalize_numpy(obj):
    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except ValueError:
            pass
    return obj


def normalize_or_die(obj):
    if isinstance(obj, dict):
        res = dict()
        for key, value in obj.items():
            assert_is_valid_key(key)
            res[key] = normalize_or_die(value)
        return res
    elif isinstance(obj, (list, tuple)):
        return list([normalize_or_die(value) for value in obj])
    return normalize_numpy(obj)


def recursive_fill_in(config, preset):
    for key in preset:
        if key not in config:
            config[key] = preset[key]
        elif isinstance(config[key], dict) and isinstance(preset[key], dict):
            recursive_fill_in(config[key], preset[key])


class _ConfigScope:
    def __init__(self, func):
        self.args, vararg_name, kw_wildcard, _, kwargs = get_argspec(func)
        assert vararg_name is None, "*args not allowed for ConfigScope functions"
        assert kw_wildcard is None, "**kwargs not allowed for ConfigScope functions"
        assert not kwargs, "default values are not allowed for ConfigScope functions"

        self._func = func
        self._body_code = get_function_body_code(func)
        self._var_docs = get_config_comments(func)
        self.__doc__ = self._func.__doc__

    def __call__(self, fixed=None, preset=None, fallback=None):
        """
        Evaluate this ConfigScope.
        This will evaluate the function body and fill the relevant local
        variables into entries into keys in this dictionary.
        :param fixed: Dictionary of entries that should stay fixed during the
                      evaluation. All of them will be part of the final config.
        :type fixed: dict
        :param preset: Dictionary of preset values that will be available
                       during the evaluation (if they are declared in the
                       function argument list). All of them will be part of the
                       final config.
        :type preset: dict
        :param fallback: Dictionary of fallback values that will be available
                         during the evaluation (if they are declared in the
                         function argument list). They will NOT be part of the
                         final config.
        :type fallback: dict
        :return: self
        :rtype: ConfigScope
        """
        cfg_locals = dogmatize(fixed or {})
        fallback = fallback or {}
        preset = preset or {}
        fallback_view = {}

        available_entries = set(preset.keys()) | set(fallback.keys())

        for arg in self.args:
            if arg not in available_entries:
                raise KeyError(
                    "'{}' not in preset for ConfigScope. "
                    "Available options are: {}".format(arg, available_entries)
                )
            if arg in preset:
                cfg_locals[arg] = preset[arg]
            else:  # arg in fallback
                fallback_view[arg] = fallback[arg]

        cfg_locals.fallback = fallback_view
        eval(self._body_code, copy.copy(self._func.__globals__), cfg_locals)

        added = cfg_locals.revelation()
        config_summary = ConfigSummary(
            added,
            cfg_locals.modified,
            cfg_locals.typechanges,
            cfg_locals.fallback_writes,
            docs=self._var_docs,
        )
        # fill in the unused presets
        recursive_fill_in(cfg_locals, preset)

        for key, value in cfg_locals.items():
            try:
                config_summary[key] = normalize_or_die(value)
            except ValueError:
                pass
        return config_summary


class ConfigSummary(dict):
    def __init__(
        self, added=(), modified=(), typechanged=(), ignored_fallbacks=(), docs=()
    ):
        super().__init__()
        self.added = set(added)
        self.modified = set(modified)  # TODO: test for this member
        self.typechanged = dict(typechanged)
        self.ignored_fallbacks = set(ignored_fallbacks)  # TODO: test
        self.docs = dict(docs)
        self.ensure_coherence()

    def update_from(self, config_mod, path=""):
        added = config_mod.added
        updated = config_mod.modified
        typechanged = config_mod.typechanged
        self.added &= {join_paths(path, a) for a in added}
        self.modified |= {join_paths(path, u) for u in updated}
        self.typechanged.update(
            {join_paths(path, k): v for k, v in typechanged.items()}
        )
        self.ensure_coherence()
        for k, v in config_mod.docs.items():
            if not self.docs.get(k, ""):
                self.docs[k] = v

    def update_add(self, config_mod, path=""):
        added = config_mod.added
        updated = config_mod.modified
        typechanged = config_mod.typechanged
        self.added |= {join_paths(path, a) for a in added}
        self.modified |= {join_paths(path, u) for u in updated}
        self.typechanged.update(
            {join_paths(path, k): v for k, v in typechanged.items()}
        )
        self.docs.update(
            {
                join_paths(path, k): v
                for k, v in config_mod.docs.items()
                if path == "" or k != "seed"
            }
        )
        self.ensure_coherence()

    def ensure_coherence(self):
        # make sure parent paths show up as updated appropriately
        self.modified |= {p for a in self.added for p in iter_prefixes(a)}
        self.modified |= {p for u in self.modified for p in iter_prefixes(u)}
        self.modified |= {p for t in self.typechanged for p in iter_prefixes(t)}

        # make sure there is no overlap
        self.added -= set(self.typechanged.keys())
        self.modified -= set(self.typechanged.keys())
        self.modified -= self.added


def get_function_body(func):
    func_code_lines, start_idx = inspect.getsourcelines(func)
    func_code = "".join(func_code_lines)
    arg = "(?:[a-zA-Z_][a-zA-Z0-9_]*)"
    arguments = r"{0}(?:\s*,\s*{0})*".format(arg)
    func_def = re.compile(
        r"^[ \t]*def[ \t]*{}[ \t]*\(\s*({})?\s*\)[ \t]*:[ \t]*\n".format(
            func.__name__, arguments
        ),
        flags=re.MULTILINE,
    )
    defs = list(re.finditer(func_def, func_code))
    assert defs
    line_offset = start_idx + func_code[: defs[0].end()].count("\n") - 1
    func_body = func_code[defs[0].end():]
    return func_body, line_offset


def is_empty_or_comment(line):
    sline = line.strip()
    return sline == "" or sline.startswith("#")


def iscomment(line):
    return line.strip().startswith("#")


def dedent_line(line, indent):
    for i, (line_sym, indent_sym) in enumerate(zip(line, indent)):
        if line_sym != indent_sym:
            start = i
            break
    else:
        start = len(indent)
    return line[start:]


def dedent_function_body(body):
    lines = body.split("\n")
    # find indentation by first line
    indent = ""
    for line in lines:
        if is_empty_or_comment(line):
            continue
        else:
            indent = re.match(r"^\s*", line).group()
            break

    out_lines = [dedent_line(line, indent) for line in lines]
    return "\n".join(out_lines)


def get_function_body_code(func):
    filename = inspect.getfile(func)
    func_body, line_offset = get_function_body(func)
    body_source = dedent_function_body(func_body)
    try:
        body_code = compile(body_source, filename, "exec", ast.PyCF_ONLY_AST)
        body_code = ast.increment_lineno(body_code, n=line_offset)
        body_code = compile(body_code, filename, "exec")
    except SyntaxError as e:
        if e.args[0] == "'return' outside function":
            filename, lineno, _, statement = e.args[1]
            raise SyntaxError(
                "No return statements allowed in ConfigScopes\n"
                "('{}' in File \"{}\", line {})".format(
                    statement.strip(), filename, lineno
                )
            )
        elif e.args[0] == "'yield' outside function":
            filename, lineno, _, statement = e.args[1]
            raise SyntaxError(
                "No yield statements allowed in ConfigScopes\n"
                "('{}' in File \"{}\", line {})".format(
                    statement.strip(), filename, lineno
                )
            )
        else:
            raise
    return body_code


IGNORED_COMMENTS = ["^pylint:", "^noinspection"]


def is_ignored(line):
    for pattern in IGNORED_COMMENTS:
        if re.match(pattern, line) is not None:
            return True
    return False


def find_doc_for(ast_entry, body_lines):
    lineno = ast_entry.lineno - 1
    line_io = io.BytesIO(body_lines[lineno].encode())
    try:
        tokens = tokenize(line_io.readline) or []
        line_comments = [t.string for t in tokens if t.type == COMMENT]

        if line_comments:
            formatted_lcs = [l[1:].strip() for l in line_comments]
            filtered_lcs = [l for l in formatted_lcs if not is_ignored(l)]
            if filtered_lcs:
                return filtered_lcs[0]
    except TokenError:
        pass

    lineno -= 1
    while lineno >= 0:
        if iscomment(body_lines[lineno]):
            comment = body_lines[lineno].strip("# ")
            if not is_ignored(comment):
                return comment
        if not body_lines[lineno].strip() == "":
            return None
        lineno -= 1
    return None


def add_doc(target, variables, body_lines):
    if isinstance(target, ast.Name):
        # if it is a variable name add it to the doc
        name = target.id
        if name not in variables:
            doc = find_doc_for(target, body_lines)
            if doc is not None:
                variables[name] = doc
    elif isinstance(target, ast.Tuple):
        # if it is a tuple then iterate the elements
        # this can happen like this:
        # a, b = 1, 2
        for e in target.elts:
            add_doc(e, variables, body_lines)


def get_config_comments(func):
    filename = inspect.getfile(func)
    func_body, line_offset = get_function_body(func)
    body_source = dedent_function_body(func_body)
    body_code = compile(body_source, filename, "exec", ast.PyCF_ONLY_AST)
    body_lines = body_source.split("\n")

    variables = {"seed": "the random seed for this experiment"}

    for ast_root in body_code.body:
        for ast_entry in [ast_root] + list(ast.iter_child_nodes(ast_root)):
            if isinstance(ast_entry, ast.Assign):
                # we found an assignment statement
                # go through all targets of the assignment
                # usually a single entry, but can be more for statements like:
                # a = b = 5
                for t in ast_entry.targets:
                    add_doc(t, variables, body_lines)

    return variables


def fallback_dict(fallback, **kwargs):
    fallback_copy = fallback.copy()
    fallback_copy.update(kwargs)
    return fallback_copy


def join_paths(*parts):
    """Join different parts together to a valid dotted path."""
    return ".".join(str(p).strip(".") for p in parts if p)


def iter_prefixes(path):
    """
    Iterate through all (non-empty) prefixes of a dotted path.
    Example
    -------
    >>> list(iter_prefixes('foo.bar.baz'))
    ['foo', 'foo.bar', 'foo.bar.baz']
    """
    split_path = path.split(".")
    for i in range(1, len(split_path) + 1):
        yield join_paths(*split_path[:i])


class DogmaticDict(dict):
    def __init__(self, fixed=None, fallback=None):
        super().__init__()
        self.typechanges = {}
        self.fallback_writes = []
        self.modified = set()
        self.fixed = fixed or {}
        self._fallback = {}
        if fallback:
            self.fallback = fallback

    @property
    def fallback(self):
        return self._fallback

    @fallback.setter
    def fallback(self, newval):
        ffkeys = set(self.fixed.keys()).intersection(set(newval.keys()))
        for k in ffkeys:
            if isinstance(self.fixed[k], DogmaticDict):
                self.fixed[k].fallback = newval[k]
            elif isinstance(self.fixed[k], dict):
                self.fixed[k] = DogmaticDict(self.fixed[k])
                self.fixed[k].fallback = newval[k]

        self._fallback = newval

    def _log_blocked_setitem(self, key, value, fixed_value):
        if type_changed(value, fixed_value):
            self.typechanges[key] = (type(value), type(fixed_value))

        if is_different(value, fixed_value):
            self.modified.add(key)

        # if both are dicts recursively collect modified and typechanges
        if isinstance(fixed_value, DogmaticDict) and isinstance(value, dict):
            for k, val in fixed_value.typechanges.items():
                self.typechanges[join_paths(key, k)] = val

            self.modified |= {join_paths(key, m) for m in fixed_value.modified}

    def __setitem__(self, key, value):
        if key not in self.fixed:
            if key in self.fallback:
                self.fallback_writes.append(key)
            return dict.__setitem__(self, key, value)

        fixed_value = self.fixed[key]
        dict.__setitem__(self, key, fixed_value)
        # if both are dicts do a recursive update
        if isinstance(fixed_value, DogmaticDict) and isinstance(value, dict):
            for k, val in value.items():
                fixed_value[k] = val

        self._log_blocked_setitem(key, value, fixed_value)

    def __getitem__(self, item):
        if dict.__contains__(self, item):
            return dict.__getitem__(self, item)
        elif item in self.fallback:
            if item in self.fixed:
                return self.fixed[item]
            else:
                return self.fallback[item]
        raise KeyError(item)

    def __contains__(self, item):
        return dict.__contains__(self, item) or (item in self.fallback)

    def get(self, k, d=None):
        if dict.__contains__(self, k):
            return dict.__getitem__(self, k)
        else:
            return self.fallback.get(k, d)

    def has_key(self, item):
        return self.__contains__(item)

    def __delitem__(self, key):
        if key not in self.fixed:
            dict.__delitem__(self, key)

    def update(self, iterable=None, **kwargs):
        if iterable is not None:
            if hasattr(iterable, "keys"):
                for key in iterable:
                    self[key] = iterable[key]
            else:
                for (key, value) in iterable:
                    self[key] = value
        for key in kwargs:
            self[key] = kwargs[key]

    def revelation(self):
        missing = set()
        for key in self.fixed:
            if not dict.__contains__(self, key):
                self[key] = self.fixed[key]
                missing.add(key)

            if isinstance(self[key], (DogmaticDict, DogmaticList)):
                missing |= {key + "." + k for k in self[key].revelation()}
        return missing


class DogmaticList(list):
    def append(self, p_object):
        pass

    def extend(self, iterable):
        pass

    def insert(self, index, p_object):
        pass

    def reverse(self):
        pass

    def sort(self, compare=None, key=None, reverse=False):
        pass

    def __iadd__(self, other):
        return self

    def __imul__(self, other):
        return self

    def __setitem__(self, key, value):
        pass

    def __setslice__(self, i, j, sequence):
        pass

    def __delitem__(self, key):
        pass

    def __delslice__(self, i, j):
        pass

    def pop(self, index=None):
        raise TypeError("Cannot pop from DogmaticList")

    def remove(self, value):
        pass

    def revelation(self):
        for obj in self:
            if isinstance(obj, (DogmaticDict, DogmaticList)):
                obj.revelation()
        return set()


SIMPLIFY_TYPE = {
    type(None): type(None),
    bool: bool,
    float: float,
    int: int,
    str: str,
    list: list,
    tuple: list,
    dict: dict,
    DogmaticDict: dict,
    DogmaticList: list,
}

NP_FLOATS = ["float", "float16", "float32", "float64", "float128"]
for npf in NP_FLOATS:
    if hasattr(np, npf):
        SIMPLIFY_TYPE[getattr(np, npf)] = float

NP_INTS = [
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]
for npi in NP_INTS:
    if hasattr(np, npi):
        SIMPLIFY_TYPE[getattr(np, npi)] = int

SIMPLIFY_TYPE[np.bool_] = bool


def type_changed(old_value, new_value):
    sot = SIMPLIFY_TYPE.get(type(old_value), type(old_value))
    snt = SIMPLIFY_TYPE.get(type(new_value), type(new_value))
    return sot != snt and old_value is not None  # ignore typechanges from None


def is_different(old_value, new_value):
    """Numpy aware comparison between two values."""
    return not np.array_equal(old_value, new_value)
