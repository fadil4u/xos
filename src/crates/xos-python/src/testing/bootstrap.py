# Loaded by Rust into the xos module (no extra imports — xos only in test files).

_REGISTRY = []


def test(fn):
    """Mark a function as an xos test (like pytest)."""
    fn._xos_is_test = True
    return fn


def parametrize(*args, **kwargs):
    """@xos.parametrize("name", [a, b, ...])"""
    if kwargs:
        raise TypeError("xos.parametrize only supports positional (name, values)")
    if len(args) < 2:
        raise TypeError("xos.parametrize needs a name and a list of values")
    name = args[0]
    values = args[1]

    def decorator(fn):
        params = getattr(fn, "_xos_parametrizations", None)
        if params is None:
            # Backward-compat: migrate legacy single-param metadata if present.
            legacy = getattr(fn, "_xos_parametrize", None)
            params = []
            if legacy:
                params.append(legacy)
            fn._xos_parametrizations = params
        fn._xos_parametrizations.append((name, list(values)))
        # Keep legacy field set so older readers still see the last param.
        fn._xos_parametrize = (name, list(values))
        return fn

    return decorator


def _collect_cases(fn):
    params = getattr(fn, "_xos_parametrizations", None)
    if not params:
        legacy = getattr(fn, "_xos_parametrize", None)
        if legacy:
            params = [legacy]
    if params:
        out = [(fn.__name__, {}, fn)]
        for name, values in params:
            next_out = []
            for test_id, kwargs, f in out:
                for v in values:
                    kw = dict(kwargs)
                    kw[name] = v
                    next_out.append((test_id, kw, f))
            out = next_out
        return out
    return [(fn.__name__, {}, fn)]


def _clear_registry():
    _REGISTRY[:] = []


def _register_module_tests(namespace):
    """Register @xos.test callables from the current module namespace."""
    for _key, obj in namespace.items():
        if callable(obj) and getattr(obj, "_xos_is_test", False):
            for test_id, kwargs, fn in _collect_cases(obj):
                _REGISTRY.append((test_id, kwargs, fn))


def _read_source_line(path, lineno):
    if not path or path.startswith("<"):
        return None
    try:
        f = open(path, "r")
        try:
            for i, line in enumerate(f, 1):
                if i == lineno:
                    return line.rstrip("\n\r")
                if i > lineno:
                    break
        finally:
            f.close()
    except Exception:
        return None
    return None


def _short_path(path):
    if "/" in path:
        return path.split("/")[-1]
    if "\\" in path:
        return path.split("\\")[-1]
    return path


def _is_test_source_path(path):
    norm = path.replace("\\", "/")
    if not path or path.startswith("<"):
        return False
    return "/tests/" in norm


def _failure_site(exc):
    """Innermost traceback frame in a user test file (not bootstrap/stdlib)."""
    tb = exc.__traceback__
    site = None
    while tb is not None:
        frame = tb.tb_frame
        path = frame.f_code.co_filename
        if _is_test_source_path(path):
            site = (path, tb.tb_lineno, frame.f_code.co_name)
        tb = tb.tb_next
    return site


def _assertion_message(exc):
    args = getattr(exc, "args", ())
    if not args or args[0] is None:
        return None
    try:
        text = str(args[0])
    except Exception:
        text = repr(args[0])
    if text:
        return text
    return None


def _format_failure(exc):
    """Format an exception without importing stdlib (traceback/sys not available)."""
    name = type(exc).__name__
    msg = str(exc)
    assert_msg = None
    if name == "AssertionError":
        assert_msg = _assertion_message(exc)
        if assert_msg and not msg:
            msg = assert_msg
    if not msg and getattr(exc, "args", ()):
        parts = []
        for a in exc.args:
            if a is None:
                continue
            try:
                parts.append(str(a))
            except Exception:
                parts.append(repr(a))
        if parts:
            msg = ", ".join(parts)
    lines = []
    site = _failure_site(exc)
    if site:
        path, lineno, func = site
        src = _read_source_line(path, lineno)
        lines.append("{} at {}:{} in {}".format(name, _short_path(path), lineno, func))
        if src:
            lines.append("  > {}".format(src))
        if assert_msg:
            lines.append("  assert: {}".format(assert_msg))
        elif msg:
            lines.append("  {}".format(msg))
    else:
        if name == "AssertionError" and not msg:
            msg = "condition was false (use assert cond, \"message\" for details)"
        lines.append("{}: {}".format(name, msg))
        if assert_msg and assert_msg != msg:
            lines.append("  assert: {}".format(assert_msg))
    tb = exc.__traceback__
    lines.append("  traceback:")
    while tb is not None:
        frame = tb.tb_frame
        code = frame.f_code
        lines.append(
            '    File "{}", line {}, in {}'.format(
                code.co_filename, tb.tb_lineno, code.co_name
            )
        )
        tb = tb.tb_next
    return "\n".join(lines)


def _run_all(filter_name=None):
    passed = 0
    failed = 0
    errors = []

    cases = list(_REGISTRY)
    if filter_name:
        cases = [c for c in cases if c[0] == filter_name]

    if not cases:
        if filter_name:
            xos.print_color(
                "&cNo tests named &f{}&c (check @xos.test and src/tests/**/*.py)".format(
                    filter_name
                )
            )
        else:
            xos.print_color("&cNo tests collected&r (check @xos.test and src/tests/**/*.py)")
        return False

    for test_id, kwargs, fn in cases:
        label = test_id
        if kwargs:
            parts = ["{}={!r}".format(k, kwargs[k]) for k in sorted(kwargs.keys())]
            label = "{} [{}]".format(test_id, ", ".join(parts))

        xos.print_color("&7▶ &f{}".format(label))
        try:
            if kwargs:
                fn(**kwargs)
            else:
                fn()
            passed += 1
            xos.print_color("  &a✓ passed&r")
        except Exception as e:
            failed += 1
            report = _format_failure(e)
            errors.append((label, report))
            xos.print_color("  &c✗ failed&r")
            for line in report.split("\n"):
                if line.startswith("  > ") or line.startswith("  assert: "):
                    xos.print_color("  &c{}&r".format(line))
                elif (
                    line.startswith("AssertionError at ")
                    or line.startswith("AssertionError:")
                    or line.startswith("NameError at ")
                    or line.startswith("NameError:")
                    or (
                        line.startswith("  ")
                        and not line.startswith("  > ")
                        and not line.startswith("  assert: ")
                        and not line.startswith("  traceback:")
                        and not line.startswith('    File "')
                    )
                ):
                    xos.print_color("  &c{}&r".format(line))
                else:
                    xos.print_color("  &8{}&r".format(line))

    xos.print("")
    if failed == 0:
        xos.print_color("&a{} passed&r".format(passed))
    else:
        xos.print_color("&a{} passed&r, &c{} failed&r".format(passed, failed))
    return failed == 0
