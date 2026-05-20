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
        fn._xos_parametrize = (name, list(values))
        return fn

    return decorator


def _collect_cases(fn):
    if getattr(fn, "_xos_parametrize", None):
        name, values = fn._xos_parametrize
        out = []
        for v in values:
            out.append((fn.__name__, {name: v}, fn))
        return out
    return [(fn.__name__, {}, fn)]


def _register_module_tests(namespace):
    for _key, obj in namespace.items():
        if callable(obj) and getattr(obj, "_xos_is_test", False):
            for test_id, kwargs, fn in _collect_cases(obj):
                _REGISTRY.append((test_id, kwargs, fn))


def _run_all():
    import traceback

    passed = 0
    failed = 0
    errors = []

    for test_id, kwargs, fn in list(_REGISTRY):
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
        except Exception:
            failed += 1
            tb = traceback.format_exc()
            errors.append((label, tb))
            xos.print_color("  &c✗ failed&r")
            for line in tb.strip().split("\n"):
                xos.print_color("  &8{}&r".format(line))

    xos.print("")
    if failed == 0:
        xos.print_color("&a{} passed&r".format(passed))
    else:
        xos.print_color("&a{} passed&r, &c{} failed&r".format(passed, failed))
    return failed == 0
