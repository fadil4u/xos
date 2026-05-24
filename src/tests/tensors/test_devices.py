import xos


@xos.test
def test_default_device():
    # for now, we always have the default device as cpu no matter what.
    x = xos.tensor([1, 2, 3])
    assert x.device == "cpu"

    # this might change in the future, but for now again it's always cpu.
    assert xos.default_device == "cpu"