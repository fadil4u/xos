import xos


@xos.test
@xos.parametrize("value", [0, 7, 3.5, True, False])
def test_tensor_scalar_string_repr(value):
    tensor = xos.tensor(value)

    if isinstance(value, bool):
        expected_dtype = "bool"
        literal = "True" if value else "False"
    elif isinstance(value, int):
        expected_dtype = "int32"
        literal = str(value)
    else:
        expected_dtype = "float32"
        literal = str(float(value))

    expected = f"xos.Tensor({literal}, dtype={expected_dtype}, device='cpu')"
    assert str(tensor) == expected, f"{str(tensor)} != {expected}"
    assert repr(tensor) == expected, f"{repr(tensor)} != {expected}"

    print(str(tensor))
