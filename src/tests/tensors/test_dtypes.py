import xos

# Hardcoded bounds (must match Rust DType::min_f64 / max_f64 in dtypes.rs).
DTYPE_MIN_MAX = [
    ("float16", -65504.0, 65504.0),
    ("float32", -3.4028234663852886e38, 3.4028234663852886e38),
    ("float64", -1.7976931348623157e308, 1.7976931348623157e308),
    ("int8", -128.0, 127.0),
    ("int16", -32768.0, 32767.0),
    ("int32", -2147483648.0, 2147483647.0),
    # i64/u64 MAX are not exactly representable as Python floats; use f64-rounded values.
    ("int64", -9223372036854775808.0, 9.223372036854776e18),
    ("uint8", 0.0, 255.0),
    ("uint16", 0.0, 65535.0),
    ("uint32", 0.0, 4294967295.0),
    ("uint64", 0.0, 1.8446744073709552e19),
    ("bool", 0.0, 1.0),
]


@xos.test
def test_dtype_min_max_constants():
    for name, expected_min, expected_max in DTYPE_MIN_MAX:
        dtype = getattr(xos, name)
        assert dtype.MIN == expected_min, name
        assert dtype.MAX == expected_max, name


@xos.test
def test_int_alias_min_max_matches_int32():
    assert xos.int.MIN == xos.int32.MIN
    assert xos.int.MAX == xos.int32.MAX


@xos.test
def test_float_alias_min_max_matches_float32():
    assert xos.float.MIN == xos.float32.MIN
    assert xos.float.MAX == xos.float32.MAX


@xos.test
def test_tensor_randomize_uint8_in_range():
    t = xos.zeros((8, 8, 3), dtype=xos.uint8)
    t.randomize()
    flat = list(t)
    assert len(flat) == 8 * 8 * 3
    for v in flat:
        assert v >= xos.uint8.MIN
        assert v <= xos.uint8.MAX
