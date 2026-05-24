import xos


def test_convolutions():
    kernel_shape = (3, 3, 3)
    dtype = xos.float32
    cpu_x = xos.random.uniform(0.0, 1.0, shape=kernel_shape, dtype=dtype, device="cpu")
    gpu_x = xos.random.uniform(0.0, 1.0, shape=kernel_shape, dtype=dtype, device="gpu")

    cpu_y = xos.ops.convolve(cpu_x, gpu_x, inplace=False)
    gpu_y = xos.ops.convolve(gpu_x, gpu_x, inplace=False)

    print(cpu_y.sum(), gpu_y.sum())

    assert cpu_y.shape == gpu_y.shape
    assert xos.allclose(cpu_y, gpu_y)