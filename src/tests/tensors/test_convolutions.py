import xos


INPLACE = xos.parametrize("inplace", [True, False])
KERNEL_SHAPES = xos.parametrize("kernel_shape", [(3, 3, 3)])
INPUT_SHAPES = xos.parametrize("input_shape", [(128, 128, 3)])
DTYPES = xos.parametrize("dtype", [xos.float32, xos.float64])


@xos.test
@INPLACE
@KERNEL_SHAPES
@INPUT_SHAPES
@DTYPES
def convolutions(
    inplace: bool,
    kernel_shape: tuple,
    input_shape: tuple,
    dtype,
):
    cpu_x = xos.random.uniform(0.0, 1.0, shape=input_shape, dtype=dtype, device="cpu")
    cpu_kernel = xos.random.uniform(0.0, 1.0, shape=kernel_shape, dtype=dtype, device="cpu")
    
    # cloning should implicitly already happen when you do this
    gpu_x = cpu_x.to("gpu")
    gpu_kernel = cpu_kernel.to("gpu")

    cpu_y = xos.ops.convolve(cpu_x, cpu_kernel, inplace=inplace)
    gpu_y = xos.ops.convolve(gpu_x, gpu_kernel, inplace=inplace)

    print(cpu_y.sum(), gpu_y.sum())
    print(cpu_y.shape, gpu_y.shape)

    assert cpu_y.shape == gpu_y.shape
    assert xos.allclose(cpu_y, gpu_y)

    # TODO: same test, but inside of an xos.Application with the Frame's tensor (graphical backing)

@xos.test
@INPLACE
@KERNEL_SHAPES
@INPUT_SHAPES
@DTYPES
def frame_convolutions(
    inplace: bool,
    kernel_shape: tuple,
    input_shape: tuple,
    dtype,
):
    headless = True

    h, w, _ = input_shape
    cpu_app = xos.Application(device="cpu", headless=headless, width=w, height=h)
    gpu_app = xos.Application(device="gpu", headless=headless, width=w, height=h)

    assert cpu_app.frame.tensor.shape == gpu_app.frame.tensor.shape
    assert cpu_app.frame.tensor.shape[0] == h
    assert cpu_app.frame.tensor.shape[1] == w
    assert cpu_app.frame.tensor.shape[2] == 4

    # Share identical randomized seed state across frame-backed CPU/GPU tensors.
    cpu_app.frame.tensor.randomize()
    gpu_app.frame.tensor[:] = cpu_app.frame.tensor.to("gpu")

    # check devices are correct
    assert cpu_app.frame.tensor.device == "cpu"
    assert gpu_app.frame.tensor.device == "gpu"

    assert cpu_app.frame.tensor.sum() > 0.0
    assert gpu_app.frame.tensor.sum() > 0.0
    assert xos.allclose(cpu_app.frame.tensor, gpu_app.frame.tensor)

    print(cpu_app.frame.tensor.sum(), gpu_app.frame.tensor.sum())

    # now try the convolution and kernel initializations
    cpu_kernel = xos.random.uniform(0.0, 1.0, shape=kernel_shape, dtype=dtype, device="cpu")
    gpu_kernel = cpu_kernel.to("gpu")

    assert cpu_kernel.sum() > 0.0
    assert gpu_kernel.sum() > 0.0
    assert xos.allclose(cpu_kernel, gpu_kernel)

    print(cpu_kernel.sum(), gpu_kernel.sum())

    # now try the convolution
    cpu_y = xos.ops.convolve(cpu_app.frame.tensor, cpu_kernel, inplace=inplace)
    gpu_y = xos.ops.convolve(gpu_app.frame.tensor, gpu_kernel, inplace=inplace)

    print(cpu_y.sum(), gpu_y.sum())
    print(cpu_y.shape, gpu_y.shape)
    assert cpu_y.shape == gpu_y.shape
    assert xos.allclose(cpu_y, gpu_y)
