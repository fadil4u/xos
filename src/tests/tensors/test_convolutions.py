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
    cpu_app = xos.Application(device="cpu", headless=headless)
    gpu_app = xos.Application(device="gpu", headless=headless)


