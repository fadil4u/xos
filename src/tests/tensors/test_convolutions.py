import xos


@xos.test
@xos.parametrize("inplace", [True, False])
def convolutions(inplace: bool):
    kernel_shape = (3, 3, 3)
    input_shape = (128, 128, 3)
    dtype = xos.float32
    
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