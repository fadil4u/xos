import xos

@xos.test
def blank_test():
    # just a blank passing test. a freebie!
    pass

@xos.test
@xos.parametrize("dtype", [xos.uint8, xos.float32])
def test_squares(dtype):
    tensor = xos.zeros((180, 180, 3), dtype=dtype)

    rects = xos.tensor([
        [10, 10, 20, 20],
        [30, 30, 40, 40],
        [50, 50, 60, 60],
        [70, 70, 80, 80],
        [90, 90, 100, 100],
        [110, 110, 120, 120],
        [130, 130, 140, 140],
        [150, 150, 160, 160],
    ])

    # TODO: test all configuration of squeeze shapes/tuple initializations and whatnot
    colors = xos.tensor([
        (255, 0, 0),  # same color for them all
    ])

    xos.rasterizer.fill_rectangles(tensor, rects, colors)

    # this should be the api for rendering the frame to the screen
    viewport = xos.render(tensor)
    # viewport.pause()
    # viewport = xos.render(tensor)
    # viewport.pause()

    # tensor.printpack()

    # viewport.render(tensor)  # this could also be called for subsequent updates to this frame, especially with pause=False for a live loop animation  TODO: testing for that

    # TODO: hard-code what the raster should look like after verifying


@xos.test
@xos.parametrize("compress", [True, False])
def test_printpack(compress):
    # print packing is a good way for us to generate and maintain test cases super easily.
    tensor = xos.zeros((32, 32, 3), dtype=xos.uint8).randomize()  # .randomize() will automatically randomize the tensor according to the max and min values of the dtype.
    packed_str = tensor.printpack(compress=compress)
    assert type(packed_str) == str
    print(packed_str)

    # make sure that we can unpack it into the original tensor. it should automatically recognize when its compressed or not as well.
    tensor2 = xos.tensor(packed_str)

    # it should be capable of knowing that its a printpacked string, and just automatically unpack and initialize the tensor from it.
    # that includes dtype, shape, data, and device. this will also be used in jsonification as well.
    assert xos.all(tensor == tensor2), f"tensor and tensor2 are not equal: {tensor} != {tensor2}"

    if compress:
        # check that len is less than non-compressed
        non_compressed_len = len(tensor.printpack(compress=False))
        assert len(packed_str) < non_compressed_len