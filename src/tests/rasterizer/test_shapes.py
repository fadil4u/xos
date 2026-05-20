import xos


@xos.test
@xos.parametrize("dtype", [xos.uint8, xos.float32])
def test_squares(dtype):
    frame = xos.zeros((100, 100, 3), dtype=dtype)

    rects = xos.tensor([
        [0, 0, 100, 100],
        [10, 10, 20, 20],
        [30, 30, 40, 40],
        [50, 50, 60, 60],
        [70, 70, 80, 80],
        [90, 90, 100, 100],
        [110, 110, 120, 120],
        [130, 130, 140, 140],
        [150, 150, 160, 160],
    ])

    # TODO: test all configuration of squeeze shapes and whatnot
    colors = xos.tensor([
        (255, 0, 0),
    ])

    xos.rasterizer.fill_rectangles(frame, rects, colors)

    # this should be the api for rendering the frame to the screen
    xos.render(frame)

    # TODO: hard-code what the raster should look like after verifying