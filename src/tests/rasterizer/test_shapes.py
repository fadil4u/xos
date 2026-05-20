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
    viewport.pause()
    # viewport = xos.render(tensor)
    # viewport.pause()

    # viewport.render(tensor)  # this could also be called for subsequent updates to this frame, especially with pause=False for a live loop animation  TODO: testing for that

    # TODO: hard-code what the raster should look like after verifying