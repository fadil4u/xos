import xos


def _assert_space_properties(space, dtype, device):
    # TODO: failure construction  cases like mismatching dimensionality etc.
    assert type(space.origin) == xos.Tensor
    assert type(space.min) == xos.Tensor
    assert type(space.max) == xos.Tensor
    assert space.dtype is dtype
    assert space.dimensionality == 3

    assert space.device == device
    assert space.origin.device == device
    assert space.min.device == device
    assert space.max.device == device


@xos.test
def test_frame_transforms():
    """Dimensionalities hardcoded to match CHW (channels, height, width) transformations of frames where it's sorta technically 3d, but
    the context of the space and transforms is more like a special case of 2d where its "copied" along the channel axis with the colors
    defining their volumes.
    """

    height = 128
    width = 256
    channels = 3
    device = "cpu"

    # spoof a viewport space and pixel space transform
    # viewport_space = xos.space()  # TODO (origins, units, dtypes, definitions per axis, scale, units, etc.)
    viewport_pixel_space = xos.space(
        origin=(0, 0, 0),
        min=(0, 0, 0),
        max=(width, height, channels),
        # units=("px", "px", "px"),  # TODO: labels later
        dtype=xos.uint8,  # cells of pixels
        device=device,
    )

    _assert_space_properties(viewport_pixel_space, xos.uint8, device)

    # normal_space = xos.space()  # TODO (origins, units, dtypes, definitions per axis, scale, units, etc.)
    normal_space = xos.space(
        origin=(0, 0, 0),
        min=(0.0, 0.0, 0.0),
        max=(1.0, 1.0, 1.0),
        # units=("px", "px", "px"),
        dtype=xos.float32,
        device=device,
    )

    _assert_space_properties(normal_space, xos.float32, device)

    # automatically generate the transformations between the spaces
    normal_to_pixels = viewport_pixel_space.into_from(normal_space)
    pixels_to_normal = normal_space.into_from(viewport_pixel_space)

    # TODO: test the transforms against their inverses
    print("spaces:")
    print(viewport_pixel_space)
    print(normal_space)

    print("transforms:")
    print(normal_to_pixels)
    print(pixels_to_normal)

    pixel_rectangles = xos.shapes.rectangles(
        # (x, y, z), (x, y, z), in this case where z is channels-span
        ((10, 10, 0), (20, 20, 2)),  # rect0
        ((30, 30, 0), (40, 40, 2)),  # rect1
        ((50, 50, 0), (800, 60, 2)),  # rect2
    )

    print(pixel_rectangles.vertices.tostring(full=True))

    normal_rectangles = pixels_to_normal.apply(pixel_rectangles)
    print(normal_rectangles.vertices.tostring(full=True))

    # TODO: other shapes besides rectangles

    # NOTE: all should be n-Dimensional and 100% tensorized/organized.
    # TODO: rotations, shears
    # TODO: translations
    # TODO: scalings
    # TODO: shears
    # TODO: perspective transformations
    # TODO: affine transformations
    # TODO: projective transformations
    # TODO: non-linear transformations
    # TODO: time-varying transformations

    # TODO: rasterize to frame and display



