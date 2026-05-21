import xos


@xos.test
def test_frame_transforms():
    """Dimensionalities hardcoded to match CHW (channels, height, width) transformations of frames where it's sorta technically 3d, but
    the context of the space and transforms is more like a special case of 2d where its "copied" along the channel axis with the colors
    defining their volumes.
    """

    height = 128
    width = 256
    channels = 3

    # spoof a viewport space and pixel space transform
    # viewport_space = xos.space()  # TODO (origins, units, dtypes, definitions per axis, scale, units, etc.)
    viewport_space = xos.space(
        origin=(0, 0, 0),
        min=(0, 0, 0),
        max=(width, height, channels),
        units=("px", "px", "px"),
        dtype=xos.uint8,  # cells of pixels
    )

    # normal_space = xos.space()  # TODO (origins, units, dtypes, definitions per axis, scale, units, etc.)
    normal_space = xos.space(
        origin=(0, 0, 0),
        min=(0.0, 0.0, 0.0),
        max=(1.0, 1.0, 1.0),
        # units=("px", "px", "px"),
        dtype=xos.float32,
    )

    # automatically generate the transformations between the spaces
    normal_to_vp = transform = viewport_space.into_from(normal_space)
    vp_to_normal = normal_space.into_from(viewport_space)

    # test the transforms and their inverses
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



