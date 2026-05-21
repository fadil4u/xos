# Loaded into xos by install_space (test files: import xos only).
# ``xos`` is injected into globals by Rust (do not ``import xos`` here).


def _format_coords(coords):
    """Format a coordinate tuple with integer-like floats as ints."""
    parts = []
    for v in coords:
        if isinstance(v, float) and v == float(int(v)):
            parts.append(str(int(v)))
        elif isinstance(v, float):
            parts.append("{:.6g}".format(v))
        else:
            parts.append(repr(v))
    return "(" + ", ".join(parts) + ")"


def _coords_tensor(values, dtype, device):
    flat = list(values)
    return xos.tensor(flat, (len(flat),), dtype=dtype, device=device)


def space(origin=(0,), min=(0,), max=(1,), dtype=None, device="cpu", units=None):
    if dtype is None:
        dtype = xos.float32
    o = tuple(origin)
    mn = tuple(min)
    mx = tuple(max)
    if not (len(o) == len(mn) == len(mx)):
        raise ValueError("space origin, min, and max must have the same length")
    dim = len(o)
    ot = _coords_tensor(o, dtype, device)
    mnt = _coords_tensor(mn, dtype, device)
    mxt = _coords_tensor(mx, dtype, device)
    return Space(ot, mnt, mxt, dtype, device, dim, units)


class Space:
    def __init__(self, origin, min, max, dtype, device, dimensionality, units=None):
        self.origin = origin
        self.min = min
        self.max = max
        self.dtype = dtype
        self.device = device
        self.dimensionality = dimensionality
        self.units = units

    def into_from(self, other):
        """Transform from ``other`` space into this (pixel) space."""
        return Transform.from_spaces(other, self)

    def to_from(self, other):
        """Alias for ``into_from`` (map coordinates from ``other`` → this space)."""
        return self.into_from(other)

    def tostring(self, full=False):
        origin = _format_coords(self.origin.astuple())
        mn = _format_coords(self.min.astuple())
        mx = _format_coords(self.max.astuple())
        base = "xos.space(origin={}, min={}, max={}, dtype={}, device={!r})".format(
            origin, mn, mx, self.dtype, self.device
        )
        if not full:
            return base
        return base

    def __str__(self):
        return self.tostring(full=False)

    def __repr__(self):
        return self.tostring(full=False)


class Transform:
    def __init__(self, from_space, to_space):
        self._from = from_space
        self._to = to_space

    @classmethod
    def from_spaces(cls, src, dst):
        return cls(src, dst)

    def _map_point(self, point):
        smin = self._from.min.astuple()
        smax = self._from.max.astuple()
        dmin = self._to.min.astuple()
        dmax = self._to.max.astuple()
        out = []
        for i, p in enumerate(point):
            span_s = smax[i] - smin[i]
            if span_s == 0.0:
                t = 0.0
            else:
                t = (float(p) - float(smin[i])) / float(span_s)
            out.append(float(dmin[i]) + t * (float(dmax[i]) - float(dmin[i])))
        return tuple(out)

    def _apply_vertices_tensor(self, verts):
        pairs = verts.astuple()
        mapped = []
        for c0, c1 in pairs:
            m0 = self._map_point(tuple(c0))
            m1 = self._map_point(tuple(c1))
            mapped.append((m0, m1))
        dim = len(pairs[0][0])
        flat = []
        for c0, c1 in mapped:
            flat.extend(c0)
            flat.extend(c1)
        return xos.tensor(
            flat, (len(mapped), 2, dim), dtype=verts.dtype, device=verts.device
        )

    def apply(self, target):
        if hasattr(target, "_map_corners"):
            return target._map_corners(self._map_point, self._to)
        if hasattr(target, "astuple"):
            return self._apply_vertices_tensor(target)
        raise TypeError(
            "Transform.apply() expects xos.shapes rectangles or a vertices tensor"
        )

    def tostring(self, full=False):
        if not full:
            return "xos.Transform(from={}, to={})".format(self._from, self._to)
        return "xos.Transform(from={}, to={})".format(
            self._from.tostring(full=True),
            self._to.tostring(full=True),
        )

    def __str__(self):
        return self.tostring(full=False)

    def __repr__(self):
        return self.tostring(full=False)


class Rectangles:
    def __init__(self, corners, dtype, device, dimensionality):
        self._corners = corners
        self._dtype = dtype
        self._device = device
        self._dimensionality = dimensionality

    def _map_corners(self, map_fn, target_space):
        mapped = []
        for c0, c1 in self._corners:
            mapped.append((map_fn(c0), map_fn(c1)))
        return Rectangles(
            mapped,
            target_space.dtype,
            target_space.device,
            self._dimensionality,
        )

    @property
    def vertices(self):
        n = len(self._corners)
        dim = self._dimensionality
        flat = []
        for c0, c1 in self._corners:
            flat.extend(c0)
            flat.extend(c1)
        return xos.tensor(
            flat, (n, 2, dim), dtype=self._dtype, device=self._device
        )

    def tostring(self, full=False):
        if not full:
            return "xos.shapes.Rectangles(n={}, dim={})".format(
                len(self._corners), self._dimensionality
            )
        parts = [
            "xos.shapes.Rectangles(n={}, dim={})".format(
                len(self._corners), self._dimensionality
            )
        ]
        for i, (c0, c1) in enumerate(self._corners):
            parts.append(
                "[{}] {}..{}".format(i, _format_coords(c0), _format_coords(c1))
            )
        parts.append("vertices=" + self.vertices.tostring(full=True))
        return " | ".join(parts)

    def __str__(self):
        return self.tostring(full=False)

    def __repr__(self):
        return self.tostring(full=False)


def rectangles(vertices=None, dtype=None, device="cpu"):
    if vertices is None:
        raise TypeError("rectangles(vertices=...) is required")
    corner_pairs = tuple(vertices)
    if not corner_pairs:
        raise TypeError("rectangles(vertices=...) needs at least one corner pair")
    dim = len(corner_pairs[0][0])
    for pair in corner_pairs:
        if len(pair) != 2:
            raise ValueError("each rectangle must be two corner tuples")
        c0, c1 = pair
        if len(c0) != dim or len(c1) != dim:
            raise ValueError("rectangle corners must match dimensionality")
    if dtype is None:
        dtype = xos.float32
    return Rectangles(
        [((tuple(a)), tuple(b)) for a, b in corner_pairs],
        dtype,
        device,
        dim,
    )


class _ShapesModule:
    rectangles = staticmethod(rectangles)


shapes = _ShapesModule()


def pixel_space_for_frame(frame, dtype=None, device=None):
    """Pixel-aligned ``xos.space`` from a frame tensor ``(height, width, ...)``."""
    shape = tuple(frame.shape)
    if len(shape) < 2:
        raise ValueError("pixel_space_for_frame needs (height, width, ...)")
    h, w = int(shape[0]), int(shape[1])
    if dtype is None:
        dtype = frame.dtype
    if device is None:
        device = frame.device
    return xos.space(
        origin=(0, 0),
        min=(0, 0),
        max=(w, h),
        dtype=dtype,
        device=device,
    )


def _vertices_from_rects(rects):
    if hasattr(rects, "vertices"):
        return rects.vertices
    return rects


def fill_rectangles(frame, rects, colors=None, space=None, viewport=None):
    """
    Fill axis-aligned rectangles on ``frame``.

    ``colors`` — ``(r,g,b)`` or ``(r,g,b,a)`` (broadcast), or per-rect tensor.

    ``space`` — when set (e.g. ``normal_space``), ``rects`` are in that coordinate
    system and are mapped into pixel space for the current ``frame`` shape (or
    ``viewport.size`` when ``viewport`` is given). When ``space`` is ``None``, ``rects``
    are already in pixel coordinates matching ``frame``.
    """
    native = xos.rasterizer._fill_rectangles_native
    if colors is None:
        raise TypeError("fill_rectangles(..., colors=...) requires colors")
    if space is None:
        native(frame, rects, colors)
        return

    if viewport is not None:
        pixel_space = viewport.pixel_space(dtype=frame.dtype, device=frame.device)
    else:
        pixel_space = pixel_space_for_frame(frame, dtype=frame.dtype, device=frame.device)

    to_pixels = pixel_space.to_from(space)
    verts = _vertices_from_rects(rects)
    pixel_verts = to_pixels.apply(verts)
    native(frame, pixel_verts, colors)
