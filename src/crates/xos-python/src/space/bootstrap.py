# Loaded into xos by install_space (test files: import xos only).


def _coords_tensor(values, dtype, device):
    import xos

    flat = list(values)
    return xos.tensor(flat, (len(flat),), dtype=dtype, device=device)


def space(origin=(0,), min=(0,), max=(1,), dtype=None, device="cpu", units=None):
    import xos

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
        return Transform.from_spaces(other, self)

    def __str__(self):
        return "xos.space(origin={}, min={}, max={}, dtype={}, device={!r})".format(
            self.origin.astuple(),
            self.min.astuple(),
            self.max.astuple(),
            self.dtype,
            self.device,
        )

    def __repr__(self):
        return self.__str__()


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

    def apply(self, shapes):
        if hasattr(shapes, "_map_corners"):
            return shapes._map_corners(self._map_point, self._to)
        raise TypeError("Transform.apply() expects shapes from xos.shapes")

    def __str__(self):
        return "xos.Transform(from={}, to={})".format(self._from, self._to)

    def __repr__(self):
        return self.__str__()


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
        return Rectangles(mapped, target_space.dtype, target_space.device, self._dimensionality)

    @property
    def vertices(self):
        import xos

        n = len(self._corners)
        dim = self._dimensionality
        flat = []
        for c0, c1 in self._corners:
            flat.extend(c0)
            flat.extend(c1)
        return xos.tensor(
            flat, (n, 2, dim), dtype=self._dtype, device=self._device
        )

    def __str__(self):
        return "xos.shapes.Rectangles(n={}, dim={})".format(
            len(self._corners), self._dimensionality
        )

    def __repr__(self):
        return self.__str__()


def rectangles(*corner_pairs):
    import xos

    if not corner_pairs:
        raise TypeError("rectangles() requires at least one ((min), (max)) pair")
    dim = len(corner_pairs[0][0])
    for pair in corner_pairs:
        if len(pair) != 2:
            raise ValueError("each rectangle must be two corner tuples")
        c0, c1 = pair
        if len(c0) != dim or len(c1) != dim:
            raise ValueError("rectangle corners must match dimensionality")
    return Rectangles(
        [((tuple(a)), tuple(b)) for a, b in corner_pairs],
        xos.float32,
        "cpu",
        dim,
    )


class _ShapesModule:
    rectangles = staticmethod(rectangles)


shapes = _ShapesModule()
