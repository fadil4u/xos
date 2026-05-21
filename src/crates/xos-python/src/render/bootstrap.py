# xos.render / Viewport — functional preview without xos.Application.
# ``xos`` is injected into globals by Rust (do not ``import xos`` here).


def _next_viewport_id():
    import builtins

    n = int(getattr(builtins, "__xos_next_viewport_id__", 0))
    builtins.__xos_next_viewport_id__ = n + 1
    return n


class Viewport:
    """Live preview of a tensor — use ``tick()`` for your own loop or ``pause()`` to block."""

    def __init__(self, viewport_id, headless, channels, dtype, device, width, height):
        self._id = int(viewport_id)
        self._headless = bool(headless)
        self._channels = int(channels)
        self._dtype = dtype
        self._device = device
        self._width = int(width)
        self._height = int(height)
        self._last_tensor = None
        self._draw_tensor = None

    def _window_open(self):
        if self._headless:
            return False
        ws = xos.frame._standalone_window_size(self._id)
        return ws is not None

    @property
    def size(self):
        """Current preview window ``(width, height)`` in pixels."""
        ws = xos.frame._standalone_window_size(self._id)
        if ws is None:
            return (self._width, self._height)
        w, h = ws
        self._width = int(w)
        self._height = int(h)
        return (self._width, self._height)

    @property
    def frame(self):
        """Fresh HWC tensor sized to the live preview window (updates after resize)."""
        w, h = self.size
        t = xos.zeros(
            (h, w, self._channels),
            dtype=self._dtype,
            device=self._device,
        )
        self._draw_tensor = t
        return t

    def pixel_space(self, dtype=None, device=None):
        """Axis-aligned pixel space for the current window size (rebuild transforms each tick)."""
        w, h = self.size
        if dtype is None:
            dtype = self._dtype
        if device is None:
            device = self._device
        return xos.space(
            origin=(0, 0),
            min=(0, 0),
            max=(w, h),
            dtype=dtype,
            device=device,
        )

    @property
    def running(self):
        return self._window_open()

    @property
    def closed(self):
        """True after the preview window close button (X) was clicked."""
        return not self._window_open()

    def tick(self, drain=True):
        """
        Pump the preview window once.

        Returns False when the window was closed. With ``drain=False``, only checks whether
        the window is still open (no event pump).
        """
        if self._headless:
            return False
        if not drain:
            return self._window_open()
        return bool(xos.frame._tick_viewport(self._id))

    def pause(self):
        """Block until the preview window is closed (like matplotlib show)."""
        if self._headless:
            return
        xos.frame._pause_viewport(self._id)

    def present(self):
        """Present the last synced frame without uploading new pixels."""
        if not self._headless:
            xos.frame._present_viewport(self._id)

    def push(self, tensor):
        """Push a new tensor image to this viewport (reuse the same window)."""
        self._last_tensor = tensor
        self._draw_tensor = tensor
        xos._sync_tensor_to_standalone(tensor, self._id)
        shape = tuple(tensor.shape)
        if len(shape) >= 2:
            self._height = int(shape[0])
            self._width = int(shape[1])
        if len(shape) >= 3:
            self._channels = int(shape[2])
        if not self._headless:
            xos.frame._present_viewport(self._id)


def _is_viewport(obj):
    return isinstance(obj, Viewport)


def render(tensor, viewport=None, headless=False):
    """
    Open or update a preview for a tensor (no Application subclass required).

    - ``xos.render(frame)`` — open a new window, return a Viewport
    - ``xos.render(frame, viewport)`` / ``xos.render(frame, viewport=vp)`` — push into an existing window
    - ``xos.render(viewport, frame)`` — same update, viewport-first order
    - ``xos.render(viewport)`` — push the tensor you last drew into (``viewport.frame`` or prior ``render(frame, viewport)``)

    Animation / resize-aware loop::

        viewport = xos.render(first_frame)
        frame = viewport.frame
        while viewport.tick():
            pixel_space = viewport.pixel_space()
            to_pixels = pixel_space.into_from(normal_space)
            xos.rasterizer.fill(frame, colors=(0, 0, 0))
            verts = to_pixels.apply(normal_rectangles.vertices)
            xos.rasterizer.fill_rectangles(frame, verts, colors=(255, 0, 0))
            xos.render(viewport)
            frame = viewport.frame
    """
    if _is_viewport(tensor):
        if viewport is not None and not _is_viewport(viewport):
            tensor, viewport = viewport, tensor
        else:
            viewport, tensor = tensor, viewport

    if viewport is not None:
        if not _is_viewport(viewport):
            raise TypeError("render(..., viewport=...) must be a Viewport from xos.render()")
        if tensor is None:
            tensor = viewport._draw_tensor or viewport._last_tensor
            if tensor is None:
                viewport.present()
                return viewport
        viewport.push(tensor)
        return viewport

    shape = tuple(tensor.shape)
    if len(shape) < 2:
        raise ValueError("render() needs a tensor with at least (height, width)")
    h = int(shape[0])
    w = int(shape[1])
    ch = int(shape[2]) if len(shape) > 2 else 3
    dtype = tensor.dtype
    device = tensor.device
    vid = _next_viewport_id()
    xos.frame._begin_standalone(vid, w, h)
    xos._sync_tensor_to_standalone(tensor, vid)
    if not headless:
        xos.frame._present_viewport(vid)
    vp = Viewport(vid, headless, ch, dtype, device, w, h)
    vp._last_tensor = tensor
    vp._draw_tensor = tensor
    return vp
