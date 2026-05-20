# xos.render / Viewport — functional preview without xos.Application.


def _next_viewport_id():
    import builtins

    n = int(getattr(builtins, "__xos_next_viewport_id__", 0))
    builtins.__xos_next_viewport_id__ = n + 1
    return n


class Viewport:
    """Live preview of a tensor. Call pause() to keep the window open until you close it."""

    def __init__(self, viewport_id, headless):
        self._id = int(viewport_id)
        self._headless = bool(headless)

    def pause(self):
        """Block until the preview window is closed (like matplotlib show)."""
        if self._headless:
            return
        xos.frame._pause_viewport(self._id)

    def render(self, tensor):
        """Push a new tensor image to this viewport."""
        xos._sync_tensor_to_standalone(tensor, self._id)
        if not self._headless:
            xos.frame._present_viewport(self._id)


def render(tensor, headless=False):
    """
    Open a preview for a tensor (no Application subclass required).

    Returns a Viewport you can update with viewport.render(tensor) and
    viewport.pause() to wait until the window closes.
    """
    shape = tuple(tensor.shape)
    if len(shape) < 2:
        raise ValueError("render() needs a tensor with at least (height, width)")
    h = int(shape[0])
    w = int(shape[1])
    vid = _next_viewport_id()
    xos.frame._begin_standalone(vid, w, h)
    xos._sync_tensor_to_standalone(tensor, vid)
    if not headless:
        xos.frame._present_viewport(vid)
    return Viewport(vid, headless)
