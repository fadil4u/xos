use xos_core::engine::keyboard::shortcuts::ShortcutAction;
use xos_core::engine::{Application, EngineState, SafeRegionBoundingRectangle, ScrollWheelUnit};
use crate::engine::py_engine_tls::{CallbackEngineStateGuard, TickEngineStateGuard};
use rustpython_vm::{
    builtins::PyBaseExceptionRef, AsObject, Interpreter, PyObjectRef, PyResult, VirtualMachine,
};

/// Format a Python exception with traceback info
fn format_python_exception(vm: &VirtualMachine, py_exc: &PyBaseExceptionRef) -> String {
    let mut output = String::new();

    // Try to show traceback info if available
    if let Some(traceback) = py_exc.traceback() {
        output.push_str("Traceback (most recent call last):\n");

        // Use the debug format which should show file/line info
        let tb_str = format!("{:?}", traceback);
        if !tb_str.is_empty() && tb_str.len() < 500 {
            // Try to extract useful info from the debug string
            for line in tb_str.lines() {
                if line.contains("File") || line.contains("line") {
                    output.push_str("  ");
                    output.push_str(line.trim());
                    output.push('\n');
                }
            }
        }
    }

    // Get exception class name
    let class_name = py_exc.class().name().to_string();

    // Try to get the exception message
    let msg = vm
        .call_method(py_exc.as_object(), "__str__", ())
        .ok()
        .and_then(|result| result.str(vm).ok().map(|s| s.to_string()))
        .unwrap_or_default();

    // Add exception info
    if !msg.is_empty() {
        output.push_str(&format!("{}: {}", class_name, msg));
    } else {
        output.push_str(&class_name);
    }

    output
}

fn log_py_runtime_error(message: &str) {
    #[cfg(target_arch = "wasm32")]
    xos_core::print(message);
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{message}");
}

fn sync_app_safe_region(
    vm: &VirtualMachine,
    app: &PyObjectRef,
    safe: &SafeRegionBoundingRectangle,
) -> PyResult<()> {
    let cls = vm.builtins.get_attr("__xos_SafeRegion_cls__", vm)?;
    let x1: PyObjectRef = vm.ctx.new_float(safe.x1 as f64).into();
    let y1: PyObjectRef = vm.ctx.new_float(safe.y1 as f64).into();
    let x2: PyObjectRef = vm.ctx.new_float(safe.x2 as f64).into();
    let y2: PyObjectRef = vm.ctx.new_float(safe.y2 as f64).into();
    let sr = cls.call((x1, y1, x2, y2), vm)?;
    app.set_attr("safe_region", sr, vm)?;
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) enum RoutedPyEvent {
    MouseDown,
    MouseUp,
    MouseMove,
    Scroll {
        dx: f32,
        dy: f32,
        unit: ScrollWheelUnit,
    },
    KeyChar(char),
    Shortcut(ShortcutAction),
}

fn build_routed_event_dict(
    vm: &VirtualMachine,
    ev: RoutedPyEvent,
    state: &EngineState,
) -> PyResult {
    let d = vm.ctx.new_dict();
    match ev {
        RoutedPyEvent::MouseDown => {
            d.set_item("kind", vm.ctx.new_str("mouse_down").into(), vm)?;
            d.set_item("x", vm.ctx.new_float(state.mouse.x as f64).into(), vm)?;
            d.set_item("y", vm.ctx.new_float(state.mouse.y as f64).into(), vm)?;
            d.set_item(
                "button",
                vm.ctx
                    .new_str(if state.mouse.is_right_clicking {
                        "right"
                    } else {
                        "left"
                    })
                    .into(),
                vm,
            )?;
            d.set_item(
                "is_left",
                vm.ctx.new_bool(state.mouse.is_left_clicking).into(),
                vm,
            )?;
            d.set_item(
                "is_right",
                vm.ctx.new_bool(state.mouse.is_right_clicking).into(),
                vm,
            )?;
        }
        RoutedPyEvent::MouseUp => {
            d.set_item("kind", vm.ctx.new_str("mouse_up").into(), vm)?;
            d.set_item("x", vm.ctx.new_float(state.mouse.x as f64).into(), vm)?;
            d.set_item("y", vm.ctx.new_float(state.mouse.y as f64).into(), vm)?;
            d.set_item(
                "button",
                vm.ctx
                    .new_str(if state.mouse.is_right_clicking {
                        "right"
                    } else {
                        "left"
                    })
                    .into(),
                vm,
            )?;
            d.set_item(
                "is_left",
                vm.ctx.new_bool(state.mouse.is_left_clicking).into(),
                vm,
            )?;
            d.set_item(
                "is_right",
                vm.ctx.new_bool(state.mouse.is_right_clicking).into(),
                vm,
            )?;
        }
        RoutedPyEvent::MouseMove => {
            d.set_item("kind", vm.ctx.new_str("mouse_move").into(), vm)?;
            d.set_item("x", vm.ctx.new_float(state.mouse.x as f64).into(), vm)?;
            d.set_item("y", vm.ctx.new_float(state.mouse.y as f64).into(), vm)?;
            d.set_item(
                "button",
                vm.ctx
                    .new_str(if state.mouse.is_right_clicking {
                        "right"
                    } else {
                        "left"
                    })
                    .into(),
                vm,
            )?;
            d.set_item(
                "is_left",
                vm.ctx.new_bool(state.mouse.is_left_clicking).into(),
                vm,
            )?;
            d.set_item(
                "is_right",
                vm.ctx.new_bool(state.mouse.is_right_clicking).into(),
                vm,
            )?;
        }
        RoutedPyEvent::Scroll { dx, dy, unit } => {
            d.set_item("kind", vm.ctx.new_str("scroll").into(), vm)?;
            d.set_item("dx", vm.ctx.new_float(dx as f64).into(), vm)?;
            d.set_item("dy", vm.ctx.new_float(dy as f64).into(), vm)?;
            let u = match unit {
                ScrollWheelUnit::Line => "line",
                ScrollWheelUnit::Pixel => "pixel",
            };
            d.set_item("unit", vm.ctx.new_str(u).into(), vm)?;
        }
        RoutedPyEvent::KeyChar(ch) => {
            d.set_item("kind", vm.ctx.new_str("key_char").into(), vm)?;
            d.set_item("char", vm.ctx.new_str(ch.to_string()).into(), vm)?;
        }
        RoutedPyEvent::Shortcut(sa) => {
            d.set_item("kind", vm.ctx.new_str("shortcut").into(), vm)?;
            let action = match sa {
                ShortcutAction::Copy => "copy",
                ShortcutAction::Cut => "cut",
                ShortcutAction::Paste => "paste",
                ShortcutAction::SelectAll => "select_all",
                ShortcutAction::Undo => "undo",
                ShortcutAction::Redo => "redo",
            };
            d.set_item("action", vm.ctx.new_str(action).into(), vm)?;
        }
    }
    Ok(d.into())
}

#[inline]
fn sync_app_mouse_from_engine(
    vm: &VirtualMachine,
    app_instance: &PyObjectRef,
    state: &EngineState,
) {
    let mouse_dict = vm.ctx.new_dict();
    let _ = mouse_dict.set_item("x", vm.ctx.new_float(state.mouse.x as f64).into(), vm);
    let _ = mouse_dict.set_item("y", vm.ctx.new_float(state.mouse.y as f64).into(), vm);
    let _ = mouse_dict.set_item(
        "is_left_clicking",
        vm.ctx.new_bool(state.mouse.is_left_clicking).into(),
        vm,
    );
    let _ = mouse_dict.set_item(
        "is_right_clicking",
        vm.ctx.new_bool(state.mouse.is_right_clicking).into(),
        vm,
    );
    let _ = app_instance.set_attr("mouse", mouse_dict, vm);
}

/// Replay pointer at the trackpad laser so `xos.ui.Text` hit-testing updates focus (`Group` sees both widgets).
fn drain_embed_synthetic_click(
    vm: &VirtualMachine,
    app_instance: &PyObjectRef,
    state: &mut EngineState,
) {
    let Some((sx, sy)) = state.embed_synthetic_click_screen.take() else {
        return;
    };

    let ox = state.mouse.x;
    let oy = state.mouse.y;
    let ol = state.mouse.is_left_clicking;
    let or_click = state.mouse.is_right_clicking;

    state.mouse.x = sx;
    state.mouse.y = sy;
    state.mouse.is_left_clicking = true;
    state.mouse.is_right_clicking = false;

    sync_app_mouse_from_engine(vm, app_instance, state);
    let _ = vm.call_method(app_instance, "on_mouse_down", (sx as f64, sy as f64));
    try_dispatch_python_on_events(vm, app_instance, state, RoutedPyEvent::MouseDown);

    state.mouse.is_left_clicking = false;
    sync_app_mouse_from_engine(vm, app_instance, state);
    let _ = vm.call_method(app_instance, "on_mouse_up", (sx as f64, sy as f64));
    try_dispatch_python_on_events(vm, app_instance, state, RoutedPyEvent::MouseUp);

    state.mouse.x = ox;
    state.mouse.y = oy;
    state.mouse.is_left_clicking = ol;
    state.mouse.is_right_clicking = or_click;
    sync_app_mouse_from_engine(vm, app_instance, state);
}

fn try_dispatch_python_on_events(
    vm: &VirtualMachine,
    app_instance: &PyObjectRef,
    state: &mut EngineState,
    ev: RoutedPyEvent,
) {
    let Ok(Some(cb)) = vm.get_attribute_opt(app_instance.clone(), "on_events") else {
        return;
    };
    if !cb.is_callable() {
        return;
    }
    let dict = match build_routed_event_dict(vm, ev, state) {
        Ok(o) => o,
        Err(e) => {
            log_py_runtime_error(&format!(
                "Failed to build routed _xos_event dict:\n{}",
                format_python_exception(vm, &e)
            ));
            return;
        }
    };
    let _evt_store = app_instance.set_attr("_xos_event", dict, vm);
    let _guard = CallbackEngineStateGuard::install(state);
    if let Err(e) = vm.call_method(app_instance, "on_events", ()) {
        log_py_runtime_error(&format!(
            "Python on_events error:\n{}",
            format_python_exception(vm, &e)
        ));
    }
    let _clear = app_instance.set_attr("_xos_event", vm.ctx.none(), vm);
}

pub const APPLICATION_CLASS_CODE: &str = r#"
def _tensor_unflatten(flat, shape):
    """Row-major flat list -> nested list for the given ``shape``."""
    shape = tuple(shape)
    if len(shape) == 0:
        if len(flat) != 1:
            raise ValueError("scalar tensor expects one element")
        return flat[0]
    if len(shape) == 1:
        n = shape[0]
        if len(flat) != n:
            raise ValueError("tensor list(): length mismatch with shape")
        return list(flat)
    d0 = shape[0]
    rest = shape[1:]
    chunk = 1
    for s in rest:
        chunk *= s
    if len(flat) != d0 * chunk:
        raise ValueError("tensor list(): length mismatch with shape")
    return [_tensor_unflatten(flat[i * chunk : (i + 1) * chunk], rest) for i in range(d0)]

def _nested_list_to_tuple(nested):
    if isinstance(nested, list):
        return tuple(_nested_list_to_tuple(x) for x in nested)
    return nested

def _format_scalar_value(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        if v == float(int(v)) and abs(v) < 1e15:
            return str(int(v))
        return "{:.6g}".format(v)
    return repr(v)

def _format_nested_values(data):
    """Compact nested tuple formatting for ``tostring(full=True)``."""
    if isinstance(data, tuple):
        if not data:
            return "()"
        if isinstance(data[0], tuple):
            return "(" + ", ".join(_format_nested_values(row) for row in data) + ")"
        return "(" + ", ".join(_format_scalar_value(x) for x in data) + ")"
    return _format_scalar_value(data)

class Tensor:
    """xos.Tensor — dict-backed tensor (``shape``, ``dtype``, ``_data``) or flat list + optional ``shape``."""
    def __init__(self, data, shape=None):
        if isinstance(data, dict):
            self._data = data
        else:
            flat = list(data)
            sh = shape if shape is not None else (len(flat),)
            self._data = {
                "shape": tuple(sh),
                "dtype": "float32",
                "device": "cpu",
                "_data": flat,
            }

    def __iter__(self):
        d = self._data.get("_data")
        if isinstance(d, list):
            return iter(d)
        if isinstance(d, (bytes, bytearray)):
            return iter(d)
        return iter([])

    def __len__(self):
        d = self._data.get("_data")
        if isinstance(d, (list, bytes, bytearray)):
            return len(d)
        return 0

    def _flat_storage(self):
        d = self._data.get("_data")
        if isinstance(d, (list, bytes, bytearray)) and len(d) > 0:
            return d
        return None

    def _ensure_flat(self):
        import xos
        if self._flat_storage() is not None:
            return
        if self._data.get("_rust_tensor") is not None and self._attach_registry_flat():
            return
        if self._data.get("_xos_gpu_conv_output"):
            xos._materialize_conv_output(self)
            return
        if (
            self._data.get("_xos_viewport_id") is not None
            or self._data.get("_xos_frame_backing")
            or self._data.get("_xos_frame_materialized")
        ):
            xos._materialize_frame_tensor(self)
            return
        xos._materialize_frame_tensor(self)

    def _flat_len(self, flat):
        return len(flat)

    def _flat_get(self, flat, i):
        if isinstance(flat, (bytes, bytearray)):
            if self.dtype == "float32":
                import struct
                return struct.unpack_from("<f", flat, i * 4)[0]
            return flat[i]
        return flat[i]

    def _attach_registry_flat(self):
        rid = self._data.get("_rust_tensor")
        if rid is None:
            return False
        import xos
        ba = xos._tensor_registry_bytes(int(rid), str(self._data.get("dtype", "float32")))
        if ba is None:
            return False
        self._data["_data"] = ba
        return True

    def _cmp_scalar(self, other):
        if (
            isinstance(other, float)
            and self._data.get("dtype") == "uint8"
            and other == 0.5
        ):
            return 127.5
        return other

    def _maybe_flush_frame(self):
        import xos
        if (
            self._data.get("_xos_viewport_id") is not None
            or self._data.get("_xos_frame_backing")
            or self._data.get("_xos_frame_materialized")
        ):
            xos._flush_frame_tensor(self)

    def _broadcast_rhs_flat(self, lshape, rshape, left, right):
        if len(left) == len(right):
            return right
        if (
            len(lshape) == 2
            and len(rshape) == 3
            and lshape[0] == rshape[0]
            and lshape[1] == rshape[1]
            and rshape[2] == 1
        ):
            return right
        if (
            len(lshape) == 3
            and len(rshape) == 2
            and lshape[0] == rshape[0]
            and lshape[1] == rshape[1]
            and lshape[2] == 1
        ):
            return right
        if (
            len(lshape) == 3
            and len(rshape) == 3
            and lshape[0] == rshape[0]
            and lshape[1] == rshape[1]
            and lshape[2] == 4
            and rshape[2] == 3
        ):
            h, w = int(lshape[0]), int(lshape[1])
            out = []
            for i in range(h * w):
                rv = right[i * 3]
                for _ in range(4):
                    out.append(rv)
            return out
        raise ValueError(
            "shape mismatch for broadcast: {} vs {} ({} vs {} elements)".format(
                lshape, rshape, len(left), len(right)
            )
        )

    def _masked_assign(self, mask_tensor, value):
        flat = self._flat_storage()
        if flat is None:
            raise TypeError("tensor has no flat storage for masked assignment")
        mask = mask_tensor._flat_storage()
        if mask is None:
            raise TypeError("boolean mask must be a tensor with flat _data")
        if self._flat_len(flat) != self._flat_len(mask):
            raise IndexError(
                "boolean mask length ({}) must equal tensor size ({})".format(
                    self._flat_len(mask), self._flat_len(flat)
                )
            )
        n = self._flat_len(flat)
        if isinstance(value, Tensor):
            vflat = value._flat_storage()
            if vflat is None:
                raise TypeError("assigned value must be scalar or same-length tensor")
            if self._flat_len(vflat) == 1:
                scalar = self._flat_get(vflat, 0)
                for i in range(n):
                    if int(self._flat_get(mask, i)) != 0:
                        if isinstance(flat, bytearray):
                            flat[i] = int(scalar) & 0xFF
                        elif isinstance(flat, bytes):
                            ba = bytearray(flat)
                            ba[i] = int(scalar) & 0xFF
                            self._data["_data"] = ba
                            flat = ba
                        else:
                            flat[i] = scalar
                return
            if self._flat_len(vflat) == n:
                for i in range(n):
                    if int(self._flat_get(mask, i)) != 0:
                        if isinstance(flat, (bytes, bytearray)):
                            if isinstance(flat, bytes):
                                flat = bytearray(flat)
                                self._data["_data"] = flat
                            flat[i] = int(self._flat_get(vflat, i)) & 0xFF
                        else:
                            flat[i] = self._flat_get(vflat, i)
                return
            raise TypeError("assigned value must be scalar or same-length tensor")
        scalar = value
        for i in range(n):
            if int(self._flat_get(mask, i)) != 0:
                if isinstance(flat, (bytes, bytearray)):
                    if isinstance(flat, bytes):
                        flat = bytearray(flat)
                        self._data["_data"] = flat
                    flat[i] = int(scalar) & 0xFF
                else:
                    flat[i] = scalar

    def _getitem_int_index(self, key):
        """Row-major integer indexing: ``t[i]``, ``t[i,j]``, … partial views or a scalar."""
        shape = tuple(self.shape)
        flat = self._data["_data"]
        if isinstance(key, int):
            indices = (key,)
        else:
            indices = key
        offset = 0
        dim = 0
        for k in indices:
            if dim >= len(shape):
                raise IndexError("too many indices for tensor")
            kk = int(k)
            if kk < 0:
                kk += shape[dim]
            if kk < 0 or kk >= shape[dim]:
                raise IndexError("tensor index out of range")
            stride = 1
            for s in shape[dim + 1 :]:
                stride *= s
            offset += kk * stride
            dim += 1
        remaining = shape[dim:]
        n = 1
        for s in remaining:
            n *= s
        subflat = flat[offset : offset + n]
        if len(remaining) == 0:
            return subflat[0]
        return self._wrap_vals(remaining, subflat)
    
    def __getitem__(self, key):
        if isinstance(key, slice) and key == slice(None, None, None):
            # Return the underlying data dict for full slice access
            return self._data
        # N-dimensional integer indexing (e.g. y[0], y[0, 0], y[0, 0, 1])
        if "_data" in self._data and isinstance(self._data["_data"], list):
            if type(key) is int:
                return self._getitem_int_index(key)
            if (
                isinstance(key, tuple)
                and len(key) > 0
                and all(type(x) is int for x in key)
            ):
                return self._getitem_int_index(key)
        if isinstance(key, tuple):
            shape = tuple(self.shape)
            flat = self._data["_data"]
            if (
                len(shape) == 3
                and len(key) == 3
                and isinstance(key[0], slice)
                and isinstance(key[1], int)
                and isinstance(key[2], int)
            ):
                rows = list(range(*key[0].indices(shape[0])))
                j = int(key[1])
                k = int(key[2])
                if j < 0:
                    j += shape[1]
                if k < 0:
                    k += shape[2]
                if j < 0 or j >= shape[1] or k < 0 or k >= shape[2]:
                    raise IndexError("tensor index out of range")
                out = []
                stride_row = shape[1] * shape[2]
                for i in rows:
                    base = i * stride_row + j * shape[2] + k
                    out.append(flat[base])
                return self._wrap_vals((len(rows),), out)
        if isinstance(key, tuple) and len(key) == 2:
            a, b = key
            shape = tuple(self.shape)
            flat = self._data["_data"]
            if isinstance(a, slice) and a.start is None and a.stop is None and a.step is None and b is None:
                if len(shape) == 1:
                    n = shape[0]
                    return self._wrap_vals((n, 1), flat)
                if len(shape) == 2:
                    r, c = shape[0], shape[1]
                    return self._wrap_vals((r, c, 1), flat)
            if isinstance(a, slice) and a.start is None and a.stop is None and a.step is None and isinstance(b, int):
                if len(shape) == 2:
                    rows, cols = shape[0], shape[1]
                    col = b
                    out = [flat[i * cols + col] for i in range(rows)]
                    return self._wrap_vals((rows,), out)
        if isinstance(key, Tensor):
            return self._gather_rows(key)
        return self._data[key]

    def _wrap_vals(self, shape, values):
        return Tensor({
            "shape": tuple(shape),
            "dtype": self.dtype,
            "device": self._data.get("device", "cpu"),
            "_data": values,
        })

    def reshape(self, new_shape):
        self._ensure_flat()
        flat = self._flat_storage()
        if flat is None:
            raise TypeError("tensor has no flat storage for reshape")
        if isinstance(flat, (bytes, bytearray)):
            flat = list(flat)
        prod = 1
        for d in new_shape:
            prod *= d
        if prod != len(flat):
            raise ValueError("reshape size mismatch")
        return self._wrap_vals(tuple(new_shape), flat)

    def unsqueeze(self, axis):
        """Insert a length-1 dimension at ``axis`` (supports negative indices)."""
        self._ensure_flat()
        flat = self._flat_storage()
        if flat is None:
            raise TypeError("tensor has no flat storage for unsqueeze")
        if isinstance(flat, (bytes, bytearray)):
            flat = list(flat)
        shape = list(self.shape)
        ndim = len(shape)
        ax = int(axis)
        if ax < 0:
            ax += ndim + 1
        if ax < 0 or ax > ndim:
            raise IndexError("unsqueeze axis out of range")
        new_shape = shape[:ax] + [1] + shape[ax:]
        return self._wrap_vals(tuple(new_shape), flat)

    def repeat(self, count, axis=-1):
        """Repeat elements along ``axis`` (numpy-style; default last axis)."""
        self._ensure_flat()
        if int(count) < 1:
            raise ValueError("repeat count must be >= 1")
        count = int(count)
        flat = self._flat_storage()
        if flat is None:
            raise TypeError("tensor has no flat storage for repeat")
        if isinstance(flat, (bytes, bytearray)):
            flat = list(flat)
        shape = list(self.shape)
        ndim = len(shape)
        ax = int(axis)
        if ax < 0:
            ax += ndim
        if ax < 0 or ax >= ndim:
            raise IndexError("repeat axis out of range")
        outer = 1
        for s in shape[:ax]:
            outer *= int(s)
        axis_len = int(shape[ax])
        inner = 1
        for s in shape[ax + 1 :]:
            inner *= int(s)
        block = axis_len * inner
        out = []
        for o in range(outer):
            base = o * block
            for a in range(axis_len):
                chunk = flat[base + a * inner : base + (a + 1) * inner]
                for _ in range(count):
                    out.extend(chunk)
        new_shape = shape[:]
        new_shape[ax] = axis_len * count
        return self._wrap_vals(tuple(new_shape), out)

    def _gather_rows(self, idx_tensor):
        """Numpy-style fancy indexing along axis 0.

        Accepts either:
          * an integer index tensor (gather rows by position; supports negative indices), or
          * a boolean mask tensor of length ``shape[0]`` (filter rows where mask is truthy).

        Works for tensors of any rank ≥ 1; the trailing dimensions are preserved as a
        contiguous block. Returns a freshly-allocated tensor of shape
        ``(k,) + self.shape[1:]`` where ``k`` is the number of selected rows.
        """
        shape = tuple(self.shape)
        if not shape:
            raise IndexError("cannot index a 0-d tensor")
        flat = self._data["_data"]
        idx_flat = idx_tensor._data["_data"]
        idx_dtype = idx_tensor._data.get("dtype", "float32")
        if idx_dtype == "bool":
            if len(idx_flat) != shape[0]:
                raise IndexError(
                    "boolean mask length ({}) must equal tensor.shape[0] ({})".format(
                        len(idx_flat), shape[0]
                    )
                )
            rows = [i for i, v in enumerate(idx_flat) if int(v) != 0]
        else:
            rows = [int(v) for v in idx_flat]
        inner_shape = shape[1:]
        inner_size = 1
        for s in inner_shape:
            inner_size *= int(s)
        n0 = int(shape[0])
        out = []
        for i in rows:
            if i < 0:
                i += n0
            if i < 0 or i >= n0:
                raise IndexError("index {} out of range for axis 0 (size {})".format(i, n0))
            base = i * inner_size
            out.extend(flat[base : base + inner_size])
        return self._wrap_vals((len(rows),) + inner_shape, out)

    def _hwc3_to_rgba(self, value):
        """Expand ``(H, W, 3)`` uint8/float tensor to ``(H, W, 4)`` with alpha 255."""
        value._ensure_flat()
        vshape = tuple(value.shape)
        flat = value._flat_storage()
        if flat is None:
            raise TypeError("expected tensor with flat _data for RGB upload")
        if isinstance(flat, (bytes, bytearray)):
            flat = list(flat)
        if len(vshape) != 3 or int(vshape[2]) != 3:
            return value
        h, w = int(vshape[0]), int(vshape[1])
        rgba = []
        for i in range(h * w):
            o = i * 3
            rgba.append(int(self._flat_get(flat, o)) & 0xFF)
            rgba.append(int(self._flat_get(flat, o + 1)) & 0xFF)
            rgba.append(int(self._flat_get(flat, o + 2)) & 0xFF)
            rgba.append(255)
        return Tensor({
            "shape": (h, w, 4),
            "dtype": value.dtype,
            "device": value._data.get("device", "cpu"),
            "_data": rgba,
        })

    def __setitem__(self, key, value):
        if isinstance(key, slice) and key == slice(None, None, None):
            # Full slice assignment
            # Check if value is a sentinel dict indicating direct fill already happened
            if isinstance(value, dict) and value.get('_direct_fill', False):
                # Data already written directly to buffer by Rust - drop stale flat cache.
                self._data.pop("_data", None)
                return
            if isinstance(value, Tensor):
                fshape = tuple(self.shape)
                vshape = tuple(value.shape)
                if len(fshape) == 3 and len(vshape) == 3 and int(fshape[2]) == 4 and int(vshape[2]) == 3:
                    value = self._hwc3_to_rgba(value)
            # Call Rust function to fill buffer (handles lists and Tensor)
            import xos
            xos.rasterizer._fill_buffer(self._data, value)
        elif isinstance(key, Tensor):
            self._ensure_flat()
            key._ensure_flat()
            self._masked_assign(key, value)
            self._maybe_flush_frame()
        else:
            self._data[key] = value

    def _wrap_like_self(self, values):
        return Tensor({
            "shape": self.shape,
            "dtype": self.dtype,
            "device": self._data.get("device", "cpu"),
            "_data": values,
        })

    def _binary_op(self, other, op):
        left = self._data.get("_data", [])
        lshape = tuple(self.shape)
        if isinstance(other, Tensor):
            right = other._data.get("_data", [])
            rshape = tuple(other.shape)
            if isinstance(right, list):
                if len(lshape) == 2 and len(rshape) == 2 and lshape[1] == 2 and rshape[0] == 1 and rshape[1] == 2 and len(right) == 2:
                    w0, w1 = right[0], right[1]
                    n = lshape[0]
                    out = []
                    for i in range(n):
                        out.append(op(left[2 * i], w0))
                        out.append(op(left[2 * i + 1], w1))
                    return self._wrap_vals(lshape, out)
                if len(left) != len(right):
                    raise ValueError(f"shape mismatch for op: {len(left)} vs {len(right)} elements")
                out = [op(a, b) for a, b in zip(left, right)]
                return self._wrap_like_self(out)
        elif isinstance(other, dict) and "_data" in other:
            right = other["_data"]
        else:
            right = other

        if isinstance(right, list):
            if len(left) != len(right):
                raise ValueError(f"shape mismatch for op: {len(left)} vs {len(right)} elements")
            out = [op(a, b) for a, b in zip(left, right)]
        else:
            out = [op(a, right) for a in left]
        return self._wrap_like_self(out)

    def _cmp_broadcast(self, other, cmp_fn):
        self._ensure_flat()
        left = self._flat_storage()
        if left is None:
            raise TypeError("tensor has no flat storage for comparison")
        lshape = tuple(self.shape)
        if isinstance(other, Tensor):
            other._ensure_flat()
            right = other._flat_storage()
            if right is None:
                raise TypeError("comparison operand has no flat storage")
            rshape = tuple(other.shape)
            if self._flat_len(left) != self._flat_len(right):
                right = self._broadcast_rhs_flat(lshape, rshape, left, right)
            elif len(lshape) == 2 and len(rshape) == 2 and lshape[0] == rshape[0] and lshape[1] == 2 and rshape[1] == 1:
                n = lshape[0]
                out = []
                for i in range(n):
                    rv = self._flat_get(right, i)
                    out.append(cmp_fn(self._flat_get(left, 2 * i), rv))
                    out.append(cmp_fn(self._flat_get(left, 2 * i + 1), rv))
                return self._wrap_vals(lshape, out)
            n = self._flat_len(left)
            return self._wrap_vals(
                lshape,
                [
                    cmp_fn(self._flat_get(left, i), self._flat_get(right, i))
                    for i in range(n)
                ],
            )
        if isinstance(other, (int, float)):
            other = self._cmp_scalar(other)
            n = self._flat_len(left)
            return self._wrap_vals(
                lshape, [cmp_fn(self._flat_get(left, i), other) for i in range(n)]
            )
        raise TypeError("unsupported comparison operand")

    def _logical_binary(self, other, combiner):
        self._ensure_flat()
        la = self._flat_storage()
        if la is None:
            raise TypeError("tensor has no flat storage for logical op")
        lshape = tuple(self.shape)
        if isinstance(other, Tensor):
            other._ensure_flat()
            lb = other._flat_storage()
            if lb is None:
                raise TypeError("logical operand has no flat storage")
            rshape = tuple(other.shape)
            lb = self._broadcast_rhs_flat(lshape, rshape, la, lb)
            n = self._flat_len(la)
            return self._wrap_vals(
                lshape,
                [
                    combiner(self._flat_get(la, i), self._flat_get(lb, i))
                    for i in range(n)
                ],
            )
        raise TypeError("unsupported logical operand")

    def __lt__(self, other):
        return self._cmp_broadcast(other, lambda a, b: 1.0 if a < b else 0.0)

    def __gt__(self, other):
        return self._cmp_broadcast(other, lambda a, b: 1.0 if a > b else 0.0)

    def __eq__(self, other):
        self._ensure_flat()
        left = self._flat_storage()
        if left is not None and self._flat_len(left) == 1:
            if isinstance(other, Tensor):
                other._ensure_flat()
                right = other._flat_storage()
                if right is not None and other._flat_len(right) == 1:
                    return self._flat_get(left, 0) == other._flat_get(right, 0)
            if isinstance(other, (int, float, bool)):
                return self._flat_get(left, 0) == float(other)
        return self._cmp_broadcast(other, lambda a, b: 1.0 if a == b else 0.0)

    def __invert__(self):
        self._ensure_flat()
        d = self._flat_storage()
        if d is None:
            raise TypeError("tensor has no flat storage for invert")
        n = self._flat_len(d)
        return self._wrap_vals(
            self.shape, [0.0 if int(self._flat_get(d, i)) != 0 else 1.0 for i in range(n)]
        )

    def __and__(self, other):
        return self._logical_binary(
            other, lambda a, b: 1.0 if (int(a) != 0 and int(b) != 0) else 0.0
        )

    def __rand__(self, other):
        return self.__and__(other)

    def __or__(self, other):
        return self._logical_binary(
            other, lambda a, b: 1.0 if (int(a) != 0 or int(b) != 0) else 0.0
        )

    def __ror__(self, other):
        return self.__or__(other)

    def __neg__(self):
        return self._wrap_vals(self.shape, [-a for a in self._data["_data"]])

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return other.__sub__(self)
        left = self._data.get("_data", [])
        return self._wrap_like_self([other - a for a in left])

    def __mul__(self, other):
        self._ensure_flat()
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    @property
    def shape(self):
        return self._data.get('shape', ())
    
    @property
    def dtype(self):
        return self._data.get('dtype', 'unknown')

    @property
    def device(self):
        return self._data.get('device', 'cpu')

    def printpack(self, compress=False):
        """Return a single-line pack string (optionally deflate-compressed). Does not print."""
        import xos
        self._ensure_flat()
        return xos._tensor_printpack(self, compress=compress)

    def randomize(self):
        """Fill all elements with random values between this tensor's dtype MIN and MAX."""
        import xos
        xos._tensor_randomize(self)
        return self

    _DEVICE_NAMES = frozenset({"cpu", "gpu", "wasm"})
    _DEVICE_ALIASES = {
        "cuda": "gpu",
        "mps": "gpu",
        "metal": "gpu",
        "wgpu": "gpu",
    }

    def _normalize_device(self, dev):
        if hasattr(dev, "type"):
            dev = dev.type
        d = str(dev).strip().lower()
        return self._DEVICE_ALIASES.get(d, d)

    def _is_device_target(self, name):
        return name in self._DEVICE_NAMES or name in self._DEVICE_ALIASES

    def _to_device(self, device):
        if self._data.get("_xos_gpu_conv_output") and self._flat_storage() is None:
            import xos
            xos._materialize_conv_output(self)
        flat = self._flat_storage()
        if flat is None:
            raise TypeError("tensor has no element data for to()")
        dev = self._normalize_device(device)
        if dev not in self._DEVICE_NAMES:
            raise ValueError(f"unsupported device for to(): {device}")
        if isinstance(flat, list):
            data = list(flat)
        elif isinstance(flat, (bytes, bytearray)):
            data = bytearray(flat)
        else:
            raise TypeError("tensor has no element data for to()")
        return Tensor({
            "shape": tuple(self.shape),
            "dtype": self.dtype,
            "device": dev,
            "_data": data,
        })

    def to(self, target=None, *, dtype=None, device=None):
        """Cast ``dtype`` and/or set ``device`` (metadata; compute still uses the active backend)."""
        if self._data.get("_xos_gpu_conv_output") and self._flat_storage() is None:
            import xos
            xos._materialize_conv_output(self)
        if self._flat_storage() is None:
            raise TypeError("tensor has no element data for to()")
        flat = self._flat_storage()
        if not isinstance(flat, list) and not isinstance(flat, (bytes, bytearray)):
            raise TypeError("tensor has no element data for to()")

        if device is not None:
            out = self._to_device(device)
            if dtype is None and target is None:
                return out
            base = out
        else:
            base = self

        if target is None and dtype is None:
            if device is not None:
                return base
            raise TypeError("to() requires a dtype or device")

        if dtype is not None:
            target = dtype
        elif isinstance(target, str):
            target = target.strip().lower()
            if self._is_device_target(target):
                return base._to_device(target)
        elif hasattr(target, "type"):
            return base._to_device(target)
        elif hasattr(target, "name"):
            target = str(target.name).strip().lower()
        else:
            raise TypeError("dtype must be a dtype object or string")

        alias = {
            "u8": "uint8",
            "byte": "uint8",
            "f32": "float32",
            "float": "float32",
            "i32": "int32",
            "int": "int32",
            "i64": "int64",
        }
        target = alias.get(target, target)

        src = base._flat_storage()
        if target == "uint8":
            out = []
            if isinstance(src, (bytes, bytearray)):
                for v in src:
                    out.append(int(v))
            else:
                for v in src:
                    iv = int(v)
                    if iv < 0:
                        iv = 0
                    elif iv > 255:
                        iv = 255
                    out.append(iv)
        elif target in ("int8", "int16", "int32", "int64", "uint16", "uint32", "uint64"):
            out = [int(v) for v in src]
        elif target in ("float16", "float32", "float64"):
            if isinstance(src, (bytes, bytearray)):
                out = [float(v) for v in src]
            else:
                out = [float(v) for v in src]
        else:
            raise ValueError(f"unsupported dtype for to(): {target}")

        return Tensor({
            "shape": tuple(base.shape),
            "dtype": target,
            "device": base._data.get("device", "cpu"),
            "_data": out,
        })
    
    def size(self):
        """Return the total number of elements (product of all dimensions)"""
        shape = self.shape
        if not shape:
            return 0
        size = 1
        for dim in shape:
            size *= dim
        return size

    def list(self):
        """Nested Python lists with the same structure as ``shape`` (row-major)."""
        if "_data" not in self._data:
            raise TypeError("tensor has no element data for list()")
        flat = self._data["_data"]
        shape = tuple(self.shape)
        return _tensor_unflatten(flat, shape)

    def tuple(self):
        """Same as ``list()`` but with tuples at each nesting level."""
        return _nested_list_to_tuple(self.list())

    def astuple(self):
        """Nested tuples for ``shape`` (alias for ``tuple()``)."""
        return self.tuple()

    def min(self, axis=None, out=None, keepdims=False, **kwargs):
        """Global minimum (numpy-style signature); reductions run in Rust. ``axis`` / ``out`` / ``keepdims`` … not implemented yet."""
        if kwargs:
            raise TypeError(
                "Tensor.min() got unexpected keyword arguments: "
                + ", ".join(sorted(kwargs.keys()))
            )
        if axis is not None:
            raise NotImplementedError("Tensor.min(axis=...) is not implemented yet")
        if out is not None:
            raise NotImplementedError("Tensor.min(out=...) is not implemented yet")
        if keepdims:
            raise NotImplementedError("Tensor.min(keepdims=True) is not implemented yet")
        import xos

        return xos._tensor_min(self)

    def max(self, axis=None, out=None, keepdims=False, **kwargs):
        """Global maximum (numpy-style); reductions run in Rust."""
        if kwargs:
            raise TypeError(
                "Tensor.max() got unexpected keyword arguments: "
                + ", ".join(sorted(kwargs.keys()))
            )
        if axis is not None:
            raise NotImplementedError("Tensor.max(axis=...) is not implemented yet")
        if out is not None:
            raise NotImplementedError("Tensor.max(out=...) is not implemented yet")
        if keepdims:
            raise NotImplementedError("Tensor.max(keepdims=True) is not implemented yet")
        import xos

        return xos._tensor_max(self)

    def index(self, text):
        """Gather characters of ``text`` at the integer positions in this tensor.

        Treats ``self`` as a flat sequence of indices. Negative indices wrap numpy-style.
        Useful with boolean masks, e.g. ``xos.arange(n)[mask].index(s)`` to subset a
        string by the positions where ``mask`` is truthy. Returns a new ``str``.
        """
        import xos

        return xos._tensor_index_string(self, str(text))

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, **kwargs):
        """Arithmetic mean over the flat buffer (numpy-style ``axis=None`` default); reductions run in Rust."""
        if kwargs:
            raise TypeError(
                "Tensor.mean() got unexpected keyword arguments: "
                + ", ".join(sorted(kwargs.keys()))
            )
        if axis is not None:
            raise NotImplementedError("Tensor.mean(axis=...) is not implemented yet")
        if dtype is not None:
            raise NotImplementedError("Tensor.mean(dtype=...) is not implemented yet")
        if out is not None:
            raise NotImplementedError("Tensor.mean(out=...) is not implemented yet")
        if keepdims:
            raise NotImplementedError("Tensor.mean(keepdims=True) is not implemented yet")
        import xos

        return xos._tensor_mean(self)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, **kwargs):
        """Arithmetic sum over the flat buffer (numpy-style ``axis=None`` default)."""
        if kwargs:
            raise TypeError(
                "Tensor.sum() got unexpected keyword arguments: "
                + ", ".join(sorted(kwargs.keys()))
            )
        if axis is not None:
            raise NotImplementedError("Tensor.sum(axis=...) is not implemented yet")
        if out is not None:
            raise NotImplementedError("Tensor.sum(out=...) is not implemented yet")
        if keepdims:
            raise NotImplementedError("Tensor.sum(keepdims=True) is not implemented yet")
        import xos
        if dtype is None:
            return xos._tensor_sum(self)
        return xos._tensor_sum(self, dtype=dtype)

    def tostring(self, full=False):
        """Human-readable string; ``full=True`` prints every element."""
        if not full:
            return self._summary_string()
        self._ensure_flat()
        if "_data" not in self._data:
            return "xos.Tensor(shape={}, dtype={}, device={!r}, empty)".format(
                self.shape, self.dtype, self.device
            )
        try:
            data = self.astuple()
        except Exception:
            return "xos.Tensor(shape={}, dtype={}, device={!r}, <opaque>)".format(
                self.shape, self.dtype, self.device
            )
        values = _format_nested_values(data)
        return "xos.Tensor(shape={}, dtype={}, device={!r}, values={})".format(
            self.shape, self.dtype, self.device, values
        )

    def _summary_string(self):
        if "_data" not in self._data:
            return "xos.Tensor(shape={}, dtype=u8)".format(self.shape)
        self._ensure_flat()
        flat = self._flat_storage()
        if flat is not None and self._flat_len(flat) == 1:
            v = self._flat_get(flat, 0)
            if self.dtype == "bool":
                scalar = "True" if int(v) != 0 else "False"
            elif self.dtype.startswith("int") or self.dtype.startswith("uint"):
                scalar = str(int(v))
            else:
                scalar = str(float(v))
            return "xos.Tensor({}, dtype={}, device={!r})".format(
                scalar, self.dtype, self.device
            )
        import xos

        try:
            mn, mx, av = xos._tensor_min_max_mean(self)
        except ValueError:
            return "xos.Tensor(shape={}, dtype={}, empty)".format(
                self.shape, self.dtype
            )
        except TypeError:
            return "xos.Tensor(shape={}, dtype={}, <opaque flat storage>)".format(
                self.shape, self.dtype
            )
        return "xos.Tensor(shape={}, dtype={}, min={:.3f}, max={:.3f}, mean={:.3f})".format(
            self.shape, self.dtype, float(mn), float(mx), float(av)
        )

    def __str__(self):
        return self._summary_string()

    def __repr__(self):
        return self.__str__()

class EngineState:
    """Snapshot of engine context for Python ``Application.on_events`` (attributes set each call)."""

class SafeRegion:
    """Device safe inset in the same normalized space as ``xos.ui.Text`` (viewport 0..1).

    Updated from the engine each tick when ``_xos_engine_bound`` is true. In standalone mode
    defaults to full viewport (0, 0, 1, 1).
    """

    __slots__ = ("x1", "y1", "x2", "y2")

    def __init__(self, x1=0.0, y1=0.0, x2=1.0, y2=1.0):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)

    @property
    def width(self):
        return float(self.x2 - self.x1)

    @property
    def height(self):
        return float(self.y2 - self.y1)

    def renormalize(self, x1=0.0, y1=0.0, x2=1.0, y2=1.0):
        """Map ``(x1,y1,x2,y2)`` in inset-local coords (0..1 within the safe rectangle) onto the
        same normalized frame space as ``xos.ui.Text``. Returns ``(x1, y1, x2, y2)``.
        """
        lx1, ly1, lx2, ly2 = float(x1), float(y1), float(x2), float(y2)
        w = self.width
        h = self.height
        return (
            self.x1 + lx1 * w,
            self.y1 + ly1 * h,
            self.x1 + lx2 * w,
            self.y1 + ly2 * h,
        )

    def __repr__(self):
        return "SafeRegion(x1={!r}, y1={!r}, x2={!r}, y2={!r})".format(
            self.x1, self.y1, self.x2, self.y2
        )


class Frame:
    """Wrapper to make frame dict behave like an object with methods"""
    def __init__(self, data):
        self._data = data
        self._tensor = Tensor(data.get('tensor', {}))
    
    @property
    def tensor(self):
        """RGBA frame tensor (``xos.Tensor``); compute uses GPU on native via Burn/WGPU."""
        return self._tensor
    
    def get_width(self):
        return self._data['width']
    
    def get_height(self):
        return self._data['height']
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value

    def clear(self, *color):
        """
        Clear the frame with an RGB or RGBA color.
        Accepts:
          - frame.clear()                      -> black
          - frame.clear((r, g, b))            -> alpha defaults to 255
          - frame.clear((r, g, b, a))
          - frame.clear(r, g, b)
          - frame.clear(r, g, b, a)
        """
        import xos

        if len(color) == 0:
            rgba = (0, 0, 0, 255)
        elif len(color) == 1 and isinstance(color[0], tuple):
            if len(color[0]) == 3:
                rgba = (color[0][0], color[0][1], color[0][2], 255)
            elif len(color[0]) == 4:
                rgba = color[0]
            else:
                raise TypeError("frame.clear color tuple must be RGB or RGBA")
        elif len(color) == 3:
            rgba = (color[0], color[1], color[2], 255)
        elif len(color) == 4:
            rgba = (color[0], color[1], color[2], color[3])
        else:
            raise TypeError("frame.clear accepts (), RGB, or RGBA")

        # Standalone preview frames carry _xos_viewport_id. Only bind/unbind here when
        # there is no active tick-owned context.
        vid = self._data.get("_xos_viewport_id")
        bound = False
        if vid is not None and not xos.frame._has_context():
            w = int(self._data["width"])
            h = int(self._data["height"])
            xos.frame._begin_standalone(int(vid), w, h)
            bound = True
        try:
            xos.rasterizer.fill(self, rgba)
        finally:
            if bound:
                xos.frame._end_standalone()

class Verbosities:
  """Debug flags for xos apps (see ``Application.verbosities``)."""
  function_calls = False

DEFAULT_MAX_FPS = 2048.0


class Application:
    """Base class for xos applications. Extend this class and implement __init__() and tick().

    Set ``function_calls = True`` on a subclass to trace method entry (including ``__init__``)
    before ``app.run()`` — instance ``app.verbosities.function_calls`` only applies after creation.

    ``self.safe_region`` is an ``xos.SafeRegion`` (``x1,y1,x2,y2`` in the same normalized space as ``xos.ui.Text``)
    refreshed each engine tick — use ``safe_region.renormalize(lx1, ly1, lx2, ly2)`` for inset-local ``0..1`` rects.

    Routed input sets ``self._xos_event`` (a dict with ``kind``, etc.) before ``on_events()`` runs and
    clears it only after your handler returns, so every component sees the same event in one call.
    Pointer kinds ``mouse_down``, ``mouse_up``, and ``mouse_move`` also include ``x``, ``y`` (frame px),
    ``button`` (``\"left\"`` / ``\"right\"``), ``is_left``, and ``is_right`` for parity with host events.
    Kinds include mouse, scroll, ``key_char``, and desktop ``shortcut`` (e.g. Cmd/Ctrl+C/V/X/A).
    Conventional order is ``self.keyboard.on_events(self)`` then ``self.text.on_events(self)`` for
    pointer, ``key_char``, and shortcuts alike — no special cases per event type are required.
    """

    function_calls = False

    def __init__(self, headless=None, device=None, width=None, height=None, max_fps=None):
        import builtins
        next_id = int(getattr(builtins, "__xos_next_viewport_id__", 0))
        builtins.__xos_next_viewport_id__ = next_id + 1
        self._xos_viewport_id = next_id
        self._xos_initialized = True
        self._xos_engine_bound = False
        self.mouse = None  # Overwritten by engine in run(); standalone set below
        self.fps = 0.0  # Frames per second derived from timestep
        self.dt = 0.0  # Last frame delta time in seconds (same source as engine timestep)
        self.t = 0  # Tick index: 0 on first tick(), then increments after each tick completes
        # F3 scale as percent/100 (0.25..5.0 for 25–500%); default 1.0 at 100%.
        # Engine syncs `xos_scale` each tick.
        self.xos_scale = 1.0
        self._xos_standalone_width = 800
        self._xos_standalone_height = 600
        self._xos_last_tick_time = None
        self._xos_ticks_completed = 0
        # Full viewport until the engine replaces this (see Rust ``sync_app_safe_region``).
        self.safe_region = SafeRegion(0.0, 0.0, 1.0, 1.0)
        if headless is not None:
            self.headless = bool(headless)
        if device is not None:
            self.device = device
        if width is not None:
            self._xos_standalone_width = int(max(1, int(width)))
        if height is not None:
            self._xos_standalone_height = int(max(1, int(height)))
        raw_max_fps = max_fps if max_fps is not None else getattr(self, "max_fps", DEFAULT_MAX_FPS)
        self.max_fps = float(raw_max_fps)

        self.verbosities = Verbosities()
        class_verb = getattr(type(self), "verbosities", None)
        if isinstance(class_verb, Verbosities):
            self.verbosities.function_calls = bool(class_verb.function_calls)
        elif bool(getattr(type(self), "function_calls", False)):
            self.verbosities.function_calls = True

        # Register before standalone frame so device/headless class attrs are visible to native ops.
        import builtins
        builtins.__xos_app_instance__ = self
        # Standalone framebuffer for __init__ drawing (uniform_fill, rasterizer, etc.).
        self._xos_init_standalone_frame()

    def _xos_init_standalone_frame(self):
        """Build a CPU frame + rasterizer context for standalone use (before tick/run)."""
        import xos

        if getattr(self, "_xos_engine_bound", False):
            return
        w = int(getattr(self, "_xos_standalone_width", 800))
        h = int(getattr(self, "_xos_standalone_height", 600))
        fd = xos.frame._begin_standalone(int(self._xos_viewport_id), w, h)
        fd["_xos_viewport_id"] = int(self._xos_viewport_id)
        self.frame = Frame(fd)
        self.mouse = {"x": 0.0, "y": 0.0, "is_left_clicking": False, "is_right_clicking": False}
        xos.rasterizer.fill(self.frame.tensor, (0, 0, 0, 255))
        xos.frame._end_standalone()

    @property
    def scale(self):
        """F3 UI scale as percent/100 (100% → 1.0, 25–500% → 0.25–5.0). Updated each tick."""
        return float(getattr(self, "xos_scale", 1.0))

    @staticmethod
    def _xos_wrap_subclass_method(app_cls_name, method_name, fn):
        if isinstance(fn, (staticmethod, classmethod)):
            return fn

        def wrapper(self, *args, **kwargs):
            import builtins
            import time
            import xos

            verb = getattr(self, "verbosities", None)
            trace = verb is not None and getattr(verb, "function_calls", False)
            depth = 0
            if trace:
                depth = int(getattr(builtins, "__xos_trace_depth__", 0))
                prefix = ("│   " * depth) + "├── "
                xos.print_color(f"{prefix}&b{app_cls_name}&f.&5{method_name}&f()")
                builtins.__xos_trace_depth__ = depth + 1
                t0 = time.perf_counter()
            try:
                return fn(self, *args, **kwargs)
            finally:
                if trace:
                    ms = (time.perf_counter() - t0) * 1000.0
                    out_prefix = ("│   " * depth) + "└── "
                    xos.print_color(
                        f"{out_prefix}&8{ms:,.2f} ms&f  &7{app_cls_name}.{method_name}&f"
                    )
                    builtins.__xos_trace_depth__ = depth

        wrapper.__name__ = getattr(fn, "__name__", method_name)
        return wrapper

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, attr in list(cls.__dict__.items()):
            if name.startswith("_"):
                continue
            if not callable(attr):
                continue
            if isinstance(attr, (staticmethod, classmethod, property)):
                continue
            setattr(
                cls,
                name,
                Application._xos_wrap_subclass_method(cls.__name__, name, attr),
            )

        user_tick = cls.__dict__.get("tick")
        if user_tick is None:
            return

        def _wrapped_tick(self, *args, **kw):
            self._xos_pre_tick()
            try:
                return user_tick(self, *args, **kw)
            finally:
                self._xos_post_tick()

        cls.tick = _wrapped_tick

    def _xos_pre_tick(self):
        import xos
        if not getattr(self, "_xos_initialized", False):
            raise RuntimeError("xos.Application.__init__() was not called. Call super().__init__() first.")

        # Engine-driven mode (normal app.run lifecycle): engine owns frame context.
        if getattr(self, "_xos_engine_bound", False):
            self.pre_tick()
            return

        # In standalone preview mode, follow live preview window size and F3 scale.
        if not bool(getattr(self, "headless", False)):
            ws = xos.frame._standalone_window_size(self._xos_viewport_id)
            if ws is not None:
                w, h = ws
                self._xos_standalone_width = int(max(1, w))
                self._xos_standalone_height = int(max(1, h))
            sp = xos.frame._standalone_ui_scale(self._xos_viewport_id)
            if sp is not None:
                self.xos_scale = float(sp)

        # Standalone timing (manual Python-driven tick loop).
        now = xos.time.perf_counter()
        last = getattr(self, "_xos_last_tick_time", None)
        if last is None:
            dt = 1.0 / 60.0
        else:
            dt = max(1e-5, now - last)
        self._xos_last_tick_time = now
        self.dt = dt
        self.fps = 1.0 / dt
        self.t = int(getattr(self, "_xos_ticks_completed", 0))

        # Standalone mode (manual Python-driven tick loop): create temporary frame context.
        frame_dict = xos.frame._begin_standalone(
            int(getattr(self, "_xos_viewport_id", 0)),
            int(getattr(self, "_xos_standalone_width", 800)),
            int(getattr(self, "_xos_standalone_height", 600)),
        )
        frame_dict["_xos_viewport_id"] = int(getattr(self, "_xos_viewport_id", 0))
        self.frame = Frame(frame_dict)
        self.mouse = {"x": 0.0, "y": 0.0, "is_left_clicking": False, "is_right_clicking": False}
        self.pre_tick()

    def _xos_post_tick(self):
        import xos
        try:
            self.post_tick()
        finally:
            if not getattr(self, "_xos_engine_bound", False):
                if not bool(getattr(self, "headless", False)):
                    xos.frame._present_standalone(int(getattr(self, "_xos_viewport_id", 0)))
                xos.frame._end_standalone()
                self._xos_ticks_completed = int(getattr(self, "_xos_ticks_completed", 0)) + 1
                # Enforced in Rust for higher precision and lower Python overhead.
                xos._standalone_tick_pace(
                    int(getattr(self, "_xos_viewport_id", 0)),
                    float(getattr(self, "max_fps", DEFAULT_MAX_FPS)),
                )
    
    def get_width(self):
        """Get the current frame width"""
        return self.frame.get_width()
    
    def get_height(self):
        """Get the current frame height"""
        return self.frame.get_height()
    
    def tick(self):
        """Called every frame.

        Base `Application` supports functional-style usage (`app = xos.Application(...); app.tick()`)
        by running the full pre/post pipeline even without a subclass override.
        """
        self._xos_pre_tick()
        try:
            return None
        finally:
            self._xos_post_tick()

    def pre_tick(self):
        """Called before each tick() in both run() and standalone tick() modes."""
        pass

    def post_tick(self):
        """Called after each tick() in both run() and standalone tick() modes."""
        pass
    
    def on_mouse_down(self, x, y):
        """Called when mouse is clicked. Override this method (optional)."""
        pass
    
    def on_mouse_up(self, x, y):
        """Called when mouse is released. Override this method (optional)."""
        pass
    
    def on_mouse_move(self, x, y):
        """Called when mouse moves. Override this method (optional)."""
        pass

    def on_scroll(self, dx, dy):
        """Called on mouse wheel / trackpad scroll. Override (optional)."""
        pass
    
    def on_screen_size_change(self, width, height):
        """Called when screen size changes. Override this method (optional)."""
        pass
    
    def run(self):
        """Run the application with the xos engine."""
        # Store self in builtins so Rust can find it from any scope
        import builtins
        builtins.__xos_app_instance__ = self
"#;

/// PyApp wraps a Python Application instance and implements the Rust Application trait
pub struct PyApp {
    interpreter: Interpreter,
    app_instance: Option<PyObjectRef>,
    /// Number of `tick()` calls that have fully finished (starts at 0; incremented after each tick).
    ticks_completed: u64,
    /// RGBA snapshot of the standalone framebuffer drawn during `Application.__init__`.
    init_frame_snapshot: Option<(Vec<u8>, usize, usize)>,
    init_viewport_id: Option<u64>,
    init_blit_applied: bool,
}

impl PyApp {
    pub fn new(interpreter: Interpreter, app_instance: PyObjectRef) -> Self {
        let mut app = Self {
            interpreter,
            app_instance: Some(app_instance.clone()),
            ticks_completed: 0,
            init_frame_snapshot: None,
            init_viewport_id: None,
            init_blit_applied: false,
        };
        // Snapshot __init__ framebuffer immediately after the script runs (before engine setup).
        let viewport_id = app.interpreter.enter(|vm| {
            crate::xos_module::python_app_viewport_id(vm, &app_instance)
        });
        if let Some(viewport_id) = viewport_id {
            app.init_viewport_id = Some(viewport_id);
            app.capture_init_frame_snapshot(viewport_id);
        }
        app
    }

    fn capture_init_frame_snapshot(&mut self, viewport_id: u64) {
        if !crate::xos_module::standalone_frame_was_drawn(viewport_id) {
            return;
        }
        if let Some(snapshot) = crate::xos_module::snapshot_standalone_init_frame(viewport_id) {
            self.init_frame_snapshot = Some(snapshot);
        }
    }

    fn try_apply_init_frame_snapshot(&mut self, state: &mut EngineState) -> bool {
        if self.init_blit_applied {
            return false;
        }
        let shape = state.frame.shape();
        let dest_h = shape[0];
        let dest_w = shape[1];
        let dest = state.frame.buffer_mut();

        let applied = if let Some((src, src_w, src_h)) = self.init_frame_snapshot.as_ref() {
            crate::xos_module::blit_rgba_init_to_buffer(src, *src_w, *src_h, dest, dest_w, dest_h)
        } else if let Some(viewport_id) = self.init_viewport_id {
            crate::xos_module::apply_standalone_init_to_engine_buffer(
                viewport_id,
                dest,
                dest_w,
                dest_h,
            )
        } else {
            false
        };

        if applied {
            if state.compute_device == xos_core::compute_device::ComputeDevice::Gpu {
                state.frame.ensure_gpu_from_cpu();
            }
            self.init_blit_applied = true;
        }
        applied
    }
}

fn read_python_device_pref(
    vm: &rustpython_vm::VirtualMachine,
    app_instance: &rustpython_vm::PyObjectRef,
) -> Option<String> {
    vm.get_attribute_opt(app_instance.clone(), "device")
        .ok()
        .flatten()
        .and_then(|obj| {
            if obj.is(&vm.ctx.none()) {
                None
            } else {
                obj.try_into_value::<String>(vm).ok()
            }
        })
}

impl Application for PyApp {
    fn setup(&mut self, state: &mut EngineState) -> Result<(), String> {
        if let Some(ref app_instance) = self.app_instance {
            let viewport_id = self.interpreter.enter(|vm| -> Result<Option<u64>, String> {
                let pref = read_python_device_pref(vm, app_instance);
                state
                    .apply_compute_device_pref(pref.as_deref())
                    .map_err(|e| e.to_string())?;

                // Create Python frame object from engine state
                let frame_dict = crate::engine::py_bindings::create_py_frame_state(
                    vm,
                    &mut state.frame,
                    state.compute_device,
                )
                .map_err(|e| format!("Failed to create frame object: {:?}", e))?;

                if let Ok(wrapper_class) = vm.builtins.get_attr("Frame", vm) {
                    // Use the newer call API instead of deprecated invoke
                    if let Ok(frame_obj) = wrapper_class.call((frame_dict.clone(),), vm) {
                        app_instance
                            .set_attr("frame", frame_obj, vm)
                            .map_err(|e| format!("Failed to set frame attribute: {:?}", e))?;
                    } else {
                        // Fallback: just use the dict directly
                        app_instance
                            .set_attr("frame", frame_dict, vm)
                            .map_err(|e| format!("Failed to set frame attribute: {:?}", e))?;
                    }
                } else {
                    // Fallback: just use the dict directly
                    app_instance
                        .set_attr("frame", frame_dict, vm)
                        .map_err(|e| format!("Failed to set frame attribute: {:?}", e))?;
                }

                // Create mouse object
                let mouse_dict = vm.ctx.new_dict();
                mouse_dict
                    .set_item("x", vm.ctx.new_float(state.mouse.x as f64).into(), vm)
                    .map_err(|e| format!("Mouse x error: {:?}", e))?;
                mouse_dict
                    .set_item("y", vm.ctx.new_float(state.mouse.y as f64).into(), vm)
                    .map_err(|e| format!("Mouse y error: {:?}", e))?;
                mouse_dict
                    .set_item(
                        "is_left_clicking",
                        vm.ctx.new_bool(state.mouse.is_left_clicking).into(),
                        vm,
                    )
                    .map_err(|e| format!("Mouse clicking error: {:?}", e))?;
                mouse_dict
                    .set_item(
                        "is_right_clicking",
                        vm.ctx.new_bool(state.mouse.is_right_clicking).into(),
                        vm,
                    )
                    .map_err(|e| format!("Mouse right click error: {:?}", e))?;

                app_instance
                    .set_attr("mouse", mouse_dict, vm)
                    .map_err(|e| format!("Failed to set mouse attribute: {:?}", e))?;
                app_instance
                    .set_attr("_xos_engine_bound", vm.ctx.new_bool(true), vm)
                    .map_err(|e| format!("Failed to set _xos_engine_bound attribute: {:?}", e))?;

                // Seed timing field so Python can read it in setup/tick.
                let timestep = state.delta_time_seconds.max(1e-5) as f64;
                app_instance
                    .set_attr("dt", vm.ctx.new_float(timestep), vm)
                    .map_err(|e| format!("Failed to set dt attribute: {:?}", e))?;
                app_instance
                    .set_attr("fps", vm.ctx.new_float(1.0 / timestep), vm)
                    .map_err(|e| format!("Failed to set fps attribute: {:?}", e))?;
                app_instance
                    .set_attr("t", vm.ctx.new_int(0usize), vm)
                    .map_err(|e| format!("Failed to set t attribute: {:?}", e))?;
                app_instance
                    .set_attr(
                        "xos_scale",
                        vm.ctx.new_float(state.ui_scale_percent as f64 / 100.0),
                        vm,
                    )
                    .map_err(|e| format!("Failed to set xos_scale attribute: {:?}", e))?;

                sync_app_safe_region(vm, app_instance, &state.frame.safe_region_boundaries)
                    .map_err(|e| format!("Failed to sync safe_region: {:?}", e))?;

                Ok(crate::xos_module::python_app_viewport_id(vm, app_instance))
            })?;

            // Refresh snapshot; display blit is deferred to tick() when the pixels buffer is active.
            if let Some(viewport_id) = viewport_id {
                self.init_viewport_id = Some(viewport_id);
                self.capture_init_frame_snapshot(viewport_id);
            }

            Ok(())
        } else {
            Err("No Python app instance".to_string())
        }
    }

    fn tick(&mut self, state: &mut EngineState) {
        if let Some(app_instance) = self.app_instance.clone() {
            // Set the frame buffer context for the rasterizer
            let shape = state.frame.shape();
            let width = shape[1];
            let height = shape[0];
            // Bind CPU staging for rasterizer/text; do not pull GPU→CPU here (sync happens once before present).
            let buffer = state.frame.staging_slice_mut_for_tick();
            crate::rasterizer::set_frame_buffer_context(buffer, width, height);

            // Apply __init__ framebuffer to the live display buffer (pixels mirror when windowed).
            let _ = self.try_apply_init_frame_snapshot(state);

            let tick_index = self.ticks_completed;
            let mut tick_failed = false;

            self.interpreter.enter(|vm| {
                #[cfg(target_arch = "wasm32")]
                crate::runtime::install_wasm_python_stdio(vm);

                // Require subclasses to call super().__init__() so base fields exist.
                let initialized_ok = match vm.get_attribute_opt(app_instance.clone(), "_xos_initialized") {
                    Ok(Some(flag_obj)) => flag_obj.clone().try_into_value::<bool>(vm).unwrap_or(false),
                    Ok(None) | Err(_) => false,
                };
                if !initialized_ok {
                    log_py_runtime_error(
                        "Python app init error:\nRuntimeError: xos.Application.__init__() was not called. \
Call super().__init__() in your app __init__ before using tick().",
                    );
                    tick_failed = true;
                    return;
                }

                // Update frame data before calling tick
                if let Ok(Some(frame_obj)) = vm.get_attribute_opt(app_instance.clone(), "frame") {
                    let _ = crate::engine::py_bindings::update_py_frame_state(
                        vm,
                        frame_obj.clone(),
                        &mut state.frame,
                        state.compute_device,
                    );

                    // Update mouse data
                    let mouse_dict = vm.ctx.new_dict();
                    let _ = mouse_dict.set_item("x", vm.ctx.new_float(state.mouse.x as f64).into(), vm);
                    let _ = mouse_dict.set_item("y", vm.ctx.new_float(state.mouse.y as f64).into(), vm);
                    let _ = mouse_dict.set_item("is_left_clicking", vm.ctx.new_bool(state.mouse.is_left_clicking).into(), vm);
                    let _ = mouse_dict.set_item("is_right_clicking", vm.ctx.new_bool(state.mouse.is_right_clicking).into(), vm);
                    let _ = app_instance.set_attr("mouse", mouse_dict, vm);

                    // Expose timestep and FPS directly to Python app.
                    let timestep = state.delta_time_seconds.max(1e-5) as f64;
                    let _ = app_instance.set_attr("dt", vm.ctx.new_float(timestep), vm);
                    let _ = app_instance.set_attr("fps", vm.ctx.new_float(1.0 / timestep), vm);

                    // Tick counter: value during tick() is N ticks completed so far (0 on first tick).
                    let _ = app_instance.set_attr("t", vm.ctx.new_int(tick_index as usize), vm);
                    let _ = app_instance.set_attr("xos_scale", vm.ctx.new_float(state.ui_scale_percent as f64 / 100.0), vm);
                    let _ = sync_app_safe_region(vm, &app_instance, &state.frame.safe_region_boundaries);

                    // Call tick (xos.ui may call into Rust with `TickEngineStateGuard` active)
                    let _tls_guard = TickEngineStateGuard::install(state);
                    if let Err(e) = vm.call_method(&app_instance, "tick", ()) {
                        let error_msg = format_python_exception(vm, &e);
                        log_py_runtime_error(&format!("Python tick error:\n{}", error_msg));
                        tick_failed = true;
                    }
                }
            });

            if tick_failed {
                // Stop ticking this Python app after the first runtime error on native. In the browser,
                // keep the RAF loop recoverable and surface the error in the console.
                #[cfg(not(target_arch = "wasm32"))]
                {
                    self.app_instance = None;
                    xos_core::engine::native_engine::request_exit();
                    log_py_runtime_error("Python app execution stopped after tick error.");
                }
            } else {
                self.ticks_completed = self.ticks_completed.saturating_add(1);
            }

            if crate::rasterizer::FRAME_CPU_WRITTEN
                .lock()
                .map(|w| *w)
                .unwrap_or(false)
            {
                state.frame.mark_cpu_staging_dirty();
            }

            // Clear the frame buffer context after tick
            crate::rasterizer::clear_frame_buffer_context();
        }
    }

    fn on_mouse_down(&mut self, state: &mut EngineState) {
        let shape = state.frame.shape();
        let _keyboard_hit = state.keyboard.onscreen.on_mouse_down(
            state.mouse.x,
            state.mouse.y,
            shape[1] as f32,
            shape[0] as f32,
        );
        if let Some(ref app_instance) = self.app_instance {
            self.interpreter.enter(|vm| {
                let x = state.mouse.x;
                let y = state.mouse.y;
                let mouse_dict = vm.ctx.new_dict();
                let _ = mouse_dict.set_item("x", vm.ctx.new_float(x as f64).into(), vm);
                let _ = mouse_dict.set_item("y", vm.ctx.new_float(y as f64).into(), vm);
                let _ = mouse_dict.set_item(
                    "is_left_clicking",
                    vm.ctx.new_bool(state.mouse.is_left_clicking).into(),
                    vm,
                );
                let _ = mouse_dict.set_item(
                    "is_right_clicking",
                    vm.ctx.new_bool(state.mouse.is_right_clicking).into(),
                    vm,
                );
                let _ = app_instance.set_attr("mouse", mouse_dict, vm);
                let _ = vm.call_method(app_instance, "on_mouse_down", (x, y));
                try_dispatch_python_on_events(vm, app_instance, state, RoutedPyEvent::MouseDown);
            });
        }
    }

    fn on_mouse_up(&mut self, state: &mut EngineState) {
        state.keyboard.onscreen.on_mouse_up();
        if let Some(ref app_instance) = self.app_instance {
            self.interpreter.enter(|vm| {
                let x = state.mouse.x;
                let y = state.mouse.y;
                let mouse_dict = vm.ctx.new_dict();
                let _ = mouse_dict.set_item("x", vm.ctx.new_float(x as f64).into(), vm);
                let _ = mouse_dict.set_item("y", vm.ctx.new_float(y as f64).into(), vm);
                let _ = mouse_dict.set_item(
                    "is_left_clicking",
                    vm.ctx.new_bool(state.mouse.is_left_clicking).into(),
                    vm,
                );
                let _ = mouse_dict.set_item(
                    "is_right_clicking",
                    vm.ctx.new_bool(state.mouse.is_right_clicking).into(),
                    vm,
                );
                let _ = app_instance.set_attr("mouse", mouse_dict, vm);
                let _ = vm.call_method(app_instance, "on_mouse_up", (x, y));
                try_dispatch_python_on_events(vm, app_instance, state, RoutedPyEvent::MouseUp);
                drain_embed_synthetic_click(vm, app_instance, state);
            });
        }
    }

    fn on_mouse_move(&mut self, state: &mut EngineState) {
        if let Some(ref app_instance) = self.app_instance {
            self.interpreter.enter(|vm| {
                let x = state.mouse.x;
                let y = state.mouse.y;
                let mouse_dict = vm.ctx.new_dict();
                let _ = mouse_dict.set_item("x", vm.ctx.new_float(x as f64).into(), vm);
                let _ = mouse_dict.set_item("y", vm.ctx.new_float(y as f64).into(), vm);
                let _ = mouse_dict.set_item(
                    "is_left_clicking",
                    vm.ctx.new_bool(state.mouse.is_left_clicking).into(),
                    vm,
                );
                let _ = mouse_dict.set_item(
                    "is_right_clicking",
                    vm.ctx.new_bool(state.mouse.is_right_clicking).into(),
                    vm,
                );
                let _ = app_instance.set_attr("mouse", mouse_dict, vm);
                let _ = vm.call_method(app_instance, "on_mouse_move", (x, y));
                try_dispatch_python_on_events(vm, app_instance, state, RoutedPyEvent::MouseMove);
            });
        }
    }

    fn on_scroll(&mut self, state: &mut EngineState, dx: f32, dy: f32, unit: ScrollWheelUnit) {
        if let Some(ref app_instance) = self.app_instance {
            self.interpreter.enter(|vm| {
                let mouse_dict = vm.ctx.new_dict();
                let _ = mouse_dict.set_item("x", vm.ctx.new_float(state.mouse.x as f64).into(), vm);
                let _ = mouse_dict.set_item("y", vm.ctx.new_float(state.mouse.y as f64).into(), vm);
                let _ = mouse_dict.set_item(
                    "is_left_clicking",
                    vm.ctx.new_bool(state.mouse.is_left_clicking).into(),
                    vm,
                );
                let _ = mouse_dict.set_item(
                    "is_right_clicking",
                    vm.ctx.new_bool(state.mouse.is_right_clicking).into(),
                    vm,
                );
                let _ = app_instance.set_attr("mouse", mouse_dict, vm);
                let _ = vm.call_method(app_instance, "on_scroll", (dx, dy));
                try_dispatch_python_on_events(
                    vm,
                    app_instance,
                    state,
                    RoutedPyEvent::Scroll { dx, dy, unit },
                );
            });
        }
    }

    fn on_key_char(&mut self, state: &mut EngineState, ch: char) {
        if let Some(ref app_instance) = self.app_instance {
            self.interpreter.enter(|vm| {
                try_dispatch_python_on_events(vm, app_instance, state, RoutedPyEvent::KeyChar(ch));
            });
        }
    }

    fn on_key_shortcut(&mut self, state: &mut EngineState, shortcut: ShortcutAction) {
        if let Some(ref app_instance) = self.app_instance {
            self.interpreter.enter(|vm| {
                try_dispatch_python_on_events(
                    vm,
                    app_instance,
                    state,
                    RoutedPyEvent::Shortcut(shortcut),
                );
            });
        }
    }

    fn on_screen_size_change(&mut self, state: &mut EngineState, width: u32, height: u32) {
        if let Some(ref app_instance) = self.app_instance {
            // Set the frame buffer context so Python can write to it
            let shape = state.frame.shape();
            let frame_width = shape[1];
            let frame_height = shape[0];
            let buffer = state.frame.staging_slice_mut_for_tick();
            crate::rasterizer::set_frame_buffer_context(
                buffer,
                frame_width,
                frame_height,
            );

            let _tls_guard = TickEngineStateGuard::install(state);
            self.interpreter.enter(|vm| {
                #[cfg(target_arch = "wasm32")]
                crate::runtime::install_wasm_python_stdio(vm);

                // Update frame data before calling the handler
                if let Ok(Some(frame_obj)) = vm.get_attribute_opt(app_instance.clone(), "frame") {
                    let _ = crate::engine::py_bindings::update_py_frame_state(
                        vm,
                        frame_obj,
                        &mut state.frame,
                        state.compute_device,
                    );
                }
                // Call the Python handler
                if let Err(e) =
                    vm.call_method(app_instance, "on_screen_size_change", (width, height))
                {
                    log_py_runtime_error(&format!(
                        "Python on_screen_size_change error:\n{}",
                        format_python_exception(vm, &e)
                    ));
                }
            });

            if crate::rasterizer::FRAME_CPU_WRITTEN
                .lock()
                .map(|w| *w)
                .unwrap_or(false)
            {
                state.frame.mark_cpu_staging_dirty();
            }

            // Clear the frame buffer context after handler completes
            crate::rasterizer::clear_frame_buffer_context();
        }
    }
}
