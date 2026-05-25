//! Python-facing tensor helpers and constructors.
//!
//! This module currently keeps Python tensor compatibility (`_data`, `shape`) and is designed
//! so we can swap internal storage to Burn-backed tensors incrementally.

use crate::dtypes::DType;
use once_cell::sync::Lazy;
use rustpython_vm::{
    builtins::{PyByteArray, PyBytes, PyDict, PyList, PyTuple},
    PyObjectRef, PyResult, VirtualMachine,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(1);
static TENSOR_REGISTRY: Lazy<Mutex<HashMap<u64, Arc<Mutex<Vec<f32>>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// CPU-side tensor storage for the Python API (`shape`, `dtype`, `device` in the emitted dict).
/// GPU-backed paths use the same [`Tensor`] name with `device` set accordingly elsewhere.
#[derive(Clone)]
pub struct Tensor {
    pub id: u64,
    pub data: Arc<Mutex<Vec<f32>>>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let id = NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed);
        let data = Arc::new(Mutex::new(data));
        if let Ok(mut reg) = TENSOR_REGISTRY.lock() {
            reg.insert(id, data.clone());
        }
        Self { id, data, shape }
    }

    pub fn to_py_dict(&self, vm: &VirtualMachine, dtype: DType) -> PyResult<PyObjectRef> {
        self.to_py_dict_on(vm, dtype, "cpu")
    }

    pub fn to_py_dict_on(
        &self,
        vm: &VirtualMachine,
        dtype: DType,
        device: &str,
    ) -> PyResult<PyObjectRef> {
        let data_guard = self.data.lock().unwrap();
        let shape = &self.shape;

        let dict = vm.ctx.new_dict();

        dict.set_item(
            "shape",
            vm.ctx
                .new_tuple(shape.iter().map(|&s| vm.ctx.new_int(s).into()).collect())
                .into(),
            vm,
        )?;

        dict.set_item("dtype", vm.ctx.new_str(dtype.name()).into(), vm)?;
        dict.set_item("device", vm.ctx.new_str(device).into(), vm)?;

        let py_data: Vec<PyObjectRef> = data_guard
            .iter()
            .map(|&f| {
                let casted = dtype.cast_from_f32(f);
                if dtype.is_float() {
                    vm.ctx.new_float(casted as f64).into()
                } else {
                    vm.ctx.new_int(casted as i64).into()
                }
            })
            .collect();
        dict.set_item("_data", vm.ctx.new_list(py_data).into(), vm)?;
        dict.set_item("_rust_tensor", vm.ctx.new_int(self.id as i64).into(), vm)?;

        Ok(dict.into())
    }

    /// Registry-backed dict without materializing millions of Python scalars (GPU simulation buffers).
    pub fn to_py_dict_registry_only(
        &self,
        vm: &VirtualMachine,
        dtype: DType,
        device: &str,
    ) -> PyResult<PyObjectRef> {
        let dict = vm.ctx.new_dict();
        dict.set_item(
            "shape",
            vm.ctx
                .new_tuple(
                    self.shape
                        .iter()
                        .map(|&s| vm.ctx.new_int(s).into())
                        .collect(),
                )
                .into(),
            vm,
        )?;
        dict.set_item("dtype", vm.ctx.new_str(dtype.name()).into(), vm)?;
        dict.set_item("device", vm.ctx.new_str(device).into(), vm)?;
        dict.set_item("_rust_tensor", vm.ctx.new_int(self.id as i64).into(), vm)?;
        dict.set_item("_xos_registry_only", vm.ctx.new_bool(true).into(), vm)?;
        Ok(dict.into())
    }
}

/// True when ``device`` should use registry-only tensors (no per-element Python list).
pub fn use_registry_only_tensor(device: &str, element_count: usize) -> bool {
    let d = device.trim().to_lowercase();
    (d == "gpu" || d == "cuda" || d == "metal" || d == "mps" || d == "wgpu") && element_count > 4096
}

/// Pack registry ``f32`` storage as a little-endian ``bytearray`` for fast element access.
pub fn tensor_registry_as_bytearray(id: u64, dtype: DType, vm: &VirtualMachine) -> Option<PyObjectRef> {
    let flat = try_get_tensor_data_by_id(id)?;
    let bytes: Vec<u8> = if dtype.is_float() {
        flat.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    } else {
        flat.iter()
            .map(|&v| v.clamp(0.0, 255.0) as u8)
            .collect()
    };
    Some(PyByteArray::new_ref(bytes, &vm.ctx).into())
}

fn try_get_tensor_data_by_id(id: u64) -> Option<Vec<f32>> {
    let reg = TENSOR_REGISTRY.lock().ok()?;
    let data = reg.get(&id)?.clone();
    drop(reg);
    let guard = data.lock().ok()?;
    Some(guard.clone())
}

fn try_get_active_frame_buffer_copy() -> Option<Vec<u8>> {
    if let Some(bytes) = crate::engine::py_engine_tls::with_tick_engine_state_mut(|state| {
        #[cfg(not(target_arch = "wasm32"))]
        state.frame.publish_gpu_to_staging();
        let buf = state.frame.staging_slice_mut_for_tick();
        Some(buf.to_vec())
    })
    .flatten()
    {
        return Some(bytes);
    }

    let buffer_guard = crate::rasterizer::CURRENT_FRAME_BUFFER.lock().ok()?;
    let width = *crate::rasterizer::CURRENT_FRAME_WIDTH.lock().ok()?;
    let height = *crate::rasterizer::CURRENT_FRAME_HEIGHT.lock().ok()?;
    let buffer_ptr = buffer_guard.as_ref()?;
    let len = width.saturating_mul(height).saturating_mul(4);
    let buffer = unsafe { std::slice::from_raw_parts(buffer_ptr.as_ptr(), len) };
    Some(buffer.to_vec())
}

/// Update registry storage for a tensor id (used by ``uniform_fill`` on simulation buffers).
pub fn write_tensor_data_by_id(id: u64, flat: &[f32]) -> bool {
    let reg = match TENSOR_REGISTRY.lock() {
        Ok(r) => r,
        Err(_) => return false,
    };
    let Some(data) = reg.get(&id) else {
        return false;
    };
    let data = data.clone();
    drop(reg);
    if let Ok(mut guard) = data.lock() {
        *guard = flat.to_vec();
        return true;
    }
    false
}

/// Extract f64 from Python int or float.
pub fn py_number_to_f64(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<f64> {
    if let Ok(f) = obj.clone().try_into_value::<f64>(vm) {
        return Ok(f);
    }
    if let Ok(i) = obj.clone().try_into_value::<i64>(vm) {
        return Ok(i as f64);
    }
    Err(vm.new_type_error("Expected a number (int or float)".to_string()))
}

/// Copy flat tensor storage (`_data` list, ``bytes``, or ``bytearray``) into a byte vector.
pub fn tensor_flat_bytes(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<u8>> {
    let mut cur = obj.clone();
    for _ in 0..12 {
        if let Some(bytes) = cur.downcast_ref::<PyBytes>() {
            return Ok(bytes.as_bytes().to_vec());
        }
        if let Some(ba) = cur.downcast_ref::<PyByteArray>() {
            return Ok(ba.borrow_buf().to_vec());
        }
        if let Some(dict) = cur.downcast_ref::<PyDict>() {
            let frame_backed = dict.get_item("_xos_frame_backing", vm).is_ok()
                || dict.get_item("_xos_viewport_id", vm).is_ok();
            if frame_backed {
                if let Some(bytes) = try_get_active_frame_buffer_copy() {
                    return Ok(bytes);
                }
            }
            if let Ok(item) = dict.get_item("_data", vm) {
                cur = item;
                continue;
            }
            if let Ok(item) = dict.get_item("data", vm) {
                cur = item;
                continue;
            }
            if let Ok(item) = dict.get_item("tensor", vm) {
                cur = item;
                continue;
            }
            if let Ok(vid_obj) = dict.get_item("_xos_viewport_id", vm) {
                if let Ok(vid) = vid_obj.try_into_value::<i64>(vm) {
                    if let Some(bytes) = crate::xos_module::standalone_frame_buffer_copy(
                        vid.max(0) as u64,
                    ) {
                        return Ok(bytes);
                    }
                }
            }
        }
        if let Ok(Some(attr)) = vm.get_attribute_opt(cur.clone(), "_data") {
            cur = attr;
            continue;
        }
        break;
    }
    let floats = tensor_flat_data_list(obj, vm)?;
    Ok(floats
        .iter()
        .map(|&v| v.clamp(0.0, 255.0) as u8)
        .collect())
}

/// Resolve raw tensor dict, Python `Tensor` wrapper, or nested `_data` to the flat `PyList` of values.
pub fn tensor_flat_data_list(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<f32>> {
    let mut cur = obj.clone();
    for _ in 0..8 {
        if let Some(bytes) = cur.downcast_ref::<PyBytes>() {
            return Ok(bytes.as_bytes().iter().map(|&b| b as f32).collect());
        }
        if let Some(ba) = cur.downcast_ref::<PyByteArray>() {
            return Ok(ba
                .borrow_buf()
                .iter()
                .map(|&b| b as f32)
                .collect());
        }
        if let Some(list) = cur.downcast_ref::<PyList>() {
            let vec = list.borrow_vec();
            // Nested `[[f32, ...]]` (shape (1, N)): flatten to match Whisper / burn `&[f32]`.
            if vec.len() == 1 {
                if let Some(inner) = vec[0].downcast_ref::<PyList>() {
                    return inner
                        .borrow_vec()
                        .iter()
                        .map(|x| py_number_to_f64(x, vm).map(|v| v as f32))
                        .collect::<Result<Vec<f32>, _>>();
                }
            }
            return vec
                .iter()
                .map(|x| py_number_to_f64(x, vm).map(|v| v as f32))
                .collect::<Result<Vec<f32>, _>>();
        }
        if let Some(dict) = cur.downcast_ref::<PyDict>() {
            let frame_backed = dict.get_item("_xos_frame_backing", vm).is_ok()
                || dict.get_item("_xos_viewport_id", vm).is_ok();
            if let Ok(id_obj) = dict.get_item("_rust_tensor", vm) {
                if let Ok(id) = id_obj.try_into_value::<i64>(vm) {
                    if let Some(v) = try_get_tensor_data_by_id(id.max(0) as u64) {
                        return Ok(v);
                    }
                }
            }
            if let Ok(item) = dict.get_item("_data", vm) {
                cur = item;
                continue;
            }
            if frame_backed {
                if let Some(bytes) = try_get_active_frame_buffer_copy() {
                    return Ok(bytes.into_iter().map(|b| b as f32).collect());
                }
            }
            // Frame-backed tensors intentionally avoid generic `data` list paths:
            // those lists can be stale metadata and force expensive full-frame copies.
            if !frame_backed {
                if let Ok(item) = dict.get_item("data", vm) {
                    cur = item;
                    continue;
                }
            }
            if let Ok(item) = dict.get_item("tensor", vm) {
                cur = item;
                continue;
            }
            if let Ok(vid_obj) = dict.get_item("_xos_viewport_id", vm) {
                if let Ok(vid) = vid_obj.try_into_value::<i64>(vm) {
                    if let Some(bytes) = crate::xos_module::standalone_frame_buffer_copy(
                        vid.max(0) as u64,
                    ) {
                        let out = bytes.into_iter().map(|b| b as f32).collect::<Vec<f32>>();
                        return Ok(out);
                    }
                }
            }
        }
        if let Ok(Some(attr)) = vm.get_attribute_opt(cur.clone(), "_data") {
            cur = attr;
            continue;
        }
        break;
    }
    Err(vm.new_type_error("tensor missing _data list".to_string()))
}

pub fn tensor_shape_tuple(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<usize>> {
    let mut cur = obj.clone();
    for _ in 0..12 {
        if let Some(dict) = cur.downcast_ref::<PyDict>() {
            if let Ok(shape_obj) = dict.get_item("shape", vm) {
                if let Some(tup) = shape_obj.downcast_ref::<PyTuple>() {
                    return tup
                        .as_slice()
                        .iter()
                        .map(|s| s.clone().try_into_value::<i32>(vm).map(|i| i as usize))
                        .collect::<Result<Vec<_>, _>>();
                }
                if let Some(lst) = shape_obj.downcast_ref::<PyList>() {
                    return lst
                        .borrow_vec()
                        .iter()
                        .map(|s| s.clone().try_into_value::<i32>(vm).map(|i| i as usize))
                        .collect::<Result<Vec<_>, _>>();
                }
            }
            if let Ok(item) = dict.get_item("tensor", vm) {
                cur = item;
                continue;
            }
        }
        if let Ok(Some(attr)) = vm.get_attribute_opt(cur.clone(), "_data") {
            cur = attr;
            continue;
        }
        if let Ok(Some(attr)) = vm.get_attribute_opt(cur.clone(), "tensor") {
            cur = attr;
            continue;
        }
        if let Ok(Some(attr)) = vm.get_attribute_opt(cur.clone(), "shape") {
            cur = attr;
            if let Some(tup) = cur.downcast_ref::<PyTuple>() {
                return tup
                    .as_slice()
                    .iter()
                    .map(|s| s.clone().try_into_value::<i32>(vm).map(|i| i as usize))
                    .collect::<Result<Vec<_>, _>>();
            }
            if let Some(lst) = cur.downcast_ref::<PyList>() {
                return lst
                    .borrow_vec()
                    .iter()
                    .map(|s| s.clone().try_into_value::<i32>(vm).map(|i| i as usize))
                    .collect::<Result<Vec<_>, _>>();
            }
        }
    }
    Err(vm.new_type_error("tensor missing shape".to_string()))
}

/// Create tensor from flat data and shape.
pub fn create_tensor_from_data(flat_data: Vec<f32>, shape: Vec<usize>, _dtype: DType) -> Tensor {
    Tensor::new(flat_data, shape)
}

/// Wrap a registry [`Tensor`] for Python (optionally without a materialized ``_data`` list).
pub fn wrap_registry_tensor(
    vm: &VirtualMachine,
    tensor: &Tensor,
    dtype: DType,
    device: &str,
    registry_only: bool,
) -> PyResult<PyObjectRef> {
    if registry_only {
        tensor.to_py_dict_registry_only(vm, dtype, device)
    } else {
        tensor.to_py_dict_on(vm, dtype, device)
    }
}
