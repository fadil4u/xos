//! Materialize / flush opaque frame tensors (no `_data` list) for numpy-style Python ops.

use rustpython_vm::builtins::{PyByteArray, PyBytes, PyDict, PyDictRef, PyList};
use rustpython_vm::{function::FuncArgs, PyObjectRef, PyResult, VirtualMachine};

use crate::tensor_buf::tensor_shape_tuple;

/// Map Python `and` / `or` between tensor expressions to element-wise `&` / `|`.
pub fn preprocess_tensor_logical_keywords(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let bytes = source.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'"' || b == b'\'' {
            let quote = b;
            out.push(b as char);
            i += 1;
            while i < bytes.len() {
                let c = bytes[i];
                out.push(c as char);
                if c == b'\\' && i + 1 < bytes.len() {
                    i += 1;
                    out.push(bytes[i] as char);
                } else if c == quote {
                    break;
                }
                i += 1;
            }
            i += 1;
            continue;
        }
        if b == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                out.push(bytes[i] as char);
                i += 1;
            }
            continue;
        }
        if i + 4 <= bytes.len() && &bytes[i..i + 4] == b" and" {
            let next = bytes.get(i + 4).copied();
            if next
                .map(|c| !c.is_ascii_alphanumeric() && c != b'_')
                .unwrap_or(true)
            {
                out.push_str(" &");
                i += 4;
                continue;
            }
        }
        if i + 3 <= bytes.len() && &bytes[i..i + 3] == b" or" {
            let next = bytes.get(i + 3).copied();
            if next
                .map(|c| !c.is_ascii_alphanumeric() && c != b'_')
                .unwrap_or(true)
            {
                out.push_str(" |");
                i += 3;
                continue;
            }
        }
        out.push(b as char);
        i += 1;
    }
    out
}

fn resolve_tensor_dict(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<PyDictRef> {
    let mut cur = obj.clone();
    for _ in 0..12 {
        if let Ok(dict) = cur.clone().downcast::<PyDict>() {
            if dict.get_item("_data", vm).is_ok()
                || dict.get_item("shape", vm).is_ok()
                || dict.get_item("_xos_viewport_id", vm).is_ok()
                || dict.get_item("_xos_frame_backing", vm).is_ok()
            {
                return Ok(dict);
            }
        }
        if let Ok(Some(attr)) = vm.get_attribute_opt(cur.clone(), "_data") {
            cur = attr;
            continue;
        }
        break;
    }
    Err(vm.new_type_error("expected a frame tensor".to_string()))
}

fn read_rgba_bytes_for_tensor(dict: &PyDictRef, vm: &VirtualMachine) -> PyResult<Vec<u8>> {
    if let Ok(vid_obj) = dict.get_item("_xos_viewport_id", vm) {
        if let Ok(vid) = vid_obj.try_into_value::<i64>(vm) {
            if let Some(bytes) =
                crate::xos_module::standalone_frame_buffer_copy(vid.max(0) as u64)
            {
                return Ok(bytes);
            }
        }
    }

    if let Some(bytes) = crate::engine::py_engine_tls::with_tick_engine_state_mut(|state| {
        #[cfg(not(target_arch = "wasm32"))]
        state.frame.publish_gpu_to_staging();
        let buf = state.frame.staging_slice_mut_for_tick();
        Some(buf.to_vec())
    })
    .flatten()
    {
        return Ok(bytes);
    }

    let buffer_guard = crate::rasterizer::CURRENT_FRAME_BUFFER
        .lock()
        .map_err(|_| vm.new_runtime_error("frame buffer lock poisoned".to_string()))?;
    let width = *crate::rasterizer::CURRENT_FRAME_WIDTH
        .lock()
        .map_err(|_| vm.new_runtime_error("frame buffer lock poisoned".to_string()))?;
    let height = *crate::rasterizer::CURRENT_FRAME_HEIGHT
        .lock()
        .map_err(|_| vm.new_runtime_error("frame buffer lock poisoned".to_string()))?;
    if let Some(buffer_ptr) = buffer_guard.as_ref() {
        let len = width.saturating_mul(height).saturating_mul(4);
        let buffer = unsafe { std::slice::from_raw_parts(buffer_ptr.as_ptr(), len) };
        return Ok(buffer.to_vec());
    }

    Err(vm.new_runtime_error(
        "No frame buffer to materialize tensor from".to_string(),
    ))
}

fn write_bytes_for_tensor(dict: &PyDictRef, bytes: &[u8], vm: &VirtualMachine) -> PyResult<()> {
    if let Ok(vid_obj) = dict.get_item("_xos_viewport_id", vm) {
        if let Ok(vid) = vid_obj.try_into_value::<i64>(vm) {
            if crate::xos_module::write_standalone_frame_buffer(vid.max(0) as u64, bytes) {
                return Ok(());
            }
        }
    }

    if crate::engine::py_engine_tls::with_tick_engine_state_mut(|state| {
        let buf = state.frame.staging_slice_mut_for_tick();
        let n = buf.len().min(bytes.len());
        buf[..n].copy_from_slice(&bytes[..n]);
        state.frame.mark_cpu_staging_dirty();
        true
    })
    .unwrap_or(false)
    {
        return Ok(());
    }

    let buffer_guard = crate::rasterizer::CURRENT_FRAME_BUFFER
        .lock()
        .map_err(|_| vm.new_runtime_error("frame buffer lock poisoned".to_string()))?;
    let width = *crate::rasterizer::CURRENT_FRAME_WIDTH
        .lock()
        .map_err(|_| vm.new_runtime_error("frame buffer lock poisoned".to_string()))?;
    let height = *crate::rasterizer::CURRENT_FRAME_HEIGHT
        .lock()
        .map_err(|_| vm.new_runtime_error("frame buffer lock poisoned".to_string()))?;
    if let Some(buffer_ptr) = buffer_guard.as_ref() {
        let len = width.saturating_mul(height).saturating_mul(4);
        let buffer = unsafe { std::slice::from_raw_parts_mut(buffer_ptr.as_ptr(), len) };
        let n = buffer.len().min(bytes.len());
        buffer[..n].copy_from_slice(&bytes[..n]);
        *crate::rasterizer::FRAME_CPU_WRITTEN
            .lock()
            .map_err(|_| vm.new_runtime_error("frame buffer lock poisoned".to_string()))? =
            true;
        return Ok(());
    }

    Err(vm.new_runtime_error(
        "No frame buffer to flush tensor into".to_string(),
    ))
}

/// Populate ``tensor._data`` from the live RGBA framebuffer (standalone or engine tick).
pub fn materialize_frame_tensor(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let tensor = args.args.first().ok_or_else(|| {
        vm.new_type_error("materialize_frame_tensor() expects a tensor".to_string())
    })?;
    let dict = resolve_tensor_dict(tensor, vm)?;
    if let Ok(existing) = dict.get_item("_data", vm) {
        if existing.downcast_ref::<PyList>().is_some()
            || existing.downcast_ref::<PyBytes>().is_some()
            || existing.downcast_ref::<PyByteArray>().is_some()
        {
            return Ok(vm.ctx.none());
        }
    }
    let bytes = read_rgba_bytes_for_tensor(&dict, vm)?;
    dict.set_item(
        "_data",
        PyByteArray::new_ref(bytes, &vm.ctx).into(),
        vm,
    )?;
    dict.set_item(
        "_xos_frame_materialized",
        vm.ctx.new_bool(true).into(),
        vm,
    )?;
    Ok(vm.ctx.none())
}

/// Write ``tensor._data`` back into the RGBA framebuffer.
pub fn flush_frame_tensor(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let tensor = args
        .args
        .first()
        .ok_or_else(|| vm.new_type_error("flush_frame_tensor() expects a tensor".to_string()))?;
    let dict = resolve_tensor_dict(tensor, vm)?;
    let bytes = crate::tensor_buf::tensor_flat_bytes(tensor, vm)?;
    let shape = tensor_shape_tuple(tensor, vm).unwrap_or_default();
    let expected = shape.iter().product::<usize>();
    if expected > 0 && bytes.len() != expected {
        return Err(vm.new_value_error(format!(
            "flush: tensor length {} does not match shape product {}",
            bytes.len(),
            expected
        )));
    }
    write_bytes_for_tensor(&dict, &bytes, vm)?;
    Ok(vm.ctx.none())
}
