use xos_core::compute_device::ComputeDevice;
use rustpython_vm::builtins::{PyByteArray, PyDict};
use rustpython_vm::{function::FuncArgs, PyObjectRef, PyResult, VirtualMachine};

use crate::device_policy;
use crate::dtypes::DType;
use crate::tensor_buf::{
    create_tensor_from_data, py_number_to_f64, tensor_flat_bytes, tensor_flat_data_list,
    tensor_shape_tuple,
    wrap_registry_tensor,
};

fn direct_fill_sentinel(vm: &VirtualMachine) -> PyResult {
    let sentinel = vm.ctx.new_dict();
    sentinel.set_item("_direct_fill", vm.ctx.new_bool(true).into(), vm)?;
    Ok(sentinel.into())
}

/// HWC `[ky, kx, in_c]` (K×K×3) → NCHW `[out_c, in_c, kh, kw]`.
fn kernel_hwc_to_nchw(kernel: &[f32], kernel_size: usize) -> Vec<f32> {
    let mut kernel_nchw = vec![0.0f32; 3 * 3 * kernel_size * kernel_size];
    for out_c in 0..3 {
        for in_c in 0..3 {
            for ky in 0..kernel_size {
                for kx in 0..kernel_size {
                    let src_idx = (ky * kernel_size + kx) * 3 + in_c;
                    let dst_idx = ((out_c * 3 + in_c) * kernel_size + ky) * kernel_size + kx;
                    kernel_nchw[dst_idx] = kernel[src_idx];
                }
            }
        }
    }
    kernel_nchw
}

/// RGB same conv on [`FrameState`]'s GPU tensor (Burn / Metal, zero-copy).
fn try_convolve_on_frame_gpu(
    kernel_nchw: &[f32],
    kernel_size: usize,
    stride: [usize; 2],
) -> bool {
    crate::engine::py_engine_tls::with_tick_engine_state_mut(|state| {
        xos_core::burn_raster::convolve_rgb_same(
            &mut state.frame,
            kernel_nchw.to_vec(),
            kernel_size,
            kernel_size,
            stride,
        )
        .is_ok()
    })
    .unwrap_or(false)
}

fn try_convolve_on_frame_gpu_out(
    kernel_nchw: Vec<f32>,
    kernel_size: usize,
    stride: [usize; 2],
) -> bool {
    crate::engine::py_engine_tls::with_tick_engine_state_mut(|state| {
        match xos_core::burn_raster::convolve_rgb_same_out(
            &mut state.frame,
            kernel_nchw,
            kernel_size,
            kernel_size,
            stride,
        ) {
            Ok(t) => {
                crate::engine::py_engine_tls::set_conv_gpu_output(t);
                true
            }
            Err(_) => false,
        }
    })
    .unwrap_or(false)
}

fn try_convolve_depthwise_on_frame_gpu(
    kernel: Vec<f32>,
    kernel_size: usize,
    stride: [usize; 2],
) -> bool {
    crate::engine::py_engine_tls::with_tick_engine_state_mut(|state| {
        xos_core::burn_raster::convolve_depthwise_rgb_same(
            &mut state.frame,
            kernel,
            kernel_size,
            kernel_size,
            stride,
        )
        .is_ok()
    })
    .unwrap_or(false)
}

fn try_convolve_depthwise_on_frame_gpu_out(
    kernel: Vec<f32>,
    kernel_size: usize,
    stride: [usize; 2],
) -> bool {
    crate::engine::py_engine_tls::with_tick_engine_state_mut(|state| {
        match xos_core::burn_raster::convolve_depthwise_rgb_same_out(
            &mut state.frame,
            kernel,
            kernel_size,
            kernel_size,
            stride,
        ) {
            Ok(t) => {
                crate::engine::py_engine_tls::set_conv_gpu_output(t);
                true
            }
            Err(_) => false,
        }
    })
    .unwrap_or(false)
}

/// Read back tick-local conv output into a registry tensor (no per-element Python list).
fn wrap_conv_registry_tensor(vm: &VirtualMachine, device: ComputeDevice) -> PyResult {
    let (shape, flat) = crate::engine::py_engine_tls::materialize_conv_gpu_output_hwc_f32()
        .ok_or_else(|| {
            vm.new_runtime_error("no GPU conv output to materialize (internal error)".to_string())
        })?;
    let tensor = create_tensor_from_data(flat, shape, DType::Float32);
    let registry_only = true;
    let dict = wrap_registry_tensor(vm, &tensor, DType::Float32, device.as_str(), registry_only)?;
    if let Ok(wrapper_class) = vm.builtins.get_attr("Tensor", vm) {
        if let Ok(wrapped) = wrapper_class.call((dict.clone(),), vm) {
            return Ok(wrapped);
        }
    }
    Ok(dict)
}

/// Opaque GPU conv result: no `_data` until slice/read materializes (one readback).
fn wrap_gpu_conv_output_tensor(vm: &VirtualMachine, device: ComputeDevice) -> PyResult {
    let (height, width, channels) =
        crate::engine::py_engine_tls::conv_gpu_output_shape().ok_or_else(|| {
            vm.new_runtime_error("no GPU conv output shape (internal error)".to_string())
        })?;
    let tensor_dict = vm.ctx.new_dict();
    tensor_dict.set_item(
        "shape",
        vm.ctx
            .new_tuple(vec![
                vm.ctx.new_int(height).into(),
                vm.ctx.new_int(width).into(),
                vm.ctx.new_int(channels).into(),
            ])
            .into(),
        vm,
    )?;
    tensor_dict.set_item("dtype", vm.ctx.new_str("float32").into(), vm)?;
    tensor_dict.set_item(
        "device",
        vm.ctx.new_str(device.as_str()).into(),
        vm,
    )?;
    tensor_dict.set_item("_xos_gpu_conv_output", vm.ctx.new_bool(true).into(), vm)?;

    if let Ok(wrapper_class) = vm.builtins.get_attr("Tensor", vm) {
        if let Ok(wrapped) = wrapper_class.call((tensor_dict.clone(),), vm) {
            return Ok(wrapped);
        }
    }
    Ok(tensor_dict.into())
}

/// Populate ``tensor._data`` from the tick-local GPU conv buffer (single readback).
pub fn materialize_conv_output(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    use rustpython_vm::builtins::{PyByteArray, PyDictRef};

    let tensor = args.args.first().ok_or_else(|| {
        vm.new_type_error("materialize_conv_output() expects a tensor".to_string())
    })?;
    let mut cur = tensor.clone();
    let dict: PyDictRef = loop {
        if let Ok(d) = cur.clone().downcast::<rustpython_vm::builtins::PyDict>() {
            if d.get_item("_xos_gpu_conv_output", vm).is_ok() {
                break d;
            }
        }
        cur = vm
            .get_attribute_opt(cur, "_data")?
            .ok_or_else(|| vm.new_type_error("expected conv output tensor".to_string()))?;
    };

    if dict.get_item("_data", vm).is_ok() {
        return Ok(vm.ctx.none());
    }

    let bytes = crate::engine::py_engine_tls::materialize_conv_gpu_output_rgba_u8()
        .ok_or_else(|| vm.new_runtime_error("no GPU conv output to materialize".to_string()))?;
    dict.set_item(
        "_data",
        PyByteArray::new_ref(bytes, &vm.ctx).into(),
        vm,
    )?;
    dict.set_item(
        "_xos_conv_materialized",
        vm.ctx.new_bool(true).into(),
        vm,
    )?;
    Ok(vm.ctx.none())
}

fn get_array_data_list(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Option<PyObjectRef>> {
    if let Ok(data_attr) = obj.get_attr("_data", vm) {
        if let Ok(inner_dict) = data_attr
            .clone()
            .downcast::<rustpython_vm::builtins::PyDict>()
        {
            if let Ok(list_obj) = inner_dict.get_item("_data", vm) {
                if list_obj
                    .downcast_ref::<rustpython_vm::builtins::PyList>()
                    .is_some()
                {
                    return Ok(Some(list_obj));
                }
            }
        }
        if data_attr
            .downcast_ref::<rustpython_vm::builtins::PyList>()
            .is_some()
        {
            return Ok(Some(data_attr));
        }
    }
    if let Ok(dict) = obj.clone().downcast::<rustpython_vm::builtins::PyDict>() {
        if let Ok(list_obj) = dict.get_item("_data", vm) {
            if list_obj
                .downcast_ref::<rustpython_vm::builtins::PyList>()
                .is_some()
            {
                return Ok(Some(list_obj));
            }
        }
    }
    if obj
        .downcast_ref::<rustpython_vm::builtins::PyList>()
        .is_some()
    {
        return Ok(Some(obj.clone()));
    }
    Ok(None)
}

/// Flatten a square K×K kernel list (handles nested ``[[...], ...]``).
fn flatten_square_kernel_list(kernel_arg: &PyObjectRef, vm: &VirtualMachine) -> PyResult<usize> {
    let kernel_list = get_array_data_list(kernel_arg, vm)?
        .and_then(|o| o.downcast::<rustpython_vm::builtins::PyList>().ok())
        .ok_or_else(|| vm.new_type_error("kernel must be a list or array".to_string()))?;
    let kernel_vec = kernel_list.borrow_vec();
    let mut flat_len = 0usize;
    for val in kernel_vec.iter() {
        if val.downcast_ref::<rustpython_vm::builtins::PyList>().is_some() {
            let row = val
                .downcast_ref::<rustpython_vm::builtins::PyList>()
                .unwrap()
                .borrow_vec();
            flat_len += row.len();
        } else {
            flat_len += 1;
        }
    }
    let k = (flat_len as f32).sqrt() as usize;
    if k * k != flat_len {
        return Err(vm.new_value_error(format!(
            "kernel must be square (KxK), got {} elements",
            flat_len
        )));
    }
    Ok(k)
}

fn depthwise_hwc_shape(
    vm: &VirtualMachine,
    shape: &[usize],
) -> PyResult<(usize, usize, usize)> {
    match shape {
        [h, w] => Ok((*h, *w, 1)),
        [h, w, c] => Ok((*h, *w, *c)),
        _ => Err(vm.new_value_error(
            "convolve_depthwise expects a 2D (H, W) or 3D (H, W, C) tensor".to_string(),
        )),
    }
}

/// Depthwise on a plain ``xos.Tensor`` (registry / list storage): upload once, Burn on tick GPU, TLS output.
fn try_depthwise_conv_tensor_gpu_out(
    h: usize,
    w: usize,
    channels: usize,
    input_hwc: &[f32],
    kernel: Vec<f32>,
    kernel_h: usize,
    kernel_w: usize,
) -> bool {
    crate::engine::py_engine_tls::with_tick_engine_state_mut(|state| {
        let device = state.frame.device();
        match xos_core::burn_raster::depthwise_conv_same_hwc(
            device,
            h,
            w,
            channels,
            input_hwc,
            &kernel,
            kernel_h,
            kernel_w,
        ) {
            Ok(t) => {
                crate::engine::py_engine_tls::set_conv_gpu_output(t);
                true
            }
            Err(_) => false,
        }
    })
    .unwrap_or(false)
}

fn parse_square_kernel(
    kernel_arg: &PyObjectRef,
    vm: &VirtualMachine,
    normalize_l1: bool,
) -> PyResult<(Vec<f32>, usize)> {
    let kernel_list = get_array_data_list(kernel_arg, vm)?
        .and_then(|o| o.downcast::<rustpython_vm::builtins::PyList>().ok())
        .ok_or_else(|| vm.new_type_error("kernel must be a list or array".to_string()))?;

    let kernel_vec = kernel_list.borrow_vec();
    let kernel_len = kernel_vec.len();
    let kernel_size = (kernel_len as f32).sqrt() as usize;
    if kernel_size * kernel_size != kernel_len {
        return Err(vm.new_value_error(format!(
            "kernel must be square (KxK), got {} elements",
            kernel_len
        )));
    }

    let mut kernel: Vec<f32> = Vec::with_capacity(kernel_len);
    for val in kernel_vec.iter() {
        kernel.push(py_number_to_f64(val, vm)? as f32);
    }
    drop(kernel_vec);

    if normalize_l1 {
        let norm: f32 = kernel.iter().map(|&x| x.abs()).sum::<f32>().max(1e-6);
        kernel.iter_mut().for_each(|x| *x /= norm);
    }

    Ok((kernel, kernel_size))
}

fn parse_rgb_kernel(
    kernel_arg: &PyObjectRef,
    vm: &VirtualMachine,
) -> PyResult<(Vec<f32>, usize)> {
    let kernel_list = get_array_data_list(kernel_arg, vm)?
        .and_then(|o| o.downcast::<rustpython_vm::builtins::PyList>().ok())
        .ok_or_else(|| vm.new_type_error("kernel must be a list or array".to_string()))?;

    let kernel_vec = kernel_list.borrow_vec();
    let kernel_len = kernel_vec.len();
    if kernel_len % 3 != 0 {
        return Err(vm.new_value_error(format!(
            "kernel length must be KxKx3 (RGB), got {}",
            kernel_len
        )));
    }
    let spatial_len = kernel_len / 3;
    let kernel_size = (spatial_len as f32).sqrt() as usize;
    if kernel_size * kernel_size * 3 != kernel_len {
        return Err(vm.new_value_error(format!(
            "kernel must be KxKx3 RGB (e.g. 3×3×3 → 27 elements), got {}",
            kernel_len
        )));
    }

    let mut kernel: Vec<f32> = Vec::with_capacity(kernel_len);
    for val in kernel_vec.iter() {
        kernel.push(py_number_to_f64(val, vm)? as f32);
    }
    drop(kernel_vec);

    let norm: f32 = kernel.iter().map(|&x| x.abs()).sum::<f32>().max(1e-6);
    kernel.iter_mut().for_each(|x| *x /= norm);

    Ok((kernel, kernel_size))
}

fn convolve_frame_rgb_same_cpu_inplace(
    image_arg: &PyObjectRef,
    vm: &VirtualMachine,
    kernel_hwc: &[f32],
    kernel_size: usize,
    stride: usize,
) -> PyResult<bool> {
    if stride != 1 {
        return Err(vm.new_value_error(
            "convolve() CPU path currently requires stride=1".to_string(),
        ));
    }

    let shape = tensor_shape_tuple(image_arg, vm)?;
    if shape.len() < 3 {
        return Err(vm.new_value_error(
            "convolve() CPU path expects frame.tensor shape (H, W, C)".to_string(),
        ));
    }
    let h = shape[0];
    let w = shape[1];
    let c = shape[2];
    if c < 3 {
        return Err(vm.new_value_error(
            "convolve() CPU path expects at least 3 channels".to_string(),
        ));
    }

    let expected = h.saturating_mul(w).saturating_mul(c);
    let src = tensor_flat_bytes(image_arg, vm)?;
    if src.len() < expected {
        return Err(vm.new_value_error(format!(
            "convolve(): tensor has {} elements, shape product is {}",
            src.len(),
            expected
        )));
    }
    let mut dst = src.clone();
    let pad = (kernel_size.saturating_sub(1)) / 2;

    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * c;
            for out_ch in 0..3 {
                let mut sum = 0.0f32;
                for ky in 0..kernel_size {
                    let sy = y as isize + ky as isize - pad as isize;
                    if sy < 0 || sy >= h as isize {
                        continue;
                    }
                    for kx in 0..kernel_size {
                        let sx = x as isize + kx as isize - pad as isize;
                        if sx < 0 || sx >= w as isize {
                            continue;
                        }
                        let src_base = (sy as usize * w + sx as usize) * c;
                        for in_ch in 0..3 {
                            let kval = kernel_hwc[(ky * kernel_size + kx) * 3 + in_ch];
                            sum += src[src_base + in_ch] as f32 * kval;
                        }
                    }
                }
                dst[base + out_ch] = sum.clamp(0.0, 255.0) as u8;
            }
        }
    }

    crate::xos_module::with_frame_write_buffer(vm, Some(image_arg), |buffer| {
        let n = buffer.len().min(dst.len());
        buffer[..n].copy_from_slice(&dst[..n]);
        crate::rasterizer::note_frame_cpu_write();
        Ok(())
    })?;
    // Keep frame tensor cache coherent for subsequent Python-side reads (e.g. `.sum()`).
    let dict = resolve_tensor_dict(image_arg, vm)?;
    dict.set_item(
        "_data",
        PyByteArray::new_ref(dst, &vm.ctx).into(),
        vm,
    )?;
    Ok(true)
}

fn convolve_tensor_rgb_same_cpu_out(
    image_arg: &PyObjectRef,
    vm: &VirtualMachine,
    kernel_hwc: &[f32],
    kernel_size: usize,
    stride: usize,
    out_device: &str,
) -> PyResult {
    if stride != 1 {
        return Err(vm.new_value_error(
            "convolve() tensor path currently requires stride=1".to_string(),
        ));
    }

    let shape = tensor_shape_tuple(image_arg, vm)?;
    if shape.len() < 3 {
        return Err(vm.new_value_error(
            "convolve() tensor path expects shape (H, W, C)".to_string(),
        ));
    }
    let h = shape[0];
    let w = shape[1];
    let c = shape[2];
    if c < 3 {
        return Err(vm.new_value_error(
            "convolve() tensor path expects at least 3 channels".to_string(),
        ));
    }

    let src = tensor_flat_data_list(image_arg, vm)?;
    let expected = h.saturating_mul(w).saturating_mul(c);
    if src.len() < expected {
        return Err(vm.new_value_error(format!(
            "convolve(): tensor has {} elements, shape product is {}",
            src.len(),
            expected
        )));
    }

    let mut out = vec![0.0f32; expected];
    let pad = (kernel_size.saturating_sub(1)) / 2;
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * c;
            for out_ch in 0..3 {
                let mut sum = 0.0f32;
                for ky in 0..kernel_size {
                    let sy = y as isize + ky as isize - pad as isize;
                    if sy < 0 || sy >= h as isize {
                        continue;
                    }
                    for kx in 0..kernel_size {
                        let sx = x as isize + kx as isize - pad as isize;
                        if sx < 0 || sx >= w as isize {
                            continue;
                        }
                        let src_base = (sy as usize * w + sx as usize) * c;
                        for in_ch in 0..3 {
                            let kval = kernel_hwc[(ky * kernel_size + kx) * 3 + in_ch];
                            sum += src[src_base + in_ch] * kval;
                        }
                    }
                }
                out[base + out_ch] = sum;
            }
            // Preserve alpha/extra channels to match frame behavior.
            for ch in 3..c {
                out[base + ch] = src[base + ch];
            }
        }
    }

    let tensor = create_tensor_from_data(out, vec![h, w, c], DType::Float32);
    let dict = tensor.to_py_dict_on(vm, DType::Float32, out_device)?;
    crate::tensors::wrap_tensor_dict(dict, vm)
}

fn resolve_tensor_dict(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<rustpython_vm::PyRef<PyDict>> {
    let mut cur = obj.clone();
    for _ in 0..12 {
        if let Ok(dict) = cur.clone().downcast::<PyDict>() {
            if dict.get_item("shape", vm).is_ok()
                || dict.get_item("_data", vm).is_ok()
                || dict.get_item("_rust_tensor", vm).is_ok()
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
    Err(vm.new_type_error("expected a Tensor".to_string()))
}

fn write_flat_to_tensor(
    tensor: &PyObjectRef,
    flat: &[f32],
    vm: &VirtualMachine,
) -> PyResult<()> {
    let dtype = crate::tensors::tensor_dtype_from_ref(tensor, vm)?;
    if let Some(id) = crate::xos_module::tensor_rust_id(tensor, vm) {
        crate::tensor_buf::write_tensor_data_by_id(id, flat);
    }
    let dict = resolve_tensor_dict(tensor, vm)?;
    if dtype == DType::UInt8 {
        let bytes: Vec<u8> = flat.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
        dict.set_item(
            "_data",
            PyByteArray::new_ref(bytes, &vm.ctx).into(),
            vm,
        )?;
    } else if dtype.is_float() {
        let py: Vec<PyObjectRef> = flat
            .iter()
            .map(|&v| vm.ctx.new_float(v as f64).into())
            .collect();
        dict.set_item("_data", vm.ctx.new_list(py).into(), vm)?;
    } else {
        let py: Vec<PyObjectRef> = flat
            .iter()
            .map(|&v| vm.ctx.new_int(v as i64).into())
            .collect();
        dict.set_item("_data", vm.ctx.new_list(py).into(), vm)?;
    }
    Ok(())
}

fn convolve_tensor_rgb_same_cpu_inplace(
    image_arg: &PyObjectRef,
    vm: &VirtualMachine,
    kernel_hwc: &[f32],
    kernel_size: usize,
    stride: usize,
) -> PyResult {
    if stride != 1 {
        return Err(vm.new_value_error(
            "convolve() tensor path currently requires stride=1".to_string(),
        ));
    }

    let shape = tensor_shape_tuple(image_arg, vm)?;
    if shape.len() < 3 {
        return Err(vm.new_value_error(
            "convolve() tensor path expects shape (H, W, C)".to_string(),
        ));
    }
    let h = shape[0];
    let w = shape[1];
    let c = shape[2];
    if c < 3 {
        return Err(vm.new_value_error(
            "convolve() tensor path expects at least 3 channels".to_string(),
        ));
    }

    let src = tensor_flat_data_list(image_arg, vm)?;
    let expected = h.saturating_mul(w).saturating_mul(c);
    if src.len() < expected {
        return Err(vm.new_value_error(format!(
            "convolve(): tensor has {} elements, shape product is {}",
            src.len(),
            expected
        )));
    }

    let mut out = src.clone();
    let pad = (kernel_size.saturating_sub(1)) / 2;
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * c;
            for out_ch in 0..3 {
                let mut sum = 0.0f32;
                for ky in 0..kernel_size {
                    let sy = y as isize + ky as isize - pad as isize;
                    if sy < 0 || sy >= h as isize {
                        continue;
                    }
                    for kx in 0..kernel_size {
                        let sx = x as isize + kx as isize - pad as isize;
                        if sx < 0 || sx >= w as isize {
                            continue;
                        }
                        let src_base = (sy as usize * w + sx as usize) * c;
                        for in_ch in 0..3 {
                            let kval = kernel_hwc[(ky * kernel_size + kx) * 3 + in_ch];
                            sum += src[src_base + in_ch] * kval;
                        }
                    }
                }
                out[base + out_ch] = sum;
            }
            for ch in 3..c {
                out[base + ch] = src[base + ch];
            }
        }
    }

    write_flat_to_tensor(image_arg, &out, vm)?;
    Ok(image_arg.clone())
}

/// xos.ops.convolve(frame.tensor, kernel, inplace=True)
///
/// Dispatches to Burn on the frame GPU tensor only (same path as the TV demo).
pub fn convolve(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let inplace = args
        .kwargs
        .get("inplace")
        .and_then(|v| v.clone().try_into_value::<bool>(vm).ok())
        .or_else(|| {
            args.kwargs
                .get("direct")
                .and_then(|v| v.clone().try_into_value::<bool>(vm).ok())
        })
        .unwrap_or(false);
    let stride = args
        .kwargs
        .get("stride")
        .and_then(|v| v.clone().try_into_value::<i32>(vm).ok())
        .unwrap_or(1)
        .max(1) as usize;

    if args.args.len() < 2 {
        return Err(vm.new_type_error(
            "convolve() requires at least 2 arguments (image, kernel)".to_string(),
        ));
    }

    let image_arg = &args.args[0];
    let kernel_arg = &args.args[1];

    if get_array_data_list(kernel_arg, vm)?
        .and_then(|o| o.downcast::<rustpython_vm::builtins::PyList>().ok())
        .is_some()
    {
        if flatten_square_kernel_list(kernel_arg, vm).is_ok() {
            return convolve_depthwise(args, vm);
        }
    }

    let frame_dev = device_policy::tensor_device_label(image_arg, vm)?;
    let kernel_dev = device_policy::tensor_device_label(kernel_arg, vm)?;
    device_policy::require_same_devices(
        vm,
        "convolve",
        &[
            ("image", frame_dev.clone()),
            ("kernel", kernel_dev.clone()),
        ],
    )?;
    let (kernel, kernel_size) = parse_rgb_kernel(kernel_arg, vm)?;
    if !device_policy::is_frame_backed_tensor(image_arg, vm) {
        if inplace {
            return convolve_tensor_rgb_same_cpu_inplace(image_arg, vm, &kernel, kernel_size, stride);
        }
        return convolve_tensor_rgb_same_cpu_out(
            image_arg,
            vm,
            &kernel,
            kernel_size,
            stride,
            &frame_dev,
        );
    }

    let engine_dev = device_policy::require_engine_device(vm, "convolve", &frame_dev)?;
    if engine_dev != ComputeDevice::Gpu {
        if inplace {
            if convolve_frame_rgb_same_cpu_inplace(image_arg, vm, &kernel, kernel_size, stride)? {
                return Ok(image_arg.clone());
            }
        } else {
            return convolve_tensor_rgb_same_cpu_out(
                image_arg,
                vm,
                &kernel,
                kernel_size,
                stride,
                &frame_dev,
            );
        }
        return Err(vm.new_runtime_error(
            "convolve() CPU frame path unavailable (internal error)".to_string(),
        ));
    }

    let kernel_nchw = kernel_hwc_to_nchw(&kernel, kernel_size);
    let stride_pair = [stride, stride];

    if inplace {
        if try_convolve_on_frame_gpu(&kernel_nchw, kernel_size, stride_pair) {
            return Ok(image_arg.clone());
        }
    } else if try_convolve_on_frame_gpu_out(kernel_nchw, kernel_size, stride_pair) {
        return wrap_gpu_conv_output_tensor(vm, engine_dev);
    }

    Err(vm.new_runtime_error(
        "convolve(): GPU frame path unavailable (call during Application.tick())".to_string(),
    ))
}

/// xos.ops.convolve_depthwise(frame.tensor, kernel)
///
/// Depthwise K×K on the frame GPU tensor (Burn). See ``example-scripts/ml/conv_depthwise.py``.
pub fn convolve_depthwise(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let args_vec = args.args;
    let inplace = args
        .kwargs
        .get("inplace")
        .and_then(|v| v.clone().try_into_value::<bool>(vm).ok())
        .or_else(|| {
            args.kwargs
                .get("direct")
                .and_then(|v| v.clone().try_into_value::<bool>(vm).ok())
        })
        .unwrap_or(false);
    let stride = args
        .kwargs
        .get("stride")
        .and_then(|v| v.clone().try_into_value::<i32>(vm).ok())
        .unwrap_or(1)
        .max(1) as usize;

    if args_vec.len() < 2 {
        return Err(vm.new_type_error(
            "convolve_depthwise() requires at least 2 arguments (image, kernel)".to_string(),
        ));
    }

    let image_arg = &args_vec[0];
    let kernel_arg = &args_vec[1];

    let image_dev = device_policy::tensor_device_label(image_arg, vm)?;
    let kernel_dev = device_policy::tensor_device_label(kernel_arg, vm)?;
    device_policy::require_same_devices(
        vm,
        "convolve_depthwise",
        &[
            ("image", image_dev.clone()),
            ("kernel", kernel_dev.clone()),
        ],
    )?;
    let engine_dev = device_policy::require_engine_device(vm, "convolve_depthwise", &image_dev)?;
    if engine_dev != ComputeDevice::Gpu {
        return Err(vm.new_runtime_error(
            "convolve_depthwise() requires a GPU engine (Burn path)".to_string(),
        ));
    }

    // Stencil weights are used as-is (neighbor counts for Game of Life, etc.).
    let (kernel, kernel_size) = parse_square_kernel(kernel_arg, vm, false)?;
    let stride_pair = [stride, stride];
    let is_frame = device_policy::is_frame_backed_tensor(image_arg, vm);

    if is_frame {
        if inplace {
            if try_convolve_depthwise_on_frame_gpu(kernel.clone(), kernel_size, stride_pair) {
                return direct_fill_sentinel(vm);
            }
        } else if try_convolve_depthwise_on_frame_gpu_out(kernel, kernel_size, stride_pair) {
            return wrap_gpu_conv_output_tensor(vm, engine_dev);
        }
        return Err(vm.new_runtime_error(
            "convolve_depthwise(): GPU frame path unavailable (call during Application.tick())"
                .to_string(),
        ));
    }

    if inplace {
        return Err(vm.new_value_error(
            "convolve_depthwise(inplace=True) is only supported for frame.tensor".to_string(),
        ));
    }

    let shape = tensor_shape_tuple(image_arg, vm)?;
    let (height, width, channels) = depthwise_hwc_shape(vm, &shape)?;
    let flat = tensor_flat_data_list(image_arg, vm)?;
    let expected = height * width * channels;
    if flat.len() != expected {
        return Err(vm.new_value_error(format!(
            "convolve_depthwise: tensor has {} elements, shape product is {}",
            flat.len(),
            expected
        )));
    }

    if try_depthwise_conv_tensor_gpu_out(
        height,
        width,
        channels,
        &flat,
        kernel,
        kernel_size,
        kernel_size,
    ) {
        return wrap_conv_registry_tensor(vm, engine_dev);
    }

    Err(vm.new_runtime_error(
        "convolve_depthwise(): GPU path unavailable (call during Application.tick())".to_string(),
    ))
}
