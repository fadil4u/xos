//! xos.tensor API functions exposed to Python.

use crate::dtypes::DType;
pub use crate::tensor_buf::{
    create_tensor_from_data, py_number_to_f64, tensor_flat_data_list, tensor_shape_tuple, Tensor,
};
use rustpython_vm::builtins::{PyByteArray, PyBytes, PyDict, PyList, PyModule};
use rustpython_vm::{function::FuncArgs, PyObjectRef, PyRef, PyResult, VirtualMachine};

/// One pass over uint8 RGBA / tensor bytes—min, max, arithmetic mean (as f64).
#[inline]
fn u8_slice_min_max_mean(b: &[u8]) -> Option<(f64, f64, f64)> {
    if b.is_empty() {
        return None;
    }
    let mut min_b = u8::MAX;
    let mut max_b = 0u8;
    let mut sum: u64 = 0;
    for &x in b {
        min_b = min_b.min(x);
        max_b = max_b.max(x);
        sum += x as u64;
    }
    let n = b.len() as f64;
    Some((min_b as f64, max_b as f64, sum as f64 / n))
}

fn pyobject_to_f64_flat(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<f64> {
    if let Ok(v) = obj.clone().try_into_value::<f64>(vm) {
        return Ok(v);
    }
    if let Ok(v) = obj.clone().try_into_value::<i64>(vm) {
        return Ok(v as f64);
    }
    if let Ok(v) = obj.clone().try_into_value::<bool>(vm) {
        return Ok(if v { 1.0 } else { 0.0 });
    }
    Err(vm.new_type_error(
        "Tensor reduction: flat storage has non-numeric element".to_string(),
    ))
}

fn pylist_min_max_mean(lst: &PyList, vm: &VirtualMachine) -> PyResult<Option<(f64, f64, f64)>> {
    let v = lst.borrow_vec();
    if v.is_empty() {
        return Ok(None);
    }
    let mut min_v = 0.0f64;
    let mut max_v = 0.0f64;
    let mut sum = 0.0f64;
    let mut first = true;
    for obj in v.iter() {
        let x = pyobject_to_f64_flat(obj, vm)?;
        if first {
            min_v = x;
            max_v = x;
            first = false;
        } else {
            min_v = min_v.min(x);
            max_v = max_v.max(x);
        }
        sum += x;
    }
    Ok(Some((min_v, max_v, sum / (v.len() as f64))))
}

fn tensor_min_max_mean_triplet(obj: PyObjectRef, vm: &VirtualMachine) -> PyResult<(f64, f64, f64)> {
    let inner = vm
        .get_attribute_opt(obj, "_data")?
        .ok_or_else(|| vm.new_type_error("Tensor reduction: missing ._data".into()))?;
    let td = inner
        .downcast_ref::<PyDict>()
        .ok_or_else(|| vm.new_type_error("Tensor reduction: ._data must be dict".into()))?;

    if !td.contains_key("_data", vm) {
        return Err(vm.new_value_error(
            "cannot reduce an empty Tensor (no flat _data buffer)".to_string(),
        ));
    }

    let storage = td.get_item("_data", vm)?;

    if let Ok(pref) = storage.clone().downcast::<PyBytes>() {
        let b = pref.as_bytes();
        return u8_slice_min_max_mean(b).ok_or_else(|| {
            vm.new_value_error(
                "zero-size array to reduction operation which has no identity".to_string(),
            )
        });
    }

    if let Some(ba) = storage.downcast_ref::<PyByteArray>() {
        let b = ba.borrow_buf();
        return u8_slice_min_max_mean(&b).ok_or_else(|| {
            vm.new_value_error(
                "zero-size array to reduction operation which has no identity".to_string(),
            )
        });
    }

    if let Some(lst) = storage.downcast_ref::<PyList>() {
        return pylist_min_max_mean(lst, vm)?.ok_or_else(|| {
            vm.new_value_error(
                "zero-size array to reduction operation which has no identity".to_string(),
            )
        });
    }

    Err(vm.new_type_error(
        "Tensor reduction: expected flat _data as bytes, bytearray, or list".to_string(),
    ))
}

fn tensor_sum_scalar(obj: PyObjectRef, vm: &VirtualMachine) -> PyResult<f64> {
    // Use the common tensor resolver so frame-backed tensors (no persistent `_data`)
    // and registry-backed tensors reduce correctly.
    let flat = tensor_flat_data_list(&obj, vm)?;
    if flat.is_empty() {
        return Err(vm.new_value_error(
            "zero-size array to reduction operation which has no identity".to_string(),
        ));
    }
    Ok(flat.iter().map(|&v| v as f64).sum())
}

fn first_arg_tensor(args: &FuncArgs, vm: &VirtualMachine, name: &str) -> PyResult<PyObjectRef> {
    args.args
        .first()
        .cloned()
        .ok_or_else(|| vm.new_type_error(format!("{name}() expects a Tensor argument")))
}

/// ``(min, max, mean)`` as float64 scalars in one native pass (for ``Tensor.__str__``).
pub fn tensor_min_max_mean(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let obj = first_arg_tensor(&args, vm, "_tensor_min_max_mean")?;
    let (mn, mx, av) = tensor_min_max_mean_triplet(obj, vm)?;
    Ok(vm
        .ctx
        .new_tuple(vec![
            vm.ctx.new_float(mn).into(),
            vm.ctx.new_float(mx).into(),
            vm.ctx.new_float(av).into(),
        ])
        .into())
}

pub fn tensor_min(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let obj = first_arg_tensor(&args, vm, "_tensor_min")?;
    let (mn, _, _) = tensor_min_max_mean_triplet(obj, vm)?;
    Ok(vm.ctx.new_float(mn).into())
}

pub fn tensor_max(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let obj = first_arg_tensor(&args, vm, "_tensor_max")?;
    let (_, mx, _) = tensor_min_max_mean_triplet(obj, vm)?;
    Ok(vm.ctx.new_float(mx).into())
}

/// `_tensor_index_string(indices, text)` — gather characters of `text` at the integer
/// positions in `indices` (treated as a flat sequence). Negative indices are wrapped
/// `numpy`-style. Returns a new `str`.
pub fn tensor_index_string(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    if args.args.len() != 2 {
        return Err(vm.new_type_error(format!(
            "_tensor_index_string() takes 2 arguments ({} given)",
            args.args.len()
        )));
    }
    let indices_flat = tensor_flat_data_list(&args.args[0], vm)?;
    let text: String = args.args[1].clone().try_into_value(vm)?;
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len() as isize;
    let mut out = String::with_capacity(indices_flat.len() * 4);
    for v in indices_flat.iter() {
        let raw = *v as isize;
        let idx = if raw < 0 { raw + n } else { raw };
        if idx < 0 || idx >= n {
            return Err(vm.new_index_error(format!(
                "index {} out of range for string of length {}",
                raw, n
            )));
        }
        out.push(chars[idx as usize]);
    }
    Ok(vm.ctx.new_str(out).into())
}

pub fn tensor_mean(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let obj = first_arg_tensor(&args, vm, "_tensor_mean")?;
    let (_, _, av) = tensor_min_max_mean_triplet(obj, vm)?;
    Ok(vm.ctx.new_float(av).into())
}

pub fn tensor_sum(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let obj = first_arg_tensor(&args, vm, "_tensor_sum")?;
    let out_dtype = if args.args.len() > 1 && !vm.is_none(&args.args[1]) {
        DType::from_py_object(&args.args[1], vm).unwrap_or(DType::Int32)
    } else if let Some(dtype_kwarg) = args.kwargs.get("dtype") {
        DType::from_py_object(dtype_kwarg, vm).unwrap_or(DType::Int32)
    } else {
        DType::Int32
    };
    let s = tensor_sum_scalar(obj, vm)?;
    let value = out_dtype.cast_from_f32(s as f32);
    let py_tensor = create_tensor_from_data(vec![value], vec![1], out_dtype);
    wrap_tensor_dict(py_tensor.to_py_dict(vm, out_dtype)?, vm)
}

pub(crate) fn wrap_tensor_dict(dict: rustpython_vm::PyObjectRef, vm: &VirtualMachine) -> PyResult {
    if let Ok(wrapper_class) = vm.builtins.get_attr("Tensor", vm) {
        if let Ok(wrapped) = wrapper_class.call((dict.clone(),), vm) {
            return Ok(wrapped);
        }
    }
    Ok(dict)
}

fn where_fn(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let args_vec = args.args;
    if args_vec.len() < 3 {
        return Err(vm.new_type_error("where() requires cond, x, y".to_string()));
    }
    let c = tensor_flat_data_list(&args_vec[0], vm)?;
    let x = tensor_flat_data_list(&args_vec[1], vm)?;
    let y = tensor_flat_data_list(&args_vec[2], vm)?;
    if c.len() != x.len() || x.len() != y.len() {
        return Err(vm.new_value_error("where(): shape mismatch".to_string()));
    }
    let shape = tensor_shape_tuple(&args_vec[1], vm)?;
    let out: Vec<f32> = c
        .iter()
        .zip(x.iter())
        .zip(y.iter())
        .map(|((&cv, &xv), &yv)| if cv != 0.0 { xv } else { yv })
        .collect();
    let dtype = DType::Float32;
    let py_tensor = create_tensor_from_data(out, shape, dtype);
    wrap_tensor_dict(py_tensor.to_py_dict(vm, dtype)?, vm)
}

fn clip_fn(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let args_vec = args.args;
    if args_vec.len() < 3 {
        return Err(vm.new_type_error("clip() requires x, min, max".to_string()));
    }
    let a = tensor_flat_data_list(&args_vec[0], vm)?;
    let lo = tensor_flat_data_list(&args_vec[1], vm)?;
    let hi = tensor_flat_data_list(&args_vec[2], vm)?;
    let shape = tensor_shape_tuple(&args_vec[0], vm)?;
    let n = a.len();
    let out = if lo.len() == n && hi.len() == n {
        a.iter()
            .zip(lo.iter())
            .zip(hi.iter())
            .map(|((&x, &l), &h)| x.max(l).min(h))
            .collect()
    } else if n % 2 == 0 && lo.len() * 2 == n && hi.len() * 2 == n {
        let rows = n / 2;
        let mut v = Vec::with_capacity(n);
        for i in 0..rows {
            let l = lo[i];
            let h = hi[i];
            v.push(a[2 * i].max(l).min(h));
            v.push(a[2 * i + 1].max(l).min(h));
        }
        v
    } else {
        return Err(vm.new_value_error("clip(): incompatible shapes".to_string()));
    };
    let dtype = DType::Float32;
    let py_tensor = create_tensor_from_data(out, shape, dtype);
    wrap_tensor_dict(py_tensor.to_py_dict(vm, dtype)?, vm)
}

fn allclose_fn(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    if args.args.len() < 2 {
        return Err(vm.new_type_error(
            "allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False) requires a and b".to_string(),
        ));
    }
    let a = &args.args[0];
    let b = &args.args[1];
    let rtol = args
        .kwargs
        .get("rtol")
        .map(|v| py_number_to_f64(v, vm))
        .transpose()?
        .unwrap_or(1e-5);
    let atol = args
        .kwargs
        .get("atol")
        .map(|v| py_number_to_f64(v, vm))
        .transpose()?
        .unwrap_or(1e-8);
    let equal_nan = args
        .kwargs
        .get("equal_nan")
        .and_then(|v| v.clone().try_into_value::<bool>(vm).ok())
        .unwrap_or(false);

    let a_shape = tensor_shape_tuple(a, vm)?;
    let b_shape = tensor_shape_tuple(b, vm)?;
    if a_shape != b_shape {
        return Ok(vm.ctx.new_bool(false).into());
    }

    let a_flat = tensor_flat_data_list(a, vm)?;
    let b_flat = tensor_flat_data_list(b, vm)?;
    if a_flat.len() != b_flat.len() {
        return Ok(vm.ctx.new_bool(false).into());
    }

    let rtol = rtol.abs();
    let atol = atol.abs();
    for (av, bv) in a_flat.iter().zip(b_flat.iter()) {
        let a64 = *av as f64;
        let b64 = *bv as f64;
        if a64 == b64 {
            continue;
        }
        if a64.is_nan() || b64.is_nan() {
            if equal_nan && a64.is_nan() && b64.is_nan() {
                continue;
            }
            return Ok(vm.ctx.new_bool(false).into());
        }
        let diff = (a64 - b64).abs();
        if diff > (atol + rtol * b64.abs()) {
            return Ok(vm.ctx.new_bool(false).into());
        }
    }
    Ok(vm.ctx.new_bool(true).into())
}

pub fn tensor_fn(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let device = crate::device_policy::tensor_device_for_constructor(&args, vm)?;
    let args_vec = args.args;
    if args_vec.is_empty() {
        return Err(vm.new_type_error("tensor() requires at least 1 argument".to_string()));
    }
    let data_arg = &args_vec[0];
    if let Ok(s) = data_arg.str(vm) {
        let text = s.to_string();
        if crate::pack::is_pack_string(&text) {
            return crate::pack::tensor_from_pack_string(&text, vm);
        }
    }
    let explicit_dtype = if args_vec.len() > 2 && !vm.is_none(&args_vec[2]) {
        Some(DType::from_py_object(&args_vec[2], vm).unwrap_or(DType::Float32))
    } else if let Some(dtype_kwarg) = args.kwargs.get("dtype") {
        Some(DType::from_py_object(dtype_kwarg, vm).unwrap_or(DType::Float32))
    } else {
        None
    };
    let looks_like_bool = || {
        if data_arg.clone().try_into_value::<bool>(vm).is_err() {
            return false;
        }
        if let Ok(s) = data_arg.str(vm) {
            let text = s.to_string();
            return text == "True" || text == "False";
        }
        false
    };
    let inferred_scalar_dtype = if explicit_dtype.is_none() {
        if looks_like_bool() {
            Some(DType::Bool)
        } else if data_arg.clone().try_into_value::<i64>(vm).is_ok() {
            Some(DType::Int32)
        } else if data_arg.clone().try_into_value::<f64>(vm).is_ok() {
            Some(DType::Float32)
        } else {
            None
        }
    } else {
        None
    };
    let dtype = explicit_dtype
        .or(inferred_scalar_dtype)
        .unwrap_or(DType::Float32);

    let mut flat_data = Vec::new();
    fn flatten_list(
        obj: &rustpython_vm::PyObjectRef,
        flat: &mut Vec<f32>,
        vm: &VirtualMachine,
    ) -> PyResult<()> {
        if let Some(list) = obj.downcast_ref::<rustpython_vm::builtins::PyList>() {
            for item in list.borrow_vec().iter() {
                flatten_list(item, flat, vm)?;
            }
        } else if let Some(tup) = obj.downcast_ref::<rustpython_vm::builtins::PyTuple>() {
            for item in tup.as_slice().iter() {
                flatten_list(item, flat, vm)?;
            }
        } else {
            flat.push(py_number_to_f64(obj, vm)? as f32);
        }
        Ok(())
    }
    if let Some(data_list) = data_arg.downcast_ref::<rustpython_vm::builtins::PyList>() {
        for item in data_list.borrow_vec().iter() {
            flatten_list(item, &mut flat_data, vm)?;
        }
    } else if let Some(data_tuple) = data_arg.downcast_ref::<rustpython_vm::builtins::PyTuple>() {
        for item in data_tuple.as_slice().iter() {
            flatten_list(item, &mut flat_data, vm)?;
        }
    } else if looks_like_bool() {
        let v = data_arg.clone().try_into_value::<bool>(vm).unwrap_or(false);
        flat_data.push(if v { 1.0 } else { 0.0 });
    } else if let Ok(v) = data_arg.clone().try_into_value::<i64>(vm) {
        flat_data.push(v as f32);
    } else if let Ok(v) = data_arg.clone().try_into_value::<f64>(vm) {
        flat_data.push(v as f32);
    } else {
        return Err(vm.new_type_error(
            "data must be a number, tuple, or list".to_string(),
        ));
    }

    fn infer_shape(obj: &rustpython_vm::PyObjectRef) -> Option<Vec<usize>> {
        if let Some(list) = obj.downcast_ref::<rustpython_vm::builtins::PyList>() {
            let items = list.borrow_vec();
            let n = items.len();
            if n == 0 {
                return Some(vec![0]);
            }
            let first = infer_shape(&items[0])?;
            for item in items.iter().skip(1) {
                if infer_shape(item)? != first {
                    return None;
                }
            }
            let mut shape = vec![n];
            shape.extend(first);
            return Some(shape);
        }
        if let Some(tup) = obj.downcast_ref::<rustpython_vm::builtins::PyTuple>() {
            let items = tup.as_slice();
            let n = items.len();
            if n == 0 {
                return Some(vec![0]);
            }
            let first = infer_shape(&items[0])?;
            for item in items.iter().skip(1) {
                if infer_shape(item)? != first {
                    return None;
                }
            }
            let mut shape = vec![n];
            shape.extend(first);
            return Some(shape);
        }
        Some(vec![])
    }

    let shape = if args_vec.len() > 1 {
        let shape_arg = &args_vec[1];
        if let Some(shape_tuple) = shape_arg.downcast_ref::<rustpython_vm::builtins::PyTuple>() {
            shape_tuple
                .as_slice()
                .iter()
                .map(|s| s.clone().try_into_value::<i32>(vm).map(|i| i as usize))
                .collect::<Result<Vec<_>, _>>()?
        } else if let Some(shape_list) = shape_arg.downcast_ref::<rustpython_vm::builtins::PyList>()
        {
            shape_list
                .borrow_vec()
                .iter()
                .map(|s| s.clone().try_into_value::<i32>(vm).map(|i| i as usize))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![flat_data.len()]
        }
    } else {
        match infer_shape(data_arg) {
            Some(s) if !s.is_empty() => s,
            _ => vec![flat_data.len()],
        }
    };
    let casted_data: Vec<f32> = flat_data.iter().map(|&v| dtype.cast_from_f32(v)).collect();
    let py_tensor = create_tensor_from_data(casted_data, shape, dtype);
    wrap_tensor_dict(py_tensor.to_py_dict_on(vm, dtype, &device)?, vm)
}

fn parse_shape_arg(shape_obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<usize>> {
    if let Some(tup) = shape_obj.downcast_ref::<rustpython_vm::builtins::PyTuple>() {
        return tup
            .as_slice()
            .iter()
            .map(|s| s.clone().try_into_value::<i32>(vm).map(|i| i as usize))
            .collect::<Result<Vec<_>, _>>();
    }
    if let Some(lst) = shape_obj.downcast_ref::<rustpython_vm::builtins::PyList>() {
        return lst
            .borrow_vec()
            .iter()
            .map(|s| s.clone().try_into_value::<i32>(vm).map(|i| i as usize))
            .collect::<Result<Vec<_>, _>>();
    }
    Err(vm.new_type_error("shape must be a tuple or list".to_string()))
}

fn dtype_from_args(args: &FuncArgs, args_vec: &[PyObjectRef], vm: &VirtualMachine) -> PyResult<DType> {
    if args_vec.len() > 1 && !vm.is_none(&args_vec[1]) {
        return DType::from_py_object(&args_vec[1], vm);
    }
    if let Some(dtype_kwarg) = args.kwargs.get("dtype") {
        return DType::from_py_object(dtype_kwarg, vm);
    }
    Ok(DType::Float32)
}

pub(crate) fn tensor_dtype_from_ref(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<DType> {
    let mut cur = obj.clone();
    for _ in 0..8 {
        if let Some(dict) = cur.downcast_ref::<rustpython_vm::builtins::PyDict>() {
            if let Ok(dtype_obj) = dict.get_item("dtype", vm) {
                return DType::from_py_object(&dtype_obj, vm);
            }
        }
        if let Ok(Some(attr)) = vm.get_attribute_opt(cur.clone(), "_data") {
            cur = attr;
            continue;
        }
        break;
    }
    Ok(DType::Float32)
}

pub fn zeros_fn(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    if args.args.is_empty() {
        return Err(vm.new_type_error("zeros() requires 1 argument (shape)".to_string()));
    }
    let device = crate::device_policy::tensor_device_for_constructor(&args, vm)?;
    let dtype = dtype_from_args(&args, &args.args, vm)?;
    let shape_arg = parse_shape_arg(&args.args[0], vm)?;
    let total: usize = shape_arg.iter().product();
    let py_tensor = create_tensor_from_data(vec![0.0f32; total], shape_arg, dtype);
    let registry_only = crate::tensor_buf::use_registry_only_tensor(&device, total);
    wrap_tensor_dict(
        crate::tensor_buf::wrap_registry_tensor(vm, &py_tensor, dtype, &device, registry_only)?,
        vm,
    )
}

pub fn zeros_like_fn(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let ref_tensor = args.args.first().ok_or_else(|| {
        vm.new_type_error("zeros_like() requires 1 argument (tensor)".to_string())
    })?;
    let shape = tensor_shape_tuple(ref_tensor, vm)?;
    let dtype = tensor_dtype_from_ref(ref_tensor, vm)?;
    let device = crate::device_policy::tensor_device_label(ref_tensor, vm)?;
    let total: usize = shape.iter().product();
    let py_tensor = create_tensor_from_data(vec![0.0f32; total], shape, dtype);
    let registry_only = crate::tensor_buf::use_registry_only_tensor(&device, total);
    wrap_tensor_dict(
        crate::tensor_buf::wrap_registry_tensor(vm, &py_tensor, dtype, &device, registry_only)?,
        vm,
    )
}

pub fn ones_fn(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    if args.args.is_empty() {
        return Err(vm.new_type_error("ones() requires 1 argument (shape)".to_string()));
    }
    let device = crate::device_policy::tensor_device_for_constructor(&args, vm)?;
    let dtype = dtype_from_args(&args, &args.args, vm)?;
    let shape_arg = parse_shape_arg(&args.args[0], vm)?;
    let total: usize = shape_arg.iter().product();
    let py_tensor = create_tensor_from_data(vec![1.0f32; total], shape_arg, dtype);
    wrap_tensor_dict(py_tensor.to_py_dict_on(vm, dtype, &device)?, vm)
}

pub fn full_fn(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    if args.args.len() < 2 {
        return Err(vm.new_type_error("full() requires shape and fill value".to_string()));
    }
    let device = crate::device_policy::tensor_device_for_constructor(&args, vm)?;
    let dtype = if args.args.len() > 2 && !vm.is_none(&args.args[2]) {
        DType::from_py_object(&args.args[2], vm)?
    } else if let Some(dtype_kwarg) = args.kwargs.get("dtype") {
        DType::from_py_object(dtype_kwarg, vm)?
    } else {
        DType::Float32
    };
    let shape_arg = parse_shape_arg(&args.args[0], vm)?;
    let fill_value = py_number_to_f64(&args.args[1], vm)? as f32;
    let total: usize = shape_arg.iter().product();
    let py_tensor = create_tensor_from_data(vec![fill_value; total], shape_arg, dtype);
    wrap_tensor_dict(py_tensor.to_py_dict_on(vm, dtype, &device)?, vm)
}

pub fn arange_fn(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let args_vec = args.args;
    if args_vec.is_empty() {
        return Err(vm.new_type_error("arange() requires at least start".to_string()));
    }
    let (start, stop, step) = if args_vec.len() == 1 {
        (0.0, py_number_to_f64(&args_vec[0], vm)?, 1.0)
    } else {
        let start = py_number_to_f64(&args_vec[0], vm)?;
        let stop = py_number_to_f64(&args_vec[1], vm)?;
        let step = if args_vec.len() > 2 {
            py_number_to_f64(&args_vec[2], vm)?
        } else {
            1.0
        };
        (start, stop, step)
    };
    if step == 0.0 {
        return Err(vm.new_value_error("arange() step must not be 0".to_string()));
    }
    let dtype = if args_vec.len() > 3 && !vm.is_none(&args_vec[3]) {
        DType::from_py_object(&args_vec[3], vm).unwrap_or(DType::Float32)
    } else if let Some(dtype_kwarg) = args.kwargs.get("dtype") {
        DType::from_py_object(dtype_kwarg, vm).unwrap_or(DType::Float32)
    } else {
        DType::Float32
    };
    let mut data = Vec::new();
    let mut v = start;
    if step > 0.0 {
        while v < stop {
            data.push(v as f32);
            v += step;
        }
    } else {
        while v > stop {
            data.push(v as f32);
            v += step;
        }
    }
    let py_tensor = create_tensor_from_data(data.clone(), vec![data.len()], dtype);
    wrap_tensor_dict(py_tensor.to_py_dict(vm, dtype)?, vm)
}

pub fn stack_fn(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let args_vec = args.args;
    if args_vec.is_empty() {
        return Err(vm.new_type_error("stack() requires a list of tensors".to_string()));
    }
    let tensors = args_vec[0]
        .downcast_ref::<rustpython_vm::builtins::PyList>()
        .ok_or_else(|| vm.new_type_error("stack() first arg must be a list".to_string()))?;
    let axis = if args_vec.len() > 1 {
        args_vec[1].clone().try_into_value::<i32>(vm).unwrap_or(0)
    } else if let Some(axis_kwarg) = args.kwargs.get("axis") {
        axis_kwarg.clone().try_into_value::<i32>(vm).unwrap_or(0)
    } else {
        0
    };
    let tensor_items = tensors.borrow_vec();
    if tensor_items.is_empty() {
        return Err(vm.new_value_error("stack() requires at least one tensor".to_string()));
    }
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for t in tensor_items.iter() {
        rows.push(tensor_flat_data_list(t, vm)?);
    }
    let n = rows[0].len();
    if rows.iter().any(|r| r.len() != n) {
        return Err(vm.new_value_error("stack() all tensors must have same length".to_string()));
    }
    let (flat, shape) = if axis == 1 {
        let mut out = vec![0.0f32; n * rows.len()];
        for i in 0..n {
            for j in 0..rows.len() {
                out[i * rows.len() + j] = rows[j][i];
            }
        }
        (out, vec![n, rows.len()])
    } else {
        let mut out = Vec::with_capacity(n * rows.len());
        for row in rows.iter() {
            out.extend_from_slice(row);
        }
        (out, vec![rows.len(), n])
    };
    let dtype = DType::Float32;
    let py_tensor = create_tensor_from_data(flat, shape, dtype);
    wrap_tensor_dict(py_tensor.to_py_dict(vm, dtype)?, vm)
}

fn resolve_tensor_dict(
    obj: &PyObjectRef,
    vm: &VirtualMachine,
) -> PyResult<rustpython_vm::PyRef<PyDict>> {
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

fn random_value_for_dtype(dtype: DType) -> f32 {
    #[cfg(target_arch = "wasm32")]
    {
        let low = dtype.min_f64();
        let high = dtype.max_f64();
        let t = js_sys::Math::random();
        let v = low + t * (high - low);
        return dtype.cast_from_f32(v as f32);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rand::Rng;
        let mut rng = rand::rng();
        let v = match dtype {
            DType::Bool => {
                if rng.random_bool(0.5) {
                    1.0
                } else {
                    0.0
                }
            }
            DType::Int8 => rng.random_range(i8::MIN..=i8::MAX) as f64,
            DType::Int16 => rng.random_range(i16::MIN..=i16::MAX) as f64,
            DType::Int32 => rng.random_range(i32::MIN..=i32::MAX) as f64,
            DType::Int64 => rng.random_range(i64::MIN..=i64::MAX) as f64,
            DType::UInt8 => rng.random_range(0u8..=u8::MAX) as f64,
            DType::UInt16 => rng.random_range(0u16..=u16::MAX) as f64,
            DType::UInt32 => rng.random_range(0u32..=u32::MAX) as f64,
            DType::UInt64 => rng.random_range(0u64..=u64::MAX) as f64,
            DType::Float16 | DType::Float32 | DType::Float64 => {
                rng.random_range(dtype.min_f64()..=dtype.max_f64())
            }
        };
        dtype.cast_from_f32(v as f32)
    }
}

fn write_flat_to_tensor(
    tensor: &PyObjectRef,
    flat: &[f32],
    dtype: DType,
    vm: &VirtualMachine,
) -> PyResult<()> {
    if let Some(id) = crate::xos_module::tensor_rust_id(tensor, vm) {
        crate::tensor_buf::write_tensor_data_by_id(id, flat);
    }
    let dict = resolve_tensor_dict(tensor, vm)?;
    let is_frame_backed = dict.get_item("_xos_frame_backing", vm).is_ok()
        || dict.get_item("_xos_viewport_id", vm).is_ok();
    if dtype == DType::UInt8 {
        let bytes: Vec<u8> = flat.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
        if is_frame_backed {
            crate::xos_module::with_frame_write_buffer(vm, Some(tensor), |buffer| {
                let n = buffer.len().min(bytes.len());
                buffer[..n].copy_from_slice(&bytes[..n]);
                crate::rasterizer::note_frame_cpu_write();
                Ok(())
            })?;
        }
        dict.set_item(
            "_data",
            PyByteArray::new_ref(bytes, &vm.ctx).into(),
            vm,
        )?;
    } else if dtype.is_float() {
        if is_frame_backed {
            let bytes: Vec<u8> = flat.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
            crate::xos_module::with_frame_write_buffer(vm, Some(tensor), |buffer| {
                let n = buffer.len().min(bytes.len());
                buffer[..n].copy_from_slice(&bytes[..n]);
                crate::rasterizer::note_frame_cpu_write();
                Ok(())
            })?;
        }
        let py: Vec<PyObjectRef> = flat
            .iter()
            .map(|&v| vm.ctx.new_float(v as f64).into())
            .collect();
        dict.set_item("_data", vm.ctx.new_list(py).into(), vm)?;
    } else {
        if is_frame_backed {
            let bytes: Vec<u8> = flat.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
            crate::xos_module::with_frame_write_buffer(vm, Some(tensor), |buffer| {
                let n = buffer.len().min(bytes.len());
                buffer[..n].copy_from_slice(&bytes[..n]);
                crate::rasterizer::note_frame_cpu_write();
                Ok(())
            })?;
        }
        let py: Vec<PyObjectRef> = flat
            .iter()
            .map(|&v| vm.ctx.new_int(v as i64).into())
            .collect();
        dict.set_item("_data", vm.ctx.new_list(py).into(), vm)?;
    }
    Ok(())
}

/// ``xos._tensor_randomize(tensor)`` — fill every element with a random value in [dtype.MIN, dtype.MAX].
pub fn tensor_randomize(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let tensor = args.args.first().ok_or_else(|| {
        vm.new_type_error("_tensor_randomize() expects a tensor".to_string())
    })?;
    let dtype = tensor_dtype_from_ref(tensor, vm)?;
    let shape = tensor_shape_tuple(tensor, vm)?;
    let n: usize = shape.iter().product();
    if n == 0 {
        return Ok(vm.ctx.none());
    }
    let flat: Vec<f32> = (0..n).map(|_| random_value_for_dtype(dtype)).collect();
    write_flat_to_tensor(tensor, &flat, dtype, vm)?;
    Ok(vm.ctx.none())
}

pub fn register_tensors_functions(module: &PyRef<PyModule>, vm: &VirtualMachine) {
    module
        .set_attr("tensor", vm.new_function("tensor", tensor_fn), vm)
        .unwrap();
    module
        .set_attr("zeros", vm.new_function("zeros", zeros_fn), vm)
        .unwrap();
    module
        .set_attr("zeros_like", vm.new_function("zeros_like", zeros_like_fn), vm)
        .unwrap();
    module
        .set_attr("ones", vm.new_function("ones", ones_fn), vm)
        .unwrap();
    module
        .set_attr("full", vm.new_function("full", full_fn), vm)
        .unwrap();
    module
        .set_attr("arange", vm.new_function("arange", arange_fn), vm)
        .unwrap();
    module
        .set_attr("stack", vm.new_function("stack", stack_fn), vm)
        .unwrap();
    module
        .set_attr("where", vm.new_function("where", where_fn), vm)
        .unwrap();
    module
        .set_attr("clip", vm.new_function("clip", clip_fn), vm)
        .unwrap();
    module
        .set_attr("allclose", vm.new_function("allclose", allclose_fn), vm)
        .unwrap();
}
