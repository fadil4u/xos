//! Tensor printpack, deflate compress/decompress, and ``xos.all``.

use crate::dtypes::DType;
use crate::tensor_buf::{
    create_tensor_from_data, tensor_flat_bytes, tensor_flat_data_list, tensor_shape_tuple,
};
use crate::tensors::{tensor_dtype_from_ref, wrap_tensor_dict};
use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use flate2::read::DeflateDecoder;
use flate2::write::DeflateEncoder;
use flate2::Compression;
use rustpython_vm::builtins::{PyByteArray, PyBytes, PyDict};
use rustpython_vm::function::FuncArgs;
use rustpython_vm::{PyObjectRef, PyRef, PyResult, VirtualMachine};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{Read, Write};

pub const PACK_PREFIX: &str = "xos.pack:";
pub const PACK_Z_PREFIX: &str = "xos.pack.z:";
pub const Z_PREFIX: &str = "xos.z:";

#[derive(Debug, Serialize, Deserialize)]
struct PackPayload {
    shape: Vec<usize>,
    dtype: String,
    device: String,
    /// Per-element values (JSON integers for integral dtypes, floats for float dtypes).
    data: Vec<Value>,
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

fn device_label_from_dict(dict: PyRef<PyDict>, vm: &VirtualMachine) -> String {
    dict.get_item("device", vm)
        .ok()
        .and_then(|o| o.str(vm).ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "cpu".to_string())
}

fn deflate_bytes(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut enc = DeflateEncoder::new(Vec::new(), Compression::default());
    enc.write_all(input)
        .map_err(|e| format!("deflate failed: {e}"))?;
    enc.finish().map_err(|e| format!("deflate finish failed: {e}"))
}

fn inflate_bytes(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut dec = DeflateDecoder::new(input);
    let mut out = Vec::new();
    dec.read_to_end(&mut out)
        .map_err(|e| format!("inflate failed: {e}"))?;
    Ok(out)
}

pub fn compress_bytes(raw: &[u8]) -> Result<String, String> {
    let compressed = deflate_bytes(raw)?;
    Ok(format!("{Z_PREFIX}{}", B64.encode(compressed)))
}

pub fn decompress_bytes(encoded: &str) -> Result<Vec<u8>, String> {
    let payload = encoded
        .strip_prefix(Z_PREFIX)
        .ok_or_else(|| "expected xos.z: prefix".to_string())?;
    let compressed = B64
        .decode(payload.trim())
        .map_err(|e| format!("base64 decode failed: {e}"))?;
    inflate_bytes(&compressed)
}

fn pack_data_values(obj: &PyObjectRef, dtype: DType, vm: &VirtualMachine) -> PyResult<Vec<Value>> {
    if dtype == DType::UInt8 {
        let bytes = tensor_flat_bytes(obj, vm)?;
        return Ok(bytes.into_iter().map(|b| Value::from(b as u64)).collect());
    }
    let flat = tensor_flat_data_list(obj, vm)?;
    if dtype.is_float() {
        Ok(flat.into_iter().map(|v| Value::from(v as f64)).collect())
    } else {
        Ok(flat
            .into_iter()
            .map(|v| Value::from(dtype.cast_from_f32(v) as i64))
            .collect())
    }
}

fn unpack_flat_from_values(values: &[Value], dtype: DType) -> Result<Vec<f32>, String> {
    values
        .iter()
        .map(|v| {
            if dtype.is_float() {
                v.as_f64()
                    .ok_or_else(|| format!("expected float in pack data, got {v}"))
                    .map(|n| dtype.cast_from_f32(n as f32))
            } else if let Some(n) = v.as_u64() {
                Ok(dtype.cast_from_f32(n as f32))
            } else if let Some(n) = v.as_i64() {
                Ok(dtype.cast_from_f32(n as f32))
            } else if let Some(n) = v.as_f64() {
                Ok(dtype.cast_from_f32(n as f32))
            } else {
                Err(format!("invalid pack data element: {v}"))
            }
        })
        .collect()
}

fn payload_from_tensor(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<PackPayload> {
    let dict = resolve_tensor_dict(obj, vm)?;
    let shape = tensor_shape_tuple(obj, vm)?;
    let dtype = tensor_dtype_from_ref(obj, vm)?;
    let device = device_label_from_dict(dict, vm);
    let data = pack_data_values(obj, dtype, vm)?;
    Ok(PackPayload {
        shape,
        dtype: dtype.name().to_string(),
        device,
        data,
    })
}

fn encode_pack(payload: &PackPayload, compress: bool) -> Result<String, String> {
    let json = serde_json::to_string(payload).map_err(|e| format!("pack json failed: {e}"))?;
    if compress {
        let compressed = deflate_bytes(json.as_bytes())?;
        Ok(format!("{PACK_Z_PREFIX}{}", B64.encode(compressed)))
    } else {
        Ok(format!("{PACK_PREFIX}{json}"))
    }
}

pub fn is_pack_string(s: &str) -> bool {
    s.starts_with(PACK_PREFIX) || s.starts_with(PACK_Z_PREFIX)
}

pub fn decode_pack_string(s: &str) -> Result<PackPayload, String> {
    let json_bytes = if let Some(rest) = s.strip_prefix(PACK_Z_PREFIX) {
        let compressed = B64
            .decode(rest.trim())
            .map_err(|e| format!("pack base64 decode failed: {e}"))?;
        inflate_bytes(&compressed)?
    } else if let Some(rest) = s.strip_prefix(PACK_PREFIX) {
        rest.as_bytes().to_vec()
    } else {
        return Err("not a printpack string".to_string());
    };
    let json = std::str::from_utf8(&json_bytes).map_err(|e| format!("pack utf8: {e}"))?;
    serde_json::from_str(json).map_err(|e| format!("pack json parse: {e}"))
}

pub fn tensor_from_pack_string(s: &str, vm: &VirtualMachine) -> PyResult<PyObjectRef> {
    let payload = decode_pack_string(s).map_err(|e| vm.new_value_error(e))?;
    let dtype = DType::from_str(&payload.dtype)
        .ok_or_else(|| vm.new_value_error(format!("unknown dtype: {}", payload.dtype)))?;
    let flat = unpack_flat_from_values(&payload.data, dtype)
        .map_err(|e| vm.new_value_error(e))?;
    let expected: usize = payload.shape.iter().product();
    if expected > 0 && flat.len() != expected {
        return Err(vm.new_value_error(format!(
            "pack data length {} does not match shape product {}",
            flat.len(),
            expected
        )));
    }

    let dict = if dtype == DType::UInt8 {
        let bytes: Vec<u8> = flat.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
        let dict = vm.ctx.new_dict();
        dict.set_item(
            "shape",
            vm.ctx
                .new_tuple(
                    payload
                        .shape
                        .iter()
                        .map(|&s| vm.ctx.new_int(s as i64).into())
                        .collect(),
                )
                .into(),
            vm,
        )?;
        dict.set_item("dtype", vm.ctx.new_str(dtype.name()).into(), vm)?;
        dict.set_item("device", vm.ctx.new_str(payload.device.as_str()).into(), vm)?;
        dict.set_item(
            "_data",
            PyByteArray::new_ref(bytes, &vm.ctx).into(),
            vm,
        )?;
        dict.into()
    } else {
        let py_tensor = create_tensor_from_data(flat, payload.shape, dtype);
        py_tensor.to_py_dict_on(vm, dtype, &payload.device)?
    };
    wrap_tensor_dict(dict, vm)
}

/// ``xos._tensor_printpack(tensor, compress=False)`` → single-line pack string.
pub fn tensor_printpack(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let tensor = args
        .args
        .first()
        .ok_or_else(|| vm.new_type_error("_tensor_printpack(tensor, compress=False)".to_string()))?;
    let compress = args
        .kwargs
        .get("compress")
        .map(|o| o.clone().try_into_value::<bool>(vm))
        .transpose()?
        .unwrap_or(false);
    let payload = payload_from_tensor(tensor, vm)?;
    let out = encode_pack(&payload, compress).map_err(|e| vm.new_runtime_error(e))?;
    Ok(vm.ctx.new_str(out).into())
}

/// ``xos.compress(data)`` — deflate + base64 with ``xos.z:`` prefix.
pub fn compress_fn(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let obj = args
        .args
        .first()
        .ok_or_else(|| vm.new_type_error("compress() requires one argument".to_string()))?;
    let raw = py_object_to_bytes(obj, vm)?;
    let out = compress_bytes(&raw).map_err(|e| vm.new_runtime_error(e))?;
    Ok(vm.ctx.new_str(out).into())
}

/// ``xos.decompress(data)`` — reverse of ``compress()``; accepts ``xos.z:`` strings or bytes.
pub fn decompress_fn(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let obj = args
        .args
        .first()
        .ok_or_else(|| vm.new_type_error("decompress() requires one argument".to_string()))?;
    let was_str = obj.str(vm).is_ok();
    let encoded = if let Ok(s) = obj.str(vm) {
        s.to_string()
    } else {
        let bytes = py_object_to_bytes(obj, vm)?;
        String::from_utf8(bytes).map_err(|e| vm.new_value_error(format!("decompress utf8: {e}")))?
    };
    let raw = decompress_bytes(&encoded).map_err(|e| vm.new_runtime_error(e))?;
    if was_str {
        let text = String::from_utf8(raw)
            .map_err(|e| vm.new_value_error(format!("decompressed bytes are not utf8: {e}")))?;
        return Ok(vm.ctx.new_str(text).into());
    }
    Ok(PyByteArray::new_ref(raw, &vm.ctx).into())
}

fn py_object_to_bytes(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<Vec<u8>> {
    if let Some(bytes) = obj.downcast_ref::<PyBytes>() {
        return Ok(bytes.as_bytes().to_vec());
    }
    if let Some(ba) = obj.downcast_ref::<PyByteArray>() {
        return Ok(ba.borrow_buf().to_vec());
    }
    if let Ok(s) = obj.str(vm) {
        return Ok(s.as_str().as_bytes().to_vec());
    }
    Err(vm.new_type_error("compress/decompress expects str, bytes, or bytearray".to_string()))
}

/// ``xos.all(tensor)`` — True when every element is non-zero (e.g. after ``tensor == other``).
pub fn all_fn(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let tensor = args
        .args
        .first()
        .ok_or_else(|| vm.new_type_error("all() requires one tensor argument".to_string()))?;
    let flat = tensor_flat_data_list(tensor, vm)?;
    if flat.is_empty() {
        return Ok(vm.ctx.new_bool(true).into());
    }
    let ok = flat.iter().all(|&v| v != 0.0);
    Ok(vm.ctx.new_bool(ok).into())
}
