//! Strict device matching for Python tensor / frame ops.

use rustpython_vm::builtins::PyDictRef;
use rustpython_vm::{AsObject, PyObjectRef, PyResult, VirtualMachine};
use xos_core::compute_device::ComputeDevice;

use crate::engine::py_engine_tls;

fn python_app_instance(vm: &VirtualMachine) -> Option<PyObjectRef> {
    vm.get_attribute_opt(vm.builtins.as_object().to_owned(), "__xos_app_instance__")
        .ok()
        .flatten()
}

/// Read `Application.device` from the active app (`None` → auto).
fn read_app_device_pref_str(vm: &VirtualMachine) -> PyResult<Option<String>> {
    let Some(app) = python_app_instance(vm) else {
        return Ok(None);
    };
    let Some(obj) = vm.get_attribute_opt(app, "device").ok().flatten() else {
        return Ok(None);
    };
    if obj.is(&vm.ctx.none()) {
        return Ok(None);
    }
    Ok(Some(obj.try_into_value::<String>(vm)?))
}

/// Resolved device for the running app when the engine is not bound (e.g. `__init__`).
pub fn app_compute_device(vm: &VirtualMachine) -> PyResult<ComputeDevice> {
    let pref = ComputeDevice::parse_pref(read_app_device_pref_str(vm)?.as_deref())
        .map_err(|e| vm.new_value_error(e))?;
    Ok(ComputeDevice::resolve_auto(pref))
}

/// Engine tick/callback device, or the app's declared device during `__init__`.
pub fn effective_compute_device(vm: &VirtualMachine) -> PyResult<ComputeDevice> {
    if let Some(d) = py_engine_tls::engine_compute_device() {
        return Ok(d);
    }
    app_compute_device(vm)
}

pub fn engine_compute_device(vm: &VirtualMachine) -> PyResult<ComputeDevice> {
    py_engine_tls::engine_compute_device().ok_or_else(|| {
        vm.new_runtime_error(
            "No active xos engine (call during Application.tick() or on_screen_size_change)"
                .to_string(),
        )
    })
}

/// Read `device` from a tensor dict / `_TensorWrapper` / frame.tensor dict.
pub fn tensor_device_label(obj: &PyObjectRef, vm: &VirtualMachine) -> PyResult<String> {
    if let Ok(data_attr) = obj.get_attr("_data", vm) {
        if let Ok(inner) = data_attr.downcast::<rustpython_vm::builtins::PyDict>() {
            if let Ok(dev) = inner.get_item("device", vm) {
                return dev.str(vm).map(|s| s.to_string());
            }
        }
    }
    if let Ok(dict) = obj.clone().downcast::<rustpython_vm::builtins::PyDict>() {
        if let Ok(dev) = dict.get_item("device", vm) {
            return dev.str(vm).map(|s| s.to_string());
        }
    }
    Ok("cpu".to_string())
}

pub fn require_same_devices(
    vm: &VirtualMachine,
    op: &str,
    labels: &[(&str, String)],
) -> PyResult<()> {
    if labels.len() < 2 {
        return Ok(());
    }
    let first = &labels[0].1;
    for (name, dev) in &labels[1..] {
        if dev != first {
            return Err(vm.new_runtime_error(format!(
                "{op}(): device mismatch — {} is on '{}', {} is on '{}' (align with \
                 Application.device or Tensor.to(device=...))",
                labels[0].0, first, name, dev
            )));
        }
    }
    Ok(())
}

/// Match tensor device to the active app/engine device; return the device to execute on.
pub fn require_engine_device(
    vm: &VirtualMachine,
    op: &str,
    tensor_dev: &str,
) -> PyResult<ComputeDevice> {
    let app_dev = effective_compute_device(vm)?;

    // `Application.__init__` uses a CPU standalone framebuffer even for GPU apps.
    if py_engine_tls::engine_compute_device().is_none() {
        if tensor_dev != "cpu" && tensor_dev != app_dev.as_str() {
            return Err(vm.new_runtime_error(format!(
                "{op}(): tensor device is '{tensor_dev}' during Application.__init__() \
                 (expected 'cpu' or '{}')",
                app_dev.as_str()
            )));
        }
        return Ok(ComputeDevice::Cpu);
    }

    if tensor_dev != app_dev.as_str() {
        return Err(vm.new_runtime_error(format!(
            "{op}(): frame/tensor device is '{tensor_dev}' but this app uses '{}' \
             (set Application.device or move tensors with .to(device=...))",
            app_dev.as_str()
        )));
    }
    Ok(app_dev)
}

pub fn tag_tensor_device(dict: &PyDictRef, device: &str, vm: &VirtualMachine) {
    let _ = dict.set_item("device", vm.ctx.new_str(device).into(), vm);
}
