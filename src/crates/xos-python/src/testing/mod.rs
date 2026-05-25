//! `xos.test` / `xos.parametrize` — bootstrap loaded from `bootstrap.py` on disk when possible.

use rustpython_vm::builtins::PyModule;
use rustpython_vm::{AsObject, PyRef, VirtualMachine};
use std::path::PathBuf;

use crate::runtime::format_python_exception;

const TESTING_BOOTSTRAP_EMBEDDED: &str = include_str!("bootstrap.py");

fn testing_bootstrap_path() -> Option<PathBuf> {
    let root = xos_core::find_xos_project_root().ok()?;
    let path = root.join("src/crates/xos-python/src/testing/bootstrap.py");
    if path.is_file() {
        Some(path)
    } else {
        None
    }
}

fn load_testing_bootstrap() -> (String, String) {
    if let Some(path) = testing_bootstrap_path() {
        if let Ok(code) = std::fs::read_to_string(&path) {
            return (code, path.to_string_lossy().to_string());
        }
    }
    (
        TESTING_BOOTSTRAP_EMBEDDED.to_string(),
        "<xos/testing/bootstrap.py>".to_string(),
    )
}

pub fn install_testing(vm: &VirtualMachine, module: PyRef<PyModule>) {
    let scope = vm.new_scope_with_builtins();
    let _ = scope
        .globals
        .set_item("xos", module.as_object().to_owned(), vm);
    let (bootstrap, filename) = load_testing_bootstrap();
    if let Err(e) = vm.run_code_string(scope.clone(), &bootstrap, filename) {
        eprintln!(
            "xos testing bootstrap failed:\n{}",
            format_python_exception(vm, &e)
        );
        return;
    }
    for name in [
        "test",
        "parametrize",
        "_clear_registry",
        "_register_module_tests",
        "_run_all",
    ] {
        if let Ok(obj) = scope.globals.get_item(name, vm) {
            let _ = module.set_attr(name, obj, vm);
        }
    }
}
