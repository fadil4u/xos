//! `xos.test` / `xos.parametrize` — bootstrap loaded from `bootstrap.py` (no recompile to edit).

use rustpython_vm::builtins::PyModule;
use rustpython_vm::{AsObject, PyRef, VirtualMachine};

use crate::runtime::format_python_exception;

const TESTING_BOOTSTRAP: &str = include_str!("bootstrap.py");

pub fn install_testing(vm: &VirtualMachine, module: PyRef<PyModule>) {
    let scope = vm.new_scope_with_builtins();
    let _ = scope
        .globals
        .set_item("xos", module.as_object().to_owned(), vm);
    if let Err(e) = vm.run_code_string(
        scope.clone(),
        TESTING_BOOTSTRAP,
        "<xos/testing/bootstrap.py>".to_string(),
    ) {
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
