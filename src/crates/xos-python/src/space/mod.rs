//! ``xos.space`` / ``xos.shapes`` — coordinate spaces and axis-aligned transforms (bootstrap).

use rustpython_vm::builtins::PyModule;
use rustpython_vm::{AsObject, PyRef, VirtualMachine};

use crate::runtime::format_python_exception;

const SPACE_BOOTSTRAP: &str = include_str!("bootstrap.py");

pub fn install_space(vm: &VirtualMachine, module: PyRef<PyModule>) {
    let scope = vm.new_scope_with_builtins();
    let _ = scope
        .globals
        .set_item("xos", module.as_object().to_owned(), vm);

    if let Err(e) = vm.run_code_string(
        scope.clone(),
        SPACE_BOOTSTRAP,
        "<xos/space/bootstrap.py>".to_string(),
    ) {
        eprintln!(
            "xos space bootstrap failed:\n{}",
            format_python_exception(vm, &e)
        );
        return;
    }

    for name in ["space", "shapes", "Space", "Transform", "Rectangles"] {
        if let Ok(obj) = scope.globals.get_item(name, vm) {
            let _ = module.set_attr(name, obj, vm);
        }
    }
}
