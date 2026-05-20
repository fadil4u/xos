//! `xos.render` — functional tensor preview (bootstrap in `bootstrap.py`).

use rustpython_vm::builtins::PyModule;
use rustpython_vm::{AsObject, PyRef, VirtualMachine};

use crate::runtime::format_python_exception;

const RENDER_BOOTSTRAP: &str = include_str!("bootstrap.py");

pub fn install_render(vm: &VirtualMachine, module: PyRef<PyModule>) {
    let scope = vm.new_scope_with_builtins();
    let _ = scope
        .globals
        .set_item("xos", module.as_object().to_owned(), vm);
    if let Err(e) = vm.run_code_string(
        scope.clone(),
        RENDER_BOOTSTRAP,
        "<xos/render/bootstrap.py>".to_string(),
    ) {
        eprintln!(
            "xos render bootstrap failed:\n{}",
            format_python_exception(vm, &e)
        );
        return;
    }
    if let Ok(render_fn) = scope.globals.get_item("render", vm) {
        let _ = module.set_attr("render", render_fn, vm);
    }
    if let Ok(viewport_cls) = scope.globals.get_item("Viewport", vm) {
        let _ = module.set_attr("Viewport", viewport_cls, vm);
    }
}
