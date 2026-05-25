//! ``xos.clipboard`` — OS clipboard text via ``xos_core::clipboard``.
//!
//! - ``xos.clipboard`` → ``get_contents()`` (str)
//! - ``xos.clipboard = text`` → ``set_contents(text)``
//!
//! RustPython modules cannot swap ``__class__`` (PyModule has a payload), so assignment
//! would only touch ``module.__dict__``. We replace the module dict with a subclass that
//! routes the ``clipboard`` key to the native helpers.

use rustpython_vm::builtins::{PyDict, PyModule};
use rustpython_vm::function::FuncArgs;
use rustpython_vm::{AsObject, PyRef, PyResult, VirtualMachine};

use crate::runtime::format_python_exception;

fn native_clipboard_get(_args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let text = xos_core::clipboard::get_contents().unwrap_or_default();
    Ok(vm.ctx.new_str(text).into())
}

fn native_clipboard_set(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let text: String = if !args.args.is_empty() {
        args.args[0].clone().try_into_value(vm)?
    } else {
        return Err(vm.new_type_error("clipboard set requires a string value".to_string()));
    };
    xos_core::clipboard::set_contents(&text)
        .map_err(|e| vm.new_os_error(format!("clipboard set failed: {e}")))?;
    Ok(vm.ctx.none())
}

const CLIPBOARD_BOOTSTRAP: &str = r#"
class _ClipboardModuleDict(dict):
    def __init__(self, get_fn, set_fn):
        super().__init__()
        self._clipboard_get = get_fn
        self._clipboard_set = set_fn

    def __setitem__(self, name, value):
        if name == 'clipboard':
            self._clipboard_set(str(value))
            return
        super().__setitem__(name, value)

    def __getitem__(self, name):
        if name == 'clipboard':
            return self._clipboard_get()
        return super().__getitem__(name)

def _clipboard_wrap_module_dict(module):
    d = module.__dict__
    get_fn = d['__native_clipboard_get']
    set_fn = d['__native_clipboard_set']
    nd = _ClipboardModuleDict(get_fn, set_fn)
    skip = frozenset(('clipboard', '__native_clipboard_get', '__native_clipboard_set'))
    for k, v in d.items():
        if k not in skip:
            nd[k] = v
    return nd
"#;

pub fn install_clipboard(vm: &VirtualMachine, module: PyRef<PyModule>) {
    let _ = module.set_attr(
        "__native_clipboard_get",
        vm.new_function("__native_clipboard_get", native_clipboard_get),
        vm,
    );
    let _ = module.set_attr(
        "__native_clipboard_set",
        vm.new_function("__native_clipboard_set", native_clipboard_set),
        vm,
    );

    let scope = vm.new_scope_with_builtins();
    let _ = scope
        .globals
        .set_item("xos", module.as_object().to_owned(), vm);

    if let Err(e) = vm.run_code_string(
        scope.clone(),
        CLIPBOARD_BOOTSTRAP,
        "<xos/clipboard.py>".to_string(),
    ) {
        eprintln!(
            "xos clipboard bootstrap failed:\n{}",
            format_python_exception(vm, &e)
        );
        return;
    }

    let wrap_fn = match scope.globals.get_item("_clipboard_wrap_module_dict", vm) {
        Ok(f) => f,
        Err(_) => return,
    };
    let new_dict_obj = match wrap_fn.call((module.as_object().to_owned(),), vm) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "xos clipboard dict wrap failed:\n{}",
                format_python_exception(vm, &e)
            );
            return;
        }
    };
    let new_dict = match new_dict_obj.downcast::<PyDict>() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("xos clipboard: wrapped dict is not a dict");
            return;
        }
    };
    if module.as_object().set_dict(new_dict).is_err() {
        eprintln!("xos clipboard: module has no dict slot");
    }
}
