use rustpython_vm::{builtins::PyModule, function::FuncArgs, PyRef, PyResult, VirtualMachine};

fn to_py_err(
    vm: &VirtualMachine,
    msg: impl Into<String>,
) -> rustpython_vm::builtins::PyBaseExceptionRef {
    vm.new_runtime_error(msg.into())
}

#[cfg(all(not(target_arch = "wasm32"), feature = "llama_ct2"))]
fn llama_forward_native(args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    let av = args.args;
    let size: String = match av.first() {
        Some(v) => v.clone().try_into_value(vm)?,
        None => {
            return Err(vm.new_type_error(
                "_forward_native(size, prompt) requires size".to_string(),
            ));
        }
    };
    let prompt: String = match av.get(1) {
        Some(v) => v.clone().try_into_value(vm)?,
        None => {
            return Err(vm.new_type_error(
                "_forward_native(size, prompt) requires prompt".to_string(),
            ));
        }
    };
    let text = xos_core::ai::chat::ct2::llama::generate_once(&size, &prompt)
        .map_err(|e| to_py_err(vm, e))?;
    Ok(vm.ctx.new_str(text).into())
}

#[cfg(any(target_arch = "wasm32", not(feature = "llama_ct2")))]
fn llama_forward_native(_args: FuncArgs, vm: &VirtualMachine) -> PyResult {
    Err(to_py_err(
        vm,
        "LLaMA CT2 inference is unavailable in this build (enable the llama_ct2 feature)",
    ))
}

pub fn make_chat_module(vm: &VirtualMachine) -> PyRef<PyModule> {
    let chat = vm.new_module("xos.ai.chat", vm.ctx.new_dict(), None);
    let llama = vm.new_module("xos.ai.chat.llama", vm.ctx.new_dict(), None);

    llama
        .set_attr(
            "_forward_native",
            vm.new_function("_forward_native", llama_forward_native),
            vm,
        )
        .ok();

    let scope = vm.new_scope_with_builtins();
    if let Ok(forward_native) = llama.get_attr("_forward_native", vm) {
        scope
            .globals
            .set_item("_forward_native", forward_native, vm)
            .ok();
    }

    let glue = r#"
class _LlamaModel:
    """CTranslate2 backend: decoder-only LLaMA inference."""
    def __init__(self, size):
        self._size = size
    def forward(self, prompt):
        return _forward_native(self._size, str(prompt))

def load(size="7b-chat"):
    # size: "7b-chat" or "13b-chat"
    return _LlamaModel(size)
"#;
    if vm
        .run_code_string(scope.clone(), glue, "<xos.ai.chat.llama>".to_string())
        .is_ok()
    {
        if let Ok(load_fn) = scope.globals.get_item("load", vm) {
            llama.set_attr("load", load_fn, vm).ok();
        }
        if let Ok(cls) = scope.globals.get_item("_LlamaModel", vm) {
            llama.set_attr("LlamaModel", cls, vm).ok();
        }
    }

    chat.set_attr("llama", llama, vm).ok();
    chat
}
