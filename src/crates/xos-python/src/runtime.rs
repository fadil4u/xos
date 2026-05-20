///! Unified Python runtime for xos
///! Handles execution of Python code in both CLI and coder environments
///! with centralized logging and error handling
use rustpython_vm::{builtins::PyBaseExceptionRef, AsObject, Interpreter, VirtualMachine};
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// `--long-name` → `long_name`; only ASCII letters, digits, underscore after mapping.
pub(crate) fn cli_flag_to_snake_name(flag: &str) -> Option<String> {
    let stripped = flag.strip_prefix("--")?;
    if stripped.is_empty() || stripped.starts_with('-') {
        return None;
    }
    let name = stripped.replace('-', "_");
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return None;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return None;
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return None;
    }
    Some(name)
}

/// Build `xos.flags` setup: unknown attributes are `False`; listed names are `True`.
fn xos_flags_setup_python(true_flag_names: &[String]) -> String {
    let mut out = String::from(
        r#"

class _XosFlags:
    def __getattr__(self, name):
        return False

_xos_flags = _XosFlags()
"#,
    );
    for name in true_flag_names {
        out.push_str(&format!("setattr(_xos_flags, '{name}', True)\n"));
    }
    out.push_str("xos.flags = _xos_flags\n");
    out
}

/// Collect `--snake-style` args after the script path into flag names (`snake_style`).
pub fn parse_script_cli_flags(rest: &[String]) -> Vec<String> {
    rest.iter()
        .filter_map(|a| cli_flag_to_snake_name(a))
        .collect()
}

/// Callback type for capturing print output
pub type PrintCallback = Arc<dyn Fn(&str) + Send + Sync>;

#[cfg(target_arch = "wasm32")]
static WASM_PRINT_SINK: Mutex<Option<PrintCallback>> = Mutex::new(None);

#[cfg(target_arch = "wasm32")]
fn wasm_emit_print(s: &str) {
    if let Ok(guard) = WASM_PRINT_SINK.lock() {
        if let Some(cb) = guard.as_ref() {
            cb(s);
            return;
        }
    }
    xos_core::print(s);
}

/// Route Python `print` / `sys.stdout` to the browser console (or an optional sink).
#[cfg(target_arch = "wasm32")]
pub fn set_wasm_print_sink(callback: Option<PrintCallback>) {
    if let Ok(mut guard) = WASM_PRINT_SINK.lock() {
        *guard = callback;
    }
}

#[cfg(target_arch = "wasm32")]
const WASM_STDIO_SETUP: &str = r#"
import sys
import builtins

class _XosWasmIO:
    def write(self, s):
        if s:
            __xos_write__(s)
        return len(s) if s else 0
    def flush(self):
        pass

_io = _XosWasmIO()
sys.stdout = _io
sys.stderr = _io

def __xos_print__(*args, sep=' ', end='\n', **kwargs):
    __xos_write__(sep.join(str(arg) for arg in args) + end)

builtins.print = __xos_print__
"#;

/// Install `sys.stdout` / `builtins.print` on the VM (RustPython leaves them unset on wasm32).
#[cfg(target_arch = "wasm32")]
pub fn install_wasm_python_stdio(vm: &VirtualMachine) {
    if vm.builtins.get_attr("__xos_write__", vm).is_err() {
        let write_fn = vm.new_function(
            "__xos_write__",
            |args: rustpython_vm::function::FuncArgs, vm: &VirtualMachine| -> rustpython_vm::PyResult {
                if let Some(text_obj) = args.args.first() {
                    if let Ok(text) = text_obj.str(vm) {
                        wasm_emit_print(&text.to_string());
                    }
                }
                Ok(vm.ctx.none())
            },
        );
        let _ = vm.builtins.set_attr("__xos_write__", write_fn, vm);
    }

    let scope = vm.new_scope_with_builtins();
    if let Err(e) = vm.run_code_string(scope, WASM_STDIO_SETUP, "<wasm-stdio>".to_string()) {
        xos_core::print(&format!("xos wasm: failed to install Python stdio: {e:?}"));
    }
}

/// Call from `Interpreter::with_init` on wasm before running app scripts.
#[cfg(target_arch = "wasm32")]
pub fn wasm_interpreter_init(vm: &VirtualMachine) {
    install_wasm_python_stdio(vm);
}

/// How Python source is compiled before execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PythonRunMode {
    /// `exec` — scripts and multiline cells (no implicit expression printing).
    #[default]
    Exec,
    /// `single` — REPL cells; last expression uses `sys.displayhook` (prints non-`None`).
    Single,
}

/// Format a Python exception with traceback (like standard Python)
pub fn format_python_exception(vm: &VirtualMachine, py_exc: &PyBaseExceptionRef) -> String {
    let mut buf = String::new();
    if vm.write_exception(&mut buf, py_exc).is_ok() {
        return buf.trim_end().to_string();
    }
    let class_name = py_exc.class().name().to_string();
    let msg_result = vm
        .call_method(py_exc.as_object(), "__str__", ())
        .ok()
        .and_then(|result| result.str(vm).ok().map(|s| s.to_string()));
    match msg_result {
        Some(msg) if !msg.trim().is_empty() => format!("{}: {}", class_name, msg),
        _ => class_name,
    }
}

/// Execute Python code with optional print capture
/// Returns (result, output_text, app_instance)
pub fn execute_python_code(
    interpreter: &Interpreter,
    code: &str,
    filename: &str,
    persistent_scope: Option<rustpython_vm::scope::Scope>,
    print_callback: Option<PrintCallback>,
    script_flags: &[String],
) -> (
    Result<(), String>,
    String,
    Option<rustpython_vm::PyObjectRef>,
    Option<rustpython_vm::scope::Scope>,
) {
    execute_python_code_with_mode(
        interpreter,
        code,
        filename,
        persistent_scope,
        print_callback,
        script_flags,
        PythonRunMode::Exec,
    )
}

pub fn execute_python_code_with_mode(
    interpreter: &Interpreter,
    code: &str,
    filename: &str,
    persistent_scope: Option<rustpython_vm::scope::Scope>,
    print_callback: Option<PrintCallback>,
    script_flags: &[String],
    run_mode: PythonRunMode,
) -> (
    Result<(), String>,
    String,
    Option<rustpython_vm::PyObjectRef>,
    Option<rustpython_vm::scope::Scope>,
) {
    let output_buffer = Arc::new(Mutex::new(String::new()));
    let output_buffer_clone = Arc::clone(&output_buffer);

    let (result, app_instance, new_scope) = interpreter.enter(|vm| {
        #[cfg(target_arch = "wasm32")]
        {
            set_wasm_print_sink(print_callback.clone());
            install_wasm_python_stdio(vm);
        }

        // Clear previous app instance from builtins
        let _ = vm
            .builtins
            .as_object()
            .to_owned()
            .del_attr("__xos_app_instance__", vm);

        // Get or create persistent scope
        let scope = if let Some(existing_scope) = persistent_scope {
            existing_scope
        } else {
            let new_scope = vm.new_scope_with_builtins();
            let _ = new_scope
                .globals
                .set_item("__name__", vm.ctx.new_str("__main__").into(), vm);
            new_scope
        };

        // Make imports resolve relative to the executed file path, not process CWD.
        if !filename.starts_with('<') {
            let script_path = PathBuf::from(filename);
            if let Some(dir) = script_path.parent() {
                let dir_str = dir.to_string_lossy().to_string();
                let _ = scope.globals.set_item(
                    "__xos_script_dir__",
                    vm.ctx.new_str(dir_str.as_str()).into(),
                    vm,
                );
            }
            let _ = scope
                .globals
                .set_item("__file__", vm.ctx.new_str(filename).into(), vm);
        }

        // Set up print capture
        let buffer_for_capture = Arc::clone(&output_buffer_clone);
        #[cfg(not(target_arch = "wasm32"))]
        let callback_clone = print_callback.clone();
        let write_output_fn = vm.new_function(
            "__write_output__",
            move |args: rustpython_vm::function::FuncArgs,
                  _vm: &rustpython_vm::VirtualMachine|
                  -> rustpython_vm::PyResult {
                if let Some(text_obj) = args.args.first() {
                    if let Ok(text) = text_obj.str(_vm) {
                        let text_str = text.to_string();

                        // Write to buffer
                        if let Ok(mut buffer) = buffer_for_capture.lock() {
                            buffer.push_str(&text_str);
                        }

                        #[cfg(target_arch = "wasm32")]
                        wasm_emit_print(&text_str);
                        #[cfg(not(target_arch = "wasm32"))]
                        if let Some(ref callback) = callback_clone {
                            callback(&text_str);
                        }
                    }
                }
                Ok(_vm.ctx.none())
            },
        );
        scope
            .globals
            .set_item("__write_output__", write_output_fn.into(), vm)
            .ok();

        let setup_code = format!(
            r#"
import builtins
import sys
import xos
# Ensure `xos` is always present without an explicit user import
# (for `xpy` and `xos py/python` execution paths).
globals()["xos"] = xos
{}
__original_print__ = builtins.print
"#,
            xos_flags_setup_python(script_flags),
        );
        let setup_code = format!(
            "{}{}",
            setup_code,
            r#"
__original_import__ = builtins.__import__

try:
    __xos_script_dir__
except NameError:
    __xos_script_dir__ = None

if __xos_script_dir__:
    # Make sibling imports (e.g. `from data import Data`) work when
    # running `xpy path/to/train.py` from any current working directory.
    if __xos_script_dir__ not in sys.path:
        sys.path.insert(0, __xos_script_dir__)

def __xos_load_local_module__(module_name):
    if not __xos_script_dir__:
        raise ModuleNotFoundError(f"No module named '{module_name}'")
    source_path = __xos_script_dir__.rstrip("/\\") + "/" + module_name + ".py"
    source_path = source_path.replace("\\", "/")
    try:
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
    except Exception:
        raise ModuleNotFoundError(f"No module named '{module_name}'")
    module = type(sys)(module_name)
    module.__file__ = source_path
    module.__name__ = module_name
    module.__package__ = None
    sys.modules[module_name] = module
    exec(compile(source, source_path, "exec"), module.__dict__)
    return module

def __xos_import__(name, globals=None, locals=None, fromlist=(), level=0):
    try:
        return __original_import__(name, globals, locals, fromlist, level)
    except ModuleNotFoundError:
        # Fallback only for top-level local modules like `from data import Data`.
        if level == 0 and "." not in name and __xos_script_dir__:
            return __xos_load_local_module__(name)
        raise

def __custom_print__(*args, sep=' ', end='\n', **kwargs):
    output = sep.join(str(arg) for arg in args) + end
    __write_output__(output)

builtins.print = __custom_print__
xos.print = __custom_print__
builtins.__import__ = __xos_import__
"#
        );

        if let Err(e) = vm.run_code_string(scope.clone(), &setup_code, "<setup>".to_string()) {
            eprintln!("Failed to set up print capture: {:?}", e);
        }

        // Run the code
        let compile_mode = match run_mode {
            PythonRunMode::Exec => rustpython_vm::compiler::Mode::Exec,
            PythonRunMode::Single => rustpython_vm::compiler::Mode::Single,
        };
        let exec_result = match vm.compile(code, compile_mode, filename.to_string()) {
            Ok(code_obj) => vm.run_code_obj(code_obj, scope.clone()),
            Err(err) => Err(vm.new_syntax_error(&err, Some(code))),
        };

        // Restore original print (wasm keeps custom print + sys.stdout for tick()).
        #[cfg(not(target_arch = "wasm32"))]
        {
            let restore_code = r#"
builtins.print = __original_print__
xos.print = __original_print__
builtins.__import__ = __original_import__
"#;
            vm.run_code_string(scope.clone(), restore_code, "<restore>".to_string())
                .ok();
        }
        #[cfg(target_arch = "wasm32")]
        {
            let restore_code = r#"
builtins.__import__ = __original_import__
"#;
            vm.run_code_string(scope.clone(), restore_code, "<restore>".to_string())
                .ok();
        }

        // Handle errors
        let result = if let Err(py_exc) = exec_result {
            let error_text = format_python_exception(vm, &py_exc);
            Err(error_text)
        } else {
            Ok(())
        };

        // Check if an xos.Application was registered
        let app_instance = vm
            .get_attribute_opt(vm.builtins.as_object().to_owned(), "__xos_app_instance__")
            .ok()
            .flatten();

        (result, app_instance, scope)
    });

    let output = output_buffer.lock().unwrap().clone();
    (result, output, app_instance, Some(new_scope))
}

fn read_python_source(path: &PathBuf) -> Result<String, std::io::Error> {
    let content = fs::read_to_string(path)?;
    Ok(crate::frame_tensor::preprocess_tensor_logical_keywords(&content))
}

/// Run a Python file (CLI mode)
pub fn run_python_file(file_path: &PathBuf, script_flags: &[String]) {
    let resolved_file_path = file_path
        .canonicalize()
        .unwrap_or_else(|_| file_path.clone());
    // Read the Python file
    let code = match read_python_source(&resolved_file_path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!(
                "❌ Error reading file {}: {}",
                resolved_file_path.display(),
                e
            );
            std::process::exit(1);
        }
    };

    // Create interpreter with xos module
    let interpreter = Interpreter::with_init(Default::default(), |vm| {
        vm.add_native_module(
            "xos".to_owned(),
            Box::new(crate::xos_module::make_module),
        );
    });

    let print_cb: PrintCallback = Arc::new(|s: &str| {
        print!("{}", s);
        let _ = io::stdout().flush();
    });

    // Execute the code
    let (result, output, _, _) = execute_python_code(
        &interpreter,
        &code,
        &resolved_file_path.to_string_lossy(),
        None,
        Some(print_cb),
        script_flags,
    );

    // Handle errors
    if let Err(error_msg) = result {
        if !output.is_empty() {
            let _ = io::stdout().flush();
        }
        eprintln!("{}", error_msg);
        std::process::exit(1);
    }
}

fn collect_test_files(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_test_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("py") {
            out.push(path);
        }
    }
}

/// Discover `src/tests/**/*.py`, register `@xos.test` functions, run them. Returns process exit code.
pub fn run_test_suite(tests_dir: &std::path::Path) -> i32 {
    let mut files = Vec::new();
    collect_test_files(tests_dir, &mut files);
    files.sort();

    if files.is_empty() {
        eprintln!("no tests found under {}", tests_dir.display());
        return 1;
    }

    let interpreter = Interpreter::with_init(Default::default(), |vm| {
        vm.add_native_module(
            "xos".to_owned(),
            Box::new(crate::xos_module::make_module),
        );
    });

    interpreter.enter(|vm| {
        let scope = vm.new_scope_with_builtins();
        let _ = scope
            .globals
            .set_item("__name__", vm.ctx.new_str("__main__").into(), vm);
        let setup = "import xos";
        if let Err(e) = vm.run_code_string(scope.clone(), setup, "<test-setup>".to_string()) {
            eprintln!(
                "test setup failed:\n{}",
                format_python_exception(vm, &e)
            );
            return 1;
        }

        let mut load_failures = 0usize;
        for path in &files {
            let code = match fs::read_to_string(path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("failed to read {}: {e}", path.display());
                    load_failures += 1;
                    continue;
                }
            };
            let label = path.to_string_lossy().to_string();
            if let Err(e) = vm.run_code_string(scope.clone(), &code, label) {
                eprintln!(
                    "failed to load {}:\n{}",
                    path.display(),
                    format_python_exception(vm, &e)
                );
                load_failures += 1;
                continue;
            }
            if let Err(e) = vm.run_code_string(
                scope.clone(),
                "xos._register_module_tests(globals())",
                "<register-tests>".to_string(),
            ) {
                eprintln!(
                    "failed to register tests from {}:\n{}",
                    path.display(),
                    format_python_exception(vm, &e)
                );
                load_failures += 1;
            }
        }

        let xos_obj = match scope.globals.get_item("xos", vm) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("xos module missing: {}", format_python_exception(vm, &e));
                return 1;
            }
        };
        let run_all = match vm.get_attribute_opt(xos_obj, "_run_all") {
            Ok(Some(f)) => f,
            Ok(None) | Err(_) => {
                eprintln!("xos._run_all missing");
                return 1;
            }
        };
        let result = match run_all.call((), vm) {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "test run failed:\n{}",
                    format_python_exception(vm, &e)
                );
                return 1;
            }
        };
        let ok = result.try_into_value::<bool>(vm).unwrap_or(false);
        if load_failures > 0 || !ok {
            1
        } else {
            0
        }
    })
}

/// Run an interactive Python console (`xpy` / `xos py` with no script).
pub fn run_python_interactive() {
    let interpreter = Interpreter::with_init(Default::default(), |vm| {
        vm.add_native_module(
            "xos".to_owned(),
            Box::new(crate::xos_module::make_module),
        );
    });

    #[cfg(not(target_arch = "wasm32"))]
    crate::repl::run(&interpreter);

    #[cfg(target_arch = "wasm32")]
    {
        let _ = interpreter;
        eprintln!("❌ interactive python is not available on wasm");
        std::process::exit(1);
    }
}

/// Whether the registered `xos.Application` instance requests headless mode.
pub fn python_app_wants_headless(
    interpreter: &Interpreter,
    app_instance: &rustpython_vm::PyObjectRef,
) -> bool {
    interpreter.enter(|vm| {
        vm.get_attribute_opt(app_instance.clone(), "headless")
            .ok()
            .flatten()
            .and_then(|obj| obj.try_into_value::<bool>(vm).ok())
            .unwrap_or(false)
    })
}

/// Run a Python application with the xos engine
pub fn run_python_app(file_path: &PathBuf, script_flags: &[String]) {
    #[cfg(not(target_arch = "wasm32"))]
    use crate::engine::pyapp::PyApp;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::staged_native_python_app::{source_declares_headless_window_app, StagedNativePythonApp};
    let resolved_file_path = file_path
        .canonicalize()
        .unwrap_or_else(|_| file_path.clone());

    // Read the Python file
    let code = match read_python_source(&resolved_file_path) {
        Ok(content) => content,
        Err(e) => {
            eprintln!(
                "❌ Error reading file {}: {}",
                resolved_file_path.display(),
                e
            );
            std::process::exit(1);
        }
    };

    let print_cb: PrintCallback = Arc::new(|s: &str| {
        print!("{}", s);
        let _ = io::stdout().flush();
    });

    #[cfg(not(target_arch = "wasm32"))]
    {
        // Run the script first so `headless` / `device` on the Application instance are known.
        let interpreter = Interpreter::with_init(Default::default(), |vm| {
            vm.add_native_module(
                "xos".to_owned(),
                Box::new(crate::xos_module::make_module),
            );
        });

        let (result, output, app_instance, _) = execute_python_code(
            &interpreter,
            &code,
            &resolved_file_path.to_string_lossy(),
            None,
            Some(print_cb.clone()),
            script_flags,
        );

        if let Err(error_msg) = result {
            if !output.is_empty() {
                let _ = io::stdout().flush();
            }
            eprintln!("{}", error_msg);
            std::process::exit(1);
        }

        if let Some(app_instance) = app_instance {
            let headless = python_app_wants_headless(&interpreter, &app_instance);

            if headless {
                interpreter.enter(|vm| {
                    let _ = app_instance.set_attr("screen", vm.ctx.new_bool(true), vm);
                });
            }

            let pyapp = PyApp::new(interpreter, app_instance);
            let result = if headless {
                xos_core::engine::start_headless_native(Box::new(pyapp), 800, 600)
            } else {
                xos_core::engine::start_native(Box::new(pyapp))
            };
            if let Err(e) = result {
                eprintln!("❌ Engine error: {}", e);
                std::process::exit(1);
            }
            return;
        }

        // Scripts that defer app construction: window-first staged bootstrap (not headless).
        if !source_declares_headless_window_app(&code) {
            match xos_core::engine::start_native(Box::new(StagedNativePythonApp::new(
                resolved_file_path.clone(),
                code.clone(),
                script_flags.to_vec(),
                print_cb.clone(),
            ))) {
                Ok(()) => return,
                Err(e) => {
                    eprintln!("❌ Engine error: {e}");
                    std::process::exit(1);
                }
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        let interpreter = Interpreter::with_init(Default::default(), |vm| {
            vm.add_native_module(
                "xos".to_owned(),
                Box::new(crate::xos_module::make_module),
            );
        });

        let (result, output, app_instance, _) = execute_python_code(
            &interpreter,
            &code,
            &resolved_file_path.to_string_lossy(),
            None,
            Some(print_cb),
            script_flags,
        );

        if let Err(error_msg) = result {
            if !output.is_empty() {
                let _ = io::stdout().flush();
            }
            eprintln!("{}", error_msg);
            std::process::exit(1);
        }

        let _ = (interpreter, app_instance, output);
        eprintln!("❌ WASM not supported for Python apps yet");
        std::process::exit(1);
    }
}
