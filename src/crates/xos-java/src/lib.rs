//! JNI host for [`xos::engine::Application`]. Native library: `xos_java` (`xos_java.dll` / `libxos_java.so` / `libxos_java.dylib`).
//!
//! Java API: `ai.xlate.xos.XosNative` in `../java/`. Build with `cargo build -p xos-java --release`.
//!
//! The engine is stored in [`thread_local`] storage: Minecraft must call these natives from the
//! **client thread** only (same as other rendering/input). That allows non-[`Send`] apps such as
//! [`CoderApp`] (RustPython is not `Send`).

use jni::objects::{GlobalRef, JClass, JObject, JObjectArray, JString, JValue};
use jni::sys::{jfloat, jint, jlong, jobject, jstring};
use jni::JNIEnv;
use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Once, OnceLock};
use xos::apps::coder::CoderApp;
use xos::apps::coder::coder::{
    clear_host_python_modules, register_host_python_module, set_host_binding_invoker,
};
use xos::engine::{
    apply_frame_view_zoom,
    f3_menu_handle_mouse_down, f3_menu_handle_mouse_move, f3_menu_handle_mouse_up, tick_f3_menu,
    tick_frame_delta, tick_frame_view_zoom, Application, CursorStyleSetter, EngineState, F3Menu,
    FrameState, KeyboardState, MouseState, SafeRegionBoundingRectangle,
};

thread_local! {
    static HOST: RefCell<Option<Host>> = RefCell::new(None);
}

static INIT_HOOKS_ONCE: Once = Once::new();
struct JavaHostBindingBridge {
    jvm: Arc<jni::JavaVM>,
    callback: GlobalRef,
}
static HOST_BINDING_BRIDGE: OnceLock<Mutex<Option<JavaHostBindingBridge>>> = OnceLock::new();

fn host_binding_bridge_slot() -> &'static Mutex<Option<JavaHostBindingBridge>> {
    HOST_BINDING_BRIDGE.get_or_init(|| Mutex::new(None))
}

fn invoke_registered_host_binding(
    module_name: &str,
    function_name: &str,
    arg0: &str,
) -> Result<Option<String>, String> {
    let (jvm, callback) = {
        let slot = host_binding_bridge_slot()
            .lock()
            .map_err(|_| "host binding bridge lock poisoned".to_string())?;
        let Some(bridge) = slot.as_ref() else {
            return Err("host binding callback not set".to_string());
        };
        (Arc::clone(&bridge.jvm), bridge.callback.clone())
    };

    let mut env = jvm
        .attach_current_thread()
        .map_err(|e| format!("failed to attach JVM thread: {e}"))?;

    let module_j = env
        .new_string(module_name)
        .map_err(|e| format!("failed to create module string: {e}"))?;
    let function_j = env
        .new_string(function_name)
        .map_err(|e| format!("failed to create function string: {e}"))?;
    let arg0_j = env
        .new_string(arg0)
        .map_err(|e| format!("failed to create arg0 string: {e}"))?;

    let result = env
        .call_method(
            callback.as_obj(),
            "invokeXosBinding",
            "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;",
            &[
                JValue::Object(&JObject::from(module_j)),
                JValue::Object(&JObject::from(function_j)),
                JValue::Object(&JObject::from(arg0_j)),
            ],
        )
        .map_err(|e| format!("host binding callback failed: {e}"))?;

    let obj = result
        .l()
        .map_err(|e| format!("invalid host binding callback return: {e}"))?;
    if obj.is_null() {
        return Ok(None);
    }
    let jstr = JString::from(obj);
    let s = env
        .get_string(&jstr)
        .map_err(|e| format!("failed to decode callback string return: {e}"))?
        .to_string_lossy()
        .to_string();
    Ok(Some(s))
}

struct Host {
    engine: EngineState,
    app: Box<dyn Application>,
    last_tick_instant: Option<std::time::Instant>,
    /// Increments once per successful `tick` (for Java UI: confirm the sim advances with Minecraft).
    tick_count: u64,
    /// Packed pixels for Minecraft `NativeImage.setPixelRGBA` (little-endian int per pixel: same as
    /// Minekov `packAbgr`). Filled after each `tick` so Java avoids per-pixel RGBA→ABGR work.
    minecraft_upload: Vec<u8>,
    /// Uniform alpha for the viewport texture (idle vs hover); set from Java before `tick`.
    minecraft_viewport_alpha: u8,
}

fn resize_minecraft_upload(minecraft_upload: &mut Vec<u8>, width: u32, height: u32) {
    let len = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    minecraft_upload.resize(len, 0);
}

/// RGBA8 → premultiply by source alpha, apply uniform `a_out`, pack as Minecraft `setPixelRGBA` int (LE).
fn pack_rgba_to_minecraft_native_image(
    rgba: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    a_out: u8,
) {
    let pixel_count = width * height;
    if rgba.len() != pixel_count * 4 || dst.len() != pixel_count * 4 {
        return;
    }
    let a = a_out as u32;
    for i in 0..pixel_count {
        let base = i * 4;
        let r = rgba[base] as u32;
        let g = rgba[base + 1] as u32;
        let b = rgba[base + 2] as u32;
        let a_in = rgba[base + 3] as u32;
        let rp = ((r * a_in + 127) / 255).min(255) as u32;
        let gp = ((g * a_in + 127) / 255).min(255) as u32;
        let bp = ((b * a_in + 127) / 255).min(255) as u32;
        let packed = (a << 24) | (bp << 16) | (gp << 8) | rp;
        dst[base..base + 4].copy_from_slice(&packed.to_le_bytes());
    }
}

fn throw(env: &mut JNIEnv, class: &str, msg: &str) {
    let _ = env.throw_new(class, msg);
}

fn ensure_xos_hooks() {
    INIT_HOOKS_ONCE.call_once(xos::init_hooks);
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_ping(env: JNIEnv, _class: JClass) -> jstring {
    match env.new_string("Hello from xos-java!") {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_setCoderScriptsDirectory(
    mut env: JNIEnv,
    _class: JClass,
    absolute_directory: JString,
) {
    let Ok(path_java) = env.get_string(&absolute_directory) else {
        throw(
            &mut env,
            "java/lang/IllegalArgumentException",
            "directory path must be a non-null string",
        );
        return;
    };

    let raw = path_java.to_string_lossy().trim().to_string();
    if raw.is_empty() {
        throw(
            &mut env,
            "java/lang/IllegalArgumentException",
            "directory path must not be empty",
        );
        return;
    }

    let dir = PathBuf::from(raw);
    if !dir.is_absolute() {
        throw(
            &mut env,
            "java/lang/IllegalArgumentException",
            "directory path must be absolute",
        );
        return;
    }
    if !dir.exists() || !dir.is_dir() {
        throw(
            &mut env,
            "java/lang/IllegalArgumentException",
            "directory path must exist and be a directory",
        );
        return;
    }

    let resolved = dir.canonicalize().unwrap_or(dir);
    std::env::set_var("XOS_CODER_DIR", resolved);
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_setHostBindingCallback(
    mut env: JNIEnv,
    _class: JClass,
    callback: jobject,
) {
    if callback.is_null() {
        if let Ok(mut slot) = host_binding_bridge_slot().lock() {
            *slot = None;
        }
        set_host_binding_invoker(None);
        return;
    }

    let callback_obj = unsafe { JObject::from_raw(callback) };
    let global_callback = match env.new_global_ref(&callback_obj) {
        Ok(r) => r,
        Err(e) => {
            throw(
                &mut env,
                "java/lang/RuntimeException",
                &format!("failed to create global callback ref: {e}"),
            );
            return;
        }
    };
    let jvm = match env.get_java_vm() {
        Ok(v) => v,
        Err(e) => {
            throw(
                &mut env,
                "java/lang/RuntimeException",
                &format!("failed to get JavaVM: {e}"),
            );
            return;
        }
    };
    if let Ok(mut slot) = host_binding_bridge_slot().lock() {
        *slot = Some(JavaHostBindingBridge {
            jvm: Arc::new(jvm),
            callback: global_callback,
        });
    }
    set_host_binding_invoker(Some(Arc::new(
        |module_name, function_name, arg0| {
            invoke_registered_host_binding(module_name, function_name, arg0)
        },
    )));
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_clearHostPythonModules(
    _env: JNIEnv,
    _class: JClass,
) {
    clear_host_python_modules();
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_registerHostPythonModule(
    mut env: JNIEnv,
    _class: JClass,
    module_name: JString,
    function_names: jobject,
) {
    let module_name = match env.get_string(&module_name) {
        Ok(v) => v.to_string_lossy().to_string(),
        Err(e) => {
            throw(
                &mut env,
                "java/lang/IllegalArgumentException",
                &format!("invalid module name: {e}"),
            );
            return;
        }
    };

    if function_names.is_null() {
        throw(
            &mut env,
            "java/lang/IllegalArgumentException",
            "functionNames array must not be null",
        );
        return;
    }
    let function_names_array = unsafe { JObjectArray::from_raw(function_names) };
    let len = match env.get_array_length(&function_names_array) {
        Ok(v) => v,
        Err(e) => {
            throw(
                &mut env,
                "java/lang/IllegalArgumentException",
                &format!("invalid functionNames array: {e}"),
            );
            return;
        }
    };
    let mut functions = Vec::new();
    for idx in 0..len {
        let obj = match env.get_object_array_element(&function_names_array, idx) {
            Ok(v) => v,
            Err(e) => {
                throw(
                    &mut env,
                    "java/lang/IllegalArgumentException",
                    &format!("failed reading functionNames[{idx}]: {e}"),
                );
                return;
            }
        };
        let jstr = JString::from(obj);
        let fname = match env.get_string(&jstr) {
            Ok(v) => v.to_string_lossy().to_string(),
            Err(e) => {
                throw(
                    &mut env,
                    "java/lang/IllegalArgumentException",
                    &format!("invalid function name at index {idx}: {e}"),
                );
                return;
            }
        };
        functions.push(fname);
    }

    if let Err(e) = register_host_python_module(&module_name, &functions) {
        throw(&mut env, "java/lang/IllegalArgumentException", &e);
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_init(
    mut env: JNIEnv,
    _class: JClass,
    width: jint,
    height: jint,
) {
    if width <= 0 || height <= 0 {
        throw(
            &mut env,
            "java/lang/IllegalArgumentException",
            "width and height must be positive",
        );
        return;
    }

    let init_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ensure_xos_hooks();
        HOST.with(|cell| {
            let mut guard = cell.borrow_mut();

            // Already initialized (e.g. Java called init again before resize): same as resize, do not re-run setup.
            if let Some(host) = guard.as_mut() {
                host.engine.resize_frame(width as u32, height as u32);
                resize_minecraft_upload(&mut host.minecraft_upload, width as u32, height as u32);
                let _ = host
                    .app
                    .on_screen_size_change(&mut host.engine, width as u32, height as u32);
                return;
            }

            let safe_region = SafeRegionBoundingRectangle::full_screen();
            let mut engine = EngineState {
                frame: FrameState::new(width as u32, height as u32, safe_region),
                compute_device: xos::compute_device::ComputeDevice::resolve_auto(None),
                mouse: MouseState {
                    x: 0.0,
                    y: 0.0,
                    dx: 0.0,
                    dy: 0.0,
                    is_left_clicking: false,
                    is_right_clicking: false,
                    style: CursorStyleSetter::new(),
                },
                keyboard: KeyboardState {
                    onscreen: xos::ui::onscreen_keyboard::OnScreenKeyboard::new(),
                    modifiers: xos::engine::KeyboardModifiers::default(),
                },
                f3_menu: F3Menu::new(),
                ui_scale_percent: 100,
                delta_time_seconds: 1.0 / 60.0,
                paused: false,
                pending_step_ticks: 0,
                paused_frame_snapshot_pending: false,
                frame_view_zoom: 1.0,
                frame_view_zoom_target: 1.0,
                frame_view_zoom_velocity: 0.0,
                frame_view_center_x: 0.5,
                frame_view_center_y: 0.5,
                f3_fps_label_override: None,
                embed_last_plain_click_screen: None,
                embed_synthetic_click_screen: None,
            };

            let mut app: Box<dyn Application> = Box::new(CoderApp::new());
            if let Err(e) = app.setup(&mut engine) {
                throw(
                    &mut env,
                    "java/lang/RuntimeException",
                    &format!("xos Application::setup failed: {e}"),
                );
                return;
            }

            let mu = vec![0; (width as usize) * (height as usize) * 4];
            *guard = Some(Host {
                engine,
                app,
                last_tick_instant: None,
                tick_count: 0,
                minecraft_upload: mu,
                minecraft_viewport_alpha: 153,
            });
        });
    }));

    if init_result.is_err() {
        throw(
            &mut env,
            "java/lang/RuntimeException",
            "xos-java init panicked; check native logs",
        );
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_shutdown(_env: JNIEnv, _class: JClass) {
    HOST.with(|cell| {
        cell.borrow_mut().take();
    });
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_tick(mut env: JNIEnv, _class: JClass) {
    let tick_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        HOST.with(|cell| {
            let mut guard = cell.borrow_mut();
            let Some(host) = guard.as_mut() else {
                throw(
                    &mut env,
                    "java/lang/IllegalStateException",
                    "xos-java not initialized; call init first",
                );
                return;
            };

            host.tick_count = host.tick_count.wrapping_add(1);
            if host.engine.paused {
                if host.engine.pending_step_ticks > 0 {
                    host.engine.pending_step_ticks = host.engine.pending_step_ticks.saturating_sub(1);
                    tick_frame_delta(&mut host.engine, &mut host.last_tick_instant);
                    host.app.tick(&mut host.engine);
                } else {
                    host.last_tick_instant = Some(std::time::Instant::now());
                }
            } else {
                tick_frame_delta(&mut host.engine, &mut host.last_tick_instant);
                host.app.tick(&mut host.engine);
            }

            tick_frame_view_zoom(&mut host.engine);
            apply_frame_view_zoom(&mut host.engine);

            // Same order as `native_engine`: draw the on-screen keyboard on top after the app tick.
            {
                let shape = host.engine.frame.shape();
                let height = shape[0] as u32;
                let width = shape[1] as u32;
                let mouse_x = host.engine.mouse.x;
                let mouse_y = host.engine.mouse.y;
                let mouse_dx = host.engine.mouse.dx;
                let mouse_dy = host.engine.mouse.dy;
                let mouse_left = host.engine.mouse.is_left_clicking;
                let mouse_right = host.engine.mouse.is_right_clicking;
                let safe_region = host.engine.frame.safe_region_boundaries.clone();
                let (buffer, keyboard) = {
                    let buffer_ptr = host.engine.frame.buffer_mut() as *mut [u8];
                    let keyboard_ptr: *mut xos::ui::onscreen_keyboard::OnScreenKeyboard =
                        &mut host.engine.keyboard.onscreen;
                    (unsafe { &mut *buffer_ptr }, unsafe { &mut *keyboard_ptr })
                };
                keyboard.tick(
                    buffer,
                    width,
                    height,
                    mouse_x,
                    mouse_y,
                    mouse_dx,
                    mouse_dy,
                    mouse_left,
                    mouse_right,
                    &safe_region,
                );
            }

            tick_f3_menu(&mut host.engine);

            let shape = host.engine.frame.shape();
            let w = shape[1];
            let h = shape[0];
            let src = host.engine.frame_buffer_mut();
            pack_rgba_to_minecraft_native_image(
                &src[..],
                &mut host.minecraft_upload,
                w,
                h,
                host.minecraft_viewport_alpha,
            );
        });
    }));

    if tick_result.is_err() {
        throw(
            &mut env,
            "java/lang/RuntimeException",
            "xos-java tick panicked; check native logs",
        );
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_getEngineTickCount(
    _env: JNIEnv,
    _class: JClass,
) -> jlong {
    HOST.with(|cell| {
        cell
            .borrow()
            .as_ref()
            .map(|h| h.tick_count as jlong)
            .unwrap_or(0)
    })
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_setMinecraftViewportAlpha(
    _env: JNIEnv,
    _class: JClass,
    alpha: jint,
) {
    let a = alpha.clamp(0, 255) as u8;
    HOST.with(|cell| {
        if let Some(host) = cell.borrow_mut().as_mut() {
            host.minecraft_viewport_alpha = a;
        }
    });
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_getFrameBuffer(
    mut env: JNIEnv,
    _class: JClass,
) -> jobject {
    HOST.with(|cell| {
        let mut guard = cell.borrow_mut();
        let Some(host) = guard.as_mut() else {
            throw(
                &mut env,
                "java/lang/IllegalStateException",
                "xos-java not initialized; call init first",
            );
            return std::ptr::null_mut();
        };

        let buffer = &mut host.minecraft_upload;
        let len = buffer.len();
        let ptr = buffer.as_mut_ptr().cast();

        // Safety: `ptr`/`len` refer to the Minecraft-packed upload buffer (same size as the frame).
        // Java must not use the direct buffer after `shutdown` or `resize` (which may reallocate).
        match unsafe { env.new_direct_byte_buffer(ptr, len) } {
            Ok(bb) => bb.into_raw(),
            Err(e) => {
                throw(
                    &mut env,
                    "java/lang/OutOfMemoryError",
                    &format!("new_direct_byte_buffer: {e}"),
                );
                std::ptr::null_mut()
            }
        }
    })
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_resize(
    mut env: JNIEnv,
    _class: JClass,
    width: jint,
    height: jint,
) {
    if width <= 0 || height <= 0 {
        throw(
            &mut env,
            "java/lang/IllegalArgumentException",
            "width and height must be positive",
        );
        return;
    }

    HOST.with(|cell| {
        let mut guard = cell.borrow_mut();
        let Some(host) = guard.as_mut() else {
            throw(
                &mut env,
                "java/lang/IllegalStateException",
                "xos-java not initialized; call init first",
            );
            return;
        };

        host.engine.resize_frame(width as u32, height as u32);
        resize_minecraft_upload(&mut host.minecraft_upload, width as u32, height as u32);
        let _ = host
            .app
            .on_screen_size_change(&mut host.engine, width as u32, height as u32);
    });
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_onMouseMove(
    mut env: JNIEnv,
    _class: JClass,
    x: jfloat,
    y: jfloat,
) {
    HOST.with(|cell| {
        let mut guard = cell.borrow_mut();
        let Some(host) = guard.as_mut() else {
            throw(
                &mut env,
                "java/lang/IllegalStateException",
                "xos-java not initialized; call init first",
            );
            return;
        };

        let prev_x = host.engine.mouse.x;
        let prev_y = host.engine.mouse.y;
        host.engine.mouse.dx = x - prev_x;
        host.engine.mouse.dy = y - prev_y;
        host.engine.mouse.x = x;
        host.engine.mouse.y = y;
        if !f3_menu_handle_mouse_move(&mut host.engine) {
            host.app.on_mouse_move(&mut host.engine);
        }
    });
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_onMouseDown(
    mut env: JNIEnv,
    _class: JClass,
    button: jint,
) {
    HOST.with(|cell| {
        let mut guard = cell.borrow_mut();
        let Some(host) = guard.as_mut() else {
            throw(
                &mut env,
                "java/lang/IllegalStateException",
                "xos-java not initialized; call init first",
            );
            return;
        };

        match button {
            0 => host.engine.mouse.is_left_clicking = true,
            1 => host.engine.mouse.is_right_clicking = true,
            _ => {}
        }
        if button == 0 {
            if !f3_menu_handle_mouse_down(&mut host.engine) {
                host.app.on_mouse_down(&mut host.engine);
            }
        } else {
            host.app.on_mouse_down(&mut host.engine);
        }
    });
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_onMouseUp(
    mut env: JNIEnv,
    _class: JClass,
    button: jint,
) {
    HOST.with(|cell| {
        let mut guard = cell.borrow_mut();
        let Some(host) = guard.as_mut() else {
            throw(
                &mut env,
                "java/lang/IllegalStateException",
                "xos-java not initialized; call init first",
            );
            return;
        };

        match button {
            0 => host.engine.mouse.is_left_clicking = false,
            1 => host.engine.mouse.is_right_clicking = false,
            _ => {}
        }
        if button == 0 {
            if !f3_menu_handle_mouse_up(&mut host.engine) {
                host.app.on_mouse_up(&mut host.engine);
            }
        } else {
            host.app.on_mouse_up(&mut host.engine);
        }
    });
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_onScroll(
    mut env: JNIEnv,
    _class: JClass,
    dx: jfloat,
    dy: jfloat,
) {
    HOST.with(|cell| {
        let mut guard = cell.borrow_mut();
        let Some(host) = guard.as_mut() else {
            throw(
                &mut env,
                "java/lang/IllegalStateException",
                "xos-java not initialized; call init first",
            );
            return;
        };

        host.app.on_scroll(&mut host.engine, dx, dy, xos::engine::ScrollWheelUnit::Pixel);
    });
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_onKeyChar(
    mut env: JNIEnv,
    _class: JClass,
    codepoint: jint,
) {
    HOST.with(|cell| {
        let mut guard = cell.borrow_mut();
        let Some(host) = guard.as_mut() else {
            throw(
                &mut env,
                "java/lang/IllegalStateException",
                "xos-java not initialized; call init first",
            );
            return;
        };

        let Ok(ch) = char::try_from(codepoint as u32) else {
            throw(
                &mut env,
                "java/lang/IllegalArgumentException",
                "invalid Unicode codepoint",
            );
            return;
        };

        host.app.on_key_char(&mut host.engine, ch);
    });
}

#[no_mangle]
pub extern "system" fn Java_ai_xlate_xos_XosNative_onF3(mut env: JNIEnv, _class: JClass) {
    HOST.with(|cell| {
        let mut guard = cell.borrow_mut();
        let Some(host) = guard.as_mut() else {
            throw(
                &mut env,
                "java/lang/IllegalStateException",
                "xos-java not initialized; call init first",
            );
            return;
        };

        host.engine.f3_menu.toggle_visible();
    });
}
