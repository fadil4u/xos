//! Thread-local pointer to the active [`EngineState`] while Python `Application.tick()` runs.
//! Used by native `xos.ui` hooks (e.g. on-screen keyboard) that must mutate engine state from
//! callbacks invoked during `tick()`.
//!
//! # Safety
//! The pointer is valid only on the engine thread, only for the dynamic extent of
//! [`TickEngineStateGuard`], and must not alias an active `&mut EngineState` borrow in Rust.

use std::cell::{Cell, RefCell};

use xos_core::compute_device::ComputeDevice;
use xos_core::engine::EngineState;
use xos_tensor::BurnTensor;

thread_local! {
    static TICK_ENGINE: Cell<Option<*mut EngineState>> = const { Cell::new(None) };
    static TICK_COMPUTE_DEVICE: Cell<Option<ComputeDevice>> = const { Cell::new(None) };
    /// Latest `convolve(..., inplace=False)` GPU result; stays on device until materialized.
    static CONV_GPU_OUTPUT: RefCell<Option<BurnTensor<3>>> = const { RefCell::new(None) };
}

#[inline]
fn set_tick_engine_state(ptr: Option<*mut EngineState>) {
    TICK_ENGINE.with(|c| c.set(ptr));
}

#[inline]
fn set_tick_compute_device(dev: Option<ComputeDevice>) {
    TICK_COMPUTE_DEVICE.with(|c| c.set(dev));
}

/// Run `f` with the [`EngineState`] installed for the current Python `tick()`.
pub fn with_tick_engine_state_mut<T>(f: impl FnOnce(&mut EngineState) -> T) -> Option<T> {
    let p = TICK_ENGINE.with(|c| c.get())?;
    Some(unsafe { f(&mut *p) })
}

/// Tick or callback path (`on_screen_size_change`, etc.).
pub fn with_engine_state_mut<T>(f: impl FnOnce(&mut EngineState) -> T) -> Option<T> {
    if let Some(p) = TICK_ENGINE.with(|c| c.get()) {
        return Some(unsafe { f(&mut *p) });
    }
    with_callback_engine_state_mut(f)
}

/// Active app compute device during tick / resize callbacks.
pub fn engine_compute_device() -> Option<ComputeDevice> {
    TICK_COMPUTE_DEVICE.with(|c| c.get()).or_else(|| {
        CALLBACK_COMPUTE_DEVICE.with(|c| c.get())
    })
}

pub struct TickEngineStateGuard {
    _private: (),
}

impl TickEngineStateGuard {
    pub fn install(state: &mut EngineState) -> Self {
        set_tick_engine_state(Some(std::ptr::from_mut(state)));
        set_tick_compute_device(Some(state.compute_device));
        Self { _private: () }
    }
}

impl Drop for TickEngineStateGuard {
    fn drop(&mut self) {
        set_tick_engine_state(None);
        set_tick_compute_device(None);
        clear_conv_gpu_output();
    }
}

/// Store the most recent GPU conv output (replaces any prior output).
pub fn set_conv_gpu_output(tensor: BurnTensor<3>) {
    CONV_GPU_OUTPUT.with(|c| *c.borrow_mut() = Some(tensor));
}

pub fn clear_conv_gpu_output() {
    CONV_GPU_OUTPUT.with(|c| *c.borrow_mut() = None);
}

/// Shape `[height, width, channels]` of the stored conv tensor (if any).
pub fn conv_gpu_output_shape() -> Option<(usize, usize, usize)> {
    CONV_GPU_OUTPUT.with(|c| {
        let t = c.borrow();
        let t = t.as_ref()?;
        let [h, w, ch] = t.dims();
        Some((h, w, ch))
    })
}

/// One host readback of the stored conv tensor into packed RGBA `u8` (HWC).
pub fn materialize_conv_gpu_output_rgba_u8() -> Option<Vec<u8>> {
    CONV_GPU_OUTPUT.with(|c| {
        let t = c.borrow();
        let t = t.as_ref()?;
        let [h, w, c_ch] = t.dims();
        if c_ch < 1 {
            return None;
        }
        let data = t.clone().into_data();
        let s = data.as_slice::<f32>().ok()?;
        let pixels = h * w;
        let mut out = Vec::with_capacity(pixels * c_ch);
        for i in 0..pixels {
            let o = i * c_ch;
            for c in 0..c_ch {
                out.push(s[o + c].clamp(0., 255.) as u8);
            }
        }
        Some(out)
    })
}

// ---------------------------------------------------------------------------
// Callback path: `Application.on_events` + component dispatch (mouse / keys)
// ---------------------------------------------------------------------------

thread_local! {
    static CALLBACK_ENGINE: Cell<Option<*mut EngineState>> = const { Cell::new(None) };
    static CALLBACK_COMPUTE_DEVICE: Cell<Option<ComputeDevice>> = const { Cell::new(None) };
}

#[inline]
fn set_callback_engine(ptr: Option<*mut EngineState>) {
    CALLBACK_ENGINE.with(|c| c.set(ptr));
}

pub fn with_callback_engine_state_mut<T>(f: impl FnOnce(&mut EngineState) -> T) -> Option<T> {
    let p = CALLBACK_ENGINE.with(|c| c.get())?;
    Some(unsafe { f(&mut *p) })
}

pub struct CallbackEngineStateGuard {
    _private: (),
}

impl CallbackEngineStateGuard {
    pub fn install(state: &mut EngineState) -> Self {
        set_callback_engine(Some(std::ptr::from_mut(state)));
        CALLBACK_COMPUTE_DEVICE.with(|c| c.set(Some(state.compute_device)));
        Self { _private: () }
    }
}

impl Drop for CallbackEngineStateGuard {
    fn drop(&mut self) {
        set_callback_engine(None);
        CALLBACK_COMPUTE_DEVICE.with(|c| c.set(None));
    }
}
