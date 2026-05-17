//! One-time Burn WGPU / cubecl setup. On wasm, must run [`ensure_initialized`] (async) before tensor ops.

#[cfg(target_arch = "wasm32")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_arch = "wasm32")]
static WGPU_READY: AtomicBool = AtomicBool::new(false);

/// Initialize the WebGPU runtime (required on wasm before `WgpuDevice` tensor ops).
#[cfg(target_arch = "wasm32")]
pub async fn ensure_initialized() {
    if WGPU_READY.load(Ordering::Acquire) {
        return;
    }
    use burn_wgpu::{graphics::WebGpu, init_setup_async, RuntimeOptions, WgpuDevice};
    let device = WgpuDevice::default();
    let _setup = init_setup_async::<WebGpu>(&device, RuntimeOptions::default()).await;
    WGPU_READY.store(true, Ordering::Release);
}

#[cfg(target_arch = "wasm32")]
#[inline]
pub fn is_initialized() -> bool {
    WGPU_READY.load(Ordering::Acquire)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn ensure_initialized() {
    use burn_wgpu::{graphics::AutoGraphicsApi, init_setup, RuntimeOptions, WgpuDevice};
    use std::sync::OnceLock;
    static INIT: OnceLock<()> = OnceLock::new();
    INIT.get_or_init(|| {
        let device = WgpuDevice::default();
        let _setup = init_setup::<AutoGraphicsApi>(&device, RuntimeOptions::default());
    });
}
