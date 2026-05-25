//! Wasm-pack builds this crate (`cdylib`). The native `xos` CLI links the root `xos` rlib.

use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn wasm_start() {
    console_error_panic_hook::set_once();
}

/// Initialize WebGPU (async) and start the selected xos app. Call after `default()` init.
#[wasm_bindgen]
pub async fn xos_launch() -> Result<(), JsValue> {
    xos_tensor::wgpu_init::ensure_initialized().await;
    xos::wasm_entry()
}
