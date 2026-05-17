//! Burn-backed tensors for xos.
//!
//! Search-friendly layout:
//! - **`BurnTensor`** — `burn::tensor::Tensor` on [`XosBackend`] (ML ops, conv, GPU math).
//! - Python-facing CPU [`Tensor`] lives in `xos-python` (`tensor_buf`); frame RGBA uses [`xos_core::engine::FrameState`].

pub mod conv;

pub use burn::tensor::backend::Backend;
pub use burn::tensor::{ElementConversion, Int, Shape, TensorData};

/// Default backend: Burn WGPU (Metal on Apple, WebGPU in the browser, Vulkan/SPIR-V elsewhere).
#[cfg(any(
    all(not(target_os = "ios"), not(target_arch = "wasm32")),
    target_os = "ios",
    target_arch = "wasm32"
))]
pub use burn_wgpu::{Wgpu, WgpuDevice};

pub use conv::{conv2d, depthwise_conv2d};

/// Default backend and device for xos tensor ops.
pub type XosBackend = Wgpu;
pub type XosDevice = WgpuDevice;

/// Label for Python `tensor.device` after GPU-backed ops.
pub fn compute_device_label() -> &'static str {
    "gpu"
}

use burn::tensor::Float;

/// Burn float tensor on the default xos backend (use this name in Rust for file search).
pub type BurnTensor<const D: usize> = burn::tensor::Tensor<XosBackend, D, Float>;
/// Alias for float tensors using the default backend
pub type FloatTensor<const D: usize> = BurnTensor<D>;
/// Alias for int tensors using the default backend
pub type IntTensor<const D: usize> = burn::tensor::Tensor<XosBackend, D, Int>;
