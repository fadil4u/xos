//! Centralized Python-side device defaults.
//!
//! Keep policy in one place so future backend auto-selection is easy to update.

/// Default device label for newly constructed Python tensors when `device=` is omitted.
pub const DEFAULT_TENSOR_DEVICE: &str = "cpu";

#[inline]
pub const fn default_tensor_device() -> &'static str {
    DEFAULT_TENSOR_DEVICE
}
