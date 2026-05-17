//! GPU presentation pipeline cache (see [`crate::gpu_present`] and `render_pending_gpu_passes`).

#[cfg(not(target_arch = "wasm32"))]
use crate::gpu_present::GpuPresentCache;

/// Per-window GPU blit pipeline and params buffer.
pub struct RasterCache {
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) gpu_present: Option<GpuPresentCache>,
}

impl RasterCache {
    pub fn new() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            gpu_present: None,
        }
    }
}

impl Default for RasterCache {
    fn default() -> Self {
        Self::new()
    }
}
