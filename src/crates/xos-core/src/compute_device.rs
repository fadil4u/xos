//! App compute device policy (`cpu` vs `gpu` frame/tensor backend).

/// Resolved compute backend for the active application.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeDevice {
    Cpu,
    Gpu,
}

impl ComputeDevice {
    pub const fn as_str(self) -> &'static str {
        match self {
            ComputeDevice::Cpu => "cpu",
            ComputeDevice::Gpu => "gpu",
        }
    }

    /// Parse an optional Python `device` preference (`None` → auto).
    pub fn parse_pref(s: Option<&str>) -> Result<Option<Self>, String> {
        match s.map(str::trim).filter(|s| !s.is_empty()) {
            None => Ok(None),
            Some("cpu") | Some("CPU") => Ok(Some(Self::Cpu)),
            Some("gpu") | Some("GPU") => Ok(Some(Self::Gpu)),
            Some(other) => Err(format!(
                "invalid device '{other}' (use 'cpu', 'gpu', or None for auto)"
            )),
        }
    }

    /// Auto: GPU (WGPU / Metal / WebGPU) on all hosts; use `Some(Cpu)` to force CPU staging paths.
    pub fn resolve_auto(pref: Option<Self>) -> Self {
        match pref {
            Some(d) => d,
            None => ComputeDevice::Gpu,
        }
    }

    #[inline]
    pub fn gpu_present_enabled(self) -> bool {
        matches!(self, ComputeDevice::Gpu)
    }
}
