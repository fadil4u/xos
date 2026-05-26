//! Chat inference: LLaMA via **CT2** (`ct2rs`) when the `llama_ct2` Cargo feature is enabled.

#[cfg(all(feature = "llama_ct2", not(target_arch = "wasm32")))]
pub mod ct2;
