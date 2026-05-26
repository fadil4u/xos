//! LLaMA via **CTranslate2** (`ct2rs`). Cache under `auth_data_dir()/models/chat/llama/{size}/`
//! (same as `xos path --data`); first use downloads files individually from HuggingFace.

pub mod llama;
mod llama_ensure;
