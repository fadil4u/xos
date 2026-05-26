//! CTranslate2 LLaMA (`ct2rs`): load CT2 decoder-only model folders from `auth_data_dir()`
//! under `models/chat/llama/{size}/` — one-shot generation for Python `xos.ai.chat.llama`.
#![cfg(all(feature = "llama_ct2", not(target_arch = "wasm32")))]

use std::cell::RefCell;
use std::path::PathBuf;

use ct2rs::tokenizers::auto::Tokenizer as AutoTokenizer;
use ct2rs::{Config, GenerationOptions, Generator};

type Ct2Generator = Generator<AutoTokenizer>;

struct CachedLlamaModel {
    key: String,
    generator: Ct2Generator,
}

thread_local! {
    static LLAMA_MODEL_CACHE: RefCell<Option<CachedLlamaModel>> = const { RefCell::new(None) };
}

/// Canonical manifest key for a given size string.
fn ct2_manifest_key(size: &str) -> Result<&'static str, String> {
    match size.trim().to_ascii_lowercase().as_str() {
        "7b-chat" | "7b-chat-ct2" => Ok("7b-chat-ct2"),
        "13b-chat" | "13b-chat-ct2" => Ok("13b-chat-ct2"),
        other => Err(format!(
            "LLaMA CT2 supports '7b-chat' and '13b-chat' (got '{other}')."
        )),
    }
}

fn resolve_model_dir(size: &str) -> Result<PathBuf, String> {
    let key = ct2_manifest_key(size)?;
    let cache = xos_auth::auth_data_dir()
        .map_err(|e| e.to_string())?
        .join("models")
        .join("chat")
        .join("llama")
        .join(key);
    if super::llama_ensure::model_ready(&cache) {
        return Ok(cache);
    }
    super::llama_ensure::ensure_llama_artifacts(key, &cache)?;
    if super::llama_ensure::model_ready(&cache) {
        return Ok(cache);
    }
    Err(format!(
        "LLaMA CT2 setup failed for '{key}' under {}. \
         Check llama_ct2_download_links.json or your network connection.",
        cache.display()
    ))
}

fn with_cached_generator<T>(
    model_dir: &std::path::Path,
    f: impl FnOnce(&Ct2Generator) -> Result<T, String>,
) -> Result<T, String> {
    let key = model_dir.display().to_string();
    LLAMA_MODEL_CACHE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let needs_load = slot.as_ref().map(|m| m.key != key).unwrap_or(true);
        if needs_load {
            let mut cfg = Config::default();
            cfg.num_threads_per_replica = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .min(8);
            cfg.tensor_parallel = false;
            let generator = Ct2Generator::new(model_dir, &cfg)
                .map_err(|e| format!("LLaMA CT2 load {}: {e}", model_dir.display()))?;
            *slot = Some(CachedLlamaModel { key, generator });
        }
        let model = slot.as_ref().expect("LLaMA CT2 cache populated");
        let result = f(&model.generator);
        *slot = None; // drop generator so CT2 thread pool shuts down
        result
    })
}

/// One-shot generation for Python `xos.ai.chat.llama.load(size).forward(prompt)`.
pub fn generate_once(size: &str, prompt: &str) -> Result<String, String> {
    let dir = resolve_model_dir(size)?;
    let prompt = format!("<s>[INST] {prompt} [/INST]");
    with_cached_generator(&dir, |gen| {
        let opts = GenerationOptions {
            include_prompt_in_result: false,
            max_length: 128,
            ..GenerationOptions::default()
        };
        let results = gen
            .generate_batch(&[prompt.as_str()], &opts, None)
            .map_err(|e| format!("LLaMA CT2 generate: {e}"))?;
        Ok(results
            .into_iter()
            .next()
            .map(|(seqs, _)| seqs.join(""))
            .unwrap_or_default())
    })
}
