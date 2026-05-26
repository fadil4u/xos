//! Download pre-converted CT2 LLaMA weight files (loose on HuggingFace — no ZIP) into
//! `auth_data_dir()/models/chat/llama/{size}/` using URLs from
//! **`llama_ct2_download_links.json`** (Rust + `ureq` only — no Python).

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use serde::Deserialize;

const MANIFEST: &str = include_str!("llama_ct2_download_links.json");
const STREAM_BUF: usize = 256 * 1024; // 256 KB chunks — keeps RAM flat for multi-GB files

type Manifest = HashMap<String, Ct2HfSource>;

#[derive(Debug, Deserialize)]
struct Ct2HfSource {
    repo: String,
    files: Vec<String>,
}

/// Files required for [`ct2rs::Generator`] with LLaMA CT2 models.
pub(crate) fn model_ready(dir: &Path) -> bool {
    dir.join("model.bin").is_file()
        && dir.join("config.json").is_file()
        && dir.join("vocabulary.json").is_file()
        && dir.join("tokenizer.json").is_file()
}

/// Stream `url` directly to `dest` on disk — never buffers the full body in RAM.
/// Creates parent directories as needed. Writes to a `.part` sidecar first; renames on success.
pub(crate) fn download_file_to_dest(url: &str, dest: &Path) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("create_dir_all {}: {e}", parent.display()))?;
    }

    // Write to a .part file so interrupted downloads don't leave a truncated file that
    // model_ready() would skip on the next run.
    let part = dest.with_extension(
        dest.extension()
            .map(|e| format!("{}.part", e.to_string_lossy()))
            .unwrap_or_else(|| "part".to_string()),
    );
    if part.exists() {
        fs::remove_file(&part).ok();
    }

    let resp = ureq::get(url)
        .set("User-Agent", "xos-llama-ct2/1.0")
        .call()
        .map_err(|e| format!("GET {url}: {e}"))?;

    let mut reader = resp.into_reader();
    let mut file = fs::File::create(&part)
        .map_err(|e| format!("create {}: {e}", part.display()))?;

    let mut buf = vec![0u8; STREAM_BUF];
    let mut total = 0u64;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("read body {url}: {e}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| format!("write {}: {e}", part.display()))?;
        total += n as u64;
    }
    drop(file);

    if total == 0 {
        return Err(format!("download {url}: server returned an empty body"));
    }

    fs::rename(&part, dest)
        .map_err(|e| format!("rename {} → {}: {e}", part.display(), dest.display()))?;
    Ok(())
}

/// Download any missing files for `manifest_key` into `out_dir`.
/// Files already present on disk are skipped individually — safe to resume after interruption.
pub(crate) fn ensure_llama_artifacts(manifest_key: &str, out_dir: &Path) -> Result<(), String> {
    if model_ready(out_dir) {
        return Ok(());
    }

    let manifest: Manifest = serde_json::from_str(MANIFEST)
        .map_err(|e| format!("llama_ct2_download_links.json: {e}"))?;
    let entry = manifest.get(manifest_key).ok_or_else(|| {
        format!(
            "no entry '{manifest_key}' in \
             src/crates/xos-core/src/ai/chat/ct2/llama_ct2_download_links.json — \
             add a key with repo and files"
        )
    })?;

    fs::create_dir_all(out_dir)
        .map_err(|e| format!("create_dir_all {}: {e}", out_dir.display()))?;

    for filename in &entry.files {
        let dest = out_dir.join(filename);
        if dest.is_file() {
            continue;
        }
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            entry.repo, filename
        );
        eprintln!("[xos-llama-ct2] Downloading {filename}…");
        download_file_to_dest(&url, &dest)?;
    }

    if !model_ready(out_dir) {
        return Err(format!(
            "LLaMA CT2 download for '{manifest_key}' completed but required files are still \
             missing (need model.bin, config.json, tokenizer.json) under {}.",
            out_dir.display()
        ));
    }

    Ok(())
}
