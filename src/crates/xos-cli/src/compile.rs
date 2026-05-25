//! CLI compile helpers: `xos compile` (release by default), `xos compile --no-release`, `--ios`, `--wasm`, `--java`.

use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const WASM_TARGET_DIR_NAME: &str = "wasm";
const WASM_MAIN_OUTPUT_DIR_NAME: &str = "main";
const WASM_ZIP_NAME: &str = "xos-wasm.zip";

fn canonical_repo_target_root(project_root: &Path) -> PathBuf {
    project_root.join("target")
}

fn canonical_repo_profile_dir(project_root: &Path, release: bool) -> PathBuf {
    canonical_repo_target_root(project_root).join(profile_dir_name(release))
}

fn windows_is_lock_contention(err: &io::Error) -> bool {
    #[cfg(windows)]
    {
        if let Some(code) = err.raw_os_error() {
            // 32: ERROR_SHARING_VIOLATION, 33: ERROR_LOCK_VIOLATION
            return code == 32 || code == 33;
        }
    }
    false
}

#[cfg(windows)]
fn windows_local_appdata_dir() -> PathBuf {
    if let Ok(v) = std::env::var("LOCALAPPDATA") {
        return PathBuf::from(v);
    }
    if let Ok(v) = std::env::var("USERPROFILE") {
        return PathBuf::from(v).join("AppData").join("Local");
    }
    PathBuf::from(r"C:\Users\Public\AppData\Local")
}

#[cfg(windows)]
fn windows_cargo_target_cache_root(lane: &str) -> PathBuf {
    windows_local_appdata_dir()
        .join("xos")
        .join("cargo-target")
        .join(lane)
}

#[cfg(windows)]
fn windows_fallback_target_root(lane: &str) -> PathBuf {
    windows_local_appdata_dir().join("xos").join("cargo-target").join(format!(
        "{}-fallback-{}",
        lane,
        std::process::id()
    ))
}

#[cfg(windows)]
fn choose_windows_target_root(base_root: PathBuf, lane: &str, release: bool) -> PathBuf {
    let lock_path = base_root.join(profile_dir_name(release)).join(".cargo-lock");
    if !lock_path.exists() {
        return base_root;
    }
    match fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&lock_path)
    {
        Ok(_) => base_root,
        Err(e) if windows_is_lock_contention(&e) => {
            let fallback = windows_fallback_target_root(lane);
            eprintln!(
                "⚠️  target lock busy at {} — using fallback build cache {}",
                lock_path.display(),
                fallback.display()
            );
            fallback
        }
        Err(_) => base_root,
    }
}

fn standard_build_target_root(project_root: &Path, release: bool) -> PathBuf {
    let base = standard_target_root(project_root);
    #[cfg(windows)]
    {
        return choose_windows_target_root(base, "standard", release);
    }
    #[cfg(not(windows))]
    {
        let _ = release;
        base
    }
}

fn java_build_target_root(project_root: &Path, release: bool) -> PathBuf {
    let base = java_target_root(project_root);
    #[cfg(windows)]
    {
        return choose_windows_target_root(base, "java", release);
    }
    #[cfg(not(windows))]
    {
        let _ = release;
        base
    }
}

#[cfg(windows)]
fn normalize_windows_path(path: PathBuf) -> PathBuf {
    let s = path.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        return PathBuf::from(rest);
    }
    path
}

#[cfg(not(windows))]
fn normalize_windows_path(path: PathBuf) -> PathBuf {
    path
}

/// Cargo `target` directory for native host builds (isolates caches from `--ios` / `--wasm` lanes).
pub fn standard_target_root(project_root: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        // App Control may block build-script executables under repository directories.
        // Build in LocalAppData and publish artifacts back into `<repo>/target/{debug|release}`.
        let _ = project_root;
        return windows_cargo_target_cache_root("standard");
    }

    #[cfg(not(windows))]
    {
    project_root.join("target").join("standard")
    }
}

/// Cargo `target` directory for Java/JNI builds (`xos-java` crate).
pub fn java_target_root(project_root: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        let _ = project_root;
        return windows_cargo_target_cache_root("java");
    }

    #[cfg(not(windows))]
    {
        project_root.join("target").join("java")
    }
}

fn profile_dir_name(release: bool) -> &'static str {
    if release {
        "release"
    } else {
        "debug"
    }
}

fn java_library_filename() -> &'static str {
    if cfg!(windows) {
        "xos_java.dll"
    } else if cfg!(target_os = "macos") {
        "libxos_java.dylib"
    } else {
        "libxos_java.so"
    }
}

fn java_library_artifact_path(project_root: &Path, release: bool) -> PathBuf {
    canonical_repo_profile_dir(project_root, release)
        .join(java_library_filename())
}

/// Canonical `xos` binary path under `<repo>/target/{debug|release}/`.
pub fn standard_xos_executable(project_root: &Path, release: bool) -> PathBuf {
    canonical_repo_profile_dir(project_root, release)
        .join(if cfg!(windows) { "xos.exe" } else { "xos" })
}

/// Release artifact (`target/standard/release/xos`).
pub fn release_xos_executable(project_root: &Path) -> PathBuf {
    standard_xos_executable(project_root, true)
}

/// Default Cargo `bin` directory (`xos`, `xpy` on PATH).
fn cargo_bin_dir_hint() -> String {
    if let Ok(cargo_home) = std::env::var("CARGO_HOME") {
        return Path::new(&cargo_home).join("bin").display().to_string();
    }
    #[cfg(windows)]
    if let Ok(userprofile) = std::env::var("USERPROFILE") {
        return Path::new(&userprofile)
            .join(".cargo")
            .join("bin")
            .display()
            .to_string();
    }
    #[cfg(not(windows))]
    if let Ok(home) = std::env::var("HOME") {
        return Path::new(&home)
            .join(".cargo")
            .join("bin")
            .display()
            .to_string();
    }
    "~/.cargo/bin".to_string()
}

fn installed_cli_executable(stem: &str) -> PathBuf {
    let name = if cfg!(windows) {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    };
    PathBuf::from(cargo_bin_dir_hint()).join(name)
}

/// Copy `src` over `dest`, replacing an existing file if present.
///
/// On **Windows**, a running `xos.exe` locks the file so plain `copy` fails (error 32 / 5). The usual
/// workaround is to **rename** the locked file (often allowed), then write the new binary to the
/// original path. The old process keeps running the renamed image; new shells pick up the update.
#[cfg(windows)]
fn copy_file_replace_windows(src: &Path, dest: &Path) -> io::Result<()> {
    const ERROR_ACCESS_DENIED: i32 = 5;
    const ERROR_SHARING_VIOLATION: i32 = 32;

    match fs::copy(src, dest) {
        Ok(_) => Ok(()),
        Err(e) => {
            let code = e.raw_os_error();
            let in_use = code == Some(ERROR_SHARING_VIOLATION) || code == Some(ERROR_ACCESS_DENIED);
            if !in_use || !dest.is_file() {
                return Err(e);
            }
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            let parent = dest
                .parent()
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "dest has no parent"))?;
            let stem = dest
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "dest file name"))?;
            let backup = parent.join(format!("{stem}.replaced-{stamp}.exe"));
            fs::rename(dest, &backup)?;
            fs::copy(src, dest)?;
            Ok(())
        }
    }
}

#[cfg(not(windows))]
fn copy_file_replace_windows(src: &Path, dest: &Path) -> io::Result<()> {
    // On macOS/Linux, avoid in-place overwrite of a running executable.
    // Writing directly to `dest` can truncate the file while the current
    // process image is still mapped. Copy to a sibling temp file first,
    // then atomically rename into place.
    let parent = dest
        .parent()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "dest has no parent"))?;
    let dest_name = dest
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "dest file name"))?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let tmp = parent.join(format!(".{dest_name}.tmp-{stamp}"));

    fs::copy(src, &tmp)?;
    fs::rename(&tmp, dest)?;
    Ok(())
}

fn copy_bins_from_profile_dir(profile_dir: &Path, dest_dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dest_dir)?;
    for stem in ["xos", "xpy", "xrs"] {
        let name = if cfg!(windows) {
            format!("{stem}.exe")
        } else {
            stem.to_string()
        };
        let from = profile_dir.join(&name);
        if !from.is_file() {
            continue;
        }
        let to = dest_dir.join(&name);
        if from == to {
            continue;
        }
        copy_file_replace_windows(&from, &to)?;
    }
    Ok(())
}

fn warn_copy_failed(label: &str, err: &io::Error) {
    eprintln!();
    eprintln!("⚠️  Compile succeeded, but could not update {label}: {err}");
}

fn warn_path_copy_failed(project_root: &Path, release: bool, err: &io::Error) {
    warn_copy_failed("PATH binaries", err);
    eprintln!(
        "   Fresh binaries (build dir): {}",
        standard_target_root(project_root).join(profile_dir_name(release)).display()
    );
    eprintln!(
        "   Canonical repo binaries: {}",
        canonical_repo_profile_dir(project_root, release).display()
    );
    eprintln!(
        "   Fix: close every running `xos` / `xpy` (and shells that started them), then run:"
    );
    eprintln!("   xos compile");
    eprintln!("   Or: cargo install --path {}", project_root.display());
}

fn run_cargo_build_verbose(project_root: &Path, target_root: &Path, release: bool) -> bool {
    let mut cmd = Command::new("cargo");
    cmd.current_dir(project_root);
    cmd.env("CARGO_TARGET_DIR", target_root.as_os_str());
    cmd.arg("build");
    if release {
        cmd.arg("--release");
    }
    cmd.args([
        "-p",
        "xos-cli",
        "--bin",
        "xos",
        "--bin",
        "xpy",
        "--bin",
        "xrs",
    ]);
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());
    cmd.status().map(|s| s.success()).unwrap_or(false)
}

/// `cargo build -p xos-cli --bins` with no compiler output — spinner line only.
fn run_cargo_build_quiet_spinner(project_root: &Path, target_root: &Path, release: bool) -> bool {
    let path_str = project_root.display().to_string();
    let profile_label = profile_dir_name(release);
    let mut cargo_cmd = Command::new("cargo");
    cargo_cmd.current_dir(project_root);
    cargo_cmd.env("CARGO_TARGET_DIR", target_root.as_os_str());
    cargo_cmd.arg("build");
    if release {
        cargo_cmd.arg("--release");
    }
    cargo_cmd.args([
        "-p",
        "xos-cli",
        "--bin",
        "xos",
        "--bin",
        "xpy",
        "--bin",
        "xrs",
    ]);
    cargo_cmd.stdout(Stdio::null());
    cargo_cmd.stderr(Stdio::piped());

    let mut child = match cargo_cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to spawn cargo: {e}");
            return false;
        }
    };

    let stderr = match child.stderr.take() {
        Some(s) => s,
        None => {
            eprintln!("cargo: stderr not piped");
            return false;
        }
    };

    let reader = thread::spawn(move || {
        let mut full = String::new();
        let mut r = BufReader::new(stderr);
        let mut line = String::new();
        loop {
            line.clear();
            match r.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => full.push_str(&line),
                Err(_) => break,
            }
        }
        full
    });

    const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
    let mut frame = 0usize;

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stderr_text = reader.join().unwrap_or_default();
                if status.success() {
                    print!(
                        "\r📁 Compiling xos ({profile_label}) in {}... ✓{}\n",
                        path_str,
                        " ".repeat(8)
                    );
                    let _ = io::stdout().flush();
                    return true;
                }
                if !stderr_text.is_empty() {
                    eprint!("{stderr_text}");
                }
                return false;
            }
            Ok(None) => {
                let ch = SPINNER[frame % SPINNER.len()];
                print!(
                    "\r📁 Compiling xos ({profile_label}) in {}... {}",
                    path_str, ch
                );
                let _ = io::stdout().flush();
                frame += 1;
                thread::sleep(Duration::from_millis(80));
            }
            Err(e) => {
                eprintln!("Failed to wait for cargo: {e}");
                let _ = reader.join();
                return false;
            }
        }
    }
}

/// Compile, then sync `target/standard/{debug|release}` → Cargo `bin` (what `xos compile` does).
///
/// - `quiet == false`: show `cargo` and copy status on stdout (verbose CLI).
/// - `quiet == true`: spinner only during compile; no copy banner; PATH warnings only if copy fails.
///
/// `None` = `cargo build` failed. `Some(true)` = copy ok. `Some(false)` = compile ok, copy failed.
fn run_compile_and_update_cargo_bin(
    project_root: &Path,
    release: bool,
    quiet: bool,
) -> Option<bool> {
    let path_str = project_root.display().to_string();
    let profile_label = profile_dir_name(release);

    if !quiet {
        println!(
            "📁 Compiling xos ({profile_label}) in {}...",
            path_str
        );
    }

    let build_target_root = standard_build_target_root(project_root, release);
    let build_profile_dir = build_target_root.join(profile_dir_name(release));
    if !quiet {
        println!("📦 Build cache: {}", build_profile_dir.display());
    }

    let compile_ok = if quiet {
        run_cargo_build_quiet_spinner(project_root, &build_target_root, release)
    } else {
        run_cargo_build_verbose(project_root, &build_target_root, release)
    };
    if !compile_ok {
        return None;
    }

    let source_profile_dir = build_profile_dir;
    let canonical_profile_dir = canonical_repo_profile_dir(project_root, release);

    let canonical_ok = match copy_bins_from_profile_dir(&source_profile_dir, &canonical_profile_dir) {
        Ok(()) => true,
        Err(e) => {
            warn_copy_failed("canonical repo binaries", &e);
            eprintln!("   Canonical path: {}", canonical_profile_dir.display());
            false
        }
    };

    if !quiet {
        println!("📁 Copying xos/xpy → {} ...", cargo_bin_dir_hint());
    }

    let dest = PathBuf::from(cargo_bin_dir_hint());
    let path_ok = match copy_bins_from_profile_dir(&source_profile_dir, &dest) {
        Ok(()) => true,
        Err(e) => {
            warn_path_copy_failed(project_root, release, &e);
            false
        }
    };

    Some(canonical_ok && path_ok)
}

pub fn find_project_root() -> PathBuf {
    match xos::find_xos_project_root() {
        Ok(p) => normalize_windows_path(p),
        Err(e) => {
            eprintln!("❌ Could not find xos project root: {e}");
            eprintln!("   Run this from inside your xos checkout, or use a `xos` binary built from that checkout.");
            std::process::exit(1);
        }
    }
}

/// Run `cargo clean` for each isolated target dir (`target/standard`, `target/ios`, `target/wasm`).
pub fn run_cargo_clean(project_root: &Path) -> bool {
    println!(
        "🧹 cargo clean (parallel target dirs) in {}...",
        project_root.display()
    );

    fn clean_target_dir(project_root: &Path, target_dir: &Path) -> Result<(), String> {
        let td = target_dir;
        if !td.exists() {
            return Ok(());
        }
        let status = Command::new("cargo")
            .args(["clean", "--target-dir"])
            .arg(td)
            .current_dir(project_root)
            .status()
            .map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!(
                "cargo clean --target-dir {} failed ({status}).",
                td.display()
            ));
        }
        Ok(())
    }

    match (|| -> Result<(), String> {
        let mut targets = vec![
            standard_target_root(project_root),
            java_target_root(project_root),
            project_root.join("target").join("ios"),
            project_root.join("target").join("wasm"),
        ];
        targets.sort();
        targets.dedup();
        for target_dir in &targets {
            clean_target_dir(project_root, target_dir)?;
        }
        Ok(())
    })() {
        Ok(()) => {
            println!("✅ cargo clean finished.");
            true
        }
        Err(e) => {
            eprintln!("❌ {e}");
            false
        }
    }
}

fn command_available(command: &str) -> bool {
    Command::new(command)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn rustup_target_installed(target: &str) -> bool {
    let output = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            stdout.lines().any(|line| line.trim() == target)
        }
        _ => false,
    }
}

fn write_wasm_index_html(output_dir: &Path) -> io::Result<()> {
    let index_html = output_dir.join("index.html");
    let html = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>xos wasm build</title>
  <style>
    html, body {
      margin: 0;
      height: 100%;
      background: #000;
      user-select: none;
      -webkit-user-select: none;
    }
    canvas {
      display: block;
    }
  </style>
</head>
<body>
  <canvas id="xos-canvas" width="256" height="256"></canvas>
  <script type="module">
    const params = new URLSearchParams(location.search);
    const stagedId = params.get("xpy_id");
    if (stagedId) {
      try {
        const base = `app_payloads/${encodeURIComponent(stagedId)}`;
        const [codeResp, flagsResp] = await Promise.all([
          fetch(`${base}/main.py`),
          fetch(`${base}/flags.txt`),
        ]);
        if (!codeResp.ok) {
          throw new Error(`main.py HTTP ${codeResp.status}`);
        }
        globalThis.__XOS_PYCODE__ = await codeResp.text();
        globalThis.__XOS_PYFLAGS__ = flagsResp.ok ? await flagsResp.text() : "";
      } catch (error) {
        console.error("xos wasm: failed to load staged python app", error);
        globalThis.__XOS_PYCODE__ = "";
        globalThis.__XOS_PYFLAGS__ = "";
      }
    }
    try {
      const wasm = await import("./pkg/xos_wasm.js");
      await wasm.default();
      await wasm.xos_launch();
      console.log("xos wasm: initialized");
    } catch (error) {
      console.error("xos wasm: failed to initialize", error);
    }
    window.addEventListener("contextmenu", (event) => event.preventDefault());
  </script>
</body>
</html>
"#;
    fs::write(index_html, html)
}

fn write_wasm_readme(output_dir: &Path) -> io::Result<()> {
    let readme = output_dir.join("README.txt");
    let text = "xos wasm output\n\nContents:\n- pkg/ (generated by wasm-pack)\n- index.html (simple web loader)\n- xos-wasm.zip (packaged output)\n\nRun locally:\n  xos app <app-name> --wasm\nOr:\n  cd target/wasm/main\n  python3 -m http.server 8080\nThen open http://localhost:8080/?app=ball\n";
    fs::write(readme, text)
}

fn zip_wasm_output(output_dir: &Path) -> bool {
    let zip_path = output_dir.join(WASM_ZIP_NAME);
    if zip_path.exists() && fs::remove_file(&zip_path).is_err() {
        eprintln!("❌ failed to remove existing zip: {}", zip_path.display());
        return false;
    }

    #[cfg(windows)]
    {
        let status = Command::new("powershell")
            .current_dir(output_dir)
            .args([
                "-NoProfile",
                "-Command",
                "Compress-Archive -Path pkg,index.html,README.txt -DestinationPath xos-wasm.zip -Force",
            ])
            .status();

        return match status {
            Ok(s) if s.success() => true,
            Ok(s) => {
                eprintln!("❌ zip packaging failed ({s}).");
                false
            }
            Err(e) => {
                eprintln!("❌ failed to run powershell for zip packaging: {e}");
                false
            }
        };
    }

    #[cfg(not(windows))]
    {
        if !command_available("zip") {
            eprintln!("❌ `zip` command not found. Install it and rerun `xos compile --wasm`.");
            return false;
        }

        let status = Command::new("zip")
            .current_dir(output_dir)
            .args(["-r", WASM_ZIP_NAME, "pkg", "index.html", "README.txt"])
            .status();

        match status {
            Ok(s) if s.success() => true,
            Ok(s) => {
                eprintln!("❌ zip packaging failed ({s}).");
                false
            }
            Err(e) => {
                eprintln!("❌ failed to run zip for packaging: {e}");
                false
            }
        }
    }
}

fn unique_wasm_staging_dir(wasm_target_dir: &Path) -> PathBuf {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    wasm_target_dir.join(format!(
        ".{}.staging-{}-{millis}",
        WASM_MAIN_OUTPUT_DIR_NAME,
        std::process::id()
    ))
}

fn publish_wasm_output(staging_dir: &Path, output_dir: &Path) -> bool {
    let backup_dir = output_dir.with_file_name(format!(
        ".{}.previous-{}",
        WASM_MAIN_OUTPUT_DIR_NAME,
        std::process::id()
    ));

    if backup_dir.exists() {
        if let Err(e) = fs::remove_dir_all(&backup_dir) {
            eprintln!(
                "❌ failed to remove old wasm backup {}: {e}",
                backup_dir.display()
            );
            return false;
        }
    }

    let had_previous = output_dir.exists();
    if had_previous {
        if let Err(e) = fs::rename(output_dir, &backup_dir) {
            eprintln!(
                "❌ failed to move previous wasm output {} aside: {e}",
                output_dir.display()
            );
            return false;
        }
    }

    if let Err(e) = fs::rename(staging_dir, output_dir) {
        eprintln!(
            "❌ failed to publish wasm output {}: {e}",
            output_dir.display()
        );
        if had_previous {
            if let Err(restore_err) = fs::rename(&backup_dir, output_dir) {
                eprintln!(
                    "❌ failed to restore previous wasm output {}: {restore_err}",
                    output_dir.display()
                );
            }
        }
        return false;
    }

    if had_previous {
        if let Err(e) = fs::remove_dir_all(&backup_dir) {
            eprintln!(
                "⚠️  published wasm output, but failed to remove backup {}: {e}",
                backup_dir.display()
            );
        }
    }

    true
}

/// Build WebAssembly output into `target/wasm/main/` and package it.
pub fn compile_wasm(clean: bool) -> bool {
    let project_root = find_project_root();
    let wasm_target_dir = project_root.join("target").join(WASM_TARGET_DIR_NAME);
    if clean && !run_cargo_clean(&project_root) {
        return false;
    }

    if !command_available("wasm-pack") {
        eprintln!("❌ `wasm-pack` not found. Install it by running `cargo install wasm-pack`.");
        return false;
    }
    if !command_available("rustup") {
        eprintln!("❌ `rustup` not found. Install rustup first: https://rustup.rs/");
        return false;
    }
    if !rustup_target_installed("wasm32-unknown-unknown") {
        eprintln!("❌ wasm target not installed: wasm32-unknown-unknown");
        eprintln!("   Run: rustup target add wasm32-unknown-unknown");
        return false;
    }
    if !command_available("zip") {
        eprintln!("❌ `zip` command not found.");
        eprintln!("   On Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y zip");
        return false;
    }

    println!("🕸️  Building wasm output...");
    eprintln!(
        "    Running wasm-pack with the same app-runtime build path used by `xos app --wasm`."
    );
    eprintln!(
        "    Artifact dir: {} (only one cargo build should use this at a time).",
        wasm_target_dir.display()
    );
    eprintln!(
        "    If this hangs on \"file lock\", cancel and run: pkill -f 'target/wasm.*cargo' ; pkill -f 'wasm-pack'"
    );

    let output_dir = wasm_target_dir.join(WASM_MAIN_OUTPUT_DIR_NAME);
    let staging_dir = unique_wasm_staging_dir(&wasm_target_dir);
    let pkg_dir = staging_dir.join("pkg");

    if staging_dir.exists() {
        if let Err(e) = fs::remove_dir_all(&staging_dir) {
            eprintln!(
                "❌ failed to clear staging directory {}: {e}",
                staging_dir.display()
            );
            return false;
        }
    }
    if let Err(e) = fs::create_dir_all(&staging_dir) {
        eprintln!(
            "❌ failed to create staging directory {}: {e}",
            staging_dir.display()
        );
        return false;
    }

    // Build the `xos-wasm` cdylib crate (not the root `xos` rlib used by the native CLI).
    let wasm_crate = project_root.join("src").join("crates").join("xos-wasm");
    let status = Command::new("wasm-pack")
        .current_dir(&project_root)
        .env("GAME_SELECTION", "ball")
        .env("CARGO_TARGET_DIR", &wasm_target_dir)
        .args([
            "build",
            wasm_crate.to_str().expect("wasm crate path is valid UTF-8"),
            "--target",
            "web",
            "--out-dir",
            &pkg_dir.display().to_string(),
            "--",
            "-p",
            "xos-wasm",
            "--no-default-features",
        ])
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            eprintln!("❌ wasm build failed ({s}).");
            let _ = fs::remove_dir_all(&staging_dir);
            return false;
        }
        Err(e) => {
            eprintln!("❌ failed to run wasm-pack: {e}");
            let _ = fs::remove_dir_all(&staging_dir);
            return false;
        }
    }

    if let Err(e) = write_wasm_index_html(&staging_dir) {
        eprintln!("❌ failed to write index.html: {e}");
        let _ = fs::remove_dir_all(&staging_dir);
        return false;
    }
    if let Err(e) = write_wasm_readme(&staging_dir) {
        eprintln!("❌ failed to write README.txt: {e}");
        let _ = fs::remove_dir_all(&staging_dir);
        return false;
    }

    if !zip_wasm_output(&staging_dir) {
        let _ = fs::remove_dir_all(&staging_dir);
        return false;
    }

    if !publish_wasm_output(&staging_dir, &output_dir) {
        let _ = fs::remove_dir_all(&staging_dir);
        return false;
    }

    println!("✅ wasm output: {}", output_dir.display());
    println!("✅ wasm zip: {}", output_dir.join(WASM_ZIP_NAME).display());
    true
}

/// Compile then copy into Cargo `bin`. `release`: `--release` (default true; use `--no-release` for debug). `verbose`: full `cargo` output.
/// With `clean`, runs [`run_cargo_clean`] first.
pub fn xos_compile_command(verbose: bool, clean: bool, release: bool) -> bool {
    let project_root = find_project_root();
    if clean && !run_cargo_clean(&project_root) {
        return false;
    }
    match run_compile_and_update_cargo_bin(&project_root, release, !verbose) {
        None => {
            eprintln!("❌ Compile failed. Exiting.");
            false
        }
        Some(path_updated) => {
            if verbose {
                let built_out = standard_xos_executable(&project_root, release);
                let installed_out = installed_cli_executable("xos");
                let profile_label = profile_dir_name(release);
                if path_updated {
                    println!(
                        "✅ Compile OK ({profile_label}). Installed CLI: {} (build artifact: {})",
                        installed_out.display(),
                        built_out.display()
                    );
                } else {
                    println!(
                        "✅ Compile OK ({profile_label}). Build artifact: {} (installed CLI update failed; see warning above)",
                        built_out.display()
                    );
                }
            }
            true
        }
    }
}

/// `clean`: run `cargo clean` before the iOS build script. `release`: pass `--release` to rustc.
pub fn compile_ios_rust(clean: bool, release: bool) -> bool {
    let profile_label = profile_dir_name(release);
    println!("🦀 Compiling Rust library for iOS ({profile_label})...");

    let project_root = find_project_root();
    let ios_target_dir = project_root.join("target").join("ios");
    if clean && !run_cargo_clean(&project_root) {
        return false;
    }

    let script_path = project_root
        .join("src")
        .join("crates")
        .join("xos-ios")
        .join("build-ios.sh");

    if !script_path.exists() {
        eprintln!("❌ build-ios.sh not found at: {}", script_path.display());
        return false;
    }

    let mut compile_cmd = Command::new("bash");
    compile_cmd.arg(&script_path);
    compile_cmd.current_dir(&project_root);
    // Keep iOS artifacts isolated so `xos compile --ios` can run concurrently
    // with non-iOS builds without contending on Cargo's target-dir lock.
    compile_cmd.env("CARGO_TARGET_DIR", ios_target_dir);
    compile_cmd.env("XOS_BUILD_RELEASE", if release { "1" } else { "0" });
    compile_cmd.stdout(Stdio::inherit());
    compile_cmd.stderr(Stdio::inherit());

    let status = compile_cmd
        .status()
        .expect("Failed to run src/crates/xos-ios/build-ios.sh");
    if !status.success() {
        eprintln!("❌ iOS compile failed. Exiting.");
        return false;
    }

    println!("✅ Rust library compiled successfully.");
    true
}

/// Build JNI dynamic library (`xos-java` crate) for Java host integrations.
pub fn compile_java(clean: bool, release: bool) -> bool {
    let profile_label = profile_dir_name(release);
    println!("☕ Compiling Rust JNI library for Java ({profile_label})...");

    let project_root = find_project_root();
    let java_target_dir = java_build_target_root(&project_root, release);
    if clean {
        println!("🧹 cargo clean --target-dir {} ...", java_target_dir.display());
        let clean_status = Command::new("cargo")
            .current_dir(&project_root)
            .args(["clean", "--target-dir"])
            .arg(&java_target_dir)
            .status();
        match clean_status {
            Ok(s) if s.success() => {}
            Ok(s) => {
                eprintln!("❌ Java clean failed ({s}).");
                return false;
            }
            Err(e) => {
                eprintln!("❌ failed to run cargo clean for Java target dir: {e}");
                return false;
            }
        }
    }

    let mut compile_cmd = Command::new("cargo");
    compile_cmd.current_dir(&project_root);
    compile_cmd.env("CARGO_TARGET_DIR", &java_target_dir);
    #[cfg(windows)]
    {
        // `xos-java` links through `ct2rs`; on Windows this lane often needs static CRT linkage.
        // Keep this scoped to `xos compile --java` so standard CLI compilation behavior stays unchanged.
        let mut rustflags = std::env::var("RUSTFLAGS").unwrap_or_default();
        if !rustflags.contains("target-feature=+crt-static") {
            if !rustflags.trim().is_empty() {
                rustflags.push(' ');
            }
            rustflags.push_str("-C target-feature=+crt-static");
        }
        compile_cmd.env("RUSTFLAGS", rustflags);
    }
    compile_cmd.arg("build");
    if release {
        compile_cmd.arg("--release");
    }
    compile_cmd.args(["-p", "xos-java"]);
    compile_cmd.stdout(Stdio::inherit());
    compile_cmd.stderr(Stdio::inherit());

    let status = match compile_cmd.status() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("❌ failed to run cargo build for xos-java: {e}");
            return false;
        }
    };
    if !status.success() {
        eprintln!("❌ Java/JNI compile failed. Exiting.");
        return false;
    }

    let built = java_target_dir
        .join(profile_dir_name(release))
        .join(java_library_filename());
    let published = java_library_artifact_path(&project_root, release);
    if built != published {
        if let Some(parent) = published.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!(
                    "❌ failed to prepare Java artifact directory {}: {e}",
                    parent.display()
                );
                return false;
            }
        }
        if let Err(e) = copy_file_replace_windows(&built, &published) {
            eprintln!(
                "❌ Java/JNI build succeeded, but publishing {} failed: {e}",
                published.display()
            );
            eprintln!("   Built artifact is at: {}", built.display());
            return false;
        }
    }

    println!(
        "✅ Java/JNI library built: {}",
        published.display()
    );
    true
}

/// CocoaPods step for the iOS app; used by [`compile_ios`].
#[allow(dead_code)]
pub fn compile_ios_swift() {
    println!("📦 Running pod install...");

    let project_root = find_project_root();
    let ios_dir = project_root.join("src").join("crates").join("xos-ios");

    if !ios_dir.exists() {
        eprintln!(
            "❌ src/crates/xos-ios directory not found at: {}",
            ios_dir.display()
        );
        std::process::exit(1);
    }

    let pod_script = ios_dir.join("pod-install.sh");
    let mut pod_cmd = if pod_script.exists() {
        let mut cmd = Command::new("bash");
        cmd.arg("./pod-install.sh");
        cmd
    } else {
        let mut cmd = Command::new("pod");
        cmd.arg("install");
        cmd.env("LANG", "en_US.UTF-8");
        cmd.env("LC_ALL", "en_US.UTF-8");
        cmd
    };

    pod_cmd.current_dir(&ios_dir);
    pod_cmd.stdout(Stdio::inherit());
    pod_cmd.stderr(Stdio::inherit());

    let pod_status = pod_cmd.status().expect("Failed to run pod install");
    if !pod_status.success() {
        eprintln!("⚠️  pod install failed.");
        eprintln!(
            "   You can manually run: cd {} && ./pod-install.sh",
            ios_dir.display()
        );
        std::process::exit(1);
    } else {
        println!("✅ Pod installation complete.");
    }
}

/// Rust static lib + `pod install` + next-step hints. For Rust-only, use [`compile_ios_rust`].
#[allow(dead_code)]
pub fn compile_ios() {
    if !compile_ios_rust(false, true) {
        std::process::exit(1);
    }
    compile_ios_swift();

    println!("📱 Next steps:");
    println!("   1. Open xos.xcworkspace in Xcode (or use: xed src/crates/xos-ios/)");
    println!("   2. Configure code signing in Xcode");
    println!("   3. Build and run on device or simulator");
}
