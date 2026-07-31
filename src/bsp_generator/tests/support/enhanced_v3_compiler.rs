//! Phase 04 — Test-only compiler supervisor for Enhanced V3.
//!
//! Bounded test-only supervisor that wraps the reconciled pinned ericw-tools
//! executables. Provides SHA-256 hash verification, profile-driven compilation,
//! and staged output collection. This is test evidence only — it does not
//! become a production compiler driver or implement staging/provenance/package
//! behavior.
//!
//! # Constraints
//!
//! - Compiler unavailability is a blocked failing test, never `ignore` or skip.
//! - Every compiler warning is fatal.
//! - Output is drained concurrently under a shared byte ceiling.
//! - Process group is terminated on timeout or failure.
//! - All stages require BSP2 magic and valid QLIT v1 output.

#![allow(dead_code)]

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// ── Profile contract ──────────────────────────────────────────────────────

/// Pinned compiler profile for ericw-tools BSP2 generation.
#[derive(Debug, Clone)]
pub struct CompilerProfile {
    pub name: String,
    pub compiler_identity: String,
    pub required_version: String,
    pub qbsp_executable: String,
    pub vis_executable: String,
    pub light_executable: String,
    pub default_qbsp_args: Vec<String>,
    pub default_vis_args: Vec<String>,
    pub default_light_args: Vec<String>,
    pub timeout_seconds: u64,
    pub max_output_size: u64,
    pub expected_hashes: BTreeMap<String, String>,
}

/// Parse the pinned compiler profile from a TOML file.
pub fn parse_compiler_profile(path: &Path) -> Result<CompilerProfile, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read profile {}: {e}", path.display()))?;
    let table: toml::Table =
        toml::from_str(&text).map_err(|e| format!("invalid TOML in profile: {e}"))?;

    let string = |key: &str| {
        table
            .get(key)
            .and_then(|value| value.as_str())
            .map(str::to_owned)
            .ok_or_else(|| format!("profile missing '{key}'"))
    };
    let integer = |key: &str| {
        table
            .get(key)
            .and_then(|value| value.as_integer())
            .ok_or_else(|| format!("profile missing '{key}'"))
    };

    let timeout_seconds = u64::try_from(integer("timeout_seconds")?)
        .map_err(|_| "profile 'timeout_seconds' must be non-negative".to_string())?;
    let max_output_size = u64::try_from(integer("max_output_size")?)
        .map_err(|_| "profile 'max_output_size' must be non-negative".to_string())?;
    if timeout_seconds == 0 || max_output_size == 0 {
        return Err("profile timeout and output ceiling must be positive".to_string());
    }

    Ok(CompilerProfile {
        name: string("name")?,
        compiler_identity: string("compiler_identity")?,
        required_version: string("required_version")?,
        qbsp_executable: string("qbsp_executable")?,
        vis_executable: string("vis_executable")?,
        light_executable: string("light_executable")?,
        default_qbsp_args: parse_string_array(&table, "default_qbsp_args")?,
        default_vis_args: parse_string_array(&table, "default_vis_args")?,
        default_light_args: parse_string_array(&table, "default_light_args")?,
        timeout_seconds,
        max_output_size,
        expected_hashes: parse_expected_hashes(&table)?,
    })
}

fn parse_string_array(table: &toml::Table, key: &str) -> Result<Vec<String>, String> {
    table
        .get(key)
        .and_then(|value| value.as_array())
        .ok_or_else(|| format!("profile missing '{key}' array"))?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("'{key}' element is not a string"))
        })
        .collect()
}

fn parse_expected_hashes(table: &toml::Table) -> Result<BTreeMap<String, String>, String> {
    let hashes = table
        .get("expected_hashes")
        .and_then(|value| value.as_table())
        .ok_or("profile missing '[expected_hashes]'")?;
    hashes
        .iter()
        .map(|(key, value)| {
            let hash = value
                .as_str()
                .ok_or_else(|| format!("expected_hashes.{key} is not a string"))?;
            if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(format!("expected_hashes.{key} is not a SHA-256 hex value"));
            }
            Ok((key.clone(), hash.to_ascii_lowercase()))
        })
        .collect()
}

// ── Tool resolution ───────────────────────────────────────────────────────

/// Default tool directory for ericw-tools.
pub fn default_tool_dir() -> PathBuf {
    let home = env::var("HOME").unwrap_or_else(|_| "/home/dhickel".to_string());
    PathBuf::from(home).join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin")
}

/// Resolve tool directory from environment or default.
pub fn resolve_tool_dir() -> PathBuf {
    env::var("ERICW_TOOLS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_tool_dir())
}

/// Check whether all required tools are present.
pub fn tools_available(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

/// Compute SHA-256 hex digest of file contents.
pub fn sha256_file(path: &Path) -> Result<String, String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(sha256_hex(&data))
}

/// Verify executable hashes against the profile.
pub fn verify_executable_hashes(
    tool_dir: &Path,
    profile: &CompilerProfile,
) -> Result<(), Vec<String>> {
    let checks = [
        (&profile.qbsp_executable, "qbsp_sha256"),
        (&profile.vis_executable, "vis_sha256"),
        (&profile.light_executable, "light_sha256"),
    ];
    let failures: Vec<_> = checks
        .into_iter()
        .filter_map(|(executable, hash_key)| {
            let path = tool_dir.join(executable);
            let expected = match profile.expected_hashes.get(hash_key) {
                Some(expected) => expected,
                None => return Some(format!("profile missing expected hash '{hash_key}'")),
            };
            match sha256_file(&path) {
                Ok(actual) if &actual == expected => None,
                Ok(actual) => Some(format!(
                    "hash mismatch for {executable}: expected {expected}, got {actual}"
                )),
                Err(error) => Some(error),
            }
        })
        .collect();
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

// ── Diagnostics ───────────────────────────────────────────────────────────

/// Compiler diagnostic classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompilerDiagnostic {
    Warning { stage: String, message: String },
    Leak { stage: String, message: String },
    MissingMiptex { stage: String, message: String },
    SkippedFill { stage: String, message: String },
    Pointfile { stage: String, message: String },
}

impl CompilerDiagnostic {
    pub fn is_fatal(&self) -> bool {
        true
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Warning { message, .. }
            | Self::Leak { message, .. }
            | Self::MissingMiptex { message, .. }
            | Self::SkippedFill { message, .. }
            | Self::Pointfile { message, .. } => message,
        }
    }
}

/// Classify all prohibited compiler diagnostics case-insensitively.
pub fn classify_diagnostics(stage: &str, stdout: &str, stderr: &str) -> Vec<CompilerDiagnostic> {
    let mut diagnostics = Vec::new();
    for line in stdout.lines().chain(stderr.lines()) {
        let lower = line.to_ascii_lowercase();
        let diagnostic = if lower.contains("warning") && !lower.contains("0 warning") {
            Some(CompilerDiagnostic::Warning {
                stage: stage.to_string(),
                message: line.to_string(),
            })
        } else if lower.contains("leaked") || lower.contains("leak detected") {
            Some(CompilerDiagnostic::Leak {
                stage: stage.to_string(),
                message: line.to_string(),
            })
        } else if lower.contains("missing miptex")
            || lower.contains("missing texture")
            || lower.contains("unable to find texture")
            || lower.contains("could not load texture")
            || lower.contains("couldn't load texture")
        {
            Some(CompilerDiagnostic::MissingMiptex {
                stage: stage.to_string(),
                message: line.to_string(),
            })
        } else if lower.contains("no entities in empty space")
            || lower.contains("no filling performed")
            || (lower.contains("skipped") && lower.contains("fill"))
        {
            Some(CompilerDiagnostic::SkippedFill {
                stage: stage.to_string(),
                message: line.to_string(),
            })
        } else if lower.contains("pointfile") || lower.contains(".pts") {
            Some(CompilerDiagnostic::Pointfile {
                stage: stage.to_string(),
                message: line.to_string(),
            })
        } else {
            None
        };
        if let Some(diagnostic) = diagnostic {
            diagnostics.push(diagnostic);
        }
    }
    diagnostics
}

// ── Bounded stage runner ──────────────────────────────────────────────────

/// Output from a single compiler stage.
#[derive(Debug, Clone)]
pub struct StageOutput {
    pub stage: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub elapsed: Duration,
    pub diagnostics: Vec<CompilerDiagnostic>,
}

struct CapturedStream {
    bytes: Vec<u8>,
}

fn drain_stream(
    mut stream: impl Read,
    combined_bytes: Arc<AtomicU64>,
    combined_limit: u64,
    exceeded: Arc<AtomicBool>,
    reader_failed: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
) -> std::io::Result<CapturedStream> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let count = match stream.read(&mut buffer) {
            Ok(count) => count,
            Err(error) => {
                reader_failed.store(true, Ordering::Release);
                done.store(true, Ordering::Release);
                return Err(error);
            }
        };
        if count == 0 {
            done.store(true, Ordering::Release);
            return Ok(CapturedStream { bytes });
        }

        if exceeded.load(Ordering::Acquire) {
            continue;
        }
        let reservation =
            combined_bytes.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current
                    .checked_add(count as u64)
                    .filter(|next| *next <= combined_limit)
            });
        if reservation.is_ok() {
            bytes.extend_from_slice(&buffer[..count]);
        } else {
            exceeded.store(true, Ordering::Release);
        }
    }
}

#[cfg(unix)]
fn configure_process_group(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    unsafe {
        command.pre_exec(|| {
            if libc::setpgid(0, 0) == -1 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
}

#[cfg(not(unix))]
fn configure_process_group(_command: &mut Command) {}

fn terminate_process_group(child: &mut Child) {
    #[cfg(unix)]
    {
        let process_group = -(child.id() as i32);
        unsafe {
            libc::kill(process_group, libc::SIGKILL);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
    }
    let _ = child.wait();
}

fn join_reader(
    reader: thread::JoinHandle<std::io::Result<CapturedStream>>,
    stage: &str,
    stream: &str,
) -> Result<CapturedStream, String> {
    reader
        .join()
        .map_err(|_| format!("{stage}: {stream} reader panicked"))?
        .map_err(|error| format!("{stage}: failed to drain {stream}: {error}"))
}

/// Run one compiler tool in its own process group.
pub fn run_stage(
    tool_dir: &Path,
    exe_name: &str,
    args: &[String],
    work_dir: &Path,
    stage_name: &str,
    timeout: Duration,
    max_output: u64,
) -> Result<StageOutput, String> {
    let executable = tool_dir.join(exe_name);
    if !executable.is_file() {
        return Err(format!("tool not found: {}", executable.display()));
    }

    let mut command = Command::new(&executable);
    command
        .args(args)
        .current_dir(work_dir)
        .env_clear()
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(path) = env::var_os("PATH") {
        command.env("PATH", path);
    }
    if let Some(home) = env::var_os("HOME") {
        command.env("HOME", home);
    }
    configure_process_group(&mut command);

    let started = Instant::now();
    let mut child = command
        .spawn()
        .map_err(|error| format!("failed to spawn {stage_name}: {error}"))?;
    let stdout = child.stdout.take().ok_or("cannot capture stdout")?;
    let stderr = child.stderr.take().ok_or("cannot capture stderr")?;

    let combined_bytes = Arc::new(AtomicU64::new(0));
    let exceeded = Arc::new(AtomicBool::new(false));
    let reader_failed = Arc::new(AtomicBool::new(false));
    let stdout_done = Arc::new(AtomicBool::new(false));
    let stderr_done = Arc::new(AtomicBool::new(false));

    let stdout_reader = {
        let combined_bytes = Arc::clone(&combined_bytes);
        let exceeded = Arc::clone(&exceeded);
        let reader_failed = Arc::clone(&reader_failed);
        let done = Arc::clone(&stdout_done);
        thread::spawn(move || {
            drain_stream(
                stdout,
                combined_bytes,
                max_output,
                exceeded,
                reader_failed,
                done,
            )
        })
    };
    let stderr_reader = {
        let combined_bytes = Arc::clone(&combined_bytes);
        let exceeded = Arc::clone(&exceeded);
        let reader_failed = Arc::clone(&reader_failed);
        let done = Arc::clone(&stderr_done);
        thread::spawn(move || {
            drain_stream(
                stderr,
                combined_bytes,
                max_output,
                exceeded,
                reader_failed,
                done,
            )
        })
    };

    let status: Result<ExitStatus, String> = loop {
        if exceeded.load(Ordering::Acquire) {
            break Err(format!(
                "{stage_name}: combined stdout/stderr exceeded output ceiling of {max_output} bytes"
            ));
        }
        if reader_failed.load(Ordering::Acquire) {
            break Err(format!("{stage_name}: stdout/stderr reader failure"));
        }
        match child.try_wait() {
            Ok(Some(status))
                if stdout_done.load(Ordering::Acquire) && stderr_done.load(Ordering::Acquire) =>
            {
                break Ok(status);
            }
            Ok(_) => {}
            Err(error) => break Err(format!("{stage_name}: failed to poll child: {error}")),
        }
        if started.elapsed() >= timeout {
            break Err(format!("{stage_name}: timeout after {timeout:?}"));
        }
        thread::sleep(Duration::from_millis(5));
    };

    let status = match status {
        Ok(status) => status,
        Err(error) => {
            terminate_process_group(&mut child);
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            return Err(error);
        }
    };

    let stdout = join_reader(stdout_reader, stage_name, "stdout")?;
    let stderr = join_reader(stderr_reader, stage_name, "stderr")?;
    let stdout = String::from_utf8_lossy(&stdout.bytes).into_owned();
    let stderr = String::from_utf8_lossy(&stderr.bytes).into_owned();

    Ok(StageOutput {
        stage: stage_name.to_string(),
        diagnostics: classify_diagnostics(stage_name, &stdout, &stderr),
        stdout,
        stderr,
        exit_code: status.code().unwrap_or(-1),
        elapsed: started.elapsed(),
    })
}

// ── Full compiler pipeline ────────────────────────────────────────────────

/// Kinds of compilation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileFailureKind {
    Runner,
    NonZeroExit,
    Diagnostic,
    MissingProduct,
    InvalidProduct,
    LeakArtifact,
}

/// A compilation failure with stage outputs.
#[derive(Debug, Clone)]
pub struct CompileFailure {
    pub kind: CompileFailureKind,
    pub message: String,
    pub stage_outputs: Vec<StageOutput>,
}

/// Successfully compiled artifacts.
#[derive(Debug, Clone)]
pub struct CompiledArtifacts {
    pub bsp_data: Vec<u8>,
    pub bsp_sha256: String,
    pub lit_data: Vec<u8>,
    pub lit_sha256: String,
    pub qbsp_output: StageOutput,
    pub vis_output: StageOutput,
    pub light_output: StageOutput,
}

fn failure(
    kind: CompileFailureKind,
    message: impl Into<String>,
    stage_outputs: &[StageOutput],
) -> CompileFailure {
    CompileFailure {
        kind,
        message: message.into(),
        stage_outputs: stage_outputs.to_vec(),
    }
}

fn run_required_stage(
    tool_dir: &Path,
    executable: &str,
    args: &[String],
    work_dir: &Path,
    stage: &str,
    profile: &CompilerProfile,
    completed: &[StageOutput],
) -> Result<StageOutput, CompileFailure> {
    let output = run_stage(
        tool_dir,
        executable,
        args,
        work_dir,
        stage,
        Duration::from_secs(profile.timeout_seconds),
        profile.max_output_size,
    )
    .map_err(|message| failure(CompileFailureKind::Runner, message, completed))?;

    let mut outputs = completed.to_vec();
    outputs.push(output.clone());
    if output.exit_code != 0 {
        return Err(failure(
            CompileFailureKind::NonZeroExit,
            format!(
                "{stage} failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
                output.exit_code, output.stdout, output.stderr
            ),
            &outputs,
        ));
    }
    if !output.diagnostics.is_empty() {
        return Err(failure(
            CompileFailureKind::Diagnostic,
            format!(
                "{stage} emitted prohibited diagnostics: {:?}",
                output.diagnostics
            ),
            &outputs,
        ));
    }
    Ok(output)
}

fn read_regular_product(
    path: &Path,
    max_size: u64,
    completed: &[StageOutput],
) -> Result<Vec<u8>, CompileFailure> {
    let metadata = path.metadata().map_err(|error| {
        failure(
            CompileFailureKind::MissingProduct,
            format!("missing compiler product {}: {error}", path.display()),
            completed,
        )
    })?;
    if !metadata.is_file() {
        return Err(failure(
            CompileFailureKind::MissingProduct,
            format!("compiler product is not a regular file: {}", path.display()),
            completed,
        ));
    }
    if metadata.len() > max_size {
        return Err(failure(
            CompileFailureKind::InvalidProduct,
            format!(
                "compiler product {} exceeds {} byte ceiling",
                path.display(),
                max_size
            ),
            completed,
        ));
    }
    std::fs::read(path).map_err(|error| {
        failure(
            CompileFailureKind::MissingProduct,
            format!("read compiler product {}: {error}", path.display()),
            completed,
        )
    })
}

fn reject_leak_artifacts(work_dir: &Path, completed: &[StageOutput]) -> Result<(), CompileFailure> {
    for name in ["generated.pts", "generated.leak.prt"] {
        let path = work_dir.join(name);
        if path.exists() {
            return Err(failure(
                CompileFailureKind::LeakArtifact,
                format!("qbsp produced leak artifact {}", path.display()),
                completed,
            ));
        }
    }
    Ok(())
}

/// Compile a .map file through the full qbsp → vis → light pipeline.
pub fn compile_map(
    map_path: &Path,
    work_dir: &Path,
    tool_dir: &Path,
    wad_path: &Path,
    palette_path: &Path,
    profile: &CompilerProfile,
) -> Result<CompiledArtifacts, CompileFailure> {
    let copy_error = |what: &str, error| {
        failure(
            CompileFailureKind::MissingProduct,
            format!("stage {what}: {error}"),
            &[],
        )
    };
    std::fs::copy(map_path, work_dir.join("generated.map"))
        .map_err(|error| copy_error("map", error))?;
    let wad_name = wad_path.file_name().ok_or_else(|| {
        failure(
            CompileFailureKind::MissingProduct,
            "WAD path has no basename",
            &[],
        )
    })?;
    std::fs::copy(wad_path, work_dir.join(wad_name)).map_err(|error| copy_error("WAD", error))?;
    std::fs::copy(palette_path, work_dir.join("palette.lmp"))
        .map_err(|error| copy_error("palette", error))?;

    let mut completed = Vec::new();
    let qbsp_args = build_args(&profile.default_qbsp_args, "generated.map");
    let qbsp = run_required_stage(
        tool_dir,
        &profile.qbsp_executable,
        &qbsp_args,
        work_dir,
        "qbsp",
        profile,
        &completed,
    )?;
    completed.push(qbsp.clone());
    reject_leak_artifacts(work_dir, &completed)?;
    let bsp_path = work_dir.join("generated.bsp");
    let qbsp_product = read_regular_product(&bsp_path, profile.max_output_size, &completed)?;
    validate_bsp2(&qbsp_product)
        .map_err(|message| failure(CompileFailureKind::InvalidProduct, message, &completed))?;

    let vis_args = build_args(&profile.default_vis_args, "generated.bsp");
    let vis = run_required_stage(
        tool_dir,
        &profile.vis_executable,
        &vis_args,
        work_dir,
        "vis",
        profile,
        &completed,
    )?;
    completed.push(vis.clone());
    let vis_product = read_regular_product(&bsp_path, profile.max_output_size, &completed)?;
    validate_bsp2(&vis_product)
        .map_err(|message| failure(CompileFailureKind::InvalidProduct, message, &completed))?;

    let light_args = build_args(&profile.default_light_args, "generated.bsp");
    let light = run_required_stage(
        tool_dir,
        &profile.light_executable,
        &light_args,
        work_dir,
        "light",
        profile,
        &completed,
    )?;
    completed.push(light.clone());

    let bsp_data = read_regular_product(&bsp_path, profile.max_output_size, &completed)?;
    validate_bsp2(&bsp_data)
        .map_err(|message| failure(CompileFailureKind::InvalidProduct, message, &completed))?;
    let lit_data = read_regular_product(
        &work_dir.join("generated.lit"),
        profile.max_output_size,
        &completed,
    )?;
    validate_lit(&lit_data)
        .map_err(|message| failure(CompileFailureKind::InvalidProduct, message, &completed))?;
    validate_lit_payload(&bsp_data, &lit_data)
        .map_err(|message| failure(CompileFailureKind::InvalidProduct, message, &completed))?;

    Ok(CompiledArtifacts {
        bsp_sha256: sha256_hex(&bsp_data),
        lit_sha256: sha256_hex(&lit_data),
        bsp_data,
        lit_data,
        qbsp_output: qbsp,
        vis_output: vis,
        light_output: light,
    })
}

fn build_args(defaults: &[String], input: &str) -> Vec<String> {
    defaults
        .iter()
        .cloned()
        .chain(std::iter::once(input.to_string()))
        .collect()
}

fn validate_bsp2(data: &[u8]) -> Result<(), String> {
    match data.get(..4) {
        Some(b"BSP2") => Ok(()),
        Some(magic) => Err(format!("BSP magic mismatch: expected BSP2, got {magic:?}")),
        None => Err("BSP file too small for magic check".to_string()),
    }
}

fn validate_lit(data: &[u8]) -> Result<(), String> {
    if data.len() < 8 {
        return Err("LIT file too small for QLIT v1 header".to_string());
    }
    if data.get(..4) != Some(b"QLIT") {
        return Err("LIT magic mismatch: expected QLIT".to_string());
    }
    let version = u32::from_le_bytes(data[4..8].try_into().expect("slice length checked"));
    if version != 1 {
        return Err(format!("LIT version {version} not supported (expected 1)"));
    }
    if data.len() == 8 {
        return Err("LIT file has no RGB payload".to_string());
    }
    Ok(())
}

fn validate_lit_payload(bsp: &[u8], lit: &[u8]) -> Result<(), String> {
    const LIGHTMAP_LUMP: usize = 8;
    const LUMP_TABLE_START: usize = 4;
    const LUMP_ENTRY_SIZE: usize = 8;
    let length_offset = LUMP_TABLE_START + LIGHTMAP_LUMP * LUMP_ENTRY_SIZE + 4;
    let length_bytes = bsp
        .get(length_offset..length_offset + 4)
        .ok_or("BSP header too small for lightmap lump")?;
    let lightmap_len = u32::from_le_bytes(length_bytes.try_into().expect("slice length checked"));
    let expected = usize::try_from(lightmap_len)
        .ok()
        .and_then(|length| length.checked_mul(3))
        .ok_or("BSP lightmap length overflow")?;
    let actual = lit.len() - 8;
    if actual != expected {
        return Err(format!(
            "LIT RGB payload is {actual} bytes; expected {expected} for {lightmap_len} BSP luxels"
        ));
    }
    Ok(())
}

/// Compute SHA-256 hex digest.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// Create a staging directory for compiler output.
pub fn create_staging_dir(label: &str) -> Result<tempfile::TempDir, String> {
    tempfile::Builder::new()
        .prefix(&format!("enhanced-v3-{label}-"))
        .tempdir()
        .map_err(|error| format!("create staging dir for {label}: {error}"))
}

/// Parse the real compiler profile from its checked-in location.
pub fn load_compiler_profile() -> Result<CompilerProfile, String> {
    let profile_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml");
    parse_compiler_profile(&profile_path)
}

/// Resolve paths for the cc0_dungeon_v2 theme assets.
pub fn theme_paths() -> (PathBuf, PathBuf) {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let theme_dir = crate_dir.join("themes/cc0_dungeon_v2");
    let wad_path = theme_dir.join("cc0_dungeon_v2.wad");
    let palette_path = theme_dir.join("palette.lmp");
    (wad_path, palette_path)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_parse_valid() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml");
        let profile = parse_compiler_profile(&path).unwrap();
        assert_eq!(profile.name, "ericw-q1-bsp2-generated");
        assert_eq!(profile.required_version, "2.0.0-alpha3");
    }

    #[test]
    fn every_warning_is_fatal() {
        let diagnostics = classify_diagnostics("qbsp", "WARNING: brush bounds out of range", "");
        assert_eq!(diagnostics.len(), 1);
        assert!(matches!(diagnostics[0], CompilerDiagnostic::Warning { .. }));
        assert!(diagnostics.iter().all(CompilerDiagnostic::is_fatal));
    }

    #[test]
    fn zero_warnings_summary_is_not_a_warning() {
        assert!(classify_diagnostics("qbsp", "Completed with 0 warnings.", "").is_empty());
    }

    #[test]
    fn validate_bsp2_magic() {
        assert!(validate_bsp2(b"BSP2").is_ok());
        assert!(validate_bsp2(b"BSP1").is_err());
        assert!(validate_bsp2(b"NOT").is_err());
    }

    #[test]
    fn validate_lit_magic() {
        let mut lit = b"QLIT".to_vec();
        lit.extend_from_slice(&1_u32.to_le_bytes());
        lit.push(0);
        assert!(validate_lit(&lit).is_ok());
    }

    #[test]
    fn theme_paths_exist() {
        let (wad, palette) = theme_paths();
        assert!(wad.exists(), "WAD not found at {}", wad.display());
        assert!(
            palette.exists(),
            "palette not found at {}",
            palette.display()
        );
    }
}
