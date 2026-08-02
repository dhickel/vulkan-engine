//! Phase 05 — Convention compiler support.
//!
//! Reuses the Phase 04 compiler supervisor for ericw-tools 2.0.0-alpha3.
//! Provides convention-specific BSP inspection, strict reload, and
//! reporting owned by the EnhancedV3 Richness conventions tests.

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

// ═══════════════════════════════════════════════════════════════════════════
// Types re-exported / shared
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConventionStatus {
    Supported,
    Unsupported,
    Deferred,
}

impl ConventionStatus {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Supported => "SUPPORTED",
            Self::Unsupported => "UNSUPPORTED",
            Self::Deferred => "DEFERRED",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConventionRow {
    pub id: String,
    pub category: String,
    pub source_construction: String,
    pub expected_transformation: String,
    pub postcompile_witness: String,
    pub status: ConventionStatus,
    pub structural_equivalent: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ConventionReport {
    pub profile_name: String,
    pub compiler_version: String,
    pub tool_dir: PathBuf,
    pub entity_count: usize,
    pub rows: Vec<ConventionRow>,
    pub supported_count: usize,
    pub unsupported_count: usize,
    pub deferred_count: usize,
}

impl ConventionReport {
    pub fn new(profile_name: &str, compiler_version: &str, tool_dir: &Path) -> Self {
        Self {
            profile_name: profile_name.to_string(),
            compiler_version: compiler_version.to_string(),
            tool_dir: tool_dir.to_path_buf(),
            entity_count: 0,
            rows: Vec::new(),
            supported_count: 0,
            unsupported_count: 0,
            deferred_count: 0,
        }
    }

    pub fn add_row(&mut self, row: ConventionRow) {
        self.rows.push(row);
    }

    pub fn recompute(&mut self) {
        self.supported_count = self
            .rows
            .iter()
            .filter(|r| r.status == ConventionStatus::Supported)
            .count();
        self.unsupported_count = self
            .rows
            .iter()
            .filter(|r| r.status == ConventionStatus::Unsupported)
            .count();
        self.deferred_count = self
            .rows
            .iter()
            .filter(|r| r.status == ConventionStatus::Deferred)
            .count();
    }

    pub fn write(&self) -> Result<(), String> {
        let report_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join(".internal-dev/debug_reports/enhanced-v3-richness");
        std::fs::create_dir_all(&report_dir).map_err(|e| format!("create report dir: {e}"))?;
        let path = report_dir.join("conventions-report.json");
        let json = serde_json::json!({
            "profile": self.profile_name,
            "compiler_version": self.compiler_version,
            "tool_dir": self.tool_dir.to_string_lossy(),
            "entity_count": self.entity_count,
            "summary": {
                "supported": self.supported_count,
                "unsupported": self.unsupported_count,
                "deferred": self.deferred_count,
            },
            "rows": self.rows.iter().map(|row| serde_json::json!({
                "id": row.id,
                "category": row.category,
                "source_construction": row.source_construction,
                "expected_transformation": row.expected_transformation,
                "postcompile_witness": row.postcompile_witness,
                "status": row.status.as_str(),
                "structural_equivalent": row.structural_equivalent,
            })).collect::<Vec<_>>(),
        });
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&json).map_err(|e| format!("serialize report: {e}"))?,
        )
        .map_err(|e| format!("write report: {e}"))?;
        eprintln!("convention report written to {}", path.display());
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Reused compiler infrastructure (mirrors enhanced_v3_proof/compiler_support)
// ═══════════════════════════════════════════════════════════════════════════

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

#[derive(Debug, Clone)]
pub struct StageOutput {
    pub stage: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub elapsed: Duration,
    pub diagnostics: Vec<String>,
}

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

#[derive(Debug, Clone)]
pub struct StrictReloadFacts {
    pub profile: String,
    pub diagnostics: usize,
    pub entities: usize,
    pub faces: usize,
    pub planes: usize,
    pub nodes: usize,
    pub leaves: usize,
    pub solid_leaves: usize,
    pub empty_leaves: usize,
    pub clipnodes: usize,
    pub lightdata_bytes: usize,
}

// ── Profile parsing ──────────────────────────────────────────────────────

pub fn parse_compiler_profile(path: &Path) -> Result<CompilerProfile, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read profile: {e}"))?;
    let table: toml::Table = toml::from_str(&text).map_err(|e| format!("invalid TOML: {e}"))?;

    let string = |key: &str| {
        table
            .get(key)
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .ok_or_else(|| format!("missing '{key}'"))
    };
    let integer = |key: &str| {
        table
            .get(key)
            .and_then(|v| v.as_integer())
            .ok_or_else(|| format!("missing '{key}'"))
    };

    let timeout_seconds = u64::try_from(integer("timeout_seconds")?)
        .map_err(|_| "timeout_seconds must be non-negative".to_string())?;
    let max_output_size = u64::try_from(integer("max_output_size")?)
        .map_err(|_| "max_output_size must be non-negative".to_string())?;

    let expected_hashes = table
        .get("expected_hashes")
        .and_then(|v| v.as_table())
        .map(|h| {
            h.iter()
                .map(|(k, v)| {
                    v.as_str()
                        .map(str::to_ascii_lowercase)
                        .ok_or_else(|| format!("expected_hashes.{k} not a string"))
                        .map(|hash| (k.clone(), hash))
                })
                .collect::<Result<BTreeMap<_, _>, _>>()
        })
        .transpose()?
        .unwrap_or_default();

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
        expected_hashes,
    })
}

fn parse_string_array(table: &toml::Table, key: &str) -> Result<Vec<String>, String> {
    table
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        })
        .ok_or_else(|| format!("missing '{key}' array"))
}

pub fn load_compiler_profile() -> Result<CompilerProfile, String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml");
    parse_compiler_profile(&path)
}

// ── Tool resolution ──────────────────────────────────────────────────────

pub fn default_tool_dir() -> PathBuf {
    let home = env::var("HOME").unwrap_or_else(|_| "/home/dhickel".to_string());
    PathBuf::from(home).join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin")
}

pub fn resolve_tool_dir() -> PathBuf {
    env::var("ERICW_TOOLS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_tool_dir())
}

pub fn tools_available(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let data = std::fs::read(path).map_err(|e| format!("read {p}: {e}", p = path.display()))?;
    Ok(sha256_hex(&data))
}

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
        .filter_map(|(exe, key)| {
            let expected = profile.expected_hashes.get(key)?;
            match sha256_file(&tool_dir.join(exe)) {
                Ok(actual) if &actual == expected => None,
                Ok(actual) => Some(format!(
                    "hash mismatch for {exe}: expected {expected}, got {actual}"
                )),
                Err(e) => Some(e),
            }
        })
        .collect();
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

pub fn theme_paths() -> (PathBuf, PathBuf) {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let theme_dir = crate_dir.join("themes/cc0_dungeon_v2");
    (
        theme_dir.join("cc0_dungeon_v2.wad"),
        theme_dir.join("palette.lmp"),
    )
}

// ── Stage runner ─────────────────────────────────────────────────────────

pub fn create_staging_dir(label: &str) -> Result<tempfile::TempDir, String> {
    tempfile::Builder::new()
        .prefix(&format!("enhanced-v3-{label}-"))
        .tempdir()
        .map_err(|e| format!("create staging dir: {e}"))
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
            Ok(c) => c,
            Err(e) => {
                reader_failed.store(true, Ordering::Release);
                done.store(true, Ordering::Release);
                return Err(e);
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
                    .filter(|n| *n <= combined_limit)
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
        let pgid = -(child.id() as i32);
        unsafe {
            libc::kill(pgid, libc::SIGKILL);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
    }
    let _ = child.wait();
}

fn classify_diagnostics(stage: &str, stdout: &str, stderr: &str) -> Vec<String> {
    let mut diags = Vec::new();
    for line in stdout.lines().chain(stderr.lines()) {
        let lower = line.to_ascii_lowercase();
        if lower.contains("warning") && !lower.contains("0 warning") {
            diags.push(format!("{stage}: {line}"));
        } else if lower.contains("leaked") || lower.contains("leak detected") {
            diags.push(format!("{stage}: {line}"));
        } else if lower.contains("missing miptex")
            || lower.contains("missing texture")
            || lower.contains("unable to find texture")
        {
            diags.push(format!("{stage}: {line}"));
        } else if lower.contains("no entities in empty space")
            || lower.contains("no filling performed")
        {
            diags.push(format!("{stage}: {line}"));
        } else if lower.contains("pointfile") || lower.contains(".pts") {
            diags.push(format!("{stage}: {line}"));
        }
    }
    diags
}

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
        .map_err(|e| format!("spawn {stage_name}: {e}"))?;
    let stdout = child.stdout.take().ok_or("cannot capture stdout")?;
    let stderr = child.stderr.take().ok_or("cannot capture stderr")?;

    let combined_bytes = Arc::new(AtomicU64::new(0));
    let exceeded = Arc::new(AtomicBool::new(false));
    let reader_failed = Arc::new(AtomicBool::new(false));
    let stdout_done = Arc::new(AtomicBool::new(false));
    let stderr_done = Arc::new(AtomicBool::new(false));

    let stdout_reader = {
        let cb = Arc::clone(&combined_bytes);
        let ex = Arc::clone(&exceeded);
        let rf = Arc::clone(&reader_failed);
        let done = Arc::clone(&stdout_done);
        thread::spawn(move || drain_stream(stdout, cb, max_output, ex, rf, done))
    };
    let stderr_reader = {
        let cb = Arc::clone(&combined_bytes);
        let ex = Arc::clone(&exceeded);
        let rf = Arc::clone(&reader_failed);
        let done = Arc::clone(&stderr_done);
        thread::spawn(move || drain_stream(stderr, cb, max_output, ex, rf, done))
    };

    let status: Result<ExitStatus, String> = loop {
        if exceeded.load(Ordering::Acquire) {
            break Err(format!(
                "{stage_name}: combined stdout/stderr exceeded {max_output} bytes"
            ));
        }
        if reader_failed.load(Ordering::Acquire) {
            break Err(format!("{stage_name}: reader failure"));
        }
        match child.try_wait() {
            Ok(Some(s))
                if stdout_done.load(Ordering::Acquire) && stderr_done.load(Ordering::Acquire) =>
            {
                break Ok(s);
            }
            Ok(_) => {}
            Err(e) => break Err(format!("{stage_name}: poll child: {e}")),
        }
        if started.elapsed() >= timeout {
            break Err(format!("{stage_name}: timeout after {timeout:?}"));
        }
        thread::sleep(Duration::from_millis(5));
    };

    let status = match status {
        Ok(s) => s,
        Err(e) => {
            terminate_process_group(&mut child);
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            return Err(e);
        }
    };

    let stdout_cap = stdout_reader
        .join()
        .map_err(|_| format!("{stage_name}: stdout reader panicked"))?
        .map_err(|e| format!("{stage_name}: drain stdout: {e}"))?;
    let stderr_cap = stderr_reader
        .join()
        .map_err(|_| format!("{stage_name}: stderr reader panicked"))?
        .map_err(|e| format!("{stage_name}: drain stderr: {e}"))?;

    let stdout = String::from_utf8_lossy(&stdout_cap.bytes).into_owned();
    let stderr = String::from_utf8_lossy(&stderr_cap.bytes).into_owned();

    Ok(StageOutput {
        stage: stage_name.to_string(),
        diagnostics: classify_diagnostics(stage_name, &stdout, &stderr),
        stdout,
        stderr,
        exit_code: status.code().unwrap_or(-1),
        elapsed: started.elapsed(),
    })
}

// ── Compiler pipeline ────────────────────────────────────────────────────

pub fn compile_map(
    map_path: &Path,
    work_dir: &Path,
    tool_dir: &Path,
    wad_path: &Path,
    palette_path: &Path,
    profile: &CompilerProfile,
) -> Result<CompiledArtifacts, String> {
    std::fs::copy(map_path, work_dir.join("generated.map"))
        .map_err(|e| format!("copy map: {e}"))?;
    let wad_name = wad_path.file_name().ok_or("WAD path has no basename")?;
    std::fs::copy(wad_path, work_dir.join(wad_name)).map_err(|e| format!("copy WAD: {e}"))?;
    std::fs::copy(palette_path, work_dir.join("palette.lmp"))
        .map_err(|e| format!("copy palette: {e}"))?;

    let timeout = Duration::from_secs(profile.timeout_seconds);
    let max_out = profile.max_output_size;

    // qbsp
    let mut qbsp_args = profile.default_qbsp_args.clone();
    qbsp_args.push("generated.map".to_string());
    let qbsp = run_stage(
        tool_dir,
        &profile.qbsp_executable,
        &qbsp_args,
        work_dir,
        "qbsp",
        timeout,
        max_out,
    )?;
    if qbsp.exit_code != 0 {
        return Err(format!(
            "qbsp failed (exit {}):\n{}\n{}",
            qbsp.exit_code, qbsp.stdout, qbsp.stderr
        ));
    }
    if !qbsp.diagnostics.is_empty() {
        return Err(format!("qbsp diagnostics: {:?}", qbsp.diagnostics));
    }

    // vis
    let mut vis_args = profile.default_vis_args.clone();
    vis_args.push("generated.bsp".to_string());
    let vis = run_stage(
        tool_dir,
        &profile.vis_executable,
        &vis_args,
        work_dir,
        "vis",
        timeout,
        max_out,
    )?;
    if vis.exit_code != 0 {
        return Err(format!(
            "vis failed (exit {}):\n{}\n{}",
            vis.exit_code, vis.stdout, vis.stderr
        ));
    }
    if !vis.diagnostics.is_empty() {
        return Err(format!("vis diagnostics: {:?}", vis.diagnostics));
    }

    // light
    let mut light_args = profile.default_light_args.clone();
    light_args.push("generated.bsp".to_string());
    let light = run_stage(
        tool_dir,
        &profile.light_executable,
        &light_args,
        work_dir,
        "light",
        timeout,
        max_out,
    )?;
    if light.exit_code != 0 {
        return Err(format!(
            "light failed (exit {}):\n{}\n{}",
            light.exit_code, light.stdout, light.stderr
        ));
    }
    if !light.diagnostics.is_empty() {
        return Err(format!("light diagnostics: {:?}", light.diagnostics));
    }

    let bsp_path = work_dir.join("generated.bsp");
    let bsp_data = std::fs::read(&bsp_path).map_err(|e| format!("read BSP: {e}"))?;
    if bsp_data.len() < 4 || &bsp_data[..4] != b"BSP2" {
        return Err("output is not BSP2".to_string());
    }

    let lit_path = work_dir.join("generated.lit");
    let lit_data = std::fs::read(&lit_path).map_err(|e| format!("read LIT: {e}"))?;
    if lit_data.len() < 8 || &lit_data[..4] != b"QLIT" {
        return Err("output LIT is not QLIT v1".to_string());
    }

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

// ── Strict reload ────────────────────────────────────────────────────────

pub fn strict_reload_with_paths(
    bsp_data: &[u8],
    lit_data: &[u8],
    wad_path: &Path,
    palette_path: &Path,
) -> Result<(bsp::BspWorld, StrictReloadFacts), String> {
    let wad_name = wad_path
        .file_name()
        .ok_or("WAD has no basename")?
        .to_string_lossy()
        .into_owned();
    let options = bsp::LoadOptions {
        strict: true,
        palette: Some(std::fs::read(palette_path).map_err(|e| format!("read palette: {e}"))?),
        lit_data: Some(lit_data.to_vec()),
        wad_archives: vec![(
            wad_name,
            std::fs::read(wad_path).map_err(|e| format!("read WAD: {e}"))?,
        )],
        texture_overrides: Vec::new(),
        source_identity: "enhanced-v3-richness-convention".to_string(),
    };
    let world = bsp::BspLoader::load(bsp_data, &options)
        .map_err(|r| format!("strict reload failed: {r}"))?;
    if !world.diagnostics.is_empty() {
        return Err(format!(
            "strict reload emitted diagnostics: {:?}",
            world.diagnostics
        ));
    }
    let facts = StrictReloadFacts {
        profile: world.profile.tag().to_string(),
        diagnostics: world.diagnostics.len(),
        entities: world.entities.len(),
        faces: world.faces.len(),
        planes: world.planes.len(),
        nodes: world.nodes.len(),
        leaves: world.leaves.len(),
        solid_leaves: world.leaves.iter().filter(|l| l.contents == -2).count(),
        empty_leaves: world.leaves.iter().filter(|l| l.contents == -1).count(),
        clipnodes: world.clipnodes.len(),
        lightdata_bytes: world.lightmap_data.len(),
    };
    Ok((world, facts))
}
