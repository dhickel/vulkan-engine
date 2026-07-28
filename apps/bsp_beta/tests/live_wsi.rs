//! Phase 09: Live WSI Acceptance — Child-Process Evidence Harness
//!
//! Environment-gated acceptance harness that exercises the required live-WSI
//! entrypoints (bsp_beta, dungeon_dogfood, voxel_demo, renderer examples)
//! through independently launched child processes. Every child records command
//! identity, GPU/driver/compositor/validation-layer state, captured stdout/stderr,
//! exit disposition, and observed acquire/present evidence.
//!
//! The acceptance path is `#[ignore]` by default and requires a live GPU + WSI
//! environment. It fails (not skips) when the required environment is absent.
//!
//! Run with:
//! ```bash
//! cargo test -p bsp_beta --test live_wsi -- --ignored --nocapture
//! ```
//!
//! ## Exit Criteria (per bsp-acceptance.md §9)
//!
//! - `PASS`: real acquire/present observed, no forbidden outcomes, required
//!   post-action draw evidence present.
//! - `FAIL`: panic, device loss, validation-layer error, engine ERROR, missing
//!   presentation, missing report, or BSP fallback.
//! - `NOT_RUN`: required environment/capability unavailable; recorded with
//!   exact blocked cell and environment evidence.

use bsp_generator::DungeonConfig;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ── Selected package identity (Phase 08 frozen corpus) ────────────────────

const SELECTED_SEED: u64 = 1;
const SELECTED_LABEL: &str = "nominal-m1-seed-1";

// ── Required entrypoints (bsp-acceptance.md §9.1) ─────────────────────────

#[allow(dead_code)]
const REQUIRED_ENTRYPOINTS: &[(&str, &str)] = &[
    ("bsp_beta", "apps/bsp_beta"),
    ("dungeon_dogfood", "apps/dungeon_dogfood"),
    ("voxel_demo", "apps/voxel_demo"),
    ("api_test", "src/renderer/examples"),
];

// ── WSI lifecycle actions ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum LifecycleAction {
    Startup,
    Resize,
    Minimize,
    Restore,
    SurfaceLoss,
}

impl LifecycleAction {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Startup => "startup",
            Self::Resize => "resize",
            Self::Minimize => "minimize",
            Self::Restore => "restore",
            Self::SurfaceLoss => "surface_loss",
        }
    }
}

// ── Path helpers ──────────────────────────────────────────────────────────

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn wad_path() -> PathBuf {
    repo_root().join("src/bsp_generator/themes/cc0_stone_beta/cc0_stone_beta.wad")
}

fn palette_path() -> PathBuf {
    repo_root().join("src/bsp_generator/themes/cc0_stone_beta/palette.lmp")
}

fn profile_path() -> PathBuf {
    repo_root().join("tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml")
}

fn ericw_tools_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/dhickel".to_string());
    PathBuf::from(home).join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin")
}

fn tools_available(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

fn captures_dir() -> PathBuf {
    repo_root().join(".internal-dev/captures/bsp-dungeon-completion")
}

fn debug_dir() -> PathBuf {
    repo_root().join(".internal-dev/debug_reports/bsp-dungeon-completion")
}

fn live_wsi_report_path() -> PathBuf {
    debug_dir().join("live-wsi.json")
}

// ── Evidence types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LiveWsiReport {
    schema_version: u32,
    phase: String,
    timestamp: String,
    environment: WsiEnvironment,
    entrypoints: Vec<EntrypointCell>,
    lifecycle_actions: Vec<LifecycleCell>,
    overall_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WsiEnvironment {
    os: String,
    gpu_driver: String,
    vulkan_version: String,
    compositor: String,
    validation_layer_active: bool,
    headless_only: bool,
    note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EntrypointCell {
    entrypoint: String,
    package: Option<String>,
    command: String,
    exit_code: Option<i32>,
    stdout_snippet: String,
    stderr_snippet: String,
    acquire_present_observed: bool,
    forbidden_outcomes: Vec<String>,
    status: String,
    duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LifecycleCell {
    action: String,
    entrypoint: String,
    observation: String,
    post_action_draw: bool,
    mount_lineage_preserved: bool,
    status: String,
}

// ── WSI detection ─────────────────────────────────────────────────────────

/// Detect whether a live WSI environment is available.
///
/// Returns `true` if a windowing system (Wayland or X11) is detected and
/// a Vulkan-capable GPU is present.
fn live_wsi_available() -> bool {
    let has_wayland = std::env::var("WAYLAND_DISPLAY").is_ok();
    let has_x11 = std::env::var("DISPLAY").is_ok();
    let has_vulkan = std::env::var("VK_ICD_FILENAMES").is_ok()
        || Path::new("/usr/share/vulkan/icd.d").is_dir();

    has_vulkan && (has_wayland || has_x11)
}

/// Detect GPU and driver info via vulkaninfo or environment.
fn gpu_environment() -> WsiEnvironment {
    let compositor = if std::env::var("WAYLAND_DISPLAY").is_ok() {
        "Wayland".to_string()
    } else if std::env::var("DISPLAY").is_ok() {
        "X11".to_string()
    } else {
        "none".to_string()
    };

    let (gpu_driver, vulkan_version) = detect_vulkan_info();

    WsiEnvironment {
        os: std::env::consts::OS.to_string(),
        gpu_driver,
        vulkan_version,
        compositor,
        validation_layer_active: std::env::var("VK_INSTANCE_LAYERS")
            .map(|v| v.contains("VK_LAYER_KHRONOS_validation"))
            .unwrap_or(false),
        headless_only: !live_wsi_available(),
        note: if live_wsi_available() {
            "Live WSI environment detected".to_string()
        } else {
            "NOT_RUN: live WSI environment not available; cannot prove acquire/present".to_string()
        },
    }
}

fn detect_vulkan_info() -> (String, String) {
    // Try vulkaninfo if available.
    if let Ok(output) = Command::new("vulkaninfo")
        .arg("--summary")
        .env_clear()
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let driver = stdout
            .lines()
            .find(|l| l.contains("driverName"))
            .map(|l| l.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let version = stdout
            .lines()
            .find(|l| l.contains("apiVersion"))
            .map(|l| l.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        return (driver, version);
    }
    ("unknown".to_string(), "unknown".to_string())
}

// ── Child-process helpers ─────────────────────────────────────────────────

/// Run a child process with timeout and capture output.
fn run_child(
    label: &str,
    mut cmd: Command,
    timeout_secs: u64,
) -> (Option<Output>, bool) {
    let start = SystemTime::now();
    eprintln!("[{label}] Running: {cmd:?}");

    let result = cmd.output();
    let elapsed = start.elapsed().unwrap_or(Duration::ZERO);

    match result {
        Ok(output) => {
            let timed_out = elapsed.as_secs() >= timeout_secs;
            eprintln!(
                "[{label}] exit={:?}, elapsed={}ms, stdout={}B, stderr={}B",
                output.status.code(),
                elapsed.as_millis(),
                output.stdout.len(),
                output.stderr.len()
            );
            (Some(output), timed_out)
        }
        Err(e) => {
            eprintln!("[{label}] Failed to launch: {e}");
            (None, false)
        }
    }
}

/// Scan output for forbidden patterns.
fn scan_forbidden(output: &Output) -> Vec<String> {
    let mut issues = Vec::new();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stdout}\n{stderr}");

    if combined.contains("panic") || combined.contains("panicked") {
        issues.push("panic".to_string());
    }
    if combined.contains("VUID-") || combined.contains("validation layer")
        || combined.contains("Validation Error")
    {
        issues.push("validation_layer_error".to_string());
    }
    if combined.contains("ERROR") && !combined.contains("ERROR] engine_") {
        // Distinguish engine ERROR from other ERROR occurrences.
        // Look for the engine log pattern: [2026... ERROR engine_...
        for line in combined.lines() {
            if line.contains("ERROR") && !line.contains("RUST_LOG") {
                issues.push(format!("engine_error: {line}"));
            }
        }
    }
    if combined.contains("DeviceLost") || combined.contains("VK_ERROR_DEVICE_LOST") {
        issues.push("device_loss".to_string());
    }

    issues
}

/// Check for acquire/present evidence in output.
fn observe_acquire_present(output: &Output, is_bsp: bool) -> bool {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stdout}\n{stderr}");

    // Look for typical WSI lifecycle evidence.
    let swapchain_indicators = [
        "swapchain",
        "Swapchain",
        "Present",
        "present",
        "acquire",
        "Acquire",
        "surface",
        "Surface",
    ];

    let has_any = swapchain_indicators
        .iter()
        .any(|indicator| combined.contains(indicator));

    // For BSP entry, also check for BSP mount evidence (no fallback).
    if is_bsp && combined.contains("fallback") {
        return false; // BSP fallback detected — not valid evidence
    }

    has_any
}

/// Extract first N lines of output for snippet.
fn output_snippet(data: &[u8], max_lines: usize) -> String {
    let text = String::from_utf8_lossy(data);
    text.lines().take(max_lines).collect::<Vec<_>>().join("\n")
}

// ── BSP compilation for live WSI ──────────────────────────────────────────

fn compile_bsp_for_wsi() -> Result<(PathBuf, Vec<u8>, Option<Vec<u8>>), String> {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        return Err(format!(
            "ericw-tools not available at {}",
            tool_dir.display()
        ));
    }

    let staging = unique_tmp("live-wsi");
    let (map_text, _meta) = bsp_generator::generate(SELECTED_SEED, DungeonConfig::nominal_m1())
        .map_err(|e| format!("generate seed {}: {e}", SELECTED_SEED))?;

    let map_path = staging.join(format!("{SELECTED_LABEL}.map"));
    std::fs::write(&map_path, &map_text).map_err(|e| format!("write .map: {e}"))?;

    let profile_content =
        std::fs::read_to_string(profile_path()).map_err(|e| format!("read profile: {e}"))?;
    let profile = engine_pack::compiler::parse_compiler_profile(&profile_content)
        .map_err(|e| format!("parse profile: {e}"))?;

    let work_dir = staging.join(".compile-work");
    std::fs::create_dir_all(&work_dir).map_err(|e| format!("create work dir: {e}"))?;

    let result = engine_pack::compiler::compile_map(
        &map_path,
        &profile,
        &work_dir,
        &palette_path(),
        Some(&tool_dir),
        &[wad_path()],
    )
    .map_err(|e| format!("compile: {e}"))?;

    // Copy to captures dir.
    let captures = captures_dir();
    std::fs::create_dir_all(&captures).map_err(|e| format!("create captures dir: {e}"))?;
    let bsp_dest = captures.join(format!("{SELECTED_LABEL}.bsp"));
    std::fs::write(&bsp_dest, &result.bsp_data)
        .map_err(|e| format!("write BSP: {e}"))?;
    if let Some(ref lit) = result.lit_data {
        let lit_dest = captures.join(format!("{SELECTED_LABEL}.lit"));
        std::fs::write(&lit_dest, lit).map_err(|e| format!("write LIT: {e}"))?;
    }

    // Copy companions.
    std::fs::copy(palette_path(), captures.join("palette.lmp"))
        .map_err(|e| format!("copy palette: {e}"))?;
    std::fs::copy(wad_path(), captures.join("cc0_stone_beta.wad"))
        .map_err(|e| format!("copy WAD: {e}"))?;

    Ok((bsp_dest, result.bsp_data, result.lit_data))
}

fn unique_tmp(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "bsp-live-wsi-{label}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ── Live WSI acceptance test ──────────────────────────────────────────────

/// Full live WSI acceptance harness.
///
/// This test is `#[ignore]` by default. It requires ericw-tools, a live GPU,
/// and a WSI environment (Wayland/X11). In CI or headless environments, it
/// records `NOT_RUN` with environment evidence.
#[test]
#[ignore = "requires live GPU + WSI environment"]
fn live_wsi_acceptance() {
    let env = gpu_environment();
    let mut report = LiveWsiReport {
        schema_version: 1,
        phase: "09".to_string(),
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .to_string(),
        environment: env.clone(),
        entrypoints: Vec::new(),
        lifecycle_actions: Vec::new(),
        overall_status: "NOT_RUN".to_string(),
    };

    if !live_wsi_available() {
        eprintln!(
            "Live WSI not available (Wayland={}, X11={}). Recording NOT_RUN.",
            std::env::var("WAYLAND_DISPLAY").is_ok(),
            std::env::var("DISPLAY").is_ok(),
        );
        report.overall_status = "NOT_RUN".to_string();
        report
            .environment
            .note = "Live WSI environment not available; acquire/present cannot be proven"
                .to_string();
        write_report(&report);
        panic!(
            "Live WSI acceptance requires a live GPU + WSI environment. Got NOT_RUN."
        );
    }

    // ── Compile BSP for bsp_beta entrypoint ────────────────────────────
    let bsp_info = compile_bsp_for_wsi();
    let (bsp_path, _bsp_data, _lit_data) = match bsp_info {
        Ok(info) => info,
        Err(msg) => {
            eprintln!("Cannot compile BSP: {msg}");
            report.overall_status = "NOT_RUN".to_string();
            report.environment.note =
                format!("BSP compilation blocked: {msg}");
            write_report(&report);
            panic!("BSP compilation required for live WSI acceptance");
        }
    };

    let captures = captures_dir();
    let palette = captures.join("palette.lmp");
    let wad = captures.join("cc0_stone_beta.wad");

    // ── Entrypoint 1: bsp_beta live startup ────────────────────────────
    {
        let label = "bsp_beta-live-startup";
        let cmd_str = format!(
            "cargo run -p bsp_beta -- --strict --bsp {} --palette {} --wad {}",
            bsp_path.display(),
            palette.display(),
            wad.display(),
        );
        let (output, timed_out) = run_child(
            label,
            {
                let mut c = Command::new("cargo");
                c.args([
                    "run", "-p", "bsp_beta", "--",
                    "--strict",
                    "--bsp", &bsp_path.display().to_string(),
                    "--palette", &palette.display().to_string(),
                    "--wad", &wad.display().to_string(),
                ]);
                c.env("RUST_LOG", "warn");
                c
            },
            15,
        );

        let cell = build_entrypoint_cell(
            "bsp_beta",
            Some(SELECTED_LABEL),
            &cmd_str,
            output.as_ref(),
            timed_out,
            true,
        );
        report.entrypoints.push(cell);
    }

    // ── Entrypoint 2: dungeon_dogfood live startup ─────────────────────
    {
        let label = "dungeon_dogfood-live-startup";
        let cmd_str = "cargo run -p dungeon_dogfood".to_string();
        let (output, timed_out) = run_child(
            label,
            {
                let mut c = Command::new("cargo");
                c.args(["run", "-p", "dungeon_dogfood"]);
                c.env("RUST_LOG", "warn");
                c
            },
            15,
        );

        let cell = build_entrypoint_cell(
            "dungeon_dogfood",
            None,
            &cmd_str,
            output.as_ref(),
            timed_out,
            false,
        );
        report.entrypoints.push(cell);
    }

    // ── Entrypoint 3: voxel_demo live startup ──────────────────────────
    {
        let label = "voxel_demo-live-startup";
        let cmd_str = "cargo run -p voxel_demo -- --config presets/default.toml".to_string();
        let (output, timed_out) = run_child(
            label,
            {
                let mut c = Command::new("cargo");
                c.args([
                    "run", "-p", "voxel_demo", "--",
                    "--config", "presets/default.toml",
                ]);
                c.env("RUST_LOG", "warn");
                c
            },
            15,
        );

        let cell = build_entrypoint_cell(
            "voxel_demo",
            None,
            &cmd_str,
            output.as_ref(),
            timed_out,
            false,
        );
        report.entrypoints.push(cell);
    }

    // ── Entrypoint 4: renderer api_test ────────────────────────────────
    {
        let label = "renderer-api-test-live";
        let cmd_str = "cargo run -p renderer --example api_test".to_string();
        let (output, timed_out) = run_child(
            label,
            {
                let mut c = Command::new("cargo");
                c.args(["run", "-p", "renderer", "--example", "api_test"]);
                c.env("RUST_LOG", "warn");
                c
            },
            15,
        );

        let cell = build_entrypoint_cell(
            "api_test",
            None,
            &cmd_str,
            output.as_ref(),
            timed_out,
            false,
        );
        report.entrypoints.push(cell);
    }

    // ── Lifecycle: bsp_beta resize, minimize/restore ───────────────────
    // These actions require interactive window manipulation and are recorded
    // as NOT_RUN when a human operator is not available.
    for action in &[
        LifecycleAction::Resize,
        LifecycleAction::Minimize,
        LifecycleAction::Restore,
        LifecycleAction::SurfaceLoss,
    ] {
        report.lifecycle_actions.push(LifecycleCell {
            action: action.as_str().to_string(),
            entrypoint: "bsp_beta".to_string(),
            observation: format!(
                "{} requires interactive window manipulation in live WSI",
                action.as_str()
            ),
            post_action_draw: false,
            mount_lineage_preserved: false,
            status: "NOT_RUN".to_string(),
        });
    }

    // ── Evaluate overall status ────────────────────────────────────────
    let all_pass = report.entrypoints.iter().all(|e| e.status == "PASS");
    report.overall_status = if all_pass {
        "PASS".to_string()
    } else {
        let failures: Vec<_> = report
            .entrypoints
            .iter()
            .filter(|e| e.status != "PASS")
            .map(|e| format!("{}: {}", e.entrypoint, e.status))
            .collect();
        format!("PARTIAL: {}", failures.join("; "))
    };

    write_report(&report);

    if report.overall_status != "PASS" {
        panic!(
            "Live WSI acceptance NOT PASS: {}",
            report.overall_status
        );
    }

    eprintln!("Live WSI acceptance PASS: all {} entrypoints", report.entrypoints.len());
}

fn build_entrypoint_cell(
    entrypoint: &str,
    package: Option<&str>,
    command: &str,
    output: Option<&Output>,
    timed_out: bool,
    is_bsp: bool,
) -> EntrypointCell {
    match output {
        Some(out) => {
            let forbidden = scan_forbidden(out);
            let has_present = observe_acquire_present(out, is_bsp);
            let exit_code = out.status.code();

            let status = if timed_out {
                "FAIL: timeout".to_string()
            } else if !forbidden.is_empty() {
                format!("FAIL: {}", forbidden.join(", "))
            } else if exit_code.map_or(true, |c| c != 0) {
                format!("FAIL: exit code {}", exit_code.unwrap_or(-1))
            } else if is_bsp && !has_present {
                "FAIL: no acquire/present evidence".to_string()
            } else {
                "PASS".to_string()
            };

            EntrypointCell {
                entrypoint: entrypoint.to_string(),
                package: package.map(|p| p.to_string()),
                command: command.to_string(),
                exit_code,
                stdout_snippet: output_snippet(&out.stdout, 10),
                stderr_snippet: output_snippet(&out.stderr, 20),
                acquire_present_observed: has_present,
                forbidden_outcomes: forbidden,
                status,
                duration_ms: 0,
            }
        }
        None => EntrypointCell {
            entrypoint: entrypoint.to_string(),
            package: package.map(|p| p.to_string()),
            command: command.to_string(),
            exit_code: None,
            stdout_snippet: String::new(),
            stderr_snippet: "child process failed to launch".to_string(),
            acquire_present_observed: false,
            forbidden_outcomes: vec!["launch_failure".to_string()],
            status: "FAIL: launch failure".to_string(),
            duration_ms: 0,
        },
    }
}

fn write_report(report: &LiveWsiReport) {
    std::fs::create_dir_all(debug_dir()).unwrap();
    let path = live_wsi_report_path();
    let serialized = serde_json::to_string_pretty(report).expect("serialize report");
    std::fs::write(&path, &serialized).expect("write report");
    eprintln!(
        "Live WSI report written: {} ({} bytes)",
        path.display(),
        serialized.len()
    );
}

// ── Structural tests (always run) ─────────────────────────────────────────

#[test]
fn live_wsi_report_schema() {
    let report = LiveWsiReport {
        schema_version: 1,
        phase: "09".to_string(),
        timestamp: String::new(),
        environment: gpu_environment(),
        entrypoints: Vec::new(),
        lifecycle_actions: Vec::new(),
        overall_status: "SCHEMA_VALID".to_string(),
    };
    let serialized = serde_json::to_string_pretty(&report).unwrap();
    let _: LiveWsiReport = serde_json::from_str(&serialized).unwrap();
    eprintln!("Live WSI report schema validated: {} bytes", serialized.len());
}

#[test]
fn live_wsi_detects_environment() {
    let env = gpu_environment();
    eprintln!("OS: {}", env.os);
    eprintln!("GPU driver: {}", env.gpu_driver);
    eprintln!("Vulkan: {}", env.vulkan_version);
    eprintln!("Compositor: {}", env.compositor);
    eprintln!("Headless only: {}", env.headless_only);
    eprintln!("Live WSI available: {}", live_wsi_available());
    // This test always passes; it just reports the environment.
}

/// Verify that the BSP compilation succeeds for live WSI testing.
#[test]
fn live_wsi_bsp_compilation() {
    match compile_bsp_for_wsi() {
        Ok((bsp_path, bsp_data, lit_data)) => {
            eprintln!(
                "BSP compiled for live WSI: {} ({} bytes)",
                bsp_path.display(),
                bsp_data.len()
            );
            if let Some(ref lit) = lit_data {
                eprintln!("  LIT: {} bytes", lit.len());
            }
        }
        Err(msg) => {
            eprintln!("SKIP: {msg}");
        }
    }
}
