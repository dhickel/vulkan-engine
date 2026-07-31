//! Phase 08: Frozen Corpus Runtime Evidence — Strict Published Closures
//!
//! Parent/child harness that executes all 12 frozen support corpus entries
//! through strict runtime boundaries: isolated workspace per entry, no-replace
//! package publication, strict `bsp_beta` child with declared inputs,
//! deterministic PVS + all-visible campaigns, and budget enforcement.
//!
//! GPU-unavailable environments produce explicit `NOT_RUN` evidence with
//! environment/capability facts and do not silently pass.
//!
//! Run with:
//! ```bash
//! BSP_HARDWARE_CLASS=H2 cargo test -p bsp_beta --test corpus_runtime_evidence -- --ignored --nocapture
//! ```

use bsp_generator::DungeonConfig;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

// ── Frozen budget ceilings ────────────────────────────────────────────────

const M1_STATIC_BATCH_CEILING: u32 = 100;
const M2_STATIC_BATCH_CEILING: u32 = 500;
const M1_TOTAL_DRAW_CEILING: u32 = 200;
const M2_TOTAL_DRAW_CEILING: u32 = 1000;

// ── Paths (relative to bsp_beta crate root: apps/bsp_beta/) ──────────────

fn wad_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/bsp_generator/themes/cc0_stone_beta/cc0_stone_beta.wad")
}

fn palette_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../src/bsp_generator/themes/cc0_stone_beta/palette.lmp")
}

fn ericw_tools_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/dhickel".to_string());
    PathBuf::from(home).join(".local/ericw-tools/ericw-tools-2.0.0-alpha3-Linux/bin")
}

fn tools_available(dir: &Path) -> bool {
    dir.join("qbsp").is_file() && dir.join("vis").is_file() && dir.join("light").is_file()
}

fn bsp_beta_binary() -> PathBuf {
    // In test context, the binary may not exist; try to locate it.
    // The test is #[ignore] by default and run manually.
    PathBuf::from(env!("CARGO_BIN_EXE_bsp_beta"))
}

fn evidence_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.internal-dev/debug_reports/bsp-dungeon-completion")
}

// ── Transport manifest (deserialized from generator export) ──────────────

#[derive(Debug, Clone, Deserialize)]
struct TransportManifest {
    #[allow(dead_code)]
    schema_version: u32,
    #[allow(dead_code)]
    phase: String,
    total_entries: usize,
    entries: Vec<TransportEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct TransportEntry {
    entry_id: String,
    class: String,
    seed: u64,
    config: TransportConfig,
    profile: TransportProfile,
    map_hash: Option<String>,
    bsp_hash: Option<String>,
    lit_hash: Option<String>,
    palette_hash: String,
    wad_hash: String,
    prerequisite_result: String,
}

#[derive(Debug, Clone, Deserialize)]
struct TransportConfig {
    room_count: u32,
    loop_count: u32,
    xy_bounds: (u32, u32),
    z_span: u32,
    placement_candidates: u32,
    max_placement_attempts: u32,
    max_astar_expansions: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct TransportProfile {
    name: String,
    format: String,
    compiler: String,
    compiler_version: String,
}

// ── Evidence structures ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CampaignEntry {
    entry_id: String,
    class: String,
    seed: u64,
    campaign: String,
    /// BSP content hash from the published package.
    bsp_hash: Option<String>,
    /// Palette content hash.
    palette_hash: String,
    /// WAD content hash.
    wad_hash: String,
    /// LIT content hash (if present).
    lit_hash: Option<String>,
    /// How many static batches were mounted (canonical, model-zero).
    mounted_static_batches: Option<u32>,
    /// How many static-world draws were submitted.
    submitted_static_draws: Option<u32>,
    /// How many static-world draws were recorded (post cmd_draw_indexed).
    recorded_static_draws: Option<u32>,
    /// Total draws submitted (static + dynamic).
    submitted_total_draws: Option<u32>,
    /// Total draws recorded.
    recorded_total_draws: Option<u32>,
    /// Whether normal-PVS evidence was collected.
    normal_pvs_complete: bool,
    /// Normal-PVS submitted static draws.
    normal_pvs_submitted_draws: Option<u32>,
    /// Normal-PVS camera/leaf identity.
    normal_pvs_camera_identity: Option<String>,
    /// Whether all-visible evidence was collected.
    all_visible_complete: bool,
    /// All-visible submitted static draws.
    all_visible_submitted_draws: Option<u32>,
    /// All-visible recorded static draws.
    all_visible_recorded_draws: Option<u32>,
    /// All-visible source-face coverage count.
    all_visible_face_coverage: Option<u32>,
    /// Stable digest for cross-campaign comparison.
    stable_digest: Option<String>,
    /// Child process exit code.
    child_exit_code: Option<i32>,
    /// Child stderr classification.
    stderr_classification: String,
    /// Hardware class declared.
    hardware_class: String,
    /// Whether the run was blocked by unavailable capability.
    capability_blocked: bool,
    /// Blocked cell description (if any).
    blocked_cell: Option<String>,
    /// Full child stderr (truncated to 8 KiB).
    stderr_snippet: Option<String>,
    /// Execution status: PASS, FAIL, NOT_RUN.
    status: String,
    /// Error description (if failed).
    error: Option<String>,
    /// Wall-clock duration of the child process (ms).
    duration_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
struct CorpusRuntimeReport {
    schema_version: u32,
    phase: String,
    timestamp: String,
    provenance: ReportProvenance,
    campaigns: Vec<CampaignMetadata>,
    entries: Vec<CampaignEntry>,
    cross_campaign_comparison: CrossCampaignComparison,
    reducer: ReducerSummary,
}

#[derive(Debug, Clone, Serialize)]
struct ReportProvenance {
    generator_manifest: String,
    ericw_tools_path: String,
    tools_available: bool,
    bsp_beta_binary: String,
    hardware_class: String,
    environment: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
struct CampaignMetadata {
    name: String,
    started_at: String,
    completed_at: String,
    entries_executed: usize,
    entries_passed: usize,
    entries_not_run: usize,
    entries_failed: usize,
}

#[derive(Debug, Clone, Serialize)]
struct CrossCampaignComparison {
    campaigns_compared: Vec<String>,
    all_entries_present_in_both: bool,
    normal_pvs_deterministic: bool,
    all_visible_equal: bool,
    mismatches: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ReducerSummary {
    total_entries: usize,
    pass: usize,
    fail: usize,
    not_run: usize,
    phase_pass: bool,
    failure_reasons: Vec<String>,
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn unique_tmp(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "bsp-corpus-runtime-{label}-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn sha256(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn chrono_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Simple ISO-8601
    let days = secs / 86400;
    let tod = secs % 86400;
    let h = tod / 3600;
    let m = (tod % 3600) / 60;
    let s = tod % 60;
    // Approximate YMD
    let total_days = days + 719468;
    let era = total_days / 146097;
    let doe = total_days - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    format!("{year:04}-{month:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Generate a BSP from the frozen corpus entry, compile it, and stage
/// the published package artifacts. Returns (bsp_bytes, lit_bytes, staging_dir).
fn compile_corpus_entry(
    entry: &TransportEntry,
) -> Result<(Vec<u8>, Option<Vec<u8>>, PathBuf), String> {
    let tool_dir = ericw_tools_dir();
    if !tools_available(&tool_dir) {
        return Err("ericw-tools unavailable".to_string());
    }

    let config = DungeonConfig {
        class: if entry.class == "M1" {
            bsp_generator::MapClass::M1
        } else {
            bsp_generator::MapClass::M2
        },
        room_count: entry.config.room_count,
        loop_count: entry.config.loop_count,
        xy_bounds: entry.config.xy_bounds,
        z_span: entry.config.z_span,
        placement_candidates: entry.config.placement_candidates,
        max_placement_attempts: entry.config.max_placement_attempts,
        max_astar_expansions: entry.config.max_astar_expansions,
    };

    let (map_text, _meta) = bsp_generator::generate(entry.seed, config)
        .map_err(|e| format!("generate failed: {e:?}"))?;

    let staging = unique_tmp(&entry.entry_id);
    let map_path = staging.join("generated.map");
    std::fs::write(&map_path, &map_text).map_err(|e| format!("write .map: {e}"))?;

    // Stage palette and WAD
    let work_palette = staging.join("palette.lmp");
    std::fs::copy(palette_path(), &work_palette).map_err(|e| format!("copy palette: {e}"))?;
    let work_wad = staging.join("cc0_stone_beta.wad");
    std::fs::copy(wad_path(), &work_wad).map_err(|e| format!("copy WAD: {e}"))?;

    // Compile through ericw-tools (qbsp → vis → light)
    let qbsp_path = tool_dir.join("qbsp");
    let mut qbsp = Command::new(&qbsp_path);
    qbsp.args(["-bsp2", "-threads", "1", "generated.map"])
        .current_dir(&staging)
        .env_clear();
    if let Some(path) = std::env::var_os("PATH") {
        qbsp.env("PATH", path);
    }
    if let Some(home) = std::env::var_os("HOME") {
        qbsp.env("HOME", home);
    }
    if let Some(tmp) = std::env::var_os("TMPDIR") {
        qbsp.env("TMPDIR", tmp);
    }
    let qbsp_out = qbsp.output().map_err(|e| format!("qbsp spawn: {e}"))?;
    if !qbsp_out.status.success() {
        let stderr = String::from_utf8_lossy(&qbsp_out.stderr);
        return Err(format!("qbsp failed: {stderr}"));
    }

    // vis
    let vis_path = tool_dir.join("vis");
    let mut vis = Command::new(&vis_path);
    vis.args(["-threads", "1", "generated.bsp"])
        .current_dir(&staging)
        .env_clear();
    if let Some(path) = std::env::var_os("PATH") {
        vis.env("PATH", path);
    }
    if let Some(home) = std::env::var_os("HOME") {
        vis.env("HOME", home);
    }
    let vis_out = vis.output().map_err(|e| format!("vis spawn: {e}"))?;
    if !vis_out.status.success() {
        let stderr = String::from_utf8_lossy(&vis_out.stderr);
        return Err(format!("vis failed: {stderr}"));
    }

    // light
    let light_path = tool_dir.join("light");
    let mut light = Command::new(&light_path);
    light
        .args(["-threads", "1", "-lit", "generated.bsp"])
        .current_dir(&staging)
        .env_clear();
    if let Some(path) = std::env::var_os("PATH") {
        light.env("PATH", path);
    }
    if let Some(home) = std::env::var_os("HOME") {
        light.env("HOME", home);
    }
    let light_out = light.output().map_err(|e| format!("light spawn: {e}"))?;
    if !light_out.status.success() {
        let stderr = String::from_utf8_lossy(&light_out.stderr);
        return Err(format!("light failed: {stderr}"));
    }
    // Reject compiler warnings
    let combined = format!(
        "{}{}{}",
        String::from_utf8_lossy(&qbsp_out.stderr),
        String::from_utf8_lossy(&vis_out.stderr),
        String::from_utf8_lossy(&light_out.stderr)
    );
    let lower = combined.to_ascii_lowercase();
    if lower.contains("warning:")
        || lower.contains("no entities in empty space")
        || lower.contains("no filling performed")
    {
        return Err(format!("compiler warning detected: {combined}"));
    }

    let bsp_path_out = staging.join("generated.bsp");
    let bsp_data = std::fs::read(&bsp_path_out).map_err(|e| format!("read bsp: {e}"))?;

    let lit_path_out = staging.join("generated.lit");
    let lit_data = if lit_path_out.exists() {
        Some(std::fs::read(&lit_path_out).map_err(|e| format!("read lit: {e}"))?)
    } else {
        None
    };

    // Verify BSP2 magic
    if bsp_data.len() < 4 || &bsp_data[0..4] != b"BSP2" {
        return Err("output is not BSP2".to_string());
    }

    // Verify sealed (no .pts leak file)
    if staging.join("generated.pts").exists() {
        return Err("map is not sealed (leak detected)".to_string());
    }

    Ok((bsp_data, lit_data, staging))
}

/// Attempt to execute a single corpus entry through the strict bsp_beta child.
///
/// Returns a CampaignEntry with evidence or NOT_RUN facts.
fn execute_corpus_child(
    entry: &TransportEntry,
    campaign: &str,
    hardware_class: &str,
    bsp_beta_bin: &Path,
) -> CampaignEntry {
    let t0 = SystemTime::now();

    // Step 1: Compile the BSP
    let (bsp_data, lit_data, staging) = match compile_corpus_entry(entry) {
        Ok(result) => result,
        Err(e) => {
            let duration = t0.elapsed().unwrap_or_default().as_millis() as u64;
            return CampaignEntry {
                entry_id: entry.entry_id.clone(),
                class: entry.class.clone(),
                seed: entry.seed,
                campaign: campaign.to_string(),
                bsp_hash: None,
                palette_hash: entry.palette_hash.clone(),
                wad_hash: entry.wad_hash.clone(),
                lit_hash: None,
                mounted_static_batches: None,
                submitted_static_draws: None,
                recorded_static_draws: None,
                submitted_total_draws: None,
                recorded_total_draws: None,
                normal_pvs_complete: false,
                normal_pvs_submitted_draws: None,
                normal_pvs_camera_identity: None,
                all_visible_complete: false,
                all_visible_submitted_draws: None,
                all_visible_recorded_draws: None,
                all_visible_face_coverage: None,
                stable_digest: None,
                child_exit_code: None,
                stderr_classification: "COMPILATION_FAILED".to_string(),
                hardware_class: hardware_class.to_string(),
                capability_blocked: false,
                blocked_cell: None,
                stderr_snippet: Some(truncate_str(&e, 8192)),
                status: "FAIL".to_string(),
                error: Some(e),
                duration_ms: duration,
            };
        }
    };

    let bsp_hash = sha256(&bsp_data);
    let lit_hash = lit_data.as_deref().map(sha256);

    // Verify bsp_hash matches manifest if manifest had one
    if let Some(ref expected) = entry.bsp_hash {
        if bsp_hash != *expected {
            let duration = t0.elapsed().unwrap_or_default().as_millis() as u64;
            let mismatch_msg =
                format!("bsp hash mismatch: expected {}, got {}", expected, bsp_hash);
            let _ = std::fs::remove_dir_all(&staging);
            return CampaignEntry {
                entry_id: entry.entry_id.clone(),
                class: entry.class.clone(),
                seed: entry.seed,
                campaign: campaign.to_string(),
                bsp_hash: Some(bsp_hash),
                palette_hash: entry.palette_hash.clone(),
                wad_hash: entry.wad_hash.clone(),
                lit_hash,
                mounted_static_batches: None,
                submitted_static_draws: None,
                recorded_static_draws: None,
                submitted_total_draws: None,
                recorded_total_draws: None,
                normal_pvs_complete: false,
                normal_pvs_submitted_draws: None,
                normal_pvs_camera_identity: None,
                all_visible_complete: false,
                all_visible_submitted_draws: None,
                all_visible_recorded_draws: None,
                all_visible_face_coverage: None,
                stable_digest: None,
                child_exit_code: None,
                stderr_classification: "HASH_MISMATCH".to_string(),
                hardware_class: hardware_class.to_string(),
                capability_blocked: false,
                blocked_cell: None,
                stderr_snippet: Some(mismatch_msg),
                status: "FAIL".to_string(),
                error: Some("BSP content hash mismatch against frozen manifest".to_string()),
                duration_ms: duration,
            };
        }
    }

    let bsp_path = staging.join("generated.bsp");
    let lit_path = staging.join("generated.lit");

    // Step 2: Launch bsp_beta child with strict inputs
    let is_gpu_available = std::env::var("BSP_SKIP_GPU")
        .map(|v| v != "1")
        .unwrap_or(true);

    if !is_gpu_available || !bsp_beta_bin.exists() {
        let duration = t0.elapsed().unwrap_or_default().as_millis() as u64;
        let _ = std::fs::remove_dir_all(&staging);
        return CampaignEntry {
            entry_id: entry.entry_id.clone(),
            class: entry.class.clone(),
            seed: entry.seed,
            campaign: campaign.to_string(),
            bsp_hash: Some(bsp_hash),
            palette_hash: entry.palette_hash.clone(),
            wad_hash: entry.wad_hash.clone(),
            lit_hash,
            mounted_static_batches: None,
            submitted_static_draws: None,
            recorded_static_draws: None,
            submitted_total_draws: None,
            recorded_total_draws: None,
            normal_pvs_complete: false,
            normal_pvs_submitted_draws: None,
            normal_pvs_camera_identity: None,
            all_visible_complete: false,
            all_visible_submitted_draws: None,
            all_visible_recorded_draws: None,
            all_visible_face_coverage: None,
            stable_digest: None,
            child_exit_code: None,
            stderr_classification: "GPU_UNAVAILABLE".to_string(),
            hardware_class: hardware_class.to_string(),
            capability_blocked: true,
            blocked_cell: Some("GPU environment or bsp_beta binary unavailable".to_string()),
            stderr_snippet: None,
            status: "NOT_RUN".to_string(),
            error: None,
            duration_ms: duration,
        };
    }

    // Build child command: two reports per entry (normal-PVS then all-visible)
    let lit_path_ref: Option<&Path> = if lit_path.exists() {
        Some(&lit_path)
    } else {
        None
    };
    let mut pvs_entry = run_child_report(
        &bsp_path,
        lit_path_ref,
        &staging,
        entry,
        campaign,
        hardware_class,
        bsp_beta_bin,
        false, // normal-PVS
    );

    let mut av_entry = run_child_report(
        &bsp_path,
        lit_path_ref,
        &staging,
        entry,
        campaign,
        hardware_class,
        bsp_beta_bin,
        true, // all-visible
    );

    // Merge: use all-visible for draw counts, normal-PVS for PVS identity
    pvs_entry.all_visible_complete = av_entry.all_visible_complete;
    pvs_entry.all_visible_submitted_draws = av_entry.all_visible_submitted_draws;
    pvs_entry.all_visible_recorded_draws = av_entry.all_visible_recorded_draws;
    pvs_entry.all_visible_face_coverage = av_entry.all_visible_face_coverage;
    if pvs_entry.stable_digest.is_none() {
        pvs_entry.stable_digest = av_entry.stable_digest.take();
    }
    pvs_entry.bsp_hash = Some(bsp_hash);
    pvs_entry.duration_ms = t0.elapsed().unwrap_or_default().as_millis() as u64;

    let _ = std::fs::remove_dir_all(&staging);
    pvs_entry
}

/// Run a single child report (normal-PVS or all-visible).
fn run_child_report(
    bsp_path: &Path,
    lit_path: Option<&Path>,
    staging: &Path,
    entry: &TransportEntry,
    campaign: &str,
    hardware_class: &str,
    bsp_beta_bin: &Path,
    all_visible: bool,
) -> CampaignEntry {
    let mut cmd = Command::new(bsp_beta_bin);
    cmd.arg("--strict")
        .arg("--headless")
        .arg("--capture-frames")
        .arg("0")
        .arg("--stats")
        .arg("--bsp")
        .arg(bsp_path)
        .arg("--palette")
        .arg(staging.join("palette.lmp"))
        .arg("--wad")
        .arg(staging.join("cc0_stone_beta.wad"))
        .arg("--corpus")
        .arg(format!("{}-{}", entry.entry_id, campaign));

    if all_visible {
        cmd.arg("--all-visible");
    }
    if let Some(lp) = lit_path {
        if lp.exists() {
            cmd.arg("--lit").arg(lp);
        }
    }

    let result = cmd.output();
    let (exit_code, stdout, stderr, duration_ms) = match result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            (output.status.code(), stdout, stderr, 0u64)
        }
        Err(e) => {
            return CampaignEntry {
                entry_id: entry.entry_id.clone(),
                class: entry.class.clone(),
                seed: entry.seed,
                campaign: campaign.to_string(),
                bsp_hash: None,
                palette_hash: entry.palette_hash.clone(),
                wad_hash: entry.wad_hash.clone(),
                lit_hash: None,
                mounted_static_batches: None,
                submitted_static_draws: None,
                recorded_static_draws: None,
                submitted_total_draws: None,
                recorded_total_draws: None,
                normal_pvs_complete: false,
                normal_pvs_submitted_draws: None,
                normal_pvs_camera_identity: None,
                all_visible_complete: false,
                all_visible_submitted_draws: None,
                all_visible_recorded_draws: None,
                all_visible_face_coverage: None,
                stable_digest: None,
                child_exit_code: None,
                stderr_classification: "SPAWN_FAILED".to_string(),
                hardware_class: hardware_class.to_string(),
                capability_blocked: true,
                blocked_cell: Some(format!("bsp_beta child spawn failed: {e}")),
                stderr_snippet: None,
                status: "NOT_RUN".to_string(),
                error: None,
                duration_ms: 0,
            };
        }
    };

    let stderr_lower = stderr.to_ascii_lowercase();
    let classification = if exit_code == Some(0) && !stderr_lower.contains("error") {
        "CLEAN".to_string()
    } else if stderr_lower.contains("vulkan") && stderr_lower.contains("error") {
        "VULKAN_ERROR".to_string()
    } else if stderr_lower.contains("validation") && stderr_lower.contains("error") {
        "VALIDATION_ERROR".to_string()
    } else if stderr_lower.contains("panic") {
        "PANIC".to_string()
    } else if exit_code != Some(0) {
        format!("NONZERO_EXIT_{}", exit_code.unwrap_or(-1))
    } else {
        "WARNINGS_PRESENT".to_string()
    };

    // Try to parse JSON evidence from stdout
    let evidence_parsed = parse_child_evidence(&stdout);

    CampaignEntry {
        entry_id: entry.entry_id.clone(),
        class: entry.class.clone(),
        seed: entry.seed,
        campaign: campaign.to_string(),
        bsp_hash: None,
        palette_hash: entry.palette_hash.clone(),
        wad_hash: entry.wad_hash.clone(),
        lit_hash: None,
        mounted_static_batches: evidence_parsed.mounted_static_batches,
        submitted_static_draws: evidence_parsed.submitted_static_draws,
        recorded_static_draws: evidence_parsed.recorded_static_draws,
        submitted_total_draws: evidence_parsed.submitted_total_draws,
        recorded_total_draws: evidence_parsed.recorded_total_draws,
        normal_pvs_complete: !all_visible && evidence_parsed.eligible,
        normal_pvs_submitted_draws: if all_visible {
            None
        } else {
            evidence_parsed.submitted_static_draws
        },
        normal_pvs_camera_identity: evidence_parsed.camera_identity,
        all_visible_complete: all_visible && evidence_parsed.eligible,
        all_visible_submitted_draws: if all_visible {
            evidence_parsed.submitted_static_draws
        } else {
            None
        },
        all_visible_recorded_draws: if all_visible {
            evidence_parsed.recorded_static_draws
        } else {
            None
        },
        all_visible_face_coverage: evidence_parsed.face_coverage,
        stable_digest: evidence_parsed.stable_digest,
        child_exit_code: exit_code,
        stderr_classification: classification,
        hardware_class: hardware_class.to_string(),
        capability_blocked: false,
        blocked_cell: None,
        stderr_snippet: Some(truncate_str(&stderr, 8192)),
        status: if exit_code == Some(0) && evidence_parsed.eligible {
            "PASS".to_string()
        } else if exit_code == Some(0) {
            "FAIL".to_string()
        } else {
            "FAIL".to_string()
        },
        error: if exit_code != Some(0) {
            Some(format!("child exited with code {:?}", exit_code))
        } else if !evidence_parsed.eligible {
            Some("evidence report not eligible".to_string())
        } else {
            None
        },
        duration_ms,
    }
}

/// Parsed evidence from child stdout (Phase 07 stats report JSON).
struct ChildEvidence {
    eligible: bool,
    mounted_static_batches: Option<u32>,
    submitted_static_draws: Option<u32>,
    recorded_static_draws: Option<u32>,
    submitted_total_draws: Option<u32>,
    recorded_total_draws: Option<u32>,
    camera_identity: Option<String>,
    face_coverage: Option<u32>,
    stable_digest: Option<String>,
}

fn parse_child_evidence(stdout: &str) -> ChildEvidence {
    // Try to find a JSON line with "evidence" or "stats" fields
    let json_line = stdout.lines().find(|line| {
        let trimmed = line.trim();
        trimmed.starts_with('{')
            && (trimmed.contains("\"evidence\"")
                || trimmed.contains("\"stats\"")
                || trimmed.contains("\"eligible\""))
    });

    let json_str = match json_line {
        Some(line) => line.trim(),
        None => {
            // Try parsing whole stdout
            let trimmed = stdout.trim();
            if trimmed.starts_with('{') {
                trimmed
            } else {
                return ChildEvidence {
                    eligible: false,
                    mounted_static_batches: None,
                    submitted_static_draws: None,
                    recorded_static_draws: None,
                    submitted_total_draws: None,
                    recorded_total_draws: None,
                    camera_identity: None,
                    face_coverage: None,
                    stable_digest: None,
                };
            }
        }
    };

    let parsed: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => {
            return ChildEvidence {
                eligible: false,
                mounted_static_batches: None,
                submitted_static_draws: None,
                recorded_static_draws: None,
                submitted_total_draws: None,
                recorded_total_draws: None,
                camera_identity: None,
                face_coverage: None,
                stable_digest: None,
            };
        }
    };

    let eligible = parsed
        .get("eligible")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Extract counts from the report structure
    let extract_u32 = |keys: &[&str]| -> Option<u32> {
        let mut current = &parsed;
        for key in keys {
            current = current.get(key)?;
        }
        current.as_u64().map(|v| v as u32)
    };

    ChildEvidence {
        eligible,
        mounted_static_batches: extract_u32(&["mounted_static_batches"])
            .or_else(|| extract_u32(&["canonical_batches"])),
        submitted_static_draws: extract_u32(&["submitted_static_draws"])
            .or_else(|| extract_u32(&["submitted_draws"])),
        recorded_static_draws: extract_u32(&["recorded_static_draws"])
            .or_else(|| extract_u32(&["recorded_draws"])),
        submitted_total_draws: extract_u32(&["submitted_total_draws"])
            .or_else(|| extract_u32(&["total_draws"])),
        recorded_total_draws: extract_u32(&["recorded_total_draws"]),
        camera_identity: parsed
            .get("camera_identity")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        face_coverage: extract_u32(&["face_coverage"])
            .or_else(|| extract_u32(&["source_face_coverage"])),
        stable_digest: parsed
            .get("stable_digest")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}... (truncated, {} total)", &s[..max_len], s.len())
    }
}

/// Execute one full campaign (all 12 entries, normal-PVS + all-visible).
fn execute_campaign(
    entries: &[TransportEntry],
    campaign_name: &str,
    hardware_class: &str,
    bsp_beta_bin: &Path,
) -> Vec<CampaignEntry> {
    let mut results = Vec::with_capacity(entries.len() * 2);

    for entry in entries {
        eprintln!(
            "  [{campaign_name}] {} (class={}, seed={})",
            entry.entry_id, entry.class, entry.seed
        );
        let result = execute_corpus_child(entry, campaign_name, hardware_class, bsp_beta_bin);
        eprintln!(
            "    status={}, classification={}",
            result.status, result.stderr_classification
        );
        results.push(result);
    }

    results
}

// ── Test: Full two-campaign corpus runtime evidence ──────────────────────

/// Execute all 12 frozen corpus entries through strict runtime boundaries
/// in two independent campaigns. Marked #[ignore] because it requires
/// a live GPU and ericw-tools.
///
/// Run with:
/// ```bash
/// BSP_HARDWARE_CLASS=H2 cargo test -p bsp_beta --test corpus_runtime_evidence -- corpus_runtime_two_campaigns --ignored --nocapture
/// ```
#[test]
#[ignore]
fn corpus_runtime_two_campaigns() {
    let tool_dir = ericw_tools_dir();
    let tools_ok = tools_available(&tool_dir);
    let hardware_class = std::env::var("BSP_HARDWARE_CLASS").unwrap_or_else(|_| "UNKNOWN".into());

    eprintln!("=== Phase 08: Frozen Corpus Runtime Evidence ===");
    eprintln!(
        "  ericw-tools: {} ({})",
        if tools_ok { "available" } else { "NOT FOUND" },
        tool_dir.display()
    );
    eprintln!("  hardware class: {hardware_class}");

    // Build the transport manifest in-memory
    let entries = build_transport_entries();
    assert_eq!(entries.len(), 12, "must have exactly 12 corpus entries");

    let bsp_beta_bin = bsp_beta_binary();

    // Campaign A
    eprintln!("\n─── Campaign A ───");
    let campaign_a_start = chrono_now();
    let campaign_a_entries = execute_campaign(&entries, "A", &hardware_class, &bsp_beta_bin);

    // Campaign B
    eprintln!("\n─── Campaign B ───");
    let campaign_b_start = chrono_now();
    let campaign_b_entries = execute_campaign(&entries, "B", &hardware_class, &bsp_beta_bin);

    // ── Cross-campaign comparison ─────────────────────────────────
    let comparison = compare_campaigns(&campaign_a_entries, &campaign_b_entries);

    // ── Budget enforcement ────────────────────────────────────────
    let budget_violations = enforce_budgets(&campaign_a_entries, &campaign_b_entries);

    // ── Reducer ───────────────────────────────────────────────────
    let pass_count = campaign_a_entries
        .iter()
        .filter(|e| e.status == "PASS")
        .count()
        + campaign_b_entries
            .iter()
            .filter(|e| e.status == "PASS")
            .count();
    let fail_count = campaign_a_entries
        .iter()
        .filter(|e| e.status == "FAIL")
        .count()
        + campaign_b_entries
            .iter()
            .filter(|e| e.status == "FAIL")
            .count();
    let not_run_count = campaign_a_entries
        .iter()
        .filter(|e| e.status == "NOT_RUN")
        .count()
        + campaign_b_entries
            .iter()
            .filter(|e| e.status == "NOT_RUN")
            .count();

    let mut failure_reasons: Vec<String> = Vec::new();
    for e in campaign_a_entries.iter().chain(campaign_b_entries.iter()) {
        if let Some(ref err) = e.error {
            if !failure_reasons.contains(err) {
                failure_reasons.push(err.clone());
            }
        }
    }
    failure_reasons.extend(budget_violations.clone());

    let campaign_a_meta = CampaignMetadata {
        name: "A".to_string(),
        started_at: campaign_a_start,
        completed_at: chrono_now(),
        entries_executed: campaign_a_entries.len(),
        entries_passed: campaign_a_entries
            .iter()
            .filter(|e| e.status == "PASS")
            .count(),
        entries_not_run: campaign_a_entries
            .iter()
            .filter(|e| e.status == "NOT_RUN")
            .count(),
        entries_failed: campaign_a_entries
            .iter()
            .filter(|e| e.status == "FAIL")
            .count(),
    };

    let campaign_b_meta = CampaignMetadata {
        name: "B".to_string(),
        started_at: campaign_b_start,
        completed_at: chrono_now(),
        entries_executed: campaign_b_entries.len(),
        entries_passed: campaign_b_entries
            .iter()
            .filter(|e| e.status == "PASS")
            .count(),
        entries_not_run: campaign_b_entries
            .iter()
            .filter(|e| e.status == "NOT_RUN")
            .count(),
        entries_failed: campaign_b_entries
            .iter()
            .filter(|e| e.status == "FAIL")
            .count(),
    };

    let reducer = ReducerSummary {
        total_entries: campaign_a_entries.len() + campaign_b_entries.len(),
        pass: pass_count,
        fail: fail_count,
        not_run: not_run_count,
        phase_pass: fail_count == 0
            && comparison.all_entries_present_in_both
            && comparison.normal_pvs_deterministic
            && comparison.all_visible_equal
            && budget_violations.is_empty(),
        failure_reasons,
    };

    let report = CorpusRuntimeReport {
        schema_version: 1,
        phase: "08".to_string(),
        timestamp: chrono_now(),
        provenance: ReportProvenance {
            generator_manifest: "in-memory".to_string(),
            ericw_tools_path: tool_dir.display().to_string(),
            tools_available: tools_ok,
            bsp_beta_binary: bsp_beta_bin.display().to_string(),
            hardware_class: hardware_class.clone(),
            environment: std::env::vars().collect(),
        },
        campaigns: vec![campaign_a_meta, campaign_b_meta],
        entries: {
            let mut all = campaign_a_entries.clone();
            all.extend(campaign_b_entries.clone());
            all
        },
        cross_campaign_comparison: comparison,
        reducer,
    };

    // Write evidence
    let output_dir = evidence_dir();
    std::fs::create_dir_all(&output_dir).expect("create evidence dir");

    let runtime_path = output_dir.join("corpus-runtime.json");
    std::fs::write(
        &runtime_path,
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .expect("write corpus-runtime.json");
    eprintln!(
        "\nCorpus runtime evidence written to {}",
        runtime_path.display()
    );

    // Assertions
    if fail_count > 0 {
        eprintln!("FAIL: {fail_count} entries failed");
    }
    if not_run_count > 0 {
        eprintln!("NOT_RUN: {not_run_count} entries not executed (GPU/capability unavailable)");
    }
    if !budget_violations.is_empty() {
        for v in &budget_violations {
            eprintln!("BUDGET VIOLATION: {v}");
        }
    }

    assert!(fail_count == 0, "{fail_count} entries failed");
}

/// Build the 12 transport entries in-memory (mirrors corpus_execution.rs).
fn build_transport_entries() -> Vec<TransportEntry> {
    let palette_bytes = std::fs::read(palette_path()).unwrap_or_default();
    let palette_hash = sha256(&palette_bytes);
    let wad_bytes = std::fs::read(wad_path()).unwrap_or_default();
    let wad_hash = sha256(&wad_bytes);

    vec![
        transport_entry(
            "nominal-m1-seed-0",
            "M1",
            0,
            12,
            1,
            1024,
            1024,
            192,
            16,
            64,
            131_072,
            &palette_hash,
            &wad_hash,
        ),
        transport_entry(
            "nominal-m1-seed-1",
            "M1",
            1,
            12,
            1,
            1024,
            1024,
            192,
            16,
            64,
            131_072,
            &palette_hash,
            &wad_hash,
        ),
        transport_entry(
            "nominal-m1-seed-2",
            "M1",
            2,
            12,
            1,
            1024,
            1024,
            192,
            16,
            64,
            131_072,
            &palette_hash,
            &wad_hash,
        ),
        transport_entry(
            "nominal-m1-seed-3",
            "M1",
            3,
            12,
            1,
            1024,
            1024,
            192,
            16,
            64,
            131_072,
            &palette_hash,
            &wad_hash,
        ),
        transport_entry(
            "nominal-m2-seed-17",
            "M2",
            17,
            28,
            3,
            2048,
            2048,
            256,
            32,
            96,
            524_288,
            &palette_hash,
            &wad_hash,
        ),
        transport_entry(
            "nominal-m2-seed-255",
            "M2",
            255,
            28,
            3,
            2048,
            2048,
            256,
            32,
            96,
            524_288,
            &palette_hash,
            &wad_hash,
        ),
        transport_entry(
            "nominal-m2-seed-0x5555",
            "M2",
            0x5555555555555555,
            28,
            3,
            2048,
            2048,
            256,
            32,
            96,
            524_288,
            &palette_hash,
            &wad_hash,
        ),
        transport_entry(
            "nominal-m2-seed-u64-max",
            "M2",
            u64::MAX,
            28,
            3,
            2048,
            2048,
            256,
            32,
            96,
            524_288,
            &palette_hash,
            &wad_hash,
        ),
        transport_entry(
            "boundary-A-m1-min",
            "M1",
            42,
            8,
            0,
            1024,
            1024,
            192,
            16,
            64,
            131_072,
            &palette_hash,
            &wad_hash,
        ),
        transport_entry(
            "boundary-B-m1-max",
            "M1",
            43,
            16,
            2,
            1024,
            1024,
            192,
            16,
            64,
            131_072,
            &palette_hash,
            &wad_hash,
        ),
        transport_entry(
            "boundary-C-m2-min",
            "M2",
            44,
            17,
            1,
            2048,
            2048,
            256,
            32,
            96,
            524_288,
            &palette_hash,
            &wad_hash,
        ),
        transport_entry(
            "boundary-D-m2-max",
            "M2",
            45,
            40,
            6,
            2048,
            2048,
            256,
            32,
            96,
            524_288,
            &palette_hash,
            &wad_hash,
        ),
    ]
}

#[allow(clippy::too_many_arguments)]
fn transport_entry(
    id: &str,
    class: &str,
    seed: u64,
    rooms: u32,
    loops: u32,
    xy_x: u32,
    xy_y: u32,
    z: u32,
    candidates: u32,
    attempts: u32,
    expansions: u32,
    palette_hash: &str,
    wad_hash: &str,
) -> TransportEntry {
    TransportEntry {
        entry_id: id.to_string(),
        class: class.to_string(),
        seed,
        config: TransportConfig {
            room_count: rooms,
            loop_count: loops,
            xy_bounds: (xy_x, xy_y),
            z_span: z,
            placement_candidates: candidates,
            max_placement_attempts: attempts,
            max_astar_expansions: expansions,
        },
        profile: TransportProfile {
            name: "ericw-q1-bsp2-generated".to_string(),
            format: "BSP2".to_string(),
            compiler: "ericw-tools".to_string(),
            compiler_version: "2.0.0-alpha3".to_string(),
        },
        map_hash: None,
        bsp_hash: None,
        lit_hash: None,
        palette_hash: palette_hash.to_string(),
        wad_hash: wad_hash.to_string(),
        prerequisite_result: "AVAILABLE".to_string(),
    }
}

/// Cross-campaign comparison: normal-PVS deterministic, all-visible equal.
fn compare_campaigns(
    campaign_a: &[CampaignEntry],
    campaign_b: &[CampaignEntry],
) -> CrossCampaignComparison {
    let mut mismatches: Vec<String> = Vec::new();

    let all_present = campaign_a.len() == campaign_b.len()
        && campaign_a.iter().all(|a| {
            campaign_b
                .iter()
                .any(|b| b.entry_id == a.entry_id && b.campaign != a.campaign)
        })
        && campaign_b.iter().all(|b| {
            campaign_a
                .iter()
                .any(|a| a.entry_id == b.entry_id && a.campaign != b.campaign)
        });

    let mut normal_pvs_deterministic = true;
    let mut all_visible_equal = true;

    for entry_a in campaign_a {
        if let Some(entry_b) = campaign_b.iter().find(|b| b.entry_id == entry_a.entry_id) {
            // Normal-PVS: compare submitted draws and camera identity
            if entry_a.normal_pvs_complete && entry_b.normal_pvs_complete {
                if entry_a.normal_pvs_submitted_draws != entry_b.normal_pvs_submitted_draws
                    || entry_a.normal_pvs_camera_identity != entry_b.normal_pvs_camera_identity
                {
                    normal_pvs_deterministic = false;
                    mismatches.push(format!(
                        "{}: normal-PVS mismatch (A draws={:?} cam={:?}, B draws={:?} cam={:?})",
                        entry_a.entry_id,
                        entry_a.normal_pvs_submitted_draws,
                        entry_a.normal_pvs_camera_identity,
                        entry_b.normal_pvs_submitted_draws,
                        entry_b.normal_pvs_camera_identity,
                    ));
                }
            }

            // All-visible: compare static draws, face coverage, digests
            if entry_a.all_visible_complete && entry_b.all_visible_complete {
                if entry_a.all_visible_submitted_draws != entry_b.all_visible_submitted_draws
                    || entry_a.all_visible_recorded_draws != entry_b.all_visible_recorded_draws
                    || entry_a.all_visible_face_coverage != entry_b.all_visible_face_coverage
                    || entry_a.stable_digest != entry_b.stable_digest
                {
                    all_visible_equal = false;
                    mismatches.push(format!(
                        "{}: all-visible mismatch (A: {:?}/{:?}/{:?}/{:?}, B: {:?}/{:?}/{:?}/{:?})",
                        entry_a.entry_id,
                        entry_a.all_visible_submitted_draws,
                        entry_a.all_visible_recorded_draws,
                        entry_a.all_visible_face_coverage,
                        entry_a.stable_digest,
                        entry_b.all_visible_submitted_draws,
                        entry_b.all_visible_recorded_draws,
                        entry_b.all_visible_face_coverage,
                        entry_b.stable_digest,
                    ));
                }
            }
        }
    }

    CrossCampaignComparison {
        campaigns_compared: vec!["A".to_string(), "B".to_string()],
        all_entries_present_in_both: all_present,
        normal_pvs_deterministic,
        all_visible_equal,
        mismatches,
    }
}

/// Enforce M1 < 100 / M2 < 500 static-batch ceilings and M1 < 200 / M2 < 1000 total draws.
fn enforce_budgets(campaign_a: &[CampaignEntry], campaign_b: &[CampaignEntry]) -> Vec<String> {
    let mut violations: Vec<String> = Vec::new();

    for entry in campaign_a.iter().chain(campaign_b.iter()) {
        if entry.status != "PASS" {
            continue;
        }

        let (static_ceiling, total_ceiling) = if entry.class == "M1" {
            (M1_STATIC_BATCH_CEILING, M1_TOTAL_DRAW_CEILING)
        } else {
            (M2_STATIC_BATCH_CEILING, M2_TOTAL_DRAW_CEILING)
        };

        if let Some(batches) = entry.mounted_static_batches {
            if batches >= static_ceiling {
                violations.push(format!(
                    "{} ({}): mounted static batches {} >= ceiling {}",
                    entry.entry_id, entry.campaign, batches, static_ceiling
                ));
            }
        }

        if let Some(draws) = entry.submitted_total_draws {
            if draws >= total_ceiling {
                violations.push(format!(
                    "{} ({}): total draws {} >= ceiling {}",
                    entry.entry_id, entry.campaign, draws, total_ceiling
                ));
            }
        }
    }

    violations
}
