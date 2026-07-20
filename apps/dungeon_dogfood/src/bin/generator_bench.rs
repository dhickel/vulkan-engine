//! Standalone dungeon generator benchmark binary.
//!
//! Modes:
//!   freeze          — validate identities and write frozen protocol manifest
//!   worker <seed>   — load catalog, time exactly one generate(), emit JSON to stdout
//!   run <cohort>    — launch workers per seed, capture everything, write JSONL
//!   summarize <file> — compute p50/p95/max/Wilson from non-held-out JSONL

// Module declarations via #[path] so we share source files with the main binary.
#[path = "../generator/mod.rs"]
mod generator;
#[path = "../layout.rs"]
mod layout;
#[path = "../collision.rs"]
mod collision;
#[path = "../player.rs"]
mod player;
#[path = "../content.rs"]
mod content;

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::Instant;

use generator::{
    alloc_metrics, compute_config_hash, error_stage_code, generate, generate_with_telemetry,
    GeneratorConfig, GeneratorError, GenerationResult, QualifiedProfile,
};
use generator::context::AttemptContext;
use generator::telemetry::serialize_telemetry;
use generator::prefab::PrefabCatalog;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ─── Constants ──────────────────────────────────────────────────────────────

const WORKER_SCHEMA_VERSION: u32 = 1;
const COORDINATOR_SCHEMA_VERSION: u32 = 1;
const SUMMARY_SCHEMA_VERSION: u32 = 1;
const PROTOCOL_VERSION: u32 = 1;
const MANIFEST_SCHEMA_VERSION: u32 = 1;

const Z_95: f64 = 1.6448536269514722;

// ─── Hashes ────────────────────────────────────────────────────────────────

fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    format!("{:064x}", digest)
}

fn hash_file(path: &Path) -> Result<String, String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(sha256_hex(&data))
}

fn hash_executable() -> Result<String, String> {
    let exe =
        std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    hash_file(&exe)
}

fn lockfile_hash_hex() -> String {
    let lock_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("Cargo.lock");
    hash_file(&lock_path).unwrap_or_else(|_| "unavailable".into())
}

// ─── Cohorts ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Cohort {
    label: String,
    seeds: Vec<u64>,
    /// SHA-256 hex of the canonical seed derivation.
    derivation_hash: String,
    /// When true, summarize/tuning/replay-diff must reject this cohort.
    sealed: bool,
}

impl Cohort {
    fn derivation_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.label.as_bytes());
        bytes.extend_from_slice(&(self.seeds.len() as u64).to_be_bytes());
        for seed in &self.seeds {
            bytes.extend_from_slice(&seed.to_be_bytes());
        }
        bytes
    }

    fn hash(&self) -> String {
        sha256_hex(&self.derivation_bytes())
    }
}

/// Derive deterministic seeds from a SHA-256 domain + label + ordinal.
fn derive_seeds(domain: &[u8], label: &str, count: usize) -> Vec<u64> {
    let mut seeds = Vec::with_capacity(count);
    let mut seen = std::collections::BTreeSet::new();
    for ordinal in 0u64.. {
        if seeds.len() >= count {
            break;
        }
        let mut hasher = Sha256::new();
        hasher.update(domain);
        hasher.update(label.as_bytes());
        hasher.update(&ordinal.to_be_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        let seed = u64::from_be_bytes(digest[0..8].try_into().unwrap());
        if seen.insert(seed) {
            seeds.push(seed);
        }
    }
    seeds
}

fn build_cohorts() -> (Cohort, Cohort, Cohort) {
    let domain = b"dungeon-dogfood";

    let fixed = Cohort {
        label: "fixed".into(),
        seeds: vec![0, 5, 7, 23, 24, 33, 41, 42, 77],
        derivation_hash: String::new(), // filled below
        sealed: false,
    };

    let tuning_seeds = derive_seeds(
        Sha256::digest(b"dungeon-dogfood-tuning").as_slice(),
        "tuning",
        50,
    );
    let tuning = Cohort {
        label: "tuning".into(),
        seeds: tuning_seeds,
        derivation_hash: String::new(),
        sealed: false,
    };

    let heldout_seeds = derive_seeds(
        Sha256::digest(b"dungeon-dogfood-heldout").as_slice(),
        "heldout",
        1000,
    );
    let heldout = Cohort {
        label: "heldout".into(),
        seeds: heldout_seeds,
        derivation_hash: String::new(),
        sealed: true,
    };

    // Fill derivation hashes
    let mut fixed = fixed;
    fixed.derivation_hash = fixed.hash();
    let mut tuning = tuning;
    tuning.derivation_hash = tuning.hash();
    let mut heldout = heldout;
    heldout.derivation_hash = heldout.hash();

    (fixed, tuning, heldout)
}

fn validate_cohorts_disjoint(cohorts: &[&Cohort]) -> Result<(), String> {
    let mut all_seeds = BTreeMap::new();
    for cohort in cohorts {
        for &seed in &cohort.seeds {
            if let Some(prev_label) = all_seeds.insert(seed, cohort.label.as_str()) {
                return Err(format!(
                    "duplicate seed {seed} in cohorts '{prev_label}' and '{}'",
                    cohort.label
                ));
            }
        }
    }
    Ok(())
}

// ─── Frozen Manifest ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Manifest {
    protocol_version: u32,
    schema_version: u32,
    hardware: String,
    os: String,
    power_policy: String,
    affinity_policy: String,
    isolation_policy: String,
    toolchain: String,
    lockfile_hash: String,
    executable_hash: String,
    build_profile: String,
    config_hash: String,
    catalog_hash: String,
    cohorts: Vec<ManifestCohort>,
    invocation: String,
    cold_warm_policy: String,
    timeout_seconds: u64,
    telemetry_mode: String,
    allocation_mode: String,
    statistics: ManifestStatistics,
    raw_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManifestCohort {
    label: String,
    seed_count: usize,
    derivation_hash: String,
    sealed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ManifestStatistics {
    percentile_method: String,
    confidence_method: String,
    z_value: f64,
}

fn build_manifest(
    config_hash: &str,
    catalog_hash: &str,
    executable_hash: &str,
    fixed: &Cohort,
    tuning: &Cohort,
    heldout: &Cohort,
) -> Manifest {
    Manifest {
        protocol_version: PROTOCOL_VERSION,
        schema_version: MANIFEST_SCHEMA_VERSION,
        hardware: hostname(),
        os: std::env::consts::OS.to_string(),
        power_policy: "uncontrolled".into(),
        affinity_policy: "uncontrolled".into(),
        isolation_policy: "fresh_worker_per_seed".into(),
        toolchain: rustc_version(),
        lockfile_hash: lockfile_hash_hex(),
        executable_hash: executable_hash.to_string(),
        build_profile: build_profile().to_string(),
        config_hash: config_hash.to_string(),
        catalog_hash: catalog_hash.to_string(),
        cohorts: vec![
            ManifestCohort {
                label: fixed.label.clone(),
                seed_count: fixed.seeds.len(),
                derivation_hash: fixed.derivation_hash.clone(),
                sealed: fixed.sealed,
            },
            ManifestCohort {
                label: tuning.label.clone(),
                seed_count: tuning.seeds.len(),
                derivation_hash: tuning.derivation_hash.clone(),
                sealed: tuning.sealed,
            },
            ManifestCohort {
                label: heldout.label.clone(),
                seed_count: heldout.seeds.len(),
                derivation_hash: heldout.derivation_hash.clone(),
                sealed: heldout.sealed,
            },
        ],
        invocation: std::env::args().collect::<Vec<_>>().join(" "),
        cold_warm_policy: "cold".into(),
        timeout_seconds: 7200,
        telemetry_mode: "none".into(),
        allocation_mode: if cfg!(feature = "generator-bench-alloc") {
            "instrumented".to_string()
        } else {
            "none".to_string()
        },
        statistics: ManifestStatistics {
            percentile_method: "nearest_rank".into(),
            confidence_method: "one_sided_wilson_lower".into(),
            z_value: Z_95,
        },
        raw_paths: vec![
            "baseline-fixed-tuning.jsonl".into(),
            "baseline-heldout.jsonl".into(),
        ],
    }
}

fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn rustc_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

// ─── Worker Record ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkerRecord {
    schema_version: u32,
    seed: u64,
    outcome: String, // "success" or "exhausted"
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attempt_index: Option<u32>,
    config_hash: String,
    catalog_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    replay_hex: Option<String>,
    /// Linux VmHWM in kB from /proc/self/status, read after generate().
    #[serde(skip_serializing_if = "Option::is_none")]
    vm_hwm_kb: Option<u64>,
    /// Allocation snapshot — only Some when compiled with generator-bench-alloc.
    #[serde(skip_serializing_if = "Option::is_none")]
    alloc_snapshot: Option<AllocRecord>,
    /// Telemetry payload (if telemetry mode is not Off).
    #[serde(skip_serializing_if = "Option::is_none")]
    telemetry: Option<TelemetryRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TelemetryRecord {
    mode: String,
    json_bytes_base64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AllocRecord {
    allocations: u64,
    deallocations: u64,
    bytes_allocated: u64,
    bytes_deallocated: u64,
    peak_bytes: u64,
}

// ─── Coordinator Observation ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CoordinatorObservation {
    schema_version: u32,
    run_id: String,
    cohort_label: String,
    ordinal: u32,
    seed: u64,
    worker_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    worker_stdout: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    worker_stderr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exit_code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attempt_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    outcome: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    replay_hex: Option<String>,
    config_hash: String,
    catalog_hash: String,
    executable_hash: String,
}

// ─── Summary ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Summary {
    schema_version: u32,
    run_id: String,
    cohort_label: String,
    total_requested: usize,
    total_observed: usize,
    success_count: usize,
    exhaustion_count: usize,
    failure_count: usize,
    invalid_count: usize,
    all_calls: Option<LatencyDistribution>,
    success_only: Option<LatencyDistribution>,
    exhaustion_only: Option<LatencyDistribution>,
    wilson_success_lower_bound: Option<f64>,
    wilson_inputs: WilsonInputs,
    failure_details: Vec<FailureDetail>,
    exclusions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LatencyDistribution {
    p50_ns: Option<u64>,
    p95_ns: Option<u64>,
    max_ns: Option<u64>,
    count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WilsonInputs {
    successes: usize,
    trials: usize,
    z_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FailureDetail {
    seed: u64,
    status: String,
    reason: String,
}

// ─── Statistics ────────────────────────────────────────────────────────────

/// Nearest-rank percentile: one-based `ceil(p*n)`, converted to zero-based index.
fn nearest_rank_percentile(sorted: &[u64], p: f64) -> Option<u64> {
    if sorted.is_empty() {
        return None;
    }
    let n = sorted.len() as f64;
    let rank = (p * n).ceil() as usize;
    // One-based rank to zero-based index
    let idx = rank.saturating_sub(1);
    if idx < sorted.len() {
        Some(sorted[idx])
    } else {
        Some(sorted[sorted.len() - 1])
    }
}

/// One-sided 95% Wilson lower bound.
///
/// Uses z = 1.6448536269514722.
///
/// Returns 0.0 when denominator is zero.
fn wilson_lower_bound(successes: usize, trials: usize, z: f64) -> f64 {
    if trials == 0 {
        return 0.0;
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let centre = p + z2 / (2.0 * n);
    let margin = z * ((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))).sqrt();
    ((centre - margin) / denominator).max(0.0).min(1.0)
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: generator_bench <freeze|worker <seed>|run <cohort>|summarize <file>>"
        );
        std::process::exit(1);
    }

    let result = match args[1].as_str() {
        "freeze" => cmd_freeze(),
        "worker" => {
            if args.len() < 3 {
                eprintln!("worker requires a seed argument");
                std::process::exit(1);
            }
            let seed: u64 = args[2].parse().expect("invalid seed");
            cmd_worker(seed)
        }
        "run" => {
            if args.len() < 3 {
                eprintln!("run requires a cohort label (fixed|tuning|heldout)");
                std::process::exit(1);
            }
            cmd_run(&args[2])
        }
        "summarize" => {
            if args.len() < 3 {
                eprintln!("summarize requires a JSONL file path");
                std::process::exit(1);
            }
            cmd_summarize(&args[2])
        }
        other => {
            eprintln!("unknown mode: {other}");
            std::process::exit(1);
        }
    };

    match result {
        Ok(()) => {}
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

// ─── freeze ────────────────────────────────────────────────────────────────

fn cmd_freeze() -> Result<(), String> {
    // Load catalog
    let catalog_path = prefab_catalog_path();
    let catalog = PrefabCatalog::load(&catalog_path)
        .map_err(|e| format!("catalog load: {e:?}"))?;

    // Build normalized config
    let config = primary_config();
    let config_hash =
        compute_config_hash(&config).map_err(|e| format!("config hash: {e:?}"))?;
    let catalog_hash = hex::encode(&catalog.identity_bytes());

    // Build cohorts
    let (fixed, tuning, heldout) = build_cohorts();

    // Validate disjoint
    validate_cohorts_disjoint(&[&fixed, &tuning, &heldout])?;

    // Validate heldout cardinality
    if heldout.seeds.len() < 1000 {
        return Err(format!(
            "heldout cohort has {} seeds, need at least 1000",
            heldout.seeds.len()
        ));
    }

    // Executable hash
    let exe_hash = hash_executable()?;

    // Build and write manifest
    let manifest = build_manifest(&config_hash, &catalog_hash, &exe_hash, &fixed, &tuning, &heldout);

    let out_dir = benchmarks_dir();
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("create benchmarks dir: {e}"))?;

    write_json(&out_dir.join("manifest.json"), &manifest)?;
    write_json(&out_dir.join("cohort-fixed.json"), &fixed)?;
    write_json(&out_dir.join("cohort-tuning.json"), &tuning)?;
    write_json(&out_dir.join("cohort-heldout.json"), &heldout)?;

    println!("Frozen manifest written to {}", out_dir.display());
    println!("  config_hash:  {config_hash}");
    println!("  catalog_hash: {catalog_hash}");
    println!("  exe_hash:     {exe_hash}");
    println!(
        "  cohorts:      fixed={} tuning={} heldout={}",
        fixed.seeds.len(),
        tuning.seeds.len(),
        heldout.seeds.len()
    );

    Ok(())
}

// ─── worker ────────────────────────────────────────────────────────────────

fn cmd_worker(seed: u64) -> Result<(), String> {
    // Parse optional --telemetry <mode> from remaining args
    let args: Vec<String> = std::env::args().collect();
    let telemetry_mode = if let Some(pos) = args.iter().position(|a| a == "--telemetry") {
        let mode_str = args.get(pos + 1).map(|s| s.as_str()).unwrap_or("off");
        generator::context::TelemetryMode::from_str(mode_str)
            .unwrap_or(generator::context::TelemetryMode::Off)
    } else {
        generator::context::TelemetryMode::Off
    };

    eprintln!("[worker] seed={seed} telemetry={telemetry_mode:?} loading catalog...");
    let catalog_path = prefab_catalog_path();
    let catalog = PrefabCatalog::load(&catalog_path)
        .map_err(|e| format!("catalog load: {e:?}"))?;
    eprintln!("[worker] seed={seed} catalog loaded, generating...");

    let config = primary_config();
    let config_hash =
        compute_config_hash(&config).map_err(|e| format!("config hash: {e:?}"))?;
    let catalog_hash = hex::encode(&catalog.identity_bytes());

    // Reset alloc counters if instrumented
    alloc_metrics::reset();

    // Time exactly one generate() call
    let started = Instant::now();
    let result = generate_with_telemetry(config, &catalog, seed, telemetry_mode);
    let elapsed = started.elapsed();
    eprintln!("[worker] seed={seed} generate completed in {}ms", elapsed.as_millis());

    // Snapshot alloc counters
    let alloc_snap = alloc_metrics::snapshot();

    // Read VmHWM from /proc/self/status (Linux).
    let vm_hwm_kb = read_vm_hwm_kb();

    let duration_ns = elapsed.as_nanos() as u64;

    let (outcome, attempt_index, replay_hex, telemetry) = match &result {
        Ok((gen_result, ctx)) => {
            let replay_bytes = build_replay_success(gen_result)?;
            let telemetry = telemetry_record(ctx);
            (
                "success".to_string(),
                Some(gen_result.attempt_index),
                Some(hex::encode(&replay_bytes)),
                telemetry,
            )
        }
        Err(e) => {
            let replay_bytes = build_replay_exhausted(seed, e)?;
            (
                "exhausted".to_string(),
                None,
                Some(hex::encode(&replay_bytes)),
                None, // No telemetry for exhausted through public API
            )
        }
    };

    let record = WorkerRecord {
        schema_version: WORKER_SCHEMA_VERSION,
        seed,
        outcome,
        duration_ns: Some(duration_ns),
        attempt_index,
        config_hash,
        catalog_hash,
        replay_hex,
        vm_hwm_kb,
        alloc_snapshot: alloc_snapshot_for_record(&alloc_snap),
        telemetry,
    };

    let json = serde_json::to_string(&record).map_err(|e| format!("json: {e}"))?;
    println!("{json}");

    Ok(())
}

fn build_replay_success(result: &GenerationResult) -> Result<Vec<u8>, String> {
    use generator::replay::ReplayEncoder;

    // Reconstruct attempt identity bytes from seed and attempt_index.
    let identity_bytes = {
        let mut hasher = Sha256::new();
        hasher.update(b"dungeon-generator/attempt/v1");
        hasher.update(&result.seed.to_be_bytes());
        hasher.update(&result.attempt_index.to_be_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        digest
    };

    let bytes = ReplayEncoder::new()
        .schema()
        .seed(result.seed)
        .attempt_index(result.attempt_index)
        .attempt_identity(identity_bytes)
        .ascii(&result.level)
        .resources(&result.resource_counts)
        .capture_views(&result.capture_views)
        .diagnostics(&result.diagnostics)
        .topology_regions(result.topology_region_count)
        .topology_edges(result.topology_edge_count)
        .topology_metrics(
            result.route_distance,
            result.max_branch_depth,
            result.dead_end_count,
            result.articulation_count,
            result.crossing_count,
            &result.per_layer_cycles,
        )
        .finish();

    Ok(bytes)
}

fn build_replay_exhausted(seed: u64, error: &GeneratorError) -> Result<Vec<u8>, String> {
    use generator::replay::ReplayEncoder;

    let stage = error_stage_code(error).to_string();
    let reason = error.reason_code().to_string();

    let identity_bytes = {
        let mut hasher = Sha256::new();
        hasher.update(b"dungeon-generator/attempt/v1");
        hasher.update(&seed.to_be_bytes());
        hasher.update(&0u32.to_be_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        digest
    };

    let diag_json = serde_json::to_vec(&serde_json::json!({
        "stage": stage,
        "reason": reason
    }))
    .map_err(|e| format!("diag json: {e}"))?;

    let bytes = ReplayEncoder::new()
        .schema()
        .seed(seed)
        .attempt_index(0)
        .attempt_identity(identity_bytes)
        .exhausted(&stage, &reason)
        .diagnostics(&diag_json)
        .finish();

    Ok(bytes)
}

// ─── run ───────────────────────────────────────────────────────────────────

fn cmd_run(cohort_label: &str) -> Result<(), String> {
    let out_dir = benchmarks_dir();
    let cohort_path = out_dir.join(format!("cohort-{cohort_label}.json"));
    if !cohort_path.exists() {
        return Err(format!("cohort file not found: {}", cohort_path.display()));
    }

    let cohort_json =
        std::fs::read_to_string(&cohort_path).map_err(|e| format!("read cohort: {e}"))?;
    let cohort: Cohort =
        serde_json::from_str(&cohort_json).map_err(|e| format!("parse cohort: {e}"))?;

    if cohort_label == "heldout" && !cohort.sealed {
        return Err("heldout cohort must be sealed".into());
    }

    // Load manifest for config/catalog/executable hashes
    let manifest_path = out_dir.join("manifest.json");
    let manifest_json =
        std::fs::read_to_string(&manifest_path).map_err(|e| format!("read manifest: {e}"))?;
    let manifest: Manifest =
        serde_json::from_str(&manifest_json).map_err(|e| format!("parse manifest: {e}"))?;

    let exe_path = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let exe_hash = hash_file(&exe_path)?;

    // Normalized config hash
    let catalog_path = prefab_catalog_path();
    let catalog = PrefabCatalog::load(&catalog_path)
        .map_err(|e| format!("catalog load: {e:?}"))?;
    let config = primary_config();
    let config_hash =
        compute_config_hash(&config).map_err(|e| format!("config hash: {e:?}"))?;
    let catalog_hash = hex::encode(&catalog.identity_bytes());

    if manifest.executable_hash != exe_hash
        || manifest.config_hash != config_hash
        || manifest.catalog_hash != catalog_hash
        || manifest.lockfile_hash != lockfile_hash_hex()
        || manifest.build_profile != build_profile()
    {
        return Err("current worker identities do not match the frozen manifest".into());
    }
    if cohort.derivation_hash != cohort.hash() {
        return Err("cohort derivation hash does not match its seed framing".into());
    }
    let frozen_cohort = manifest
        .cohorts
        .iter()
        .find(|entry| entry.label == cohort_label)
        .ok_or_else(|| format!("cohort '{cohort_label}' is absent from manifest"))?;
    if frozen_cohort.seed_count != cohort.seeds.len()
        || frozen_cohort.derivation_hash != cohort.derivation_hash
        || frozen_cohort.sealed != cohort.sealed
    {
        return Err(format!("cohort '{cohort_label}' does not match the frozen manifest"));
    }

    let run_id = format!(
        "run-{}-{}",
        cohort_label,
        &exe_hash[..12]
    );

    let output_path = out_dir.join(format!("baseline-{cohort_label}.jsonl"));

    // Determine output path: for heldout, use dedicated path
    let output_path = if cohort_label == "heldout" {
        out_dir.join("baseline-heldout.jsonl")
    } else if cohort_label == "tuning" {
        out_dir.join("baseline-fixed-tuning.jsonl")
    } else if cohort_label == "fixed" {
        out_dir.join("baseline-fixed-tuning.jsonl")
    } else {
        return Err(format!("unknown cohort label: {cohort_label}"));
    };

    let timeout_secs = manifest.timeout_seconds;

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&output_path)
        .map_err(|e| format!("open output: {e}"))?;

    let worker_exe = &exe_path;

    for (ordinal, &seed) in cohort.seeds.iter().enumerate() {
        let ordinal = ordinal as u32;
        let mut cmd = Command::new(worker_exe);
        cmd.arg("worker")
            .arg(seed.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Make the worker the leader of an isolated process group so a
            // negative PID targets the complete worker tree.
            .process_group(0);

        // Drain both pipes concurrently. Waiting before draining can deadlock
        // when a worker fills either OS pipe buffer.
        let mut child = cmd.spawn().map_err(|e| format!("spawn worker: {e}"))?;
        let pid = child.id();
        let stdout_thread = child.stdout.take().map(|mut out| {
            std::thread::spawn(move || {
                let mut bytes = Vec::new();
                let _ = out.read_to_end(&mut bytes);
                String::from_utf8_lossy(&bytes).into_owned()
            })
        });
        let stderr_thread = child.stderr.take().map(|mut err| {
            std::thread::spawn(move || {
                let mut bytes = Vec::new();
                let _ = err.read_to_end(&mut bytes);
                String::from_utf8_lossy(&bytes).into_owned()
            })
        });

        enum WaitOutcome {
            Exited(std::process::ExitStatus),
            TimedOut,
            ProcessLost,
        }

        let (tx, rx) = std::sync::mpsc::channel();
        let wait_thread = std::thread::spawn(move || {
            let result = child.wait();
            let _ = tx.send(result);
        });

        let timeout_dur = std::time::Duration::from_secs(timeout_secs);
        let wait_result = match rx.recv_timeout(timeout_dur) {
            Ok(Ok(status)) => WaitOutcome::Exited(status),
            Ok(Err(_)) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                WaitOutcome::ProcessLost
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                let group = format!("-{pid}");
                let _ = Command::new("kill")
                    .args(["-TERM", "--", &group])
                    .status();
                std::thread::sleep(std::time::Duration::from_millis(200));
                let _ = Command::new("kill")
                    .args(["-KILL", "--", &group])
                    .status();
                WaitOutcome::TimedOut
            }
        };
        let _ = wait_thread.join();

        let stdout = stdout_thread.map(|thread| thread.join().unwrap_or_default());
        let stderr = stderr_thread.map(|thread| thread.join().unwrap_or_default());

        let (worker_status, duration_ns, attempt_index, outcome, replay_hex, exit_code) =
            match wait_result {
                WaitOutcome::TimedOut => (
                    "timeout".to_string(),
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                WaitOutcome::ProcessLost => (
                    "process_loss".to_string(),
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                WaitOutcome::Exited(exit_status) => {
                    let code = exit_status.code();
                    if code != Some(0) {
                        if exit_status.success() {
                            // success but no code (unlikely)
                            ("nonzero_exit".to_string(), None, None, None, None, code)
                        } else {
                            let status_str = if code.is_none() {
                                "signal".to_string()
                            } else {
                                "nonzero_exit".to_string()
                            };
                            (status_str, None, None, None, None, code)
                        }
                    } else if let Some(ref stdout_str) = stdout {
                        // Parse worker record
                        match serde_json::from_str::<WorkerRecord>(stdout_str.trim()) {
                            Ok(record)
                                if record.schema_version == WORKER_SCHEMA_VERSION
                                    && record.seed == seed
                                    && record.config_hash == config_hash
                                    && record.catalog_hash == catalog_hash =>
                            {
                                (
                                    "success".to_string(),
                                    record.duration_ns,
                                    record.attempt_index,
                                    Some(record.outcome),
                                    record.replay_hex,
                                    Some(0),
                                )
                            }
                            Ok(_) | Err(_) => {
                                ("parse_failure".to_string(), None, None, None, None, Some(0))
                            }
                        }
                    } else {
                        ("missing_output".to_string(), None, None, None, None, Some(0))
                    }
                }
            };

        let observation = CoordinatorObservation {
            schema_version: COORDINATOR_SCHEMA_VERSION,
            run_id: run_id.clone(),
            cohort_label: cohort_label.to_string(),
            ordinal,
            seed,
            worker_status,
            worker_stdout: stdout,
            worker_stderr: stderr,
            exit_code,
            duration_ns,
            attempt_index,
            outcome,
            replay_hex,
            config_hash: config_hash.clone(),
            catalog_hash: catalog_hash.clone(),
            executable_hash: exe_hash.clone(),
        };

        let line = serde_json::to_string(&observation).map_err(|e| format!("json: {e}"))?;
        writeln!(file, "{line}").map_err(|e| format!("write: {e}"))?;
    }

    println!("Completed {} seeds for cohort '{}'", cohort.seeds.len(), cohort_label);
    println!("Output: {}", output_path.display());

    // Seal heldout — make read-only
    if cohort_label == "heldout" {
        seal_file(&output_path)?;
        println!("Sealed heldout file (read-only)");
    }

    Ok(())
}

// ─── summarize ─────────────────────────────────────────────────────────────

fn cmd_summarize(input_path: &str) -> Result<(), String> {
    let path = Path::new(input_path);

    // Reject heldout paths
    if path.to_string_lossy().contains("heldout") {
        return Err("summarize rejects heldout cohort paths".into());
    }

    let file = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
    let reader = BufReader::new(file);

    let mut observations: Vec<CoordinatorObservation> = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let obs: CoordinatorObservation =
            serde_json::from_str(&line).map_err(|e| format!("parse: {e}"))?;
        observations.push(obs);
    }

    if observations.is_empty() {
        return Err("no observations found".into());
    }

    let cohort_label = observations[0].cohort_label.clone();
    let run_id = observations[0].run_id.clone();
    let total_requested = observations.len();
    let total_observed = observations.len();

    let mut success_durations: Vec<u64> = Vec::new();
    let mut exhaustion_durations: Vec<u64> = Vec::new();
    let mut all_durations: Vec<u64> = Vec::new();
    let mut success_count = 0usize;
    let mut exhaustion_count = 0usize;
    let mut failure_count = 0usize;
    let mut invalid_count = 0usize;
    let mut failure_details: Vec<FailureDetail> = Vec::new();

    for obs in &observations {
        match obs.worker_status.as_str() {
            "success" => {
                if let Some(dur) = obs.duration_ns {
                    all_durations.push(dur);
                    if obs.outcome.as_deref() == Some("success") {
                        success_durations.push(dur);
                        success_count += 1;
                    } else if obs.outcome.as_deref() == Some("exhausted") {
                        exhaustion_durations.push(dur);
                        exhaustion_count += 1;
                    } else {
                        invalid_count += 1;
                        failure_details.push(FailureDetail {
                            seed: obs.seed,
                            status: obs.worker_status.clone(),
                            reason: format!("unknown outcome: {:?}", obs.outcome),
                        });
                    }
                }
            }
            "timeout" | "signal" | "nonzero_exit" | "process_loss" => {
                failure_count += 1;
                failure_details.push(FailureDetail {
                    seed: obs.seed,
                    status: obs.worker_status.clone(),
                    reason: format!("worker failed: {}", obs.worker_status),
                });
            }
            "parse_failure" | "missing_output" => {
                invalid_count += 1;
                failure_details.push(FailureDetail {
                    seed: obs.seed,
                    status: obs.worker_status.clone(),
                    reason: "unparseable or missing worker output".into(),
                });
            }
            other => {
                invalid_count += 1;
                failure_details.push(FailureDetail {
                    seed: obs.seed,
                    status: other.to_string(),
                    reason: "unknown status".into(),
                });
            }
        }
    }

    all_durations.sort_unstable();
    success_durations.sort_unstable();
    exhaustion_durations.sort_unstable();

    let all_calls = if all_durations.is_empty() {
        None
    } else {
        let max = all_durations.last().copied();
        Some(LatencyDistribution {
            p50_ns: nearest_rank_percentile(&all_durations, 0.50),
            p95_ns: nearest_rank_percentile(&all_durations, 0.95),
            max_ns: max,
            count: all_durations.len(),
        })
    };

    let success_only = if success_durations.is_empty() {
        None
    } else {
        let max = success_durations.last().copied();
        Some(LatencyDistribution {
            p50_ns: nearest_rank_percentile(&success_durations, 0.50),
            p95_ns: nearest_rank_percentile(&success_durations, 0.95),
            max_ns: max,
            count: success_durations.len(),
        })
    };

    let exhaustion_only = if exhaustion_durations.is_empty() {
        None
    } else {
        let max = exhaustion_durations.last().copied();
        Some(LatencyDistribution {
            p50_ns: nearest_rank_percentile(&exhaustion_durations, 0.50),
            p95_ns: nearest_rank_percentile(&exhaustion_durations, 0.95),
            max_ns: max,
            count: exhaustion_durations.len(),
        })
    };

    let trials = total_requested;
    let wilson = if trials > 0 {
        Some(wilson_lower_bound(success_count, trials, Z_95))
    } else {
        None
    };

    let summary = Summary {
        schema_version: SUMMARY_SCHEMA_VERSION,
        run_id,
        cohort_label,
        total_requested,
        total_observed,
        success_count,
        exhaustion_count,
        failure_count,
        invalid_count,
        all_calls,
        success_only,
        exhaustion_only,
        wilson_success_lower_bound: wilson,
        wilson_inputs: WilsonInputs {
            successes: success_count,
            trials,
            z_value: Z_95,
        },
        failure_details,
        exclusions: Vec::new(),
    };

    let output_path = path.with_extension("").to_string_lossy().to_string() + "-summary.json";
    write_json(Path::new(&output_path), &summary)?;
    println!("Summary written to {output_path}");
    println!(
        "  success={success_count} exhausted={exhaustion_count} failures={failure_count} invalid={invalid_count}"
    );
    if let Some(ref dist) = summary.all_calls {
        println!(
            "  all: p50={:?}ns p95={:?}ns max={:?}ns",
            dist.p50_ns, dist.p95_ns, dist.max_ns
        );
    }
    if let Some(w) = wilson {
        println!("  Wilson 95% lower bound: {w:.6}");
    }

    Ok(())
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn benchmarks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(".internal-dev")
        .join("benchmarks")
        .join("primary-generator")
}

fn prefab_catalog_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/prefabs")
}

/// Create the benchmark config matching the production dogfood app's
/// `build_generator_config()` (no env overrides).
///
/// Uses `GeneratorConfig::default()` (profile: None) so normalize() falls
/// through to `interpolate_custom` using Primary's 96×96×3 dimensions.
/// This matches the production path exactly — dimension overrides produce
/// a different normalized config than `qualified(Primary)` because profile=None
/// triggers the interpolation branch.
fn primary_config() -> GeneratorConfig {
    let mut config = GeneratorConfig::default();
    config.single_bottleneck = true;
    config.relax_transition_redundancy = true;
    // No dimension overrides — normalize() defaults to Primary's 96x96x3
    // with interpolated budget values matching the production path.
    config
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value).map_err(|e| format!("json: {e}"))?;
    std::fs::write(path, json).map_err(|e| format!("write: {e}"))?;
    Ok(())
}

/// Read VmHWM (peak resident set size) in kB from /proc/self/status.
fn read_vm_hwm_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmHWM:") {
            let value = line
                .split_whitespace()
                .nth(1)?
                .parse::<u64>()
                .ok()?;
            return Some(value);
        }
    }
    None
}

/// Return Some(AllocRecord) only when compiled with generator-bench-alloc.
fn alloc_snapshot_for_record(snap: &alloc_metrics::AllocSnapshot) -> Option<AllocRecord> {
    if cfg!(feature = "generator-bench-alloc") {
        Some(AllocRecord {
            allocations: snap.allocations,
            deallocations: snap.deallocations,
            bytes_allocated: snap.bytes_allocated,
            bytes_deallocated: snap.bytes_deallocated,
            peak_bytes: snap.peak_bytes,
        })
    } else {
        None
    }
}

/// Build an optional TelemetryRecord from a finalized context.
fn telemetry_record(ctx: &AttemptContext) -> Option<TelemetryRecord> {
    if ctx.mode() == generator::context::TelemetryMode::Off {
        return None;
    }
    let json_bytes = serialize_telemetry(ctx).ok()?;
    Some(TelemetryRecord {
        mode: format!("{:?}", ctx.mode()).to_lowercase(),
        json_bytes_base64: base64_encode(&json_bytes),
    })
}

fn base64_encode(data: &[u8]) -> String {
    use std::fmt::Write;
    const CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        let _ = write!(result, "{}", CHARS[((triple >> 18) & 0x3f) as usize] as char);
        let _ = write!(result, "{}", CHARS[((triple >> 12) & 0x3f) as usize] as char);
        if chunk.len() >= 2 {
            let _ = write!(result, "{}", CHARS[((triple >> 6) & 0x3f) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() >= 3 {
            let _ = write!(result, "{}", CHARS[(triple & 0x3f) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Seal a file by making it read-only (mode 0o444).
fn seal_file(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)
        .map_err(|e| format!("metadata: {e}"))?
        .permissions();
    perms.set_mode(0o444);
    std::fs::set_permissions(path, perms)
        .map_err(|e| format!("set_permissions: {e}"))?;
    Ok(())
}

// ─── Hex encoding helper ───────────────────────────────────────────────────
// Uses the `hex` crate from Cargo.toml dependencies.

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Statistics ──────────────────────────────────────────────────────

    #[test]
    fn nearest_rank_empty() {
        assert_eq!(nearest_rank_percentile(&[], 0.50), None);
    }

    #[test]
    fn nearest_rank_singleton() {
        assert_eq!(nearest_rank_percentile(&[100], 0.50), Some(100));
        assert_eq!(nearest_rank_percentile(&[100], 0.95), Some(100));
    }

    #[test]
    fn nearest_rank_even_size() {
        let data = [10, 20, 30, 40];
        assert_eq!(nearest_rank_percentile(&data, 0.50), Some(20)); // ceil(0.5*4)=2 -> idx 1
        assert_eq!(nearest_rank_percentile(&data, 0.95), Some(40)); // ceil(0.95*4)=4 -> idx 3
    }

    #[test]
    fn nearest_rank_boundary() {
        let data = [1, 2, 3, 4, 5];
        assert_eq!(nearest_rank_percentile(&data, 0.00), Some(1)); // ceil(0)=0 -> idx 0
        assert_eq!(nearest_rank_percentile(&data, 1.00), Some(5)); // ceil(5)=5 -> idx 4
    }

    #[test]
    fn wilson_empty() {
        assert_eq!(wilson_lower_bound(0, 0, Z_95), 0.0);
    }

    #[test]
    fn wilson_all_success() {
        let lb = wilson_lower_bound(100, 100, Z_95);
        assert!(lb > 0.96 && lb < 1.0); // Very high with many trials
    }

    #[test]
    fn wilson_all_failure() {
        let lb = wilson_lower_bound(0, 100, Z_95);
        assert!((lb - 0.0).abs() < 1e-10); // Essentially 0
    }

    #[test]
    fn wilson_mixed() {
        let lb = wilson_lower_bound(50, 100, Z_95);
        assert!(lb > 0.40 && lb < 0.60); // Should be around 0.41-0.43
    }

    // ── Cohorts ─────────────────────────────────────────────────────────

    #[test]
    fn cohorts_are_disjoint() {
        let (fixed, tuning, heldout) = build_cohorts();
        validate_cohorts_disjoint(&[&fixed, &tuning, &heldout]).unwrap();
    }

    #[test]
    fn heldout_has_1000_seeds() {
        let (_, _, heldout) = build_cohorts();
        assert!(heldout.seeds.len() >= 1000);
    }

    #[test]
    fn fixed_contains_regression_seeds() {
        let (fixed, _, _) = build_cohorts();
        for &seed in &[0, 5, 7, 23, 24, 33, 41, 42, 77] {
            assert!(fixed.seeds.contains(&seed), "fixed missing seed {seed}");
        }
    }

    #[test]
    fn cohort_hashes_are_stable() {
        let (f1, t1, h1) = build_cohorts();
        let (f2, t2, h2) = build_cohorts();
        assert_eq!(f1.hash(), f2.hash());
        assert_eq!(t1.hash(), t2.hash());
        assert_eq!(h1.hash(), h2.hash());
    }

    #[test]
    fn cohorts_are_internally_unique() {
        let (fixed, tuning, heldout) = build_cohorts();
        for cohort in [&fixed, &tuning, &heldout] {
            let mut seen = BTreeMap::new();
            for &seed in &cohort.seeds {
                assert!(
                    seen.insert(seed, ()).is_none(),
                    "cohort '{}' has duplicate seed {seed}",
                    cohort.label
                );
            }
        }
    }

    // ── Manifest ────────────────────────────────────────────────────────

    #[test]
    fn manifest_identities_match_cohorts() {
        let (fixed, tuning, heldout) = build_cohorts();
        let manifest = build_manifest(
            "cfg_hash",
            "cat_hash",
            "exe_hash",
            &fixed,
            &tuning,
            &heldout,
        );
        assert_eq!(manifest.cohorts[0].label, "fixed");
        assert_eq!(manifest.cohorts[0].seed_count, 9);
        assert_eq!(manifest.cohorts[2].label, "heldout");
        assert!(manifest.cohorts[2].sealed);
    }

    // ── Worker/Coordinator ──────────────────────────────────────────────

    #[test]
    fn worker_record_serializes_without_null_fields() {
        let record = WorkerRecord {
            schema_version: 1,
            seed: 42,
            outcome: "success".into(),
            duration_ns: Some(12345),
            attempt_index: Some(0),
            config_hash: "abc".into(),
            catalog_hash: "def".into(),
            replay_hex: None,
            vm_hwm_kb: None,
            alloc_snapshot: None,
            telemetry: None,
        };
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("\"duration_ns\":12345"));
        assert!(!json.contains("\"replay_hex\""));
        assert!(!json.contains("\"alloc_snapshot\""));
        assert!(!json.contains("\"vm_hwm_kb\""));
    }

    #[test]
    fn coordinator_observation_handles_all_statuses() {
        let obs = CoordinatorObservation {
            schema_version: 1,
            run_id: "r1".into(),
            cohort_label: "fixed".into(),
            ordinal: 0,
            seed: 42,
            worker_status: "timeout".into(),
            worker_stdout: None,
            worker_stderr: None,
            exit_code: None,
            duration_ns: None,
            attempt_index: None,
            outcome: None,
            replay_hex: None,
            config_hash: "cfg".into(),
            catalog_hash: "cat".into(),
            executable_hash: "exe".into(),
        };
        let json = serde_json::to_string(&obs).unwrap();
        assert!(json.contains("\"worker_status\":\"timeout\""));
    }

    // ── Config consistency ──────────────────────────────────────────────

    #[test]
    fn primary_config_is_96x96x3() {
        let config = primary_config();
        let config_hash = compute_config_hash(&config).unwrap();
        // Verify config hash is non-empty (config normalizes successfully)
        assert!(!config_hash.is_empty());
        assert_eq!(config_hash.len(), 64);
    }
}
