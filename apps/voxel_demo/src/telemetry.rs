//! Telemetry recording for generation, meshing, and upload timing.
//!
//! `PhaseTiming` is a JSON-serializable struct that captures wall-clock
//! measurements for each stage of the cave-to-scene pipeline. A
//! `TelemetryRecorder` writes JSONL records to a configurable path.

use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

// ─── Environment ───────────────────────────────────────────────────────────

/// Build environment metadata.
#[derive(Debug, Clone, Serialize)]
pub struct BuildEnv {
    /// Rust toolchain triple, e.g. "x86_64-unknown-linux-gnu".
    pub target_triple: String,
    /// Build profile ("debug" or "release").
    pub profile: String,
    /// Operating system name.
    pub os: String,
}

impl Default for BuildEnv {
    fn default() -> Self {
        Self {
            target_triple: std::env::consts::ARCH.to_string()
                + "-"
                + std::env::consts::FAMILY
                + "-"
                + std::env::consts::OS,
            profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
            .to_string(),
            os: std::env::consts::OS.to_string(),
        }
    }
}

// ─── PhaseTiming ───────────────────────────────────────────────────────────

/// Timing and metadata for one generation-mesh-partition pipeline run.
#[derive(Debug, Clone, Serialize)]
pub struct PhaseTiming {
    /// Preset name (e.g. "default", "cavernous").
    pub preset: String,
    /// RNG seed.
    pub seed: u64,
    /// Cubic resolution.
    pub resolution: u32,

    // ── Timings (milliseconds) ─────────────────────────────────────────
    /// Generation wall-clock (caverns + tunnels + maze).
    pub generation_ms: f64,
    /// MC33 extraction wall-clock.
    pub mc33_ms: f64,
    /// Wall/floor partition wall-clock.
    pub partition_ms: f64,
    /// Mesh-to-CpuMesh conversion wall-clock.
    pub conversion_ms: f64,
    /// Total CPU time (generation + MC33 + partition + conversion).
    pub total_cpu_ms: f64,

    // ── Geometry counts ────────────────────────────────────────────────
    /// Number of wall triangles after partition.
    pub wall_triangles: usize,
    /// Number of floor triangles after partition.
    pub floor_triangles: usize,
    /// Total lattice voxels (resolution³).
    pub total_voxels: u64,

    // ── Generator metadata ─────────────────────────────────────────────
    /// Number of sites placed.
    pub site_count: usize,
    /// Number of spline edges.
    pub spline_edges: usize,
    /// Number of maze links created.
    pub maze_links: usize,
    /// Number of point lights derived.
    pub light_count: usize,
    /// Number of viewpoints derived.
    pub viewpoint_count: usize,

    // ── Environment ────────────────────────────────────────────────────
    /// Build environment metadata.
    pub build_env: BuildEnv,

    // ── Lifecycle ──────────────────────────────────────────────────────
    /// Request identity for this generation.
    pub request_id: String,

    // ── Upload timing ──────────────────────────────────────────────────
    /// Mesh upload wall-clock to GPU (milliseconds).
    pub upload_ms: f64,
    /// Material creation wall-clock (milliseconds).
    pub material_create_ms: f64,
}

impl PhaseTiming {
    /// Create a new PhaseTiming with the given preset/seed/resolution and default env/request.
    pub fn new(preset: &str, seed: u64, resolution: u32) -> Self {
        Self {
            preset: preset.to_string(),
            seed,
            resolution,
            generation_ms: 0.0,
            mc33_ms: 0.0,
            partition_ms: 0.0,
            conversion_ms: 0.0,
            total_cpu_ms: 0.0,
            wall_triangles: 0,
            floor_triangles: 0,
            total_voxels: 0,
            site_count: 0,
            spline_edges: 0,
            maze_links: 0,
            light_count: 0,
            viewpoint_count: 0,
            build_env: BuildEnv::default(),
            request_id: String::new(),
            upload_ms: 0.0,
            material_create_ms: 0.0,
        }
    }
}

/// A record written to the JSONL file for a single campaign run.
#[derive(Debug, Clone, Serialize)]
pub struct CampaignRecord {
    /// Preset name.
    pub preset: String,
    /// RNG seed.
    pub seed: u64,
    /// Cubic resolution.
    pub resolution: u32,
    /// Whether validation passed.
    pub passed: bool,
    /// Error message if validation failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Timing data (present on success).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<PhaseTiming>,
    /// Build environment metadata.
    pub build_env: BuildEnv,
}

// ─── TelemetryRecorder ─────────────────────────────────────────────────────

/// Appends JSONL records to a file.
pub struct TelemetryRecorder {
    writer: BufWriter<File>,
}

impl TelemetryRecorder {
    /// Open (or create) a JSONL file at `path`. Appends to existing files.
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("failed to open telemetry file {}: {e}", path.display()))?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Write a single campaign record as one JSON line.
    pub fn record(&mut self, record: &CampaignRecord) -> Result<(), String> {
        let line =
            serde_json::to_string(record).map_err(|e| format!("JSON serialization failed: {e}"))?;
        writeln!(self.writer, "{line}").map_err(|e| format!("write failed: {e}"))?;
        self.writer
            .flush()
            .map_err(|e| format!("flush failed: {e}"))?;
        Ok(())
    }
}
