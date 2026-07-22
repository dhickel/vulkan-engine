//! Telemetry recording for generation, meshing, and upload timing.
//!
//! `PhaseTiming` is a JSON-serializable struct that captures wall-clock
//! measurements for each stage of the cave-to-scene pipeline. A
//! `TelemetryRecorder` writes JSONL records to a configurable path.

use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

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
        let line = serde_json::to_string(record)
            .map_err(|e| format!("JSON serialization failed: {e}"))?;
        writeln!(self.writer, "{line}")
            .map_err(|e| format!("write failed: {e}"))?;
        self.writer
            .flush()
            .map_err(|e| format!("flush failed: {e}"))?;
        Ok(())
    }
}
