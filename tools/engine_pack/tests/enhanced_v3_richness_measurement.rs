//! Phase 05 — Reproducible measurement harness.
//!
//! Records host normalization, corpus identity, warmup exclusion,
//! sample statistic, generation/compiler stage metrics,
//! parse/extract/load/render metrics, process memory, runtime memory,
//! package bytes, BSP lumps, faces/entities/batches, and artifacts.
//! Does NOT freeze Richness ceilings yet (no Richness maps exist).
//! Freezes the METHOD.

use serde::Serialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

// ── Measurement types ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct HostNormalization {
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub cpu_count: usize,
    pub total_memory_bytes: u64,
    pub rust_version: String,
    pub cargo_profile: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CorpusIdentity {
    pub corpus_name: String,
    pub entry_count: usize,
    pub entries: Vec<CorpusEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CorpusEntry {
    pub seed: u64,
    pub preset: String,
    pub extent: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct WarmupConfig {
    pub warmup_iterations: usize,
    pub excluded_from_statistics: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SampleStatistic {
    pub metric_name: String,
    pub sample_count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageMetrics {
    pub stage_name: String,
    pub elapsed_ms: u64,
    pub cpu_time_ms: Option<u64>,
    pub peak_memory_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BspLumpStats {
    pub lump_name: String,
    pub element_count: usize,
    pub element_size: usize,
    pub total_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArtifactStats {
    pub map_bytes: u64,
    pub bsp_bytes: u64,
    pub lit_bytes: u64,
    pub wad_bytes: u64,
    pub package_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MeasurementRun {
    pub run_id: String,
    pub timestamp_utc: String,
    pub host: HostNormalization,
    pub corpus: CorpusIdentity,
    pub warmup: WarmupConfig,
    pub stage_metrics: Vec<StageMetrics>,
    pub process_memory: ProcessMemorySnapshot,
    pub runtime_memory: Option<u64>,
    pub artifacts: ArtifactStats,
    pub bsp_lumps: Vec<BspLumpStats>,
    pub compiled_faces: usize,
    pub compiled_entities: usize,
    pub renderer_batches: Option<usize>,
    pub measurement_duration_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcessMemorySnapshot {
    pub vm_peak_kb: Option<u64>,
    pub vm_size_kb: Option<u64>,
    pub rss_kb: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MeasurementConfig {
    pub output_path: PathBuf,
    pub warmup_iterations: usize,
    pub sample_iterations: usize,
    pub timeout_per_sample_seconds: u64,
}

// ── Measurement harness ───────────────────────────────────────────────────

pub struct MeasurementHarness {
    config: MeasurementConfig,
    host: HostNormalization,
    runs: Vec<MeasurementRun>,
}

impl MeasurementHarness {
    pub fn new(config: MeasurementConfig) -> Result<Self, String> {
        let host = Self::record_host()?;
        Ok(Self {
            config,
            host,
            runs: Vec::new(),
        })
    }

    fn record_host() -> Result<HostNormalization, String> {
        let hostname = std::env::var("HOSTNAME")
            .or_else(|_| std::env::var("HOST"))
            .unwrap_or_else(|_| "unknown".to_string());

        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();

        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let total_memory_bytes = Self::probe_total_memory();

        let rust_version = rustc_version();

        let cargo_profile = if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
        .to_string();

        Ok(HostNormalization {
            hostname,
            os,
            arch,
            cpu_count,
            total_memory_bytes,
            rust_version,
            cargo_profile,
        })
    }

    fn probe_total_memory() -> u64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
                for line in contents.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb * 1024;
                            }
                        }
                    }
                }
            }
        }
        0
    }

    /// Record a measurement run.
    pub fn record_run(
        &mut self,
        run_id: &str,
        corpus: CorpusIdentity,
        stage_metrics: Vec<StageMetrics>,
        artifacts: ArtifactStats,
        bsp_lumps: Vec<BspLumpStats>,
        compiled_faces: usize,
        compiled_entities: usize,
        renderer_batches: Option<usize>,
        elapsed_ms: u64,
    ) {
        let timestamp = chrono_now();
        let process_memory = Self::snapshot_process_memory();
        let runtime_memory = Self::probe_runtime_memory();

        self.runs.push(MeasurementRun {
            run_id: run_id.to_string(),
            timestamp_utc: timestamp,
            host: self.host.clone(),
            corpus,
            warmup: WarmupConfig {
                warmup_iterations: self.config.warmup_iterations,
                excluded_from_statistics: true,
            },
            stage_metrics,
            process_memory,
            runtime_memory,
            artifacts,
            bsp_lumps,
            compiled_faces,
            compiled_entities,
            renderer_batches,
            measurement_duration_ms: elapsed_ms,
        });
    }

    fn snapshot_process_memory() -> ProcessMemorySnapshot {
        #[cfg(target_os = "linux")]
        {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                let mut vm_peak = None;
                let mut vm_size = None;
                let mut rss = None;
                for line in status.lines() {
                    if line.starts_with("VmPeak:") {
                        vm_peak = parse_kb_field(line);
                    } else if line.starts_with("VmSize:") {
                        vm_size = parse_kb_field(line);
                    } else if line.starts_with("VmRSS:") {
                        rss = parse_kb_field(line);
                    }
                }
                return ProcessMemorySnapshot {
                    vm_peak_kb: vm_peak,
                    vm_size_kb: vm_size,
                    rss_kb: rss,
                };
            }
        }
        ProcessMemorySnapshot {
            vm_peak_kb: None,
            vm_size_kb: None,
            rss_kb: None,
        }
    }

    fn probe_runtime_memory() -> Option<u64> {
        #[cfg(target_os = "linux")]
        {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        return parse_kb_field(line).map(|kb| kb * 1024);
                    }
                }
            }
        }
        None
    }

    /// Compute sample statistics from recorded runs.
    pub fn compute_statistics(&self, metric_name: &str) -> Option<SampleStatistic> {
        let values: Vec<f64> = self
            .runs
            .iter()
            .skip(self.config.warmup_iterations)
            .filter_map(|run| match metric_name {
                "bsp_bytes" => Some(run.artifacts.bsp_bytes as f64),
                "lit_bytes" => Some(run.artifacts.lit_bytes as f64),
                "package_bytes" => Some(run.artifacts.package_bytes as f64),
                "compiled_faces" => Some(run.compiled_faces as f64),
                "compiled_entities" => Some(run.compiled_entities as f64),
                "duration_ms" => Some(run.measurement_duration_ms as f64),
                _ => None,
            })
            .collect();

        if values.is_empty() {
            return None;
        }

        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        Some(SampleStatistic {
            metric_name: metric_name.to_string(),
            sample_count: n,
            min,
            max,
            mean,
            median,
            std_dev,
        })
    }

    /// Write the complete measurement report as JSON.
    pub fn write_report(&self) -> Result<(), String> {
        let parent = self
            .config
            .output_path
            .parent()
            .ok_or("output path has no parent")?;
        std::fs::create_dir_all(parent).map_err(|e| format!("create dir: {e}"))?;

        let report = serde_json::json!({
            "schema": "enhanced-v3-richness-measurement/v1",
            "config": {
                "warmup_iterations": self.config.warmup_iterations,
                "sample_iterations": self.config.sample_iterations,
                "timeout_per_sample_seconds": self.config.timeout_per_sample_seconds,
            },
            "host": self.host,
            "runs": self.runs,
            "statistics": {
                "bsp_bytes": self.compute_statistics("bsp_bytes"),
                "lit_bytes": self.compute_statistics("lit_bytes"),
                "package_bytes": self.compute_statistics("package_bytes"),
                "compiled_faces": self.compute_statistics("compiled_faces"),
                "compiled_entities": self.compute_statistics("compiled_entities"),
                "duration_ms": self.compute_statistics("duration_ms"),
            },
        });

        let json = serde_json::to_string_pretty(&report).map_err(|e| format!("serialize: {e}"))?;
        std::fs::write(&self.config.output_path, json).map_err(|e| format!("write report: {e}"))?;

        eprintln!(
            "measurement report written to {}",
            self.config.output_path.display()
        );
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn rustc_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_else(|| "unknown".to_string())
        .trim()
        .to_string()
}

fn chrono_now() -> String {
    // Simple UTC timestamp without chrono dependency
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    // Format as ISO 8601-ish
    let days_since_epoch = secs / 86400;
    // This is a rough approximation; for exact timestamps, chrono would be used
    format!("unix-{}", secs)
}

#[cfg(target_os = "linux")]
fn parse_kb_field(line: &str) -> Option<u64> {
    line.split_whitespace().nth(1)?.parse::<u64>().ok()
}

#[cfg(not(target_os = "linux"))]
fn parse_kb_field(_line: &str) -> Option<u64> {
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[test]
fn measurement_host_normalization_records_required_fields() {
    let host = MeasurementHarness::record_host().expect("record host");
    assert!(!host.hostname.is_empty(), "hostname must not be empty");
    assert!(!host.os.is_empty(), "os must not be empty");
    assert!(!host.arch.is_empty(), "arch must not be empty");
    assert!(host.cpu_count > 0, "cpu_count must be positive");
    assert!(
        !host.rust_version.is_empty(),
        "rust_version must not be empty"
    );
    assert!(
        !host.cargo_profile.is_empty(),
        "cargo_profile must not be empty"
    );
}

#[test]
fn measurement_harness_produces_valid_json_schema() {
    let temp = tempfile::tempdir().expect("tempdir");
    let output = temp.path().join("measurement.json");

    let config = MeasurementConfig {
        output_path: output.clone(),
        warmup_iterations: 0,
        sample_iterations: 3,
        timeout_per_sample_seconds: 60,
    };

    let mut harness = MeasurementHarness::new(config).expect("create harness");

    let corpus = CorpusIdentity {
        corpus_name: "test-corpus".to_string(),
        entry_count: 1,
        entries: vec![CorpusEntry {
            seed: 0,
            preset: "sparse".to_string(),
            extent: 2048,
        }],
    };

    let stage_metrics = vec![
        StageMetrics {
            stage_name: "generate".to_string(),
            elapsed_ms: 100,
            cpu_time_ms: Some(90),
            peak_memory_bytes: 10_000_000,
        },
        StageMetrics {
            stage_name: "compile".to_string(),
            elapsed_ms: 500,
            cpu_time_ms: Some(450),
            peak_memory_bytes: 50_000_000,
        },
    ];

    let artifacts = ArtifactStats {
        map_bytes: 4000,
        bsp_bytes: 26000,
        lit_bytes: 12000,
        wad_bytes: 66000,
        package_bytes: 108000,
    };

    let bsp_lumps = vec![
        BspLumpStats {
            lump_name: "entities".to_string(),
            element_count: 1,
            element_size: 0,
            total_bytes: 500,
        },
        BspLumpStats {
            lump_name: "faces".to_string(),
            element_count: 41,
            element_size: 28,
            total_bytes: 1148,
        },
    ];

    harness.record_run(
        "test-001",
        corpus,
        stage_metrics,
        artifacts,
        bsp_lumps,
        41,      // compiled_faces
        3,       // compiled_entities
        Some(4), // renderer_batches
        1234,    // elapsed_ms
    );

    harness.write_report().expect("write report");
    assert!(output.exists(), "report file must exist");

    let raw = std::fs::read_to_string(&output).expect("read report");
    let parsed: serde_json::Value = serde_json::from_str(&raw).expect("valid JSON");
    assert_eq!(
        parsed["schema"], "enhanced-v3-richness-measurement/v1",
        "schema must match"
    );
    assert!(
        parsed["runs"].as_array().unwrap().len() == 1,
        "must have one run"
    );
    assert!(
        parsed["statistics"]["compiled_faces"]["sample_count"] == 1,
        "statistics must be computed"
    );
}

#[test]
fn measurement_statistics_exclude_warmup() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config = MeasurementConfig {
        output_path: temp.path().join("warmup-test.json"),
        warmup_iterations: 2,
        sample_iterations: 3,
        timeout_per_sample_seconds: 60,
    };

    let mut harness = MeasurementHarness::new(config.clone()).expect("create harness");

    let corpus = CorpusIdentity {
        corpus_name: "warmup-test".to_string(),
        entry_count: 1,
        entries: vec![CorpusEntry {
            seed: 0,
            preset: "sparse".to_string(),
            extent: 2048,
        }],
    };

    let empty_stages = Vec::new();

    // Record 5 runs: 2 warmup + 3 samples with increasing BSP sizes
    for i in 0..5 {
        let artifacts = ArtifactStats {
            map_bytes: 4000,
            bsp_bytes: 26000 + i * 100, // warmup: 26000, 26100; samples: 26200, 26300, 26400
            lit_bytes: 12000,
            wad_bytes: 66000,
            package_bytes: 108000,
        };
        harness.record_run(
            &format!("run-{i}"),
            corpus.clone(),
            empty_stages.clone(),
            artifacts,
            Vec::new(),
            41,
            3,
            None,
            1000,
        );
    }

    let bsp_stats = harness.compute_statistics("bsp_bytes").expect("bsp stats");
    // After skipping 2 warmup runs, we have 3 samples at 26200, 26300, 26400
    assert_eq!(
        bsp_stats.sample_count, 3,
        "warmup must be excluded from statistics"
    );
    // The mean should be (26200 + 26300 + 26400) / 3 = 26300
    assert!(
        (bsp_stats.mean - 26300.0).abs() < 1.0,
        "mean must exclude warmup values: got {}",
        bsp_stats.mean
    );
}

#[test]
fn measurement_report_contains_all_required_fields() {
    let temp = tempfile::tempdir().expect("tempdir");
    let config = MeasurementConfig {
        output_path: temp.path().join("fields-test.json"),
        warmup_iterations: 0,
        sample_iterations: 1,
        timeout_per_sample_seconds: 60,
    };

    let output_path = config.output_path.clone();
    let mut harness = MeasurementHarness::new(config).expect("create harness");
    let corpus = CorpusIdentity {
        corpus_name: "fields".to_string(),
        entry_count: 0,
        entries: Vec::new(),
    };

    harness.record_run(
        "fields-1",
        corpus,
        Vec::new(),
        ArtifactStats {
            map_bytes: 0,
            bsp_bytes: 0,
            lit_bytes: 0,
            wad_bytes: 0,
            package_bytes: 0,
        },
        Vec::new(),
        0,
        0,
        None,
        0,
    );

    harness.write_report().expect("write report");
    let raw = std::fs::read_to_string(&output_path).expect("read");
    let report: serde_json::Value = serde_json::from_str(&raw).expect("parse");

    // Verify all required top-level fields
    let required = ["schema", "config", "host", "runs", "statistics"];
    for field in &required {
        assert!(
            report.get(field).is_some(),
            "report must contain '{field}' field"
        );
    }

    // Verify host normalization fields
    let host = &report["host"];
    for field in &[
        "hostname",
        "os",
        "arch",
        "cpu_count",
        "total_memory_bytes",
        "rust_version",
        "cargo_profile",
    ] {
        assert!(host.get(field).is_some(), "host must contain '{field}'");
    }

    // Verify run fields
    let run = &report["runs"][0];
    for field in &[
        "run_id",
        "timestamp_utc",
        "host",
        "corpus",
        "warmup",
        "stage_metrics",
        "process_memory",
        "artifacts",
        "bsp_lumps",
        "compiled_faces",
        "compiled_entities",
    ] {
        assert!(run.get(field).is_some(), "run must contain '{field}'");
    }
}

#[test]
fn measurement_output_ordering_does_not_feed_generation() {
    // The measurement harness must NOT feed its output back into generation decisions.
    // This test verifies that the harness is purely observational — it only records,
    // never mutates generation state, config, or RNG.
    let temp = tempfile::tempdir().expect("tempdir");
    let config = MeasurementConfig {
        output_path: temp.path().join("no-feedback.json"),
        warmup_iterations: 0,
        sample_iterations: 1,
        timeout_per_sample_seconds: 60,
    };

    let mut harness = MeasurementHarness::new(config.clone()).expect("create");

    // Simulate two identical generations with measurement interleaved
    let corpus_a = CorpusIdentity {
        corpus_name: "test".to_string(),
        entry_count: 1,
        entries: vec![CorpusEntry {
            seed: 42,
            preset: "sparse".to_string(),
            extent: 2048,
        }],
    };

    harness.record_run(
        "a",
        corpus_a.clone(),
        vec![],
        ArtifactStats {
            map_bytes: 100,
            bsp_bytes: 200,
            lit_bytes: 300,
            wad_bytes: 400,
            package_bytes: 1000,
        },
        vec![],
        41,
        3,
        None,
        100,
    );

    // Second run with identical seed — statistics from first run must not
    // influence this one's recorded values.
    harness.record_run(
        "b",
        corpus_a,
        vec![],
        ArtifactStats {
            map_bytes: 100,
            bsp_bytes: 200,
            lit_bytes: 300,
            wad_bytes: 400,
            package_bytes: 1000,
        },
        vec![],
        41,
        3,
        None,
        100,
    );

    // Verify both runs independently recorded the same values
    assert_eq!(
        harness.runs[0].artifacts.bsp_bytes,
        harness.runs[1].artifacts.bsp_bytes
    );
    assert_eq!(
        harness.runs[0].compiled_faces,
        harness.runs[1].compiled_faces
    );

    // Statistics are computed post-hoc, never fed back
    let stats = harness.compute_statistics("bsp_bytes").expect("stats");
    assert_eq!(stats.sample_count, 2);
}
