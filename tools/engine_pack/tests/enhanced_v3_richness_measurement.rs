//! Phase 05 — fail-closed, reproducible Richness measurement method.
//!
//! This freezes the measurement method, not Richness ceilings. No Richness
//! map corpus exists yet, so every observation in these tests is synthetic.

use serde::Serialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

const SCHEMA_ID: &str = "enhanced-v3-richness-measurement/v1";
const REQUIRED_STAGES: [&str; 8] = [
    "generate", "qbsp", "vis", "light", "parse", "extract", "load", "render",
];
const REQUIRED_LUMPS: [&str; 15] = [
    "entities",
    "planes",
    "miptex",
    "vertices",
    "visibility",
    "nodes",
    "texinfo",
    "faces",
    "lighting",
    "clipnodes",
    "leaves",
    "marksurfaces",
    "edges",
    "surfedges",
    "models",
];

#[derive(Debug, Clone, Serialize)]
pub struct HostNormalization {
    pub hostname: String,
    pub os: String,
    pub arch: String,
    pub cpu_count: usize,
    pub total_memory_bytes: u64,
    pub rust_version: String,
    pub cargo_profile: String,
    /// Exact environment contract for measured child processes.
    pub normalized_environment: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CorpusIdentity {
    pub corpus_name: String,
    pub entry_count: usize,
    pub ordered_sha256: String,
    pub entries: Vec<CorpusEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct CorpusEntry {
    pub seed: u64,
    pub preset: String,
    pub extent: u32,
}

impl CorpusIdentity {
    pub fn new(
        corpus_name: impl Into<String>,
        mut entries: Vec<CorpusEntry>,
    ) -> Result<Self, String> {
        let corpus_name = corpus_name.into();
        if corpus_name.trim().is_empty() || entries.is_empty() {
            return Err("corpus name and entries are required".into());
        }
        entries.sort();
        if entries.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("corpus entries must be unique".into());
        }
        for entry in &entries {
            if entry.preset.trim().is_empty() || entry.extent == 0 {
                return Err("corpus entries require preset and nonzero extent".into());
            }
        }
        let mut hasher = Sha256::new();
        hasher.update(b"enhanced-v3-richness-measurement-corpus/v1\0");
        hasher.update(corpus_name.as_bytes());
        hasher.update([0]);
        for entry in &entries {
            hasher.update(entry.seed.to_le_bytes());
            hasher.update((entry.preset.len() as u64).to_le_bytes());
            hasher.update(entry.preset.as_bytes());
            hasher.update(entry.extent.to_le_bytes());
        }
        Ok(Self {
            corpus_name,
            entry_count: entries.len(),
            ordered_sha256: format!("{:x}", hasher.finalize()),
            entries,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SampleKind {
    WarmupExcluded,
    Measured,
}

#[derive(Debug, Clone, Serialize)]
pub struct SampleStatistic {
    pub metric_name: String,
    pub sample_count: usize,
    pub selected_statistic: &'static str,
    pub selected_value: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub p95_nearest_rank: f64,
    pub population_std_dev: f64,
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
    /// Zero only for variable-width byte lumps.
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
    pub map_sha256: String,
    pub bsp_sha256: String,
    pub lit_sha256: String,
    pub package_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProcessMemorySnapshot {
    pub vm_peak_bytes: Option<u64>,
    pub vm_size_bytes: Option<u64>,
    pub rss_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunObservation {
    pub stage_metrics: Vec<StageMetrics>,
    pub process_memory: ProcessMemorySnapshot,
    pub runtime_memory_bytes: Option<u64>,
    pub artifacts: ArtifactStats,
    pub bsp_lumps: Vec<BspLumpStats>,
    pub compiled_faces: usize,
    pub compiled_entities: usize,
    pub renderer_batches: Option<usize>,
    pub measurement_duration_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MeasurementRun {
    pub run_id: String,
    pub sequence: usize,
    pub sample_kind: SampleKind,
    pub timestamp_utc: String,
    pub corpus: CorpusIdentity,
    pub stage_metrics: Vec<StageMetrics>,
    pub process_memory: ProcessMemorySnapshot,
    pub runtime_memory_bytes: u64,
    pub artifacts: ArtifactStats,
    pub bsp_lumps: Vec<BspLumpStats>,
    pub compiled_faces: usize,
    pub compiled_entities: usize,
    pub renderer_batches: usize,
    pub measurement_duration_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MeasurementConfig {
    pub output_path: PathBuf,
    pub warmup_iterations: usize,
    pub sample_iterations: usize,
    pub timeout_per_sample_seconds: u64,
}

pub struct MeasurementHarness {
    config: MeasurementConfig,
    host: HostNormalization,
    runs: Vec<MeasurementRun>,
}

impl MeasurementHarness {
    pub fn new(config: MeasurementConfig) -> Result<Self, String> {
        if config.sample_iterations == 0 || config.timeout_per_sample_seconds == 0 {
            return Err("sample_iterations and timeout must be nonzero".into());
        }
        Ok(Self {
            config,
            host: Self::record_host()?,
            runs: Vec::new(),
        })
    }

    fn record_host() -> Result<HostNormalization, String> {
        let total_memory_bytes = probe_total_memory();
        let rust_version = rustc_version();
        if total_memory_bytes == 0 || rust_version == "unknown" {
            return Err("host memory and rustc version must be observable".into());
        }
        Ok(HostNormalization {
            hostname: std::env::var("HOSTNAME")
                .or_else(|_| std::env::var("HOST"))
                .unwrap_or_else(|_| "unknown".into()),
            os: std::env::consts::OS.into(),
            arch: std::env::consts::ARCH.into(),
            cpu_count: std::thread::available_parallelism().map_or(1, |count| count.get()),
            total_memory_bytes,
            rust_version,
            cargo_profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
            .into(),
            normalized_environment: BTreeMap::from([
                ("LANG".into(), "C".into()),
                ("LC_ALL".into(), "C".into()),
                ("RAYON_NUM_THREADS".into(), "1".into()),
                ("TZ".into(), "UTC".into()),
            ]),
        })
    }

    pub fn record_run(
        &mut self,
        run_id: &str,
        corpus: CorpusIdentity,
        mut observation: RunObservation,
    ) -> Result<(), String> {
        let expected = self.config.warmup_iterations + self.config.sample_iterations;
        if self.runs.len() >= expected {
            return Err(format!("received more than {expected} configured runs"));
        }
        if run_id.trim().is_empty() || self.runs.iter().any(|run| run.run_id == run_id) {
            return Err("run_id must be nonempty and unique".into());
        }
        if self.runs.first().is_some_and(|run| run.corpus != corpus) {
            return Err("every run must use the identical canonical corpus".into());
        }
        validate_observation(&observation, self.config.timeout_per_sample_seconds)?;

        observation.stage_metrics.sort_by_key(|stage| {
            REQUIRED_STAGES
                .iter()
                .position(|required| *required == stage.stage_name)
                .expect("validated stage")
        });
        observation.bsp_lumps.sort_by_key(|lump| {
            REQUIRED_LUMPS
                .iter()
                .position(|required| *required == lump.lump_name)
                .expect("validated lump")
        });
        let runtime_memory_bytes = observation
            .runtime_memory_bytes
            .expect("validated runtime memory");
        let renderer_batches = observation
            .renderer_batches
            .expect("validated renderer batches");
        let sequence = self.runs.len();
        self.runs.push(MeasurementRun {
            run_id: run_id.into(),
            sequence,
            sample_kind: if sequence < self.config.warmup_iterations {
                SampleKind::WarmupExcluded
            } else {
                SampleKind::Measured
            },
            timestamp_utc: unix_timestamp_utc(),
            corpus,
            stage_metrics: observation.stage_metrics,
            process_memory: observation.process_memory,
            runtime_memory_bytes,
            artifacts: observation.artifacts,
            bsp_lumps: observation.bsp_lumps,
            compiled_faces: observation.compiled_faces,
            compiled_entities: observation.compiled_entities,
            renderer_batches,
            measurement_duration_ms: observation.measurement_duration_ms,
        });
        Ok(())
    }

    fn metric_values(&self, metric_name: &str) -> Result<Vec<f64>, String> {
        let values = self
            .runs
            .iter()
            .filter(|run| run.sample_kind == SampleKind::Measured)
            .map(|run| match metric_name {
                "map_bytes" => Some(run.artifacts.map_bytes as f64),
                "bsp_bytes" => Some(run.artifacts.bsp_bytes as f64),
                "lit_bytes" => Some(run.artifacts.lit_bytes as f64),
                "wad_bytes" => Some(run.artifacts.wad_bytes as f64),
                "package_bytes" => Some(run.artifacts.package_bytes as f64),
                "compiled_faces" => Some(run.compiled_faces as f64),
                "compiled_entities" => Some(run.compiled_entities as f64),
                "renderer_batches" => Some(run.renderer_batches as f64),
                "runtime_memory_bytes" => Some(run.runtime_memory_bytes as f64),
                "process_vm_peak_bytes" => run.process_memory.vm_peak_bytes.map(|v| v as f64),
                "process_vm_size_bytes" => run.process_memory.vm_size_bytes.map(|v| v as f64),
                "process_rss_bytes" => run.process_memory.rss_bytes.map(|v| v as f64),
                "measurement_duration_ms" => Some(run.measurement_duration_ms as f64),
                name if name.starts_with("stage.") && name.ends_with(".elapsed_ms") => {
                    let stage_name = name
                        .trim_start_matches("stage.")
                        .trim_end_matches(".elapsed_ms");
                    run.stage_metrics
                        .iter()
                        .find(|stage| stage.stage_name == stage_name)
                        .map(|stage| stage.elapsed_ms as f64)
                }
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| format!("metric {metric_name} missing from a measured run"))?;
        if values.len() != self.config.sample_iterations {
            return Err(format!(
                "metric {metric_name} has {} samples, expected {}",
                values.len(),
                self.config.sample_iterations
            ));
        }
        Ok(values)
    }

    pub fn compute_statistics(&self, metric_name: &str) -> Result<SampleStatistic, String> {
        let values = self.metric_values(metric_name)?;
        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };
        let p95_index = ((count as f64 * 0.95).ceil() as usize).saturating_sub(1);
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / count as f64;
        Ok(SampleStatistic {
            metric_name: metric_name.into(),
            sample_count: count,
            selected_statistic: "median",
            selected_value: median,
            min: sorted[0],
            max: sorted[count - 1],
            mean,
            median,
            p95_nearest_rank: sorted[p95_index],
            population_std_dev: variance.sqrt(),
        })
    }

    fn finish_report(&self) -> Result<Value, String> {
        let expected = self.config.warmup_iterations + self.config.sample_iterations;
        if self.runs.len() != expected {
            return Err(format!(
                "recorded {} runs, expected {expected}",
                self.runs.len()
            ));
        }
        let statistics: BTreeMap<String, SampleStatistic> = required_metric_names()
            .into_iter()
            .map(|name| self.compute_statistics(&name).map(|stats| (name, stats)))
            .collect::<Result<_, _>>()?;
        let report = json!({
            "schema": SCHEMA_ID,
            "method": {
                "clock": "monotonic_elapsed",
                "environment_applied_to_all_child_stages": true,
                "sample_statistic": "median",
                "p95_definition": "nearest_rank",
                "std_dev_definition": "population",
                "warmup_excluded": true,
                "output_observational_only": true
            },
            "config": {
                "warmup_iterations": self.config.warmup_iterations,
                "sample_iterations": self.config.sample_iterations,
                "timeout_per_sample_seconds": self.config.timeout_per_sample_seconds
            },
            "host": self.host,
            "runs": self.runs,
            "statistics": statistics
        });
        validate_report_schema(&report)?;
        Ok(report)
    }

    pub fn write_report(&self) -> Result<(), String> {
        let report = self.finish_report()?;
        let parent = self
            .config
            .output_path
            .parent()
            .ok_or("output path has no parent")?;
        std::fs::create_dir_all(parent).map_err(|error| format!("create report dir: {error}"))?;
        let temporary = self.config.output_path.with_extension("json.tmp");
        let bytes =
            serde_json::to_vec_pretty(&report).map_err(|error| format!("serialize: {error}"))?;
        std::fs::write(&temporary, bytes)
            .map_err(|error| format!("write temporary report: {error}"))?;
        std::fs::rename(&temporary, &self.config.output_path)
            .map_err(|error| format!("publish report atomically: {error}"))
    }
}

fn validate_observation(observation: &RunObservation, timeout_seconds: u64) -> Result<(), String> {
    let stage_names: Vec<&str> = observation
        .stage_metrics
        .iter()
        .map(|stage| stage.stage_name.as_str())
        .collect();
    let actual: BTreeSet<_> = stage_names.iter().copied().collect();
    let required: BTreeSet<_> = REQUIRED_STAGES.into_iter().collect();
    if actual != required || stage_names.len() != REQUIRED_STAGES.len() {
        return Err(format!(
            "required stages are incomplete or duplicated: {stage_names:?}"
        ));
    }
    let timeout_ms = timeout_seconds
        .checked_mul(1_000)
        .ok_or("timeout overflow")?;
    for stage in &observation.stage_metrics {
        if stage.elapsed_ms == 0
            || stage.elapsed_ms > timeout_ms
            || stage.cpu_time_ms.is_none()
            || stage.peak_memory_bytes == 0
        {
            return Err(format!(
                "stage {} has incomplete/out-of-range metrics",
                stage.stage_name
            ));
        }
    }
    let lump_names: Vec<&str> = observation
        .bsp_lumps
        .iter()
        .map(|lump| lump.lump_name.as_str())
        .collect();
    let actual_lumps: BTreeSet<_> = lump_names.iter().copied().collect();
    let required_lumps: BTreeSet<_> = REQUIRED_LUMPS.into_iter().collect();
    if actual_lumps != required_lumps || lump_names.len() != REQUIRED_LUMPS.len() {
        return Err("all fifteen unique BSP lumps are required".into());
    }
    if observation.bsp_lumps.iter().any(|lump| {
        lump.total_bytes == 0
            || (lump.element_size > 0
                && lump.element_count.checked_mul(lump.element_size) != Some(lump.total_bytes))
    }) {
        return Err("BSP lump byte/count accounting is inconsistent".into());
    }
    let artifact_sizes = [
        observation.artifacts.map_bytes,
        observation.artifacts.bsp_bytes,
        observation.artifacts.lit_bytes,
        observation.artifacts.wad_bytes,
        observation.artifacts.package_bytes,
    ];
    let hashes = [
        &observation.artifacts.map_sha256,
        &observation.artifacts.bsp_sha256,
        &observation.artifacts.lit_sha256,
        &observation.artifacts.package_sha256,
    ];
    if artifact_sizes.contains(&0)
        || hashes
            .iter()
            .any(|hash| hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
        || observation.compiled_faces == 0
        || observation.compiled_entities == 0
        || observation.renderer_batches == Some(0)
        || observation.renderer_batches.is_none()
        || observation.runtime_memory_bytes == Some(0)
        || observation.runtime_memory_bytes.is_none()
        || observation.measurement_duration_ms == 0
        || observation.measurement_duration_ms > timeout_ms
        || observation.process_memory.vm_peak_bytes == Some(0)
        || observation.process_memory.vm_size_bytes == Some(0)
        || observation.process_memory.rss_bytes == Some(0)
        || observation.process_memory.vm_peak_bytes.is_none()
        || observation.process_memory.vm_size_bytes.is_none()
        || observation.process_memory.rss_bytes.is_none()
    {
        return Err("required artifact/count/memory observation is absent or invalid".into());
    }
    Ok(())
}

/// Machine-readable schema declaration shipped with the frozen method.
fn measurement_json_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "type": "object",
        "additionalProperties": false,
        "required": ["schema", "method", "config", "host", "runs", "statistics"],
        "properties": {
            "schema": {"const": SCHEMA_ID},
            "method": {"type": "object"},
            "config": {"type": "object"},
            "host": {"type": "object"},
            "runs": {"type": "array", "minItems": 1},
            "statistics": {"type": "object"}
        }
    })
}

fn required_metric_names() -> Vec<String> {
    let mut names: Vec<String> = [
        "map_bytes",
        "bsp_bytes",
        "lit_bytes",
        "wad_bytes",
        "package_bytes",
        "compiled_faces",
        "compiled_entities",
        "renderer_batches",
        "runtime_memory_bytes",
        "process_vm_peak_bytes",
        "process_vm_size_bytes",
        "process_rss_bytes",
        "measurement_duration_ms",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    names.extend(
        REQUIRED_STAGES
            .iter()
            .map(|stage| format!("stage.{stage}.elapsed_ms")),
    );
    names
}

fn validate_report_schema(report: &Value) -> Result<(), String> {
    let schema = measurement_json_schema();
    let object = report.as_object().ok_or("report must be an object")?;
    let expected: BTreeSet<&str> = ["schema", "method", "config", "host", "runs", "statistics"]
        .into_iter()
        .collect();
    let actual: BTreeSet<&str> = object.keys().map(String::as_str).collect();
    if actual != expected || report["schema"] != SCHEMA_ID {
        return Err("report top-level schema is closed and required".into());
    }
    let declared: BTreeSet<&str> = schema["required"]
        .as_array()
        .ok_or("schema required must be an array")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .ok_or("schema required item must be a string")
        })
        .collect::<Result<_, _>>()?;
    if declared != expected {
        return Err("machine-readable schema and validator required sets diverge".into());
    }
    require_exact_object_fields(
        &report["method"],
        &[
            "clock",
            "environment_applied_to_all_child_stages",
            "sample_statistic",
            "p95_definition",
            "std_dev_definition",
            "warmup_excluded",
            "output_observational_only",
        ],
    )?;
    require_exact_object_fields(
        &report["config"],
        &[
            "warmup_iterations",
            "sample_iterations",
            "timeout_per_sample_seconds",
        ],
    )?;
    require_exact_object_fields(
        &report["host"],
        &[
            "hostname",
            "os",
            "arch",
            "cpu_count",
            "total_memory_bytes",
            "rust_version",
            "cargo_profile",
            "normalized_environment",
        ],
    )?;
    let runs = report["runs"].as_array().ok_or("runs must be an array")?;
    if runs.is_empty() {
        return Err("runs must not be empty".into());
    }
    for run in runs {
        require_exact_object_fields(
            run,
            &[
                "run_id",
                "sequence",
                "sample_kind",
                "timestamp_utc",
                "corpus",
                "stage_metrics",
                "process_memory",
                "runtime_memory_bytes",
                "artifacts",
                "bsp_lumps",
                "compiled_faces",
                "compiled_entities",
                "renderer_batches",
                "measurement_duration_ms",
            ],
        )?;
        require_exact_object_fields(
            &run["corpus"],
            &["corpus_name", "entry_count", "ordered_sha256", "entries"],
        )?;
        for entry in run["corpus"]["entries"]
            .as_array()
            .ok_or("corpus entries must be an array")?
        {
            require_exact_object_fields(entry, &["seed", "preset", "extent"])?;
        }
        let stages = run["stage_metrics"]
            .as_array()
            .ok_or("stage_metrics must be an array")?;
        let stage_names: Vec<&str> = stages
            .iter()
            .map(|stage| {
                require_exact_object_fields(
                    stage,
                    &[
                        "stage_name",
                        "elapsed_ms",
                        "cpu_time_ms",
                        "peak_memory_bytes",
                    ],
                )?;
                stage["stage_name"]
                    .as_str()
                    .ok_or_else(|| "stage_name must be a string".to_string())
            })
            .collect::<Result<_, _>>()?;
        if stage_names != REQUIRED_STAGES {
            return Err("run stages violate frozen order/coverage".into());
        }
        require_exact_object_fields(
            &run["process_memory"],
            &["vm_peak_bytes", "vm_size_bytes", "rss_bytes"],
        )?;
        require_exact_object_fields(
            &run["artifacts"],
            &[
                "map_bytes",
                "bsp_bytes",
                "lit_bytes",
                "wad_bytes",
                "package_bytes",
                "map_sha256",
                "bsp_sha256",
                "lit_sha256",
                "package_sha256",
            ],
        )?;
        let lumps = run["bsp_lumps"]
            .as_array()
            .ok_or("bsp_lumps must be an array")?;
        let lump_names: Vec<&str> = lumps
            .iter()
            .map(|lump| {
                require_exact_object_fields(
                    lump,
                    &["lump_name", "element_count", "element_size", "total_bytes"],
                )?;
                lump["lump_name"]
                    .as_str()
                    .ok_or_else(|| "lump_name must be a string".to_string())
            })
            .collect::<Result<_, _>>()?;
        if lump_names != REQUIRED_LUMPS {
            return Err("run lumps violate frozen order/coverage".into());
        }
    }
    let sample_iterations = report["config"]["sample_iterations"]
        .as_u64()
        .ok_or("sample_iterations must be an integer")? as usize;
    let statistics = report["statistics"]
        .as_object()
        .ok_or("statistics object")?;
    let actual_metrics: BTreeSet<&str> = statistics.keys().map(String::as_str).collect();
    let metric_names = required_metric_names();
    let required_metrics: BTreeSet<&str> = metric_names.iter().map(String::as_str).collect();
    if actual_metrics != required_metrics {
        return Err("statistics metric coverage is incomplete or extended".into());
    }
    for (metric, statistic) in statistics {
        require_exact_object_fields(
            statistic,
            &[
                "metric_name",
                "sample_count",
                "selected_statistic",
                "selected_value",
                "min",
                "max",
                "mean",
                "median",
                "p95_nearest_rank",
                "population_std_dev",
            ],
        )?;
        if statistic["metric_name"] != metric.as_str()
            || statistic["sample_count"].as_u64() != Some(sample_iterations as u64)
            || statistic["selected_statistic"] != "median"
        {
            return Err(format!("statistic {metric} violates schema contract"));
        }
    }
    Ok(())
}

fn require_exact_object_fields(value: &Value, required: &[&str]) -> Result<(), String> {
    let object: &Map<String, Value> = value.as_object().ok_or("expected object")?;
    let actual: BTreeSet<&str> = object.keys().map(String::as_str).collect();
    let required: BTreeSet<&str> = required.iter().copied().collect();
    if actual != required {
        return Err(format!(
            "closed object fields differ: required={required:?}, actual={actual:?}"
        ));
    }
    Ok(())
}

fn probe_total_memory() -> u64 {
    #[cfg(target_os = "linux")]
    if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
        if let Some(kib) = contents
            .lines()
            .find(|line| line.starts_with("MemTotal:"))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u64>().ok())
        {
            return kib.saturating_mul(1024);
        }
    }
    0
}

fn rustc_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn unix_timestamp_utc() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("unix:{seconds}:UTC")
}

fn complete_corpus(entries: Vec<CorpusEntry>) -> CorpusIdentity {
    CorpusIdentity::new("phase-05-test-corpus", entries).expect("valid corpus")
}

fn complete_observation(offset: u64) -> RunObservation {
    let stage_metrics = REQUIRED_STAGES
        .iter()
        .enumerate()
        .map(|(index, name)| StageMetrics {
            stage_name: (*name).into(),
            elapsed_ms: 10 + index as u64 + offset,
            cpu_time_ms: Some(8 + index as u64),
            peak_memory_bytes: 1_000_000 + index as u64,
        })
        .collect();
    let bsp_lumps = REQUIRED_LUMPS
        .iter()
        .enumerate()
        .map(|(index, name)| BspLumpStats {
            lump_name: (*name).into(),
            element_count: 1 + index,
            element_size: 4,
            total_bytes: (1 + index) * 4,
        })
        .collect();
    let hash = |byte: u8| format!("{byte:02x}").repeat(32);
    RunObservation {
        stage_metrics,
        process_memory: ProcessMemorySnapshot {
            vm_peak_bytes: Some(8_000_000 + offset),
            vm_size_bytes: Some(7_000_000 + offset),
            rss_bytes: Some(6_000_000 + offset),
        },
        runtime_memory_bytes: Some(5_000_000 + offset),
        artifacts: ArtifactStats {
            map_bytes: 4_000,
            bsp_bytes: 26_000 + offset,
            lit_bytes: 12_000,
            wad_bytes: 66_000,
            package_bytes: 108_000,
            map_sha256: hash(0x11),
            bsp_sha256: hash(0x22),
            lit_sha256: hash(0x33),
            package_sha256: hash(0x44),
        },
        bsp_lumps,
        compiled_faces: 41,
        compiled_entities: 3,
        renderer_batches: Some(4),
        measurement_duration_ms: 1_000 + offset,
    }
}

fn test_config(output_path: PathBuf, warmups: usize, samples: usize) -> MeasurementConfig {
    MeasurementConfig {
        output_path,
        warmup_iterations: warmups,
        sample_iterations: samples,
        timeout_per_sample_seconds: 60,
    }
}

#[test]
fn host_normalization_records_and_freezes_child_environment() {
    let host = MeasurementHarness::record_host().expect("record host");
    assert!(!host.hostname.is_empty());
    assert!(!host.os.is_empty());
    assert!(!host.arch.is_empty());
    assert!(host.cpu_count > 0 && host.total_memory_bytes > 0);
    assert!(host.rust_version.starts_with("rustc "));
    assert_eq!(
        host.normalized_environment,
        BTreeMap::from([
            ("LANG".into(), "C".into()),
            ("LC_ALL".into(), "C".into()),
            ("RAYON_NUM_THREADS".into(), "1".into()),
            ("TZ".into(), "UTC".into()),
        ])
    );
}

#[test]
fn report_validates_schema_and_exact_observation_coverage() {
    let temp = tempfile::tempdir().expect("tempdir");
    let output = temp.path().join("measurement.json");
    let mut harness = MeasurementHarness::new(test_config(output.clone(), 1, 2)).expect("harness");
    let corpus = complete_corpus(vec![CorpusEntry {
        seed: 7,
        preset: "sparse".into(),
        extent: 2048,
    }]);
    harness
        .record_run("warmup", corpus.clone(), complete_observation(900))
        .expect("warmup");
    harness
        .record_run("sample-a", corpus.clone(), complete_observation(0))
        .expect("sample a");
    harness
        .record_run("sample-b", corpus, complete_observation(200))
        .expect("sample b");
    harness.write_report().expect("write complete report");
    let report: Value =
        serde_json::from_slice(&std::fs::read(output).expect("read report")).expect("JSON");
    validate_report_schema(&report).expect("schema-valid");
    assert_eq!(report["runs"][0]["sample_kind"], "warmup_excluded");
    assert_eq!(report["runs"][1]["sample_kind"], "measured");
    assert_eq!(report["statistics"]["bsp_bytes"]["sample_count"], 2);
    assert_eq!(
        report["statistics"]["bsp_bytes"]["selected_value"],
        26_100.0
    );
    assert!(measurement_json_schema()["additionalProperties"] == false);

    let mut extended = report.clone();
    extended["runs"][0]
        .as_object_mut()
        .expect("run object")
        .insert("unapproved".into(), Value::Bool(true));
    assert!(
        validate_report_schema(&extended)
            .unwrap_err()
            .contains("closed object fields differ"),
        "nested report objects must reject extensions"
    );
    let mut incomplete = report;
    incomplete["runs"][0]["stage_metrics"]
        .as_array_mut()
        .expect("stages")
        .pop();
    assert!(
        validate_report_schema(&incomplete)
            .unwrap_err()
            .contains("frozen order/coverage"),
        "missing required stages must fail schema validation"
    );
}

#[test]
fn warmups_are_excluded_and_statistics_definitions_are_frozen() {
    let temp = tempfile::tempdir().expect("tempdir");
    let mut harness = MeasurementHarness::new(test_config(temp.path().join("stats.json"), 2, 3))
        .expect("harness");
    let corpus = complete_corpus(vec![CorpusEntry {
        seed: 1,
        preset: "moderate".into(),
        extent: 2048,
    }]);
    for (index, offset) in [9_000, 8_000, 0, 100, 1_000].into_iter().enumerate() {
        harness
            .record_run(
                &format!("run-{index}"),
                corpus.clone(),
                complete_observation(offset),
            )
            .expect("record");
    }
    let stats = harness.compute_statistics("bsp_bytes").expect("stats");
    assert_eq!(stats.sample_count, 3);
    assert_eq!(stats.median, 26_100.0);
    assert_eq!(stats.p95_nearest_rank, 27_000.0);
    assert_eq!(stats.selected_statistic, "median");
    assert_eq!(stats.selected_value, stats.median);
}

#[test]
fn required_observations_and_run_count_fail_closed() {
    let temp = tempfile::tempdir().expect("tempdir");
    let corpus = complete_corpus(vec![CorpusEntry {
        seed: 2,
        preset: "rich".into(),
        extent: 3072,
    }]);
    let mut missing_stage = complete_observation(0);
    missing_stage
        .stage_metrics
        .retain(|stage| stage.stage_name != "render");
    let mut harness = MeasurementHarness::new(test_config(temp.path().join("missing.json"), 0, 1))
        .expect("harness");
    assert!(harness
        .record_run("missing-stage", corpus.clone(), missing_stage)
        .unwrap_err()
        .contains("required stages"));

    let mut missing_runtime = complete_observation(0);
    missing_runtime.runtime_memory_bytes = None;
    assert!(harness
        .record_run("missing-runtime", corpus.clone(), missing_runtime)
        .is_err());
    assert!(harness.write_report().unwrap_err().contains("expected 1"));

    harness
        .record_run("complete", corpus, complete_observation(0))
        .expect("complete");
    harness.write_report().expect("complete report");
}

#[test]
fn corpus_and_observation_order_are_canonical_and_never_generation_input() {
    let entries = vec![
        CorpusEntry {
            seed: 42,
            preset: "sparse".into(),
            extent: 2048,
        },
        CorpusEntry {
            seed: 7,
            preset: "rich".into(),
            extent: 3072,
        },
    ];
    let mut reversed = entries.clone();
    reversed.reverse();
    let corpus_a = complete_corpus(entries);
    let corpus_b = complete_corpus(reversed);
    assert_eq!(
        corpus_a, corpus_b,
        "input order must not alter corpus identity"
    );

    // Generation decisions consume only request/corpus identity. Observation
    // arrays are accepted later by record_run and cannot enter this boundary.
    let generation_identity = |seed: u64, corpus: &CorpusIdentity| {
        let mut hash = Sha256::new();
        hash.update(b"generation-decision/v1\0");
        hash.update(seed.to_le_bytes());
        hash.update(corpus.ordered_sha256.as_bytes());
        format!("{:x}", hash.finalize())
    };
    let before = generation_identity(99, &corpus_a);

    let temp = tempfile::tempdir().expect("tempdir");
    let mut harness = MeasurementHarness::new(test_config(temp.path().join("order.json"), 0, 1))
        .expect("harness");
    let mut observation = complete_observation(0);
    observation.stage_metrics.reverse();
    observation.bsp_lumps.reverse();
    harness
        .record_run("ordered", corpus_b.clone(), observation)
        .expect("record");
    assert_eq!(
        harness.runs[0]
            .stage_metrics
            .iter()
            .map(|stage| stage.stage_name.as_str())
            .collect::<Vec<_>>(),
        REQUIRED_STAGES
    );
    assert_eq!(
        harness.runs[0]
            .bsp_lumps
            .iter()
            .map(|lump| lump.lump_name.as_str())
            .collect::<Vec<_>>(),
        REQUIRED_LUMPS
    );
    harness.write_report().expect("report");
    assert_eq!(before, generation_identity(99, &corpus_b));
}
