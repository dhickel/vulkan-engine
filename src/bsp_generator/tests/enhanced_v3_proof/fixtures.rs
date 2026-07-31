//! Typed focused fixtures for the Phase 06 compiler + spatial proof.
//!
//! Defines 4 fixture map files with expected BSP2/lightmap/witness outcomes.
//! All fixtures are minimal, well-formed Quake .map files targeting
//! cc0_dungeon_v2.wad and the ericw-tools BSP2 compiler.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ── Fixture paths ─────────────────────────────────────────────────────────

/// Resolve the crate root directory.
pub fn crate_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

/// Path to the compiler profile TOML.
pub fn compiler_profile_path() -> PathBuf {
    crate_dir().join("../../tools/bsp_authoring/ericw-q1-bsp2-generated-profile.toml")
}

/// Path to the fixture cases TOML.
pub fn fixture_cases_path() -> PathBuf {
    crate_dir().join("tests/fixtures/enhanced_v3_proof/fixture-cases.toml")
}

/// Theme directory for cc0_dungeon_v2.
pub fn theme_dir() -> PathBuf {
    crate_dir().join("themes/cc0_dungeon_v2")
}

/// WAD path.
pub fn wad_path() -> PathBuf {
    theme_dir().join("cc0_dungeon_v2.wad")
}

/// Palette path.
pub fn palette_path() -> PathBuf {
    theme_dir().join("palette.lmp")
}

/// Path for a named fixture .map file.
pub fn fixture_map_path(name: &str) -> PathBuf {
    crate_dir().join(format!("tests/fixtures/enhanced_v3_proof/{name}.map"))
}

/// Debug report output path.
pub fn report_path() -> PathBuf {
    crate_dir()
        .join("../../.internal-dev/debug_reports/enhanced-v3-proof/compiler-spatial-report.json")
}

// ── Fixture case definition ───────────────────────────────────────────────

/// A single fixture case from fixture-cases.toml.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureCase {
    /// Unique case identifier.
    pub id: String,
    /// Base name of the .map file (without .map extension).
    pub map_file: String,
    /// Human-readable description.
    pub description: String,
    /// Whether BSP2 magic is expected.
    pub expect_bsp2: bool,
    /// Whether .lit output is expected (requires -lit in light args).
    pub expect_lit: bool,
    /// Minimum number of solid leaves expected.
    pub min_solid_leaves: u32,
    /// Minimum number of empty leaves expected.
    pub min_empty_leaves: u32,
    /// Minimum number of clipnodes expected.
    pub min_clipnodes: u32,
    /// Witness outcomes: (witness_id, description, expected_pass).
    pub witnesses: Vec<WitnessSpec>,
}

/// A spatial witness specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WitnessSpec {
    /// Witness identifier.
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Whether this witness is expected to pass.
    pub expected_pass: bool,
    /// Query coordinates (in Quake units).
    pub query_coords: Vec<[f32; 3]>,
    /// Tolerance in Quake units.
    pub tolerance: f32,
}

/// All fixture cases parsed from fixture-cases.toml.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureCasesFile {
    pub schema: String,
    pub profile: String,
    pub cases: Vec<FixtureCase>,
}

/// Load and parse the fixture cases TOML file.
pub fn load_fixture_cases() -> Result<FixtureCasesFile, String> {
    let path = fixture_cases_path();
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read fixture cases {}: {e}", path.display()))?;
    toml::from_str(&text).map_err(|e| format!("invalid fixture cases TOML: {e}"))
}

/// Load a fixture .map file as a string.
pub fn load_fixture_map(name: &str) -> Result<String, String> {
    let path = fixture_map_path(name);
    std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read fixture map {}: {e}", path.display()))
}

// ── Spatial witness result ────────────────────────────────────────────────

/// Result of a single spatial witness query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessResult {
    pub id: String,
    pub description: String,
    pub expected_pass: bool,
    pub actual_pass: bool,
    pub coordinates: Vec<[f32; 3]>,
    pub details: Vec<WitnessDetail>,
}

/// Detail from a witness query point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessDetail {
    pub point: [f32; 3],
    pub contents: String,
    pub is_solid: bool,
    pub matched_expectation: bool,
}

// ── Compiler spatial report ───────────────────────────────────────────────

/// The complete compiler-spatial-report.json structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerSpatialReport {
    pub schema: String,
    pub timestamp: String,
    pub profile_name: String,
    pub profile_version: String,
    pub tool_dir: String,
    pub tools_available: bool,
    pub executable_hashes_verified: bool,
    pub env_identity: EnvIdentitySnapshot,
    pub results: Vec<FixtureResult>,
    pub summary: ReportSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvIdentitySnapshot {
    pub home: String,
    pub path: String,
    pub lang: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureResult {
    pub case_id: String,
    pub map_file: String,
    pub status: FixtureStatus,
    pub bsp_sha256: Option<String>,
    pub lit_sha256: Option<String>,
    pub bsp_size: Option<u64>,
    pub lit_size: Option<u64>,
    pub compilation_time_ms: Option<u64>,
    pub diagnostics: Vec<String>,
    pub stage_outputs: Vec<StageOutputSnapshot>,
    pub witness_results: Vec<WitnessResult>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum FixtureStatus {
    Pass,
    Fail,
    NotRun,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageOutputSnapshot {
    pub stage: String,
    pub exit_code: i32,
    pub elapsed_ms: u64,
    pub stderr_summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub not_run: usize,
    pub skipped: usize,
    pub total_witnesses: usize,
    pub witnesses_passed: usize,
    pub witnesses_failed: usize,
}

impl CompilerSpatialReport {
    /// Create a new empty report.
    pub fn new(
        profile_name: &str,
        profile_version: &str,
        tool_dir: &Path,
        tools_available: bool,
        hashes_verified: bool,
        home: &str,
        path: &str,
        lang: &str,
    ) -> Self {
        CompilerSpatialReport {
            schema: "enhanced-v3-compiler-spatial-report/v1".to_string(),
            timestamp: iso8601_now(),
            profile_name: profile_name.to_string(),
            profile_version: profile_version.to_string(),
            tool_dir: tool_dir.display().to_string(),
            tools_available,
            executable_hashes_verified: hashes_verified,
            env_identity: EnvIdentitySnapshot {
                home: home.to_string(),
                path: path.to_string(),
                lang: lang.to_string(),
            },
            results: Vec::new(),
            summary: ReportSummary {
                total: 0,
                passed: 0,
                failed: 0,
                not_run: 0,
                skipped: 0,
                total_witnesses: 0,
                witnesses_passed: 0,
                witnesses_failed: 0,
            },
        }
    }

    /// Add a fixture result.
    pub fn add_result(&mut self, result: FixtureResult) {
        self.results.push(result);
    }

    /// Recompute the summary from results.
    pub fn recompute_summary(&mut self) {
        let total = self.results.len();
        let passed = self
            .results
            .iter()
            .filter(|r| r.status == FixtureStatus::Pass)
            .count();
        let failed = self
            .results
            .iter()
            .filter(|r| r.status == FixtureStatus::Fail)
            .count();
        let not_run = self
            .results
            .iter()
            .filter(|r| r.status == FixtureStatus::NotRun)
            .count();
        let skipped = self
            .results
            .iter()
            .filter(|r| r.status == FixtureStatus::Skipped)
            .count();

        let mut total_witnesses = 0;
        let mut witnesses_passed = 0;
        let mut witnesses_failed = 0;
        for r in &self.results {
            total_witnesses += r.witness_results.len();
            witnesses_passed += r.witness_results.iter().filter(|w| w.actual_pass).count();
            witnesses_failed += r.witness_results.iter().filter(|w| !w.actual_pass).count();
        }

        self.summary = ReportSummary {
            total,
            passed,
            failed,
            not_run,
            skipped,
            total_witnesses,
            witnesses_passed,
            witnesses_failed,
        };
    }

    /// Write to the debug report path.
    pub fn write(&self) -> Result<(), String> {
        let path = report_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("create report dir: {e}"))?;
        }
        let json =
            serde_json::to_string_pretty(self).map_err(|e| format!("serialize report: {e}"))?;
        std::fs::write(&path, &json)
            .map_err(|e| format!("write report {}: {e}", path.display()))?;
        Ok(())
    }
}

/// ISO-8601 timestamp.
fn iso8601_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let h = time_of_day / 3600;
    let min = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;

    let d = days as i64 + 719468;
    let era = if d >= 0 { d } else { d - 146096 } / 146097;
    let doe = d - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    format!("{year:04}-{month:02}-{day:02}T{h:02}:{min:02}:{s:02}Z")
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_cases_file_exists() {
        let path = fixture_cases_path();
        assert!(
            path.exists(),
            "fixture-cases.toml must exist at {}",
            path.display()
        );
    }

    #[test]
    fn fixture_cases_parse_valid() {
        let cases = load_fixture_cases().expect("parse fixture-cases.toml");
        assert_eq!(cases.schema, "enhanced-v3-fixture-cases/v1");
        assert!(!cases.cases.is_empty(), "must have at least one case");
        for case in &cases.cases {
            assert!(!case.id.is_empty(), "case must have id");
            assert!(!case.map_file.is_empty(), "case must have map_file");
        }
    }

    #[test]
    fn all_fixture_maps_exist() {
        let cases = load_fixture_cases().expect("parse fixture-cases.toml");
        for case in &cases.cases {
            let path = fixture_map_path(&case.map_file);
            assert!(
                path.exists(),
                "fixture map {} missing for case {}",
                path.display(),
                case.id
            );
            let map_text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read fixture map {}: {e}", path.display()));
            assert!(
                map_text.contains("worldspawn"),
                "fixture {} must contain worldspawn",
                case.id
            );
        }
    }

    #[test]
    fn theme_assets_exist() {
        assert!(wad_path().exists(), "cc0_dungeon_v2.wad must exist");
        assert!(palette_path().exists(), "palette.lmp must exist");
    }

    #[test]
    fn report_writes_to_debug_reports_dir() {
        let mut report = CompilerSpatialReport::new(
            "test",
            "0.0.0",
            Path::new("/tmp"),
            true,
            true,
            "/home/test",
            "/usr/bin",
            "en_US.UTF-8",
        );
        report.add_result(FixtureResult {
            case_id: "test-case".to_string(),
            map_file: "test.map".to_string(),
            status: FixtureStatus::Pass,
            bsp_sha256: Some("abc123".to_string()),
            lit_sha256: None,
            bsp_size: Some(12345),
            lit_size: None,
            compilation_time_ms: Some(500),
            diagnostics: vec![],
            stage_outputs: vec![],
            witness_results: vec![],
        });
        report.recompute_summary();

        // Write to a temp file instead of the real report path
        let tmp = tempfile::tempdir().unwrap();
        let tmp_report = tmp.path().join("report.json");
        let json = serde_json::to_string_pretty(&report).unwrap();
        std::fs::write(&tmp_report, &json).unwrap();

        // Parse back
        let back: CompilerSpatialReport =
            serde_json::from_str(&std::fs::read_to_string(&tmp_report).unwrap()).unwrap();
        assert_eq!(back.results.len(), 1);
        assert_eq!(back.summary.total, 1);
        assert_eq!(back.summary.passed, 1);
    }
}
