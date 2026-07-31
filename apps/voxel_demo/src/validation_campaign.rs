//! Validation campaign runner for v2 presets.
//!
//! Reads seed-corpus.toml and preset-gates.toml from test_data/v2/,
//! then runs the full generation → MC33 → partition pipeline for each
//! (preset, seed, resolution) combination, verifying:
//!
//!  1. Config validation passes
//!  2. Generator produces expected site/spline/maze counts
//!  3. Shell integrity (all layers solid)
//!  4. Reachability (flood-fill from spawn)
//!  5. 5 core viewpoints + 9 light anchors in air
//!  6. MC33 mesh extraction succeeds
//!  7. Partition succeeds
//!  8. Triangle conservation (wall + floor > 0)
//!  9. Timing captured via PhaseTiming
//!
//! Results are written to a JSONL file.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use serde::Deserialize;

use crate::cave_gen::generators::topology_first::generate_v2;
use crate::cave_gen::generators::verify_shell_multi;
use crate::cave_gen::lattice::VoxelWorld;
use crate::cave_gen::metrics::flood_fill_air;
use crate::config::{
    compute_geometry_identity, compute_scene_config_identity, get_embedded_preset,
    known_catalog_ids, normalize_document, resolve_asset_ref, DocumentSource, ResolvedAppConfig,
    ResolvedAssetRef, RuntimeOptions,
};

use crate::scene_package::build_scene_package;
use crate::telemetry::{CampaignRecord, PhaseTiming, TelemetryRecorder};
use crate::validate::validate_preset_document;

// ─── Seed Corpus ───────────────────────────────────────────────────────────

/// A single entry in the seed corpus.
#[derive(Debug, Clone, Deserialize)]
struct CorpusEntry {
    label: String,
    #[serde(deserialize_with = "deserialize_u64_flex")]
    seed: u64,
    resolutions: Vec<u32>,
    #[allow(dead_code)]
    description: String,
}

/// Deserialize a u64 from either a TOML integer or a decimal string.
fn deserialize_u64_flex<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;
    struct U64FlexVisitor;
    impl<'de> de::Visitor<'de> for U64FlexVisitor {
        type Value = u64;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a u64 integer or decimal string")
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<u64, E> {
            Ok(v)
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<u64, E> {
            u64::try_from(v).map_err(|_| E::custom("negative seed"))
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<u64, E> {
            v.parse().map_err(|_| E::custom("invalid u64 string"))
        }
    }
    deserializer.deserialize_any(U64FlexVisitor)
}

/// The seed-corpus.toml file.
#[derive(Debug, Clone, Deserialize)]
struct SeedCorpus {
    #[allow(dead_code)]
    schema: SchemaHeader,
    corpus: Vec<CorpusEntry>,
}

/// A preset gate entry with actionable limits.
#[derive(Debug, Clone, Deserialize)]
struct GateEntry {
    preset: String,
    resolutions: Vec<u32>,
    min_interior: u32,
    max_mc33_triangles: u64,
    max_byte_estimate: u64,
    expected_sites: Option<usize>,
    expected_spline: Option<usize>,
    expected_maze: Option<usize>,
}

/// The preset-gates.toml file.
#[derive(Debug, Clone, Deserialize)]
struct PresetGates {
    #[allow(dead_code)]
    schema: SchemaHeader,
    gates: Vec<GateEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct SchemaHeader {
    #[allow(dead_code)]
    version: u32,
    #[allow(dead_code)]
    description: String,
}

// ─── Campaign Plan ─────────────────────────────────────────────────────────

/// A single test case in the campaign.
#[derive(Debug, Clone)]
struct CampaignCase {
    preset_name: String,
    seed: u64,
    resolution: u32,
}

/// Build the list of campaign cases from seed-corpus + preset-gates.
fn build_campaign_plan() -> Result<Vec<CampaignCase>, String> {
    let corpus_path = test_data_path("v2/seed-corpus.toml");
    let gates_path = test_data_path("v2/preset-gates.toml");

    let corpus_toml = std::fs::read_to_string(&corpus_path)
        .map_err(|e| format!("read {}: {e}", corpus_path.display()))?;
    let gates_toml = std::fs::read_to_string(&gates_path)
        .map_err(|e| format!("read {}: {e}", gates_path.display()))?;

    let corpus: SeedCorpus =
        toml::from_str(&corpus_toml).map_err(|e| format!("parse seed-corpus.toml: {e}"))?;
    let gates: PresetGates =
        toml::from_str(&gates_toml).map_err(|e| format!("parse preset-gates.toml: {e}"))?;

    let mut cases = Vec::new();

    for gate in &gates.gates {
        for entry in &corpus.corpus {
            for &resolution in &entry.resolutions {
                if gate.resolutions.contains(&resolution) {
                    cases.push(CampaignCase {
                        preset_name: gate.preset.clone(),
                        seed: entry.seed,
                        resolution,
                    });
                }
            }
        }
    }

    Ok(cases)
}

fn test_data_path(relative: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("test_data").join(relative)
}

// ─── Campaign Runner ───────────────────────────────────────────────────────

/// Run a single campaign case: generate, validate, mesh, partition, record.
fn run_case(case: &CampaignCase) -> CampaignRecord {
    let (_, preset_doc) = match get_embedded_preset(&case.preset_name) {
        Some(v) => v,
        None => {
            return CampaignRecord {
                preset: case.preset_name.clone(),
                seed: case.seed,
                resolution: case.resolution,
                passed: false,
                error: Some(format!("unknown preset: '{}'", case.preset_name)),
                timing: None,
                build_env: crate::telemetry::BuildEnv::default(),
            };
        }
    };

    let mut doc = preset_doc.clone();
    doc.generator.seed = case.seed;
    doc.generator.resolution = case.resolution;

    // 1. Normalize & validate config
    if let Err(e) = normalize_document(&mut doc) {
        return CampaignRecord {
            preset: case.preset_name.clone(),
            seed: case.seed,
            resolution: case.resolution,
            passed: false,
            error: Some(format!("normalize: {e}")),
            timing: None,
            build_env: crate::telemetry::BuildEnv::default(),
        };
    }

    let errors = validate_preset_document(&doc);
    if !errors.is_empty() {
        let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        return CampaignRecord {
            preset: case.preset_name.clone(),
            seed: case.seed,
            resolution: case.resolution,
            passed: false,
            error: Some(format!("validation: {}", msgs.join("; "))),
            timing: None,
            build_env: crate::telemetry::BuildEnv::default(),
        };
    }

    let source_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let catalog_ids = known_catalog_ids();
    let resolve = |asset_ref: &crate::config::AssetRef| -> Result<ResolvedAssetRef, String> {
        resolve_asset_ref(asset_ref, &source_dir, catalog_ids)
    };

    let resolved_wall_albedo = match resolve(&doc.materials.wall.albedo) {
        Ok(v) => v,
        Err(e) => {
            return CampaignRecord {
                preset: case.preset_name.clone(),
                seed: case.seed,
                resolution: case.resolution,
                passed: false,
                error: Some(format!("asset resolve: {e}")),
                timing: None,
                build_env: crate::telemetry::BuildEnv::default(),
            };
        }
    };
    let resolved_wall_normal = resolve(&doc.materials.wall.normal).unwrap();
    let resolved_wall_roughness = resolve(&doc.materials.wall.roughness).unwrap();
    let resolved_wall_ao = resolve(&doc.materials.wall.ao).unwrap();
    let resolved_floor_albedo = resolve(&doc.materials.floor.albedo).unwrap();
    let resolved_floor_normal = resolve(&doc.materials.floor.normal).unwrap();
    let resolved_floor_roughness = resolve(&doc.materials.floor.roughness).unwrap();
    let resolved_floor_ao = resolve(&doc.materials.floor.ao).unwrap();

    let geometry_identity =
        compute_geometry_identity(doc.generator_version, doc.rng_version, &doc.generator);
    let scene_config_identity = compute_scene_config_identity(
        &geometry_identity,
        &doc.generator,
        &doc.materials.wall,
        &doc.materials.floor,
        &resolved_wall_albedo,
        &resolved_wall_normal,
        &resolved_wall_roughness,
        &resolved_wall_ao,
        &resolved_floor_albedo,
        &resolved_floor_normal,
        &resolved_floor_roughness,
        &resolved_floor_ao,
    );

    let resolved = ResolvedAppConfig {
        document: doc.clone(),
        runtime: RuntimeOptions {
            light_budget: 9,
            headless: true,
            capture_dir: None,
            env_path: None,
        },
        source: DocumentSource::Embedded {
            name: case.preset_name.clone(),
        },
        resolved_wall_albedo,
        resolved_wall_normal,
        resolved_wall_roughness,
        resolved_wall_ao,
        resolved_floor_albedo,
        resolved_floor_normal,
        resolved_floor_roughness,
        resolved_floor_ao,
        geometry_identity: geometry_identity.clone(),
        scene_config_identity: scene_config_identity.clone(),
        asset_digests: Vec::new(),
    };

    // 2. Build scene package (runs generation + MC33 + partition)
    let t_total = Instant::now();
    let package = match build_scene_package(&resolved) {
        Ok(pkg) => pkg,
        Err(e) => {
            return CampaignRecord {
                preset: case.preset_name.clone(),
                seed: case.seed,
                resolution: case.resolution,
                passed: false,
                error: Some(format!("scene package: {e}")),
                timing: None,
                build_env: crate::telemetry::BuildEnv::default(),
            };
        }
    };
    let total_cpu_ms = t_total.elapsed().as_secs_f64() * 1000.0;

    // 3. Additional structural checks that go beyond build_scene_package
    let gen_config = &resolved.document.generator;

    // 3a. Shell integrity (re-verify post-build)
    // We need a world to check — build_scene_package consumes the world,
    // so we run a separate generation for structural-only checks.
    let mut check_world = VoxelWorld::new(case.resolution, case.resolution, case.resolution);
    check_world.fill_solid();
    let gen_result = match generate_v2(gen_config, &mut check_world, case.seed) {
        Ok(r) => r,
        Err(e) => {
            return CampaignRecord {
                preset: case.preset_name.clone(),
                seed: case.seed,
                resolution: case.resolution,
                passed: false,
                error: Some(format!("generation (check): {e}")),
                timing: None,
                build_env: crate::telemetry::BuildEnv::default(),
            };
        }
    };

    // Shell integrity
    if !verify_shell_multi(&check_world, gen_config.shell_thickness) {
        return CampaignRecord {
            preset: case.preset_name.clone(),
            seed: case.seed,
            resolution: case.resolution,
            passed: false,
            error: Some("shell breach detected".to_string()),
            timing: None,
            build_env: crate::telemetry::BuildEnv::default(),
        };
    }

    // 3b. Reachability: flood-fill from spawn site
    let spawn_site = if gen_result.spawn_index < gen_result.sites.len() {
        gen_result.sites[gen_result.spawn_index]
    } else if !gen_result.sites.is_empty() {
        gen_result.sites[0]
    } else {
        return CampaignRecord {
            preset: case.preset_name.clone(),
            seed: case.seed,
            resolution: case.resolution,
            passed: false,
            error: Some("no sites placed".to_string()),
            timing: None,
            build_env: crate::telemetry::BuildEnv::default(),
        };
    };
    let reachable = flood_fill_air(
        check_world.density(),
        spawn_site.x,
        spawn_site.y,
        spawn_site.z,
    );
    if reachable.is_empty() {
        return CampaignRecord {
            preset: case.preset_name.clone(),
            seed: case.seed,
            resolution: case.resolution,
            passed: false,
            error: Some("spawn site is not in air".to_string()),
            timing: None,
            build_env: crate::telemetry::BuildEnv::default(),
        };
    }

    // 3c. All sites reachable from spawn
    let reachable_set: HashSet<usize> = reachable.into_iter().collect();
    for site in &gen_result.sites {
        let idx = xyz_to_idx(check_world.dims(), site.x, site.y, site.z);
        if !reachable_set.contains(&idx) {
            return CampaignRecord {
                preset: case.preset_name.clone(),
                seed: case.seed,
                resolution: case.resolution,
                passed: false,
                error: Some(format!("site '{}' unreachable from spawn", site.label)),
                timing: None,
                build_env: crate::telemetry::BuildEnv::default(),
            };
        }
    }

    // 3d. 5 core viewpoints in air
    for vp in &gen_result.viewpoints {
        if !is_in_air(check_world.dims(), check_world.density(), vp.x, vp.y, vp.z) {
            return CampaignRecord {
                preset: case.preset_name.clone(),
                seed: case.seed,
                resolution: case.resolution,
                passed: false,
                error: Some(format!(
                    "viewpoint anchor {} at ({},{},{}) is inside solid",
                    vp.id, vp.x, vp.y, vp.z
                )),
                timing: None,
                build_env: crate::telemetry::BuildEnv::default(),
            };
        }
    }

    // 3e. Light anchors in air
    for la in &gen_result.light_anchors {
        if !is_in_air(check_world.dims(), check_world.density(), la.x, la.y, la.z) {
            return CampaignRecord {
                preset: case.preset_name.clone(),
                seed: case.seed,
                resolution: case.resolution,
                passed: false,
                error: Some(format!(
                    "light anchor {} at ({},{},{}) is inside solid",
                    la.id, la.x, la.y, la.z
                )),
                timing: None,
                build_env: crate::telemetry::BuildEnv::default(),
            };
        }
    }

    // 3f. Verify triangle conservation: wall + floor > 0
    if package.wall_triangles == 0 && package.floor_triangles == 0 {
        return CampaignRecord {
            preset: case.preset_name.clone(),
            seed: case.seed,
            resolution: case.resolution,
            passed: false,
            error: Some("zero triangles in both wall and floor partitions".to_string()),
            timing: None,
            build_env: crate::telemetry::BuildEnv::default(),
        };
    }

    // 3g. Verify spline+edge count against config
    let spline_count = gen_result
        .serializable_edges
        .iter()
        .filter(|e| {
            matches!(
                e.kind,
                crate::cave_gen::generators::RouteKind::SplineBackbone
                    | crate::cave_gen::generators::RouteKind::SplineExtra
            )
        })
        .count();
    let maze_count = gen_result
        .serializable_edges
        .iter()
        .filter(|e| matches!(e.kind, crate::cave_gen::generators::RouteKind::Maze))
        .count();

    // 4. Gate limit checks
    let gates_path = test_data_path("v2/preset-gates.toml");
    if let Ok(gates_toml) = std::fs::read_to_string(&gates_path) {
        if let Ok(gates) = toml::from_str::<PresetGates>(&gates_toml) {
            for gate in &gates.gates {
                if gate.preset == case.preset_name && gate.resolutions.contains(&case.resolution) {
                    // Check gate limits
                    if let Some(expected_sites) = gate.expected_sites {
                        if gen_result.sites.len() != expected_sites {
                            return CampaignRecord {
                                preset: case.preset_name.clone(),
                                seed: case.seed,
                                resolution: case.resolution,
                                passed: false,
                                error: Some(format!(
                                    "gate: expected {expected_sites} sites, got {}",
                                    gen_result.sites.len()
                                )),
                                timing: None,
                                build_env: crate::telemetry::BuildEnv::default(),
                            };
                        }
                    }
                    if let Some(expected_spline) = gate.expected_spline {
                        if spline_count != expected_spline {
                            return CampaignRecord {
                                preset: case.preset_name.clone(),
                                seed: case.seed,
                                resolution: case.resolution,
                                passed: false,
                                error: Some(format!(
                                    "gate: expected {expected_spline} spline edges, got {spline_count}"
                                )),
                                timing: None,
                build_env: crate::telemetry::BuildEnv::default(),
                            };
                        }
                    }
                    if let Some(expected_maze) = gate.expected_maze {
                        if maze_count != expected_maze {
                            return CampaignRecord {
                                preset: case.preset_name.clone(),
                                seed: case.seed,
                                resolution: case.resolution,
                                passed: false,
                                error: Some(format!(
                                    "gate: expected {expected_maze} maze links, got {maze_count}"
                                )),
                                timing: None,
                                build_env: crate::telemetry::BuildEnv::default(),
                            };
                        }
                    }
                }
            }
        }
    }

    // 5. Build timing record
    let timing = PhaseTiming {
        preset: case.preset_name.clone(),
        seed: case.seed,
        resolution: case.resolution,
        generation_ms: package.generation_time_ms as f64,
        mc33_ms: package.mesh_time_ms as f64,
        partition_ms: package.partition_time_ms as f64,
        conversion_ms: 0.0,
        total_cpu_ms,
        wall_triangles: package.wall_triangles,
        floor_triangles: package.floor_triangles,
        total_voxels: package.total_voxels,
        site_count: gen_result.sites.len(),
        spline_edges: spline_count,
        maze_links: maze_count,
        light_count: package.lights.len(),
        viewpoint_count: package.viewpoints.len(),
        build_env: crate::telemetry::BuildEnv::default(),
        request_id: format!(
            "campaign-{}-{}-{}",
            case.preset_name, case.seed, case.resolution
        ),
        upload_ms: 0.0,
        material_create_ms: 0.0,
    };

    CampaignRecord {
        preset: case.preset_name.clone(),
        seed: case.seed,
        resolution: case.resolution,
        passed: true,
        error: None,
        timing: Some(timing),
        build_env: crate::telemetry::BuildEnv::default(),
    }
}

// ─── Helpers ───────────────────────────────────────────────────────────────

fn xyz_to_idx((w, h, _d): (u32, u32, u32), x: u32, y: u32, z: u32) -> usize {
    (x as usize) + (y as usize) * (w as usize) + (z as usize) * (w as usize) * (h as usize)
}

fn is_in_air(
    (w, h, d): (u32, u32, u32),
    lattice: &crate::cave_gen::lattice::DenseLattice<i8>,
    x: u32,
    y: u32,
    z: u32,
) -> bool {
    if x >= w || y >= h || z >= d {
        return false; // out of bounds → considered solid
    }
    lattice.get(x, y, z).map_or(false, |&den| den >= 0)
}

// ─── Public Entrypoint ─────────────────────────────────────────────────────

/// Run the full validation campaign and write results to a JSONL file.
///
/// Returns the number of passed and total cases.
pub fn run_campaign(output_path: &std::path::Path) -> Result<(usize, usize), String> {
    let cases = build_campaign_plan()?;
    let total = cases.len();
    let mut recorder = TelemetryRecorder::open(output_path)?;
    let mut passed = 0usize;

    log::info!(
        "Validation campaign: {} cases across {} presets",
        total,
        cases
            .iter()
            .map(|c| c.preset_name.as_str())
            .collect::<HashSet<_>>()
            .len()
    );

    for case in &cases {
        log::info!(
            "  Testing preset={} seed={} resolution={}",
            case.preset_name,
            case.seed,
            case.resolution
        );
        let record = run_case(case);
        if record.passed {
            passed += 1;
            if let Some(ref t) = record.timing {
                log::info!(
                    "    ✓ PASS ({}ms gen, {}ms mc33, {}ms partition, {} wall tris, {} floor tris)",
                    t.generation_ms as u64,
                    t.mc33_ms as u64,
                    t.partition_ms as u64,
                    t.wall_triangles,
                    t.floor_triangles
                );
            }
        } else {
            log::error!(
                "    ✗ FAIL: {}",
                record.error.as_deref().unwrap_or("unknown")
            );
        }
        recorder.record(&record)?;
    }

    log::info!("Campaign complete: {passed}/{total} passed");
    Ok((passed, total))
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Fast structural campaign: test all presets at 64³ with seed 0 only.
    /// This is a deterministic, bounded validation of core structural
    /// invariants. The full corpus+profile campaign is run separately.
    #[test]
    fn structural_campaign_64_seed0_all_presets() {
        let presets = ["default", "cavernous", "mazy", "tight"];
        let mut passed = 0;
        let mut failed = Vec::new();

        for &preset_name in &presets {
            let case = CampaignCase {
                preset_name: preset_name.to_string(),
                seed: 0,
                resolution: 64,
            };
            let record = run_case(&case);
            if record.passed {
                passed += 1;
                if let Some(ref t) = record.timing {
                    // Basic sanity on geometry
                    assert!(
                        t.wall_triangles + t.floor_triangles > 0,
                        "{preset_name}: no triangles"
                    );
                    assert!(t.total_voxels > 0, "{preset_name}: zero voxels");
                    assert!(t.site_count >= 5, "{preset_name}: fewer than 5 sites");
                    assert!(
                        t.viewpoint_count >= 5,
                        "{preset_name}: fewer than 5 viewpoints"
                    );
                    // Exactly 9 generator-anchor lights: 5 CoreLight + 4 BackboneLight.
                    assert_eq!(t.light_count, 9, "{preset_name}: expected exactly 9 lights");
                }
            } else {
                failed.push(format!(
                    "{preset_name}: {}",
                    record.error.as_deref().unwrap_or("unknown")
                ));
            }
        }

        if !failed.is_empty() {
            panic!(
                "{}/{} structural tests failed:\n{}",
                failed.len(),
                presets.len(),
                failed.join("\n")
            );
        }
        assert_eq!(
            passed, 4,
            "expected all 4 presets to pass structural campaign"
        );
    }

    #[test]
    fn structural_campaign_96_seed0_all_presets() {
        let presets = ["default", "cavernous", "mazy", "tight"];
        let mut passed = 0;
        let mut failed = Vec::new();

        for &preset_name in &presets {
            let case = CampaignCase {
                preset_name: preset_name.to_string(),
                seed: 0,
                resolution: 96,
            };
            let record = run_case(&case);
            if record.passed {
                passed += 1;
            } else {
                failed.push(format!(
                    "{preset_name}: {}",
                    record.error.as_deref().unwrap_or("unknown")
                ));
            }
        }

        if !failed.is_empty() {
            panic!(
                "{}/{} structural tests at 96³ failed:\n{}",
                failed.len(),
                presets.len(),
                failed.join("\n")
            );
        }
        assert_eq!(passed, 4, "expected all 4 presets to pass at 96³");
    }

    #[test]
    fn structural_campaign_128_seed0_all_presets() {
        let presets = ["default", "cavernous", "mazy", "tight"];
        let mut passed = 0;
        let mut failed = Vec::new();

        for &preset_name in &presets {
            let case = CampaignCase {
                preset_name: preset_name.to_string(),
                seed: 0,
                resolution: 128,
            };
            let record = run_case(&case);
            if record.passed {
                passed += 1;
            } else {
                failed.push(format!(
                    "{preset_name}: {}",
                    record.error.as_deref().unwrap_or("unknown")
                ));
            }
        }

        if !failed.is_empty() {
            panic!(
                "{}/{} structural tests at 128³ failed:\n{}",
                failed.len(),
                presets.len(),
                failed.join("\n")
            );
        }
        assert_eq!(passed, 4, "expected all 4 presets to pass at 128³");
    }

    #[test]
    fn full_campaign_record_to_jsonl() {
        let output_path =
            std::path::PathBuf::from(std::env::var("CAMPAIGN_OUTPUT").unwrap_or_else(|_| {
                ".internal-dev/debug_reports/voxel-demo-v2-profile.jsonl".to_string()
            }));
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let (passed, total) = run_campaign(&output_path).expect("campaign should succeed");
        assert_eq!(passed, total, "all campaign cases must pass");
    }
}
