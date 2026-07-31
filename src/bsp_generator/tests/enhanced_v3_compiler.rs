//! Phase 06 — compiler and compiled-spatial proof.
//!
//! Compiler-produced fixture qualification and synthetic `bsp` query tests are
//! intentionally separate. A synthetic world can prove `point_contents`
//! traversal, but it can never qualify qbsp, BSP2/LIT artifacts, strict reload,
//! or a focused fixture.

mod enhanced_v3_proof;

use enhanced_v3_proof::compiler::{self, CompileFailure, CompilerProfile, StageOutput};
use enhanced_v3_proof::fixtures::{
    self, CompilerSpatialReport, FixtureCase, FixtureResult, FixtureStatus, StageOutputSnapshot,
    StrictReloadFacts, SyntheticQueryPipelineProof, WitnessResult,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn stage_snapshot(output: &StageOutput) -> StageOutputSnapshot {
    let combined = format!("{}\n{}", output.stdout, output.stderr);
    StageOutputSnapshot {
        stage: output.stage.clone(),
        exit_code: output.exit_code,
        elapsed_ms: output.elapsed.as_millis() as u64,
        output_sha256: compiler::sha256_hex(combined.as_bytes()),
        output_excerpt: bounded_excerpt(&combined, 4096),
        diagnostics: output
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.message().to_string())
            .collect(),
    }
}

fn bounded_excerpt(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let excerpt: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{excerpt}\n[diagnostic excerpt truncated]")
    } else {
        excerpt
    }
}

fn known_ericw_hull_limitation(failure: &CompileFailure, brush_count: usize) -> Option<String> {
    let qbsp = failure.stage_outputs.last()?;
    let combined = format!("{}\n{}", qbsp.stdout, qbsp.stderr).to_ascii_lowercase();
    (qbsp.stage == "qbsp"
        && qbsp.exit_code == -1
        && combined.contains("processing hull"))
    .then(|| {
        format!(
            "Pinned ericw-tools qbsp terminated by signal during hull computation even after the focus fixture was expanded to {brush_count} world brushes. Compiler qualification remains FAIL; no synthetic world was substituted."
        )
    })
}

fn unavailable_fixture(case: &FixtureCase, brush_count: usize, reason: String) -> FixtureResult {
    FixtureResult {
        case_id: case.id.clone(),
        map_file: case.map_file.clone(),
        status: FixtureStatus::NotRun,
        source_brush_count: brush_count,
        bsp_sha256: None,
        lit_sha256: None,
        bsp_size: None,
        lit_size: None,
        compilation_time_ms: None,
        diagnostics: vec![reason],
        tool_limitation: None,
        stage_outputs: Vec::new(),
        strict_reload: None,
        witness_results: Vec::new(),
    }
}

fn failed_fixture(
    case: &FixtureCase,
    brush_count: usize,
    diagnostic: String,
    tool_limitation: Option<String>,
    stage_outputs: Vec<StageOutputSnapshot>,
) -> FixtureResult {
    FixtureResult {
        case_id: case.id.clone(),
        map_file: case.map_file.clone(),
        status: FixtureStatus::Fail,
        source_brush_count: brush_count,
        bsp_sha256: None,
        lit_sha256: None,
        bsp_size: None,
        lit_size: None,
        compilation_time_ms: None,
        diagnostics: vec![diagnostic],
        tool_limitation,
        stage_outputs,
        strict_reload: None,
        witness_results: Vec::new(),
    }
}

fn run_fixture_case(
    case: &FixtureCase,
    profile: &CompilerProfile,
    tool_dir: &Path,
    unavailable_reason: Option<&str>,
) -> FixtureResult {
    let map_path = fixtures::fixture_map_path(&case.map_file);
    let map = match fixtures::load_fixture_map(&case.map_file) {
        Ok(map) => map,
        Err(error) => return failed_fixture(case, 0, error, None, Vec::new()),
    };
    let brush_count = fixtures::source_brush_count(&map);
    if brush_count < case.minimum_source_brushes {
        return failed_fixture(
            case,
            brush_count,
            format!(
                "canonical source has {brush_count} world brushes; fixture requires at least {} to avoid the known small-map qbsp hull path",
                case.minimum_source_brushes
            ),
            None,
            Vec::new(),
        );
    }
    if let Some(reason) = unavailable_reason {
        return unavailable_fixture(case, brush_count, reason.to_string());
    }

    let staging = match compiler::create_staging_dir(&case.id) {
        Ok(staging) => staging,
        Err(error) => return failed_fixture(case, brush_count, error, None, Vec::new()),
    };
    let started = Instant::now();
    let compiled = match compiler::compile_map(
        &map_path,
        staging.path(),
        tool_dir,
        &fixtures::wad_path(),
        &fixtures::palette_path(),
        profile,
    ) {
        Ok(compiled) => compiled,
        Err(failure) => {
            let limitation = known_ericw_hull_limitation(&failure, brush_count);
            let snapshots = failure.stage_outputs.iter().map(stage_snapshot).collect();
            return failed_fixture(
                case,
                brush_count,
                bounded_excerpt(&failure.message, 4096),
                limitation,
                snapshots,
            );
        }
    };
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let stage_outputs = [
        &compiled.qbsp_output,
        &compiled.vis_output,
        &compiled.light_output,
    ]
    .into_iter()
    .map(stage_snapshot)
    .collect();

    let (world, strict_reload) = match compiler_produced_world_queries::strict_reload(
        &compiled.bsp_data,
        &compiled.lit_data,
        &fixtures::wad_path(),
        &fixtures::palette_path(),
    ) {
        Ok(result) => result,
        Err(error) => {
            return FixtureResult {
                case_id: case.id.clone(),
                map_file: case.map_file.clone(),
                status: FixtureStatus::Fail,
                source_brush_count: brush_count,
                bsp_sha256: Some(compiled.bsp_sha256),
                lit_sha256: Some(compiled.lit_sha256),
                bsp_size: Some(compiled.bsp_data.len() as u64),
                lit_size: Some(compiled.lit_data.len() as u64),
                compilation_time_ms: Some(elapsed_ms),
                diagnostics: vec![error],
                tool_limitation: None,
                stage_outputs,
                strict_reload: None,
                witness_results: Vec::new(),
            };
        }
    };

    let witness_results = compiler_produced_world_queries::run_witnesses(case, &world);
    let mut diagnostics = Vec::new();
    if strict_reload.solid_leaves < case.min_solid_leaves as usize {
        diagnostics.push(format!(
            "strict BSP has {} solid leaves; expected at least {}",
            strict_reload.solid_leaves, case.min_solid_leaves
        ));
    }
    if strict_reload.empty_leaves < case.min_empty_leaves as usize {
        diagnostics.push(format!(
            "strict BSP has {} empty leaves; expected at least {}",
            strict_reload.empty_leaves, case.min_empty_leaves
        ));
    }
    if strict_reload.clipnodes < case.min_clipnodes as usize {
        diagnostics.push(format!(
            "strict BSP has {} clipnodes; expected at least {}",
            strict_reload.clipnodes, case.min_clipnodes
        ));
    }
    if witness_results.iter().any(|witness| !witness.actual_pass) {
        diagnostics.push("one or more compiler-produced spatial witnesses failed".to_string());
    }

    FixtureResult {
        case_id: case.id.clone(),
        map_file: case.map_file.clone(),
        status: if diagnostics.is_empty() {
            FixtureStatus::Pass
        } else {
            FixtureStatus::Fail
        },
        source_brush_count: brush_count,
        bsp_sha256: Some(compiled.bsp_sha256),
        lit_sha256: Some(compiled.lit_sha256),
        bsp_size: Some(compiled.bsp_data.len() as u64),
        lit_size: Some(compiled.lit_data.len() as u64),
        compilation_time_ms: Some(elapsed_ms),
        diagnostics,
        tool_limitation: None,
        stage_outputs,
        strict_reload: Some(strict_reload),
        witness_results,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Synthetic bsp-crate query pipeline tests — no compiler qualification
// ═══════════════════════════════════════════════════════════════════════════

mod synthetic_world_queries {
    use super::*;
    use bsp::PointContents;
    use glam::Vec3;

    fn world() -> bsp::BspWorld {
        let planes = vec![bsp::lumps::Plane {
            normal: Vec3::X,
            dist: 0.0,
            plane_type: 0,
        }];
        let nodes = vec![bsp::lumps::Node {
            plane_id: 0,
            children: [-1, -2],
            mins: [0; 3],
            maxs: [0; 3],
            face_id: 0,
            face_num: 0,
        }];
        let leaves = vec![
            bsp::lumps::Leaf {
                contents: -2,
                visofs: 0,
                mins: [0; 3],
                maxs: [0; 3],
                mark_id: 0,
                mark_num: 0,
                ambient: [0; 4],
            },
            bsp::lumps::Leaf {
                contents: -1,
                visofs: 0,
                mins: [0; 3],
                maxs: [0; 3],
                mark_id: 0,
                mark_num: 0,
                ambient: [0; 4],
            },
        ];
        bsp::BspWorld {
            profile: bsp::profile::BspProfile::Bsp29,
            entity_raw: b"{\"classname\" \"worldspawn\"}\0".to_vec(),
            entities: vec![bsp::Entity {
                source_index: 0,
                raw: b"{\"classname\" \"worldspawn\"}".to_vec(),
                key_values: Vec::new(),
                class: bsp::EntityClass::Worldspawn,
            }],
            planes,
            vertices: vec![Vec3::ZERO],
            nodes,
            leaves,
            faces: Vec::new(),
            models: Vec::new(),
            texinfos: Vec::new(),
            edges: Vec::new(),
            surfedges: Vec::new(),
            markfaces: Vec::new(),
            clipnodes: Vec::new(),
            miptex_data: Vec::new(),
            lightmap_data: Vec::new(),
            vis_data: Vec::new(),
            bspx: None,
            bspx_rgb_lighting: None,
            palette: None,
            colored_light_source: bsp::companions::ColoredLightSource::Monochrome,
            lit_data: None,
            wad_archives: Vec::new(),
            content_hash: [0; 32],
            source_identity: "synthetic-query-only".to_string(),
            diagnostics: Vec::new(),
        }
    }

    pub(super) fn prove_query_pipeline() -> SyntheticQueryPipelineProof {
        let world = world();
        let queries = [
            (Vec3::new(1.0, 0.0, 0.0), PointContents::Solid),
            (Vec3::new(-1.0, 0.0, 0.0), PointContents::Empty),
        ];
        let passed = queries
            .iter()
            .filter(|(point, expected)| {
                bsp::point_contents(*point, &world.nodes, &world.leaves, &world.planes) == *expected
            })
            .count();
        SyntheticQueryPipelineProof {
            status: if passed == queries.len() {
                FixtureStatus::Pass
            } else {
                FixtureStatus::Fail
            },
            scope: "bsp crate point_contents traversal only; not compiler, reload, fixture, BSP, or LIT qualification".to_string(),
            queries_total: queries.len(),
            queries_passed: passed,
        }
    }

    #[test]
    fn synthetic_world_proves_only_point_contents_traversal() {
        let proof = prove_query_pipeline();
        assert_eq!(proof.status, FixtureStatus::Pass);
        assert!(proof.scope.contains("not compiler"));
        assert_eq!(proof.queries_total, proof.queries_passed);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Compiler-produced world qualification tests
// ═══════════════════════════════════════════════════════════════════════════

mod compiler_produced_world_queries {
    use super::*;
    use bsp::{BspLoader, LoadOptions, QuakeToEngine};

    pub(super) fn strict_reload(
        bsp_data: &[u8],
        lit_data: &[u8],
        wad_path: &Path,
        palette_path: &Path,
    ) -> Result<(bsp::BspWorld, StrictReloadFacts), String> {
        let wad_name = wad_path
            .file_name()
            .ok_or("WAD path has no basename")?
            .to_string_lossy()
            .into_owned();
        let options = LoadOptions {
            strict: true,
            palette: Some(
                std::fs::read(palette_path)
                    .map_err(|error| format!("read strict-load palette: {error}"))?,
            ),
            lit_data: Some(lit_data.to_vec()),
            wad_archives: vec![(
                wad_name,
                std::fs::read(wad_path)
                    .map_err(|error| format!("read strict-load WAD: {error}"))?,
            )],
            texture_overrides: Vec::new(),
            source_identity: "enhanced-v3-compiler-fixture".to_string(),
        };
        let world = BspLoader::load(bsp_data, &options)
            .map_err(|report| format!("strict BSP reload failed: {report}"))?;
        if world.profile != bsp::profile::BspProfile::Bsp2 {
            return Err(format!(
                "strict reload profile is {}, expected bsp2",
                world.profile.tag()
            ));
        }
        if !world.diagnostics.is_empty() {
            return Err(format!(
                "strict BSP reload emitted diagnostics: {:?}",
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
            solid_leaves: world
                .leaves
                .iter()
                .filter(|leaf| leaf.contents == -2)
                .count(),
            empty_leaves: world
                .leaves
                .iter()
                .filter(|leaf| leaf.contents == -1)
                .count(),
            clipnodes: world.clipnodes.len(),
            lightdata_bytes: world.lightmap_data.len(),
        };
        Ok((world, facts))
    }

    pub(super) fn run_witnesses(case: &FixtureCase, world: &bsp::BspWorld) -> Vec<WitnessResult> {
        let transform = QuakeToEngine::default();
        case.witnesses
            .iter()
            .map(|witness| {
                let details: Vec<_> = witness
                    .query_coords
                    .iter()
                    .map(|coordinate| {
                        let point = transform.position(coordinate[0], coordinate[1], coordinate[2]);
                        let contents =
                            bsp::point_contents(point, &world.nodes, &world.leaves, &world.planes);
                        let is_solid = contents.is_solid();
                        let matched_expectation = if witness.expected_pass {
                            !is_solid
                        } else {
                            is_solid
                        };
                        fixtures::WitnessDetail {
                            point: *coordinate,
                            contents: format!("{contents:?}"),
                            is_solid,
                            matched_expectation,
                        }
                    })
                    .collect();
                WitnessResult {
                    id: witness.id.clone(),
                    description: witness.description.clone(),
                    expected_pass: witness.expected_pass,
                    actual_pass: details.iter().all(|detail| detail.matched_expectation),
                    coordinates: witness.query_coords.clone(),
                    details,
                }
            })
            .collect()
    }

    #[test]
    fn compiler_spatial_proof_full() {
        let profile_path = fixtures::compiler_profile_path();
        let profile = compiler::parse_compiler_profile(&profile_path)
            .unwrap_or_else(|error| panic!("parse {}: {error}", profile_path.display()));
        assert_eq!(profile.name, "ericw-q1-bsp2-generated");
        assert_eq!(profile.required_version, "2.0.0-alpha3");

        let tool_dir = compiler::resolve_tool_dir();
        let tools_present = compiler::tools_available(&tool_dir);
        let hash_result = if tools_present {
            Some(compiler::verify_executable_hashes(&tool_dir, &profile))
        } else {
            None
        };
        let unavailable_reason = match hash_result {
            None => Some(format!(
                "required ericw-tools executables are unavailable at {}",
                tool_dir.display()
            )),
            Some(Ok(())) => None,
            Some(Err(errors)) => Some(format!(
                "ericw-tools executable hashes are not authorized: {}",
                errors.join("; ")
            )),
        };
        let hashes_verified = tools_present && unavailable_reason.is_none();

        let env_identity = compiler::EnvIdentity::record(&tool_dir);
        let mut report = CompilerSpatialReport::new(
            &profile.name,
            &profile.required_version,
            &tool_dir,
            tools_present,
            hashes_verified,
            &env_identity.home,
            &env_identity.path,
            &env_identity.lang,
        );
        report.set_synthetic_query_pipeline(synthetic_world_queries::prove_query_pipeline());

        let cases = fixtures::load_fixture_cases().expect("load fixture cases");
        for case in &cases.cases {
            report.add_result(run_fixture_case(
                case,
                &profile,
                &tool_dir,
                unavailable_reason.as_deref(),
            ));
        }
        report.recompute_summary();
        report.write().expect("write compiler spatial report");

        assert_eq!(
            report.synthetic_query_pipeline.status,
            FixtureStatus::Pass,
            "the independent bsp query-pipeline proof failed"
        );
        for result in &report.results {
            if result.status == FixtureStatus::Pass {
                assert!(result.bsp_sha256.is_some(), "PASS without BSP hash");
                assert!(result.lit_sha256.is_some(), "PASS without LIT hash");
                assert!(result.strict_reload.is_some(), "PASS without strict reload");
                assert!(
                    result
                        .witness_results
                        .iter()
                        .all(|witness| witness.actual_pass),
                    "PASS with failed compiler-produced witness"
                );
            } else if result.strict_reload.is_none() {
                assert!(
                    result.witness_results.is_empty(),
                    "compiler-failed fixture must not receive synthetic fixture witnesses"
                );
            }
        }

        if unavailable_reason.is_some() {
            assert_eq!(report.compiler_qualification_status, FixtureStatus::NotRun);
            eprintln!(
                "compiler qualification NOT_RUN: {}",
                unavailable_reason.expect("checked above")
            );
            return;
        }

        assert_eq!(
            report.phase_status,
            FixtureStatus::Pass,
            "phase harness failed; report: {}",
            fixtures::report_path().display()
        );
        if report.compiler_qualification_status == FixtureStatus::Fail {
            assert!(
                report
                    .results
                    .iter()
                    .filter(|result| result.status == FixtureStatus::Fail)
                    .all(|result| result.tool_limitation.is_some()),
                "an unapproved compiler/reload/witness failure cannot pass the phase"
            );
            eprintln!(
                "compiler qualification FAIL due to documented ericw-tools hull limitation; synthetic query pipeline PASS remains separately scoped"
            );
        } else {
            assert_eq!(report.compiler_qualification_status, FixtureStatus::Pass);
            eprintln!(
                "compiler qualification PASS: {} compiler-produced BSP2/LIT worlds strict-reloaded and passed witnesses",
                report.summary.passed
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Runner and failure-closure tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(unix)]
fn write_executable(path: &Path, script: &str) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::write(path, script).unwrap();
    let mut permissions = path.metadata().unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).unwrap();
}

#[cfg(unix)]
#[test]
fn runner_drains_stdout_and_stderr_concurrently() {
    let temp = tempfile::tempdir().unwrap();
    write_executable(
        &temp.path().join("emit"),
        "#!/bin/sh\ni=0\nwhile [ $i -lt 2048 ]; do printf 1234567890; printf abcdefghij >&2; i=$((i+1)); done\n",
    );
    let output = compiler::run_stage(
        temp.path(),
        "emit",
        &[],
        temp.path(),
        "concurrent-drain",
        std::time::Duration::from_secs(5),
        50_000,
    )
    .expect("both full pipes must be drained without deadlock");
    assert_eq!(output.exit_code, 0);
    assert_eq!(output.stdout.len() + output.stderr.len(), 40_960);
}

#[cfg(unix)]
#[test]
fn runner_enforces_one_combined_output_ceiling() {
    let temp = tempfile::tempdir().unwrap();
    write_executable(
        &temp.path().join("overflow"),
        "#!/bin/sh\nprintf 12345678\nprintf abcdefgh >&2\n",
    );
    let error = compiler::run_stage(
        temp.path(),
        "overflow",
        &[],
        temp.path(),
        "combined-overflow",
        std::time::Duration::from_secs(5),
        12,
    )
    .unwrap_err();
    assert!(error.contains("combined stdout/stderr"), "{error}");
}

#[cfg(unix)]
#[test]
fn runner_timeout_terminates_process_group() {
    let temp = tempfile::tempdir().unwrap();
    write_executable(
        &temp.path().join("tree"),
        "#!/bin/sh\nsleep 30 &\necho $! > descendant.pid\nwait\n",
    );
    let error = compiler::run_stage(
        temp.path(),
        "tree",
        &[],
        temp.path(),
        "process-group-timeout",
        std::time::Duration::from_millis(100),
        1024,
    )
    .unwrap_err();
    assert!(error.contains("timeout"), "{error}");

    let pid: u32 = std::fs::read_to_string(temp.path().join("descendant.pid"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let proc_status = PathBuf::from(format!("/proc/{pid}/stat"));
    for _ in 0..100 {
        let no_live_descendant = match std::fs::read_to_string(&proc_status) {
            Ok(stat) => stat.split_whitespace().nth(2) == Some("Z"),
            Err(_) => true,
        };
        if no_live_descendant {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("descendant {pid} remained live after process-group timeout");
}

#[cfg(unix)]
#[test]
fn compiler_stops_after_generic_warning() {
    let temp = tempfile::tempdir().unwrap();
    for tool in ["qbsp", "vis", "light"] {
        let script = if tool == "qbsp" {
            "#!/bin/sh\necho 'WARNING: owner has not approved this warning'\nexit 0\n"
        } else {
            "#!/bin/sh\necho ran > later-stage-ran\nexit 0\n"
        };
        write_executable(&temp.path().join(tool), script);
    }
    let map = temp.path().join("input.map");
    let wad = temp.path().join("input.wad");
    let palette = temp.path().join("input.lmp");
    std::fs::write(&map, "map").unwrap();
    std::fs::write(&wad, "wad").unwrap();
    std::fs::write(&palette, "palette").unwrap();
    let work = tempfile::tempdir().unwrap();
    let profile = CompilerProfile {
        name: "test".to_string(),
        compiler_identity: "test".to_string(),
        required_version: "test".to_string(),
        qbsp_executable: "qbsp".to_string(),
        vis_executable: "vis".to_string(),
        light_executable: "light".to_string(),
        default_qbsp_args: Vec::new(),
        default_vis_args: Vec::new(),
        default_light_args: Vec::new(),
        timeout_seconds: 5,
        max_output_size: 1024,
        expected_hashes: Default::default(),
    };
    let failure = compiler::compile_map(&map, work.path(), temp.path(), &wad, &palette, &profile)
        .unwrap_err();
    assert_eq!(failure.kind, compiler::CompileFailureKind::Diagnostic);
    assert_eq!(failure.stage_outputs.len(), 1);
    assert!(!work.path().join("later-stage-ran").exists());
}

#[test]
fn synthetic_pass_cannot_promote_compiler_failure() {
    let mut report =
        CompilerSpatialReport::new("test", "test", Path::new("/tmp"), true, true, "", "", "");
    report.set_synthetic_query_pipeline(SyntheticQueryPipelineProof {
        status: FixtureStatus::Pass,
        scope: "query only".to_string(),
        queries_total: 2,
        queries_passed: 2,
    });
    report.add_result(FixtureResult {
        case_id: "compiler-failed".to_string(),
        map_file: "compiler-failed".to_string(),
        status: FixtureStatus::Fail,
        source_brush_count: 74,
        bsp_sha256: None,
        lit_sha256: None,
        bsp_size: None,
        lit_size: None,
        compilation_time_ms: None,
        diagnostics: vec!["qbsp failed".to_string()],
        tool_limitation: None,
        stage_outputs: Vec::new(),
        strict_reload: None,
        witness_results: Vec::new(),
    });
    report.recompute_summary();
    assert_eq!(report.compiler_qualification_status, FixtureStatus::Fail);
    assert_eq!(report.phase_status, FixtureStatus::Fail);
    assert_eq!(report.summary.passed, 0);
}

#[test]
fn documented_tool_gap_does_not_become_compiler_pass() {
    let mut report =
        CompilerSpatialReport::new("test", "test", Path::new("/tmp"), true, true, "", "", "");
    report.set_synthetic_query_pipeline(SyntheticQueryPipelineProof {
        status: FixtureStatus::Pass,
        scope: "query only".to_string(),
        queries_total: 2,
        queries_passed: 2,
    });
    report.add_result(FixtureResult {
        case_id: "known-tool-gap".to_string(),
        map_file: "known-tool-gap".to_string(),
        status: FixtureStatus::Fail,
        source_brush_count: 74,
        bsp_sha256: None,
        lit_sha256: None,
        bsp_size: None,
        lit_size: None,
        compilation_time_ms: None,
        diagnostics: vec!["qbsp exit -1".to_string()],
        tool_limitation: Some("ericw hull computation limitation".to_string()),
        stage_outputs: Vec::new(),
        strict_reload: None,
        witness_results: Vec::new(),
    });
    report.recompute_summary();
    assert_eq!(report.compiler_qualification_status, FixtureStatus::Fail);
    assert_eq!(report.summary.passed, 0);
    assert_eq!(report.phase_status, FixtureStatus::Pass);
    assert_eq!(
        report.known_tool_limitation.observed_case_ids,
        ["known-tool-gap"]
    );
}

#[test]
fn profile_and_baseline_inputs_remain_present() {
    assert!(fixtures::compiler_profile_path().is_file());
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(crate_dir.join("tests/enhanced_v3_integrated.rs").is_file());
    assert!(crate_dir.join("tests/enhanced_v3_baseline.rs").is_file());
}
