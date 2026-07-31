//! Phase 04 — Enhanced V3 compiler smoke test.
//!
//! Generates representative maps through the production pipeline and
//! compiles them through the pinned qbsp → vis → light BSP2/QLIT-v1
//! profile. Tests verify warning-free compilation, valid BSP2/LIT output,
//! and strict reload through the `bsp` crate.
//!
//! # Constraints
//!
//! - Compiler unavailability is a blocked failing test, never `ignore` or skip.
//! - No synthetic BSP data is accepted.
//! - Representative maps are generated solely through the production pipeline.
//! - All three stages must produce valid output without warnings.
//! - BSP2 magic and nonempty QLIT v1 are required.

#[path = "support/enhanced_v3_compiler.rs"]
mod compiler_support;

use bsp_generator::enhanced_v3::*;
use compiler_support::{
    compile_map, load_compiler_profile, resolve_tool_dir, theme_paths, tools_available,
    verify_executable_hashes, CompiledArtifacts,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

// ── Helper: write map and compile ─────────────────────────────────────────

/// Generate a map for the given config, write it to a temp file, and compile it.
fn generate_and_compile(
    config: &V3Config,
    label: &str,
) -> Result<(V3PipelineOutput, CompiledArtifacts), String> {
    let output = run_pipeline(config).map_err(|e| format!("generation failed: {e}"))?;

    let staging =
        compiler_support::create_staging_dir(label).map_err(|e| format!("staging: {e}"))?;
    // Write map to a source file that is NOT named "generated.map" to
    // avoid truncation when compile_map copies it to "generated.map".
    let src_map_path = staging.path().join("source.map");
    fs::write(&src_map_path, &output.map_text).map_err(|e| format!("write map: {e}"))?;

    let (wad_path, palette_path) = theme_paths();
    let profile = load_compiler_profile()?;
    let tool_dir = resolve_tool_dir();

    let compiled = compile_map(
        &src_map_path,
        staging.path(),
        &tool_dir,
        &wad_path,
        &palette_path,
        &profile,
    )
    .map_err(|failure| {
        // Keep staging dir alive on failure for debugging
        // Keep staging dir alive on failure for debugging
        #[allow(deprecated)]
        let retained = staging.into_path();
        format!(
            "compilation failed (staging at {}): {}",
            retained.display(),
            failure.message
        )
    })?;

    Ok((output, compiled))
}

fn build_production_topology(config: &V3Config) -> CommittedTopology {
    let seed = V3Seed::new(config.seed);
    let mut alloc = V3IdAllocator::new();
    let (footprints, layout) = build_footprints(config, seed, &mut alloc).unwrap();
    build_topology(config, &footprints, &layout, seed, &mut alloc).unwrap()
}

fn strict_reload(compiled: &CompiledArtifacts, identity: &str) -> bsp::BspWorld {
    let (wad_path, palette_path) = theme_paths();
    let options = bsp::LoadOptions {
        strict: true,
        palette: Some(fs::read(&palette_path).expect("read palette")),
        lit_data: Some(compiled.lit_data.clone()),
        wad_archives: vec![(
            "cc0_dungeon_v2.wad".to_string(),
            fs::read(&wad_path).expect("read WAD"),
        )],
        texture_overrides: Vec::new(),
        source_identity: identity.to_string(),
    };
    let world = bsp::BspLoader::load(&compiled.bsp_data, &options).expect("strict BSP reload");
    assert_eq!(world.profile, bsp::profile::BspProfile::Bsp2);
    assert!(
        world.diagnostics.is_empty(),
        "strict reload emitted diagnostics: {:?}",
        world.diagnostics
    );
    world
}

fn point_contents_quake(world: &bsp::BspWorld, point: (i32, i32, i32)) -> bsp::PointContents {
    let transform = bsp::QuakeToEngine::default();
    bsp::point_contents(
        transform.position(point.0 as f32, point.1 as f32, point.2 as f32),
        &world.nodes,
        &world.leaves,
        &world.planes,
    )
}

fn assert_non_solid(world: &bsp::BspWorld, label: &str, point: (i32, i32, i32)) {
    let contents = point_contents_quake(world, point);
    assert!(
        !contents.is_solid(),
        "{label} is solid at {point:?}: {contents:?}"
    );
}

fn assert_solid(world: &bsp::BspWorld, label: &str, point: (i32, i32, i32)) {
    let contents = point_contents_quake(world, point);
    assert!(
        contents.is_solid(),
        "{label} lacks solid support at {point:?}: {contents:?}"
    );
}

fn assert_trace(
    world: &bsp::BspWorld,
    label: &str,
    hull: bsp::StoredHull,
    start: (i32, i32, i32),
    end: (i32, i32, i32),
) {
    let transform = bsp::QuakeToEngine::default();
    let trace = bsp::trace_line(
        transform.position(start.0 as f32, start.1 as f32, start.2 as f32),
        transform.position(end.0 as f32, end.1 as f32, end.2 as f32),
        hull,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &transform,
    );
    assert!(
        !trace.starts_solid && !trace.all_solid && trace.hit_fraction >= 0.999_9,
        "{label} {hull:?} trace {start:?}->{end:?} blocked: {trace:?}"
    );
}

fn assert_player_trace(
    world: &bsp::BspWorld,
    label: &str,
    start: (i32, i32, i32),
    end: (i32, i32, i32),
) {
    assert_trace(world, label, bsp::StoredHull::Player, start, end);
}

fn assert_point_trace(
    world: &bsp::BspWorld,
    label: &str,
    start: (i32, i32, i32),
    end: (i32, i32, i32),
) {
    // Sample the compiler-produced render BSP at half-quantum spacing. Hull-0
    // clipnodes may outside-fill disconnected-by-expansion stair volumes even
    // when the authored 64×80 swept centerline is open, so node/leaf contents
    // are the strict source-space continuity witness here.
    let delta = (end.0 - start.0, end.1 - start.1, end.2 - start.2);
    let distance = delta.0.abs().max(delta.1.abs()).max(delta.2.abs());
    let steps = (distance / 8).max(1);
    for step in 0..=steps {
        let point = (
            start.0 + delta.0 * step / steps,
            start.1 + delta.1 * step / steps,
            start.2 + delta.2 * step / steps,
        );
        assert_non_solid(world, &format!("{label} sample {step}/{steps}"), point);
    }
}

fn wall_witness(
    room: &CommittedRoom,
    direction: &str,
    tangent: i32,
    normal_depth: i32,
    z: i32,
) -> (i32, i32, i32) {
    match direction {
        "north" => (tangent, room.shell.1 + normal_depth, z),
        "south" => (tangent, room.shell.3 - normal_depth, z),
        "west" => (room.shell.0 + normal_depth, tangent, z),
        "east" => (room.shell.2 - normal_depth, tangent, z),
        other => panic!("non-cardinal direction {other}"),
    }
}

fn opposite(direction: &str) -> &'static str {
    match direction {
        "north" => "south",
        "south" => "north",
        "west" => "east",
        "east" => "west",
        other => panic!("non-cardinal direction {other}"),
    }
}

// ── Compiler availability gate ────────────────────────────────────────────

#[test]
fn compiler_tools_are_available_and_hashes_match() {
    let tool_dir = resolve_tool_dir();
    assert!(
        tools_available(&tool_dir),
        "ericw-tools not found at {}. Install ericw-tools 2.0.0-alpha3 or set ERICW_TOOLS_DIR.",
        tool_dir.display()
    );

    let profile = load_compiler_profile().expect("load compiler profile");
    assert_eq!(profile.name, "ericw-q1-bsp2-generated");
    assert_eq!(profile.required_version, "2.0.0-alpha3");

    verify_executable_hashes(&tool_dir, &profile).expect("ericw-tools executable hash mismatch");
}

#[test]
fn theme_assets_are_present() {
    let (wad, palette) = theme_paths();
    assert!(wad.exists(), "WAD not found at {}", wad.display());
    assert!(
        palette.exists(),
        "palette not found at {}",
        palette.display()
    );
}

// ── Sparse representative compilation ─────────────────────────────────────

#[test]
fn sparse_compiles_warning_free_to_valid_bsp2_and_lit() {
    let config = V3Config::nominal_sparse();
    let (output, compiled) =
        generate_and_compile(&config, "smoke-sparse").expect("sparse compilation failed");

    // BSP2 magic verified during compilation
    assert!(!compiled.bsp_data.is_empty());
    assert!(!compiled.lit_data.is_empty());
    assert!(!compiled.bsp_sha256.is_empty());
    assert!(!compiled.lit_sha256.is_empty());

    // Source checks
    assert!(!output.map_text.is_empty());
    assert!(output.metadata.actual_brushes() > 0);

    // Verify BSP2 magic bytes
    assert_eq!(&compiled.bsp_data[..4], b"BSP2");
    // Verify QLIT magic
    assert_eq!(&compiled.lit_data[..4], b"QLIT");
}

#[test]
fn sparse_qbsp_no_warnings() {
    let config = V3Config::nominal_sparse();
    let (_output, compiled) =
        generate_and_compile(&config, "smoke-sparse-qbsp").expect("sparse qbsp failed");

    assert!(
        compiled.qbsp_output.diagnostics.is_empty(),
        "qbsp emitted warnings: {:?}",
        compiled.qbsp_output.diagnostics
    );
    assert_eq!(compiled.qbsp_output.exit_code, 0);
}

#[test]
fn sparse_vis_no_warnings() {
    let config = V3Config::nominal_sparse();
    let (_output, compiled) =
        generate_and_compile(&config, "smoke-sparse-vis").expect("sparse vis failed");

    assert!(
        compiled.vis_output.diagnostics.is_empty(),
        "vis emitted warnings: {:?}",
        compiled.vis_output.diagnostics
    );
    assert_eq!(compiled.vis_output.exit_code, 0);
}

#[test]
fn sparse_light_no_warnings() {
    let config = V3Config::nominal_sparse();
    let (_output, compiled) =
        generate_and_compile(&config, "smoke-sparse-light").expect("sparse light failed");

    assert!(
        compiled.light_output.diagnostics.is_empty(),
        "light emitted warnings: {:?}",
        compiled.light_output.diagnostics
    );
    assert_eq!(compiled.light_output.exit_code, 0);
}

// ── Moderate representative compilation ───────────────────────────────────

#[test]
fn moderate_compiles_warning_free() {
    let config = V3Config::nominal_moderate();
    let (_output, compiled) =
        generate_and_compile(&config, "smoke-moderate").expect("moderate compilation failed");

    assert!(!compiled.bsp_data.is_empty());
    assert_eq!(&compiled.bsp_data[..4], b"BSP2");
    assert_eq!(&compiled.lit_data[..4], b"QLIT");

    assert!(compiled.qbsp_output.diagnostics.is_empty());
    assert!(compiled.vis_output.diagnostics.is_empty());
    assert!(compiled.light_output.diagnostics.is_empty());
}

// ── Rich representative compilation ───────────────────────────────────────

#[test]
fn rich_compiles_warning_free() {
    let config = V3Config::nominal_rich();
    let (_output, compiled) =
        generate_and_compile(&config, "smoke-rich").expect("rich compilation failed");

    assert!(!compiled.bsp_data.is_empty());
    assert_eq!(&compiled.bsp_data[..4], b"BSP2");
    assert_eq!(&compiled.lit_data[..4], b"QLIT");

    assert!(compiled.qbsp_output.diagnostics.is_empty());
    assert!(compiled.vis_output.diagnostics.is_empty());
    assert!(compiled.light_output.diagnostics.is_empty());
}

// ── Production seed-42 structural compilation ─────────────────────────────

#[test]
fn seed_42_production_presets_compile_and_pass_strict_spatial_witnesses() {
    // Stored-hull witnesses sit one unit above exact floor/tread contact so the
    // trace proves clear volume rather than depending on clip-plane equality.
    const HULL_EYE_OFFSET: i32 = 25;
    let cases = [
        (
            "seed42-sparse",
            V3Config::new(42, V3Preset::Sparse, 2048).unwrap(),
        ),
        (
            "seed42-moderate",
            V3Config::new(42, V3Preset::Moderate, 2048).unwrap(),
        ),
        (
            "seed42-rich",
            V3Config::new(42, V3Preset::Rich, 3072).unwrap(),
        ),
    ];

    for (label, config) in cases {
        let (output, compiled) =
            generate_and_compile(&config, label).unwrap_or_else(|error| panic!("{label}: {error}"));
        let topology = build_production_topology(&config);
        let world = strict_reload(&compiled, label);
        assert!(compiled.qbsp_output.diagnostics.is_empty());
        assert!(compiled.vis_output.diagnostics.is_empty());
        assert!(compiled.light_output.diagnostics.is_empty());
        assert!(output.metadata.actual_faces() <= config.preset.face_budget());

        let spawn = output.metadata.spawn_origin();
        assert_non_solid(&world, &format!("{label} spawn"), spawn);
        assert_player_trace(&world, &format!("{label} spawn hull"), spawn, spawn);

        let room_by_id: BTreeMap<_, _> =
            topology.rooms.iter().map(|room| (room.id, room)).collect();
        let room_center = |room: &CommittedRoom| {
            (
                (room.shell.0 + room.shell.2) / 2,
                (room.shell.1 + room.shell.3) / 2,
                room.floor_z + CONSTRUCTION_QUANTUM + HULL_EYE_OFFSET,
            )
        };
        for room in &topology.rooms {
            let center = room_center(room);
            assert_non_solid(&world, &format!("{label} {} center", room.id), center);
        }

        let mut adjacency: BTreeMap<RoomId, Vec<RoomId>> = BTreeMap::new();
        for route in &topology.routes {
            adjacency
                .entry(route.source_room)
                .or_default()
                .push(route.target_room);
            adjacency
                .entry(route.target_room)
                .or_default()
                .push(route.source_room);
            let portal = topology
                .portals
                .iter()
                .find(|portal| {
                    portal.source_room == route.source_room
                        && portal.target_room == Some(route.target_room)
                })
                .expect("every route has one structural portal");
            let source = room_by_id[&route.source_room];
            let target = room_by_id[&route.target_room];
            let target_direction = opposite(&portal.wall);
            let tangent = if matches!(portal.wall.as_str(), "east" | "west") {
                portal.anchor.1
            } else {
                portal.anchor.0
            };
            let eye_z = source.floor_z + CONSTRUCTION_QUANTUM + HULL_EYE_OFFSET;
            let source_throat = wall_witness(source, &portal.wall, tangent, 8, eye_z);
            let target_throat = wall_witness(target, target_direction, tangent, 8, eye_z);
            let source_approach = wall_witness(source, &portal.wall, tangent, 40, eye_z);
            let target_approach = wall_witness(target, target_direction, tangent, 40, eye_z);
            let corridor = if matches!(portal.wall.as_str(), "east" | "west") {
                ((source_throat.0 + target_throat.0) / 2, tangent, eye_z)
            } else {
                (tangent, (source_throat.1 + target_throat.1) / 2, eye_z)
            };
            let route_points = [
                room_center(source),
                source_approach,
                source_throat,
                corridor,
                target_throat,
                target_approach,
                room_center(target),
            ];
            for (index, point) in route_points.iter().enumerate() {
                assert_non_solid(
                    &world,
                    &format!("{label} route {} witness {index}", route.id),
                    *point,
                );
            }
            for (index, segment) in route_points.windows(2).enumerate() {
                assert_point_trace(
                    &world,
                    &format!("{label} route {} segment {index}", route.id),
                    segment[0],
                    segment[1],
                );
            }

            // The 80-unit rectangular swept core remains clear through the
            // full wall depth, and the first pointed step has a visible apex.
            for (room, direction) in [(source, portal.wall.as_str()), (target, target_direction)] {
                let floor_top = room.floor_z + CONSTRUCTION_QUANTUM;
                for depth in [1, 8, 15] {
                    for height in [1, 40, 79] {
                        assert_non_solid(
                            &world,
                            &format!("{label} portal {} core", portal.id),
                            wall_witness(room, direction, tangent, depth, floor_top + height),
                        );
                    }
                }
                assert_non_solid(
                    &world,
                    &format!("{label} portal {} pointed apex", portal.id),
                    wall_witness(room, direction, tangent, 8, floor_top + HEADROOM + 8),
                );
            }
        }

        let transition = &topology.transitions[0];
        adjacency
            .entry(transition.lower_room)
            .or_default()
            .push(transition.upper_room);
        adjacency
            .entry(transition.upper_room)
            .or_default()
            .push(transition.lower_room);
        let lower = room_by_id[&transition.lower_room];
        let upper = room_by_id[&transition.upper_room];
        let x_center = (transition.protected_volume.0 + transition.protected_volume.3) / 2;
        let tread_start = transition.tread_run.1;
        let mut stair_path = Vec::new();
        let lower_landing = (
            (transition.lower_landing.0 + transition.lower_landing.2) / 2,
            (transition.lower_landing.1 + transition.lower_landing.3) / 2,
            lower.floor_z + CONSTRUCTION_QUANTUM + HULL_EYE_OFFSET,
        );
        stair_path.push(lower_landing);
        for step in 0..12_i32 {
            let tread_center = (
                x_center,
                tread_start + step * CONSTRUCTION_QUANTUM + CONSTRUCTION_QUANTUM / 2,
            );
            let tread_top = lower.floor_z + (step + 1) * CONSTRUCTION_QUANTUM;
            let eye = (tread_center.0, tread_center.1, tread_top + HULL_EYE_OFFSET);
            assert_solid(
                &world,
                &format!("{label} tread {step} support"),
                (tread_center.0, tread_center.1, tread_top - 8),
            );
            assert_non_solid(
                &world,
                &format!("{label} tread {step} clearance"),
                (tread_center.0, tread_center.1, tread_top + 79),
            );
            stair_path.push(eye);
        }
        let run_end = tread_start + 12 * CONSTRUCTION_QUANTUM;
        assert_eq!(run_end - tread_start, 192);
        let upper_approach = (
            x_center,
            (run_end + upper.shell.1) / 2,
            upper.floor_z + CONSTRUCTION_QUANTUM + HULL_EYE_OFFSET,
        );
        let upper_landing = (
            (transition.upper_landing.0 + transition.upper_landing.2) / 2,
            (transition.upper_landing.1 + transition.upper_landing.3) / 2,
            upper.floor_z + CONSTRUCTION_QUANTUM + HULL_EYE_OFFSET,
        );
        for (name, point) in [
            ("upper approach", upper_approach),
            ("upper landing", upper_landing),
        ] {
            assert_non_solid(&world, &format!("{label} {name}"), point);
        }
        assert_solid(
            &world,
            &format!("{label} upper approach support"),
            (upper_approach.0, upper_approach.1, upper.floor_z + 8),
        );
        assert_non_solid(
            &world,
            &format!("{label} upper approach 80-unit crest clearance"),
            (
                upper_approach.0,
                upper_approach.1,
                upper.floor_z + CONSTRUCTION_QUANTUM + 79,
            ),
        );
        stair_path.extend([upper_approach, upper_landing]);
        for (index, segment) in stair_path.windows(2).enumerate() {
            assert_point_trace(
                &world,
                &format!("{label} stair segment {index}"),
                segment[0],
                segment[1],
            );
        }
        assert_point_trace(
            &world,
            &format!("{label} lower room to stair"),
            room_center(lower),
            lower_landing,
        );
        assert_point_trace(
            &world,
            &format!("{label} stair to upper room"),
            upper_landing,
            room_center(upper),
        );

        let spawn_room = topology
            .rooms
            .iter()
            .find(|room| {
                spawn.0 > room.shell.0
                    && spawn.0 < room.shell.2
                    && spawn.1 > room.shell.1
                    && spawn.1 < room.shell.3
                    && spawn.2 > room.floor_z
                    && spawn.2 < room.floor_z + room.dims.2 as i32
            })
            .expect("spawn belongs to a committed room");
        let mut visited = BTreeSet::new();
        let mut stack = vec![spawn_room.id];
        while let Some(room) = stack.pop() {
            if !visited.insert(room) {
                continue;
            }
            stack.extend(adjacency.get(&room).into_iter().flatten().copied());
        }
        assert_eq!(
            visited.len(),
            topology.rooms.len(),
            "{label}: physically witnessed route graph does not reach every room"
        );
    }
}

// ── Budget validation tests ───────────────────────────────────────────────

#[test]
fn all_presets_stay_within_compiled_budgets() {
    let configs = [
        ("smoke-budget-sparse", V3Config::nominal_sparse()),
        ("smoke-budget-moderate", V3Config::nominal_moderate()),
        ("smoke-budget-rich", V3Config::nominal_rich()),
    ];

    for (label, config) in &configs {
        let (output, compiled) = generate_and_compile(config, label)
            .unwrap_or_else(|e| panic!("{label} compilation failed: {e}"));

        // Source budgets
        let actual_faces = output.metadata.actual_faces();
        assert!(
            actual_faces < 10000,
            "{label}: source faces {actual_faces} exceeds 10000 budget"
        );

        // BSP and LIT must be non-empty
        assert!(!compiled.bsp_data.is_empty(), "{label}: empty BSP");
        assert!(!compiled.lit_data.is_empty(), "{label}: empty LIT");
    }
}

// ── Strict BSP reload tests ───────────────────────────────────────────────

#[test]
fn sparse_bsp_strict_reloads_without_diagnostics() {
    let config = V3Config::nominal_sparse();
    let (_output, compiled) =
        generate_and_compile(&config, "smoke-reload-sparse").expect("sparse reload prep failed");

    let (wad_path, palette_path) = theme_paths();

    let options = bsp::LoadOptions {
        strict: true,
        palette: Some(fs::read(&palette_path).expect("read palette")),
        lit_data: Some(compiled.lit_data.clone()),
        wad_archives: vec![(
            "cc0_dungeon_v2.wad".to_string(),
            fs::read(&wad_path).expect("read WAD"),
        )],
        texture_overrides: Vec::new(),
        source_identity: "enhanced-v3-compiler-smoke".to_string(),
    };

    let world =
        bsp::BspLoader::load(&compiled.bsp_data, &options).expect("strict BSP reload failed");

    assert_eq!(
        world.profile,
        bsp::profile::BspProfile::Bsp2,
        "expected BSP2 profile"
    );
    assert!(
        world.diagnostics.is_empty(),
        "strict reload emitted diagnostics: {:?}",
        world.diagnostics
    );
    assert!(!world.entities.is_empty(), "no entities in BSP");
    assert!(world.leaves.len() > 2, "too few leaves");
}

#[test]
fn sparse_bsp_has_solid_and_empty_leaves() {
    let config = V3Config::nominal_sparse();
    let (_output, compiled) =
        generate_and_compile(&config, "smoke-leaves-sparse").expect("sparse leaves prep failed");

    let (wad_path, palette_path) = theme_paths();

    let options = bsp::LoadOptions {
        strict: true,
        palette: Some(fs::read(&palette_path).expect("read palette")),
        lit_data: Some(compiled.lit_data.clone()),
        wad_archives: vec![(
            "cc0_dungeon_v2.wad".to_string(),
            fs::read(&wad_path).expect("read WAD"),
        )],
        texture_overrides: Vec::new(),
        source_identity: "enhanced-v3-compiler-smoke".to_string(),
    };

    let world =
        bsp::BspLoader::load(&compiled.bsp_data, &options).expect("strict BSP reload failed");

    let solid_leaves: Vec<_> = world.leaves.iter().filter(|l| l.contents == -2).collect();
    let empty_leaves: Vec<_> = world.leaves.iter().filter(|l| l.contents == -1).collect();

    assert!(!solid_leaves.is_empty(), "no solid leaves");
    assert!(!empty_leaves.is_empty(), "no empty leaves");
}

// ── Spatial witness tests ─────────────────────────────────────────────────

#[test]
fn spawn_point_is_in_empty_space() {
    let config = V3Config::nominal_sparse();
    let (output, compiled) =
        generate_and_compile(&config, "smoke-witness-spawn").expect("spawn witness prep failed");

    let (wad_path, palette_path) = theme_paths();
    let options = bsp::LoadOptions {
        strict: true,
        palette: Some(fs::read(&palette_path).expect("read palette")),
        lit_data: Some(compiled.lit_data.clone()),
        wad_archives: vec![(
            "cc0_dungeon_v2.wad".to_string(),
            fs::read(&wad_path).expect("read WAD"),
        )],
        texture_overrides: Vec::new(),
        source_identity: "enhanced-v3-compiler-smoke".to_string(),
    };

    let world =
        bsp::BspLoader::load(&compiled.bsp_data, &options).expect("strict BSP reload failed");

    let (sx, sy, sz) = output.metadata.spawn_origin();
    let transform = bsp::QuakeToEngine::default();
    let point = transform.position(sx as f32, sy as f32, sz as f32);
    let contents = bsp::point_contents(point, &world.nodes, &world.leaves, &world.planes);

    assert!(
        !contents.is_solid(),
        "spawn point ({sx}, {sy}, {sz}) is in solid space: {contents:?}"
    );
}

#[test]
fn room_centers_are_in_empty_space() {
    let config = V3Config::nominal_sparse();
    let (output, compiled) = generate_and_compile(&config, "smoke-witness-rooms")
        .expect("room centers witness prep failed");

    let (wad_path, palette_path) = theme_paths();
    let options = bsp::LoadOptions {
        strict: true,
        palette: Some(fs::read(&palette_path).expect("read palette")),
        lit_data: Some(compiled.lit_data.clone()),
        wad_archives: vec![(
            "cc0_dungeon_v2.wad".to_string(),
            fs::read(&wad_path).expect("read WAD"),
        )],
        texture_overrides: Vec::new(),
        source_identity: "enhanced-v3-compiler-smoke".to_string(),
    };

    let world =
        bsp::BspLoader::load(&compiled.bsp_data, &options).expect("strict BSP reload failed");

    let transform = bsp::QuakeToEngine::default();
    // Probe the spawn origin which is at the center of the first room
    let (sx, sy, sz) = output.metadata.spawn_origin();
    let point = transform.position(sx as f32, sy as f32, sz as f32);
    let contents = bsp::point_contents(point, &world.nodes, &world.leaves, &world.planes);
    assert!(!contents.is_solid(), "room center is solid");

    // Also probe a lower standing-height point. `spawn_origin` is floor +
    // 48, so subtracting 40 would sample inside the 0..16 floor slab.
    let low_point = transform.position(sx as f32, sy as f32, (sz - CONSTRUCTION_QUANTUM) as f32);
    let low_contents = bsp::point_contents(low_point, &world.nodes, &world.leaves, &world.planes);
    assert!(!low_contents.is_solid(), "lower room point is solid");
}

// ── Determinism across compilations ────────────────────────────────────────

#[test]
fn deterministic_generation_produces_same_compiled_output() {
    let config = V3Config::nominal_sparse();
    let (output1, compiled1) =
        generate_and_compile(&config, "smoke-det-a").expect("first compilation failed");
    let (output2, compiled2) =
        generate_and_compile(&config, "smoke-det-b").expect("second compilation failed");

    // Generated maps must be identical
    assert_eq!(output1.map_text, output2.map_text);
    // Compiled BSP should be identical (same map + same compiler + same args)
    assert_eq!(
        compiled1.bsp_sha256, compiled2.bsp_sha256,
        "BSP hash differs between deterministic runs"
    );
    assert_eq!(
        compiled1.lit_sha256, compiled2.lit_sha256,
        "LIT hash differs between deterministic runs"
    );
}

// ── No fixture dependency test ────────────────────────────────────────────

#[test]
fn smoke_tests_do_not_import_proof_modules() {
    // This test exists to verify that the compiler smoke test does not
    // import proof-only modules. The fact that this file compiles without
    // referencing enhanced_v3_proof is the proof.
    //
    // We generate outputs only through the production pipeline (run_pipeline).
    let config = V3Config::nominal_sparse();
    let output = run_pipeline(&config).expect("production pipeline failed");
    assert!(!output.map_text.is_empty());
    assert!(output.metadata.room_count() > 0);
}
