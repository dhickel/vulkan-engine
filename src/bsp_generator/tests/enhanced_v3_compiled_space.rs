//! Enhanced V3 production compiled-space qualification.
//!
//! This target deliberately uses the same pinned compiler support as the
//! compiler-smoke target. It never estimates room centers from global bounds,
//! substitutes source brush counts for renderer batches, or treats an
//! unavailable compiler as a passing result.

#[path = "support/enhanced_v3_compiler.rs"]
mod compiler_support;

use bsp_generator::enhanced_v3::*;
use compiler_support::{
    compile_map, load_compiler_profile, resolve_tool_dir, theme_paths, CompiledArtifacts,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

const HULL_EYE_OFFSET: i32 = 25;

fn compile_and_reload(
    config: &V3Config,
    label: &str,
) -> (V3PipelineOutput, CommittedTopology, bsp::BspWorld) {
    let output =
        run_pipeline(config).unwrap_or_else(|error| panic!("{label}: generation failed: {error}"));
    let mut alloc = V3IdAllocator::new();
    let seed = V3Seed::new(config.seed);
    let (footprints, layout) = build_footprints(config, seed, &mut alloc)
        .unwrap_or_else(|error| panic!("{label}: footprints: {error}"));
    let topology = build_topology(config, &footprints, &layout, seed, &mut alloc)
        .unwrap_or_else(|error| panic!("{label}: topology: {error}"));

    let staging = compiler_support::create_staging_dir(label)
        .unwrap_or_else(|error| panic!("{label}: staging: {error}"));
    let source = staging.path().join("source.map");
    fs::write(&source, &output.map_text)
        .unwrap_or_else(|error| panic!("{label}: write map: {error}"));
    let (wad, palette) = theme_paths();
    let compiled = compile_map(
        &source,
        staging.path(),
        &resolve_tool_dir(),
        &wad,
        &palette,
        &load_compiler_profile().expect("load pinned compiler profile"),
    )
    .unwrap_or_else(|failure| panic!("{label}: pinned compiler failed: {}", failure.message));
    let world = strict_reload(&compiled, label);
    (output, topology, world)
}

fn strict_reload(compiled: &CompiledArtifacts, identity: &str) -> bsp::BspWorld {
    let (wad, palette) = theme_paths();
    let options = bsp::LoadOptions {
        strict: true,
        palette: Some(fs::read(palette).expect("read palette")),
        lit_data: Some(compiled.lit_data.clone()),
        wad_archives: vec![(
            "cc0_dungeon_v2.wad".to_string(),
            fs::read(wad).expect("read WAD"),
        )],
        texture_overrides: Vec::new(),
        source_identity: identity.to_string(),
    };
    let world = bsp::BspLoader::load(&compiled.bsp_data, &options).expect("strict BSP reload");
    assert_eq!(world.profile, bsp::profile::BspProfile::Bsp2);
    assert!(
        world.diagnostics.is_empty(),
        "strict reload diagnostics: {:?}",
        world.diagnostics
    );
    world
}

fn point_contents(world: &bsp::BspWorld, point: (i32, i32, i32)) -> bsp::PointContents {
    let transform = bsp::QuakeToEngine::default();
    bsp::point_contents(
        transform.position(point.0 as f32, point.1 as f32, point.2 as f32),
        &world.nodes,
        &world.leaves,
        &world.planes,
    )
}

fn assert_clear(world: &bsp::BspWorld, label: &str, point: (i32, i32, i32)) {
    let contents = point_contents(world, point);
    assert!(
        !contents.is_solid(),
        "{label} is solid at {point:?}: {contents:?}"
    );
}

fn assert_solid(world: &bsp::BspWorld, label: &str, point: (i32, i32, i32)) {
    let contents = point_contents(world, point);
    assert!(
        contents.is_solid(),
        "{label} is not solid at {point:?}: {contents:?}"
    );
}

fn wall_witness(
    room: &CommittedRoom,
    direction: &str,
    tangent: i32,
    depth: i32,
    z: i32,
) -> (i32, i32, i32) {
    match direction {
        "north" => (tangent, room.shell.1 + depth, z),
        "south" => (tangent, room.shell.3 - depth, z),
        "west" => (room.shell.0 + depth, tangent, z),
        "east" => (room.shell.2 - depth, tangent, z),
        other => panic!("non-cardinal portal direction {other}"),
    }
}

fn opposite(direction: &str) -> &'static str {
    match direction {
        "north" => "south",
        "south" => "north",
        "west" => "east",
        "east" => "west",
        other => panic!("non-cardinal portal direction {other}"),
    }
}

fn room_center(room: &CommittedRoom) -> (i32, i32, i32) {
    (
        (room.shell.0 + room.shell.2) / 2,
        (room.shell.1 + room.shell.3) / 2,
        room.floor_z + CONSTRUCTION_QUANTUM + HULL_EYE_OFFSET,
    )
}

#[test]
fn production_matrix_is_total_deterministic_and_within_source_face_budget() {
    for preset in [V3Preset::Sparse, V3Preset::Moderate, V3Preset::Rich] {
        for extent in [1024, 2048, 3072] {
            let mut maps = BTreeSet::new();
            for seed in [0, 42, 99, 255] {
                let config = V3Config::new(seed, preset, extent).expect("valid matrix config");
                let first = run_pipeline(&config).unwrap_or_else(|error| {
                    panic!("{preset:?} seed={seed} extent={extent}: {error}")
                });
                let second = run_pipeline(&config).expect("deterministic replay");
                assert_eq!(first.map_text, second.map_text);
                assert_eq!(first.metadata, second.metadata);
                assert!(first.metadata.actual_faces() < 10_000);
                maps.insert(first.map_text);
            }
            assert!(
                maps.len() > 1,
                "seed does not vary {preset:?} at extent {extent}"
            );
        }
    }
}

#[test]
fn seed_42_production_presets_compile_reload_and_preserve_spatial_witnesses() {
    let cases = [
        (
            "compiled-space-sparse",
            V3Config::new(42, V3Preset::Sparse, 2048).unwrap(),
        ),
        (
            "compiled-space-moderate",
            V3Config::new(42, V3Preset::Moderate, 2048).unwrap(),
        ),
        (
            "compiled-space-rich",
            V3Config::new(42, V3Preset::Rich, 3072).unwrap(),
        ),
    ];

    for (label, config) in cases {
        let (output, topology, world) = compile_and_reload(&config, label);
        assert!(output.metadata.actual_faces() < 10_000);
        assert!(world.faces.len() < 10_000, "{label}: compiled face budget");
        assert!(world.entities.len() < 300, "{label}: entity budget");
        assert_clear(
            &world,
            &format!("{label} spawn"),
            output.metadata.spawn_origin(),
        );

        let rooms: BTreeMap<_, _> = topology.rooms.iter().map(|room| (room.id, room)).collect();
        for room in &topology.rooms {
            assert_clear(
                &world,
                &format!("{label} room {}", room.id),
                room_center(room),
            );
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
                .expect("route has committed portal");
            let source = rooms[&route.source_room];
            let target = rooms[&route.target_room];
            let target_direction = opposite(&portal.wall);
            let tangent = if matches!(portal.wall.as_str(), "east" | "west") {
                portal.anchor.1
            } else {
                portal.anchor.0
            };
            let eye = source.floor_z + CONSTRUCTION_QUANTUM + HULL_EYE_OFFSET;
            let source_throat = wall_witness(source, &portal.wall, tangent, 8, eye);
            let target_throat = wall_witness(target, target_direction, tangent, 8, eye);
            let corridor = if matches!(portal.wall.as_str(), "east" | "west") {
                ((source_throat.0 + target_throat.0) / 2, tangent, eye)
            } else {
                (tangent, (source_throat.1 + target_throat.1) / 2, eye)
            };
            for (index, point) in [
                room_center(source),
                source_throat,
                corridor,
                target_throat,
                room_center(target),
            ]
            .into_iter()
            .enumerate()
            {
                assert_clear(
                    &world,
                    &format!("{label} route {} point {index}", route.id),
                    point,
                );
            }

            // The 64x80 core is open through the complete 16-unit wall depth;
            // the five tangent samples include both sides one unit inside it.
            for (room, direction) in [(source, portal.wall.as_str()), (target, target_direction)] {
                let floor_top = room.floor_z + CONSTRUCTION_QUANTUM;
                for tangent_offset in [-31, -16, 0, 16, 31] {
                    for depth in [1, 8, 15] {
                        for height in [1, 40, 79] {
                            assert_clear(
                                &world,
                                &format!("{label} portal {} core", portal.id),
                                wall_witness(
                                    room,
                                    direction,
                                    tangent + tangent_offset,
                                    depth,
                                    floor_top + height,
                                ),
                            );
                        }
                    }
                }
                // Moderate/Rich (and the shared structural lowerer) preserve
                // a genuinely open pointed apex above the rectangular core.
                assert_clear(
                    &world,
                    &format!("{label} portal {} apex", portal.id),
                    wall_witness(room, direction, tangent, 8, floor_top + HEADROOM + 8),
                );
                assert_solid(
                    &world,
                    &format!("{label} portal {} arch shoulder", portal.id),
                    wall_witness(room, direction, tangent - 24, 8, floor_top + HEADROOM + 8),
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
        let lower = rooms[&transition.lower_room];
        let upper = rooms[&transition.upper_room];
        assert_eq!(transition.tread_boxes.len(), 12);
        assert_eq!(transition.tread_run.3 - transition.tread_run.1, 192);
        let x = (transition.tread_run.0 + transition.tread_run.2) / 2;
        for (step, tread) in transition.tread_boxes.iter().enumerate() {
            let y = (tread.1 + tread.4) / 2;
            assert_solid(
                &world,
                &format!("{label} stair tread {step}"),
                (x, y, tread.5 - 8),
            );
            assert_clear(
                &world,
                &format!("{label} stair headroom {step}"),
                (x, y, tread.5 + 79),
            );
        }
        let upper_y = (transition.upper_approach.1 + transition.upper_approach.3) / 2;
        assert_solid(
            &world,
            &format!("{label} upper approach support"),
            (x, upper_y, upper.floor_z + 8),
        );
        assert_clear(
            &world,
            &format!("{label} upper crest headroom"),
            (x, upper_y, upper.floor_z + CONSTRUCTION_QUANTUM + 79),
        );
        assert_clear(
            &world,
            &format!("{label} lower transition landing"),
            (
                x,
                (transition.lower_landing.1 + transition.lower_landing.3) / 2,
                lower.floor_z + CONSTRUCTION_QUANTUM + HULL_EYE_OFFSET,
            ),
        );

        let spawn_room = topology
            .rooms
            .iter()
            .find(|room| {
                let (x, y, z) = output.metadata.spawn_origin();
                x > room.shell.0
                    && x < room.shell.2
                    && y > room.shell.1
                    && y < room.shell.3
                    && z > room.floor_z
                    && z < room.floor_z + room.dims.2 as i32
            })
            .expect("spawn belongs to a committed room");
        let mut visited = BTreeSet::new();
        let mut pending = vec![spawn_room.id];
        while let Some(room) = pending.pop() {
            if visited.insert(room) {
                pending.extend(adjacency.get(&room).into_iter().flatten().copied());
            }
        }
        assert_eq!(
            visited.len(),
            topology.rooms.len(),
            "{label}: all rooms must connect from spawn"
        );
    }
}
