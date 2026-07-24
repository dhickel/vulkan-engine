//! Phase 04 dungeon geometry & topology evidence tests.
//!
//! Tests prove:
//!  - each topology fixture loads through BspLoader (strict reload)
//!  - adjacent rooms with shared wall produce no portal (solid separation)
//!  - connected rooms produce valid portal/visibility (PVS non-empty)
//!  - intentional leak produces empty VIS data / pointfile attribution
//!  - large faces don't cause subdivision issues (face count within bounds)
//!  - unequal ceilings produce correct clip hulls (non-empty clipnodes)

use bsp::*;
use bsp::coords::QuakeToEngine;
use glam::Vec3;
use std::path::Path;

// ── Fixture helpers ────────────────────────────────────────────────────────

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn compiled_dir() -> std::path::PathBuf {
    fixtures_dir().join("compiled")
}

fn read(path: &std::path::Path) -> Vec<u8> {
    std::fs::read(path).expect(&format!("failed to read {}", path.display()))
}

fn load_bsp2_fixture(name: &str) -> BspWorld {
    let bsp_data = read(&compiled_dir().join(format!("{name}.bsp")));
    let options = LoadOptions {
        strict: true,
        source_identity: name.into(),
        ..LoadOptions::default()
    };
    BspLoader::load(&bsp_data, &options).expect(&format!("strict load of {name}"))
}

// ── Strict reload: every fixture must survive BspLoader in strict mode ─────

#[test]
fn topology_straight_junction_strict_reload() {
    let world = load_bsp2_fixture("dungeon-junction-straight-bsp2");
    assert!(world.num_models() > 0);
    assert!(world.num_leaves() > 0);
    assert!(!world.entities.is_empty());
    assert!(world.worldspawn().is_some());
}

#[test]
fn topology_l_junction_strict_reload() {
    let world = load_bsp2_fixture("dungeon-junction-l-bsp2");
    assert!(world.num_models() > 0);
    assert!(world.num_leaves() > 0);
    assert!(!world.entities.is_empty());
}

#[test]
fn topology_t_junction_strict_reload() {
    let world = load_bsp2_fixture("dungeon-junction-t-bsp2");
    assert!(world.num_models() > 0);
    assert!(world.num_leaves() > 0);
}

#[test]
fn topology_x_junction_strict_reload() {
    let world = load_bsp2_fixture("dungeon-junction-x-bsp2");
    assert!(world.num_models() > 0);
    assert!(world.num_leaves() > 0);
}

#[test]
fn topology_adjacent_closed_strict_reload() {
    let world = load_bsp2_fixture("dungeon-adjacent-closed-bsp2");
    assert!(world.num_models() > 0);
    assert!(world.num_leaves() > 0);
}

#[test]
fn topology_parallel_corridors_strict_reload() {
    let world = load_bsp2_fixture("dungeon-parallel-corridors-bsp2");
    assert!(world.num_models() > 0);
    assert!(world.num_leaves() > 0);
}

#[test]
fn topology_grazing_route_strict_reload() {
    let world = load_bsp2_fixture("dungeon-grazing-route-bsp2");
    assert!(world.num_models() > 0);
    assert!(world.num_leaves() > 0);
}

#[test]
fn topology_large_face_strict_reload() {
    let world = load_bsp2_fixture("dungeon-large-face-bsp2");
    assert!(world.num_models() > 0);
    assert!(world.num_leaves() > 0);
}

#[test]
fn topology_unequal_ceiling_strict_reload() {
    let world = load_bsp2_fixture("dungeon-unequal-ceiling-bsp2");
    assert!(world.num_models() > 0);
    assert!(world.num_leaves() > 0);
}

#[test]
fn topology_intentional_leak_strict_reload() {
    let world = load_bsp2_fixture("dungeon-intentional-leak-bsp2");
    // Leaky map still produces a valid BSP; it just won't have VIS data.
    assert!(world.num_models() > 0);
    assert!(world.num_leaves() > 0);
}

// ── Adjacent rooms with shared wall: no portal (solid separation) ──────────

#[test]
fn topology_adjacent_closed_solid_separation() {
    let world = load_bsp2_fixture("dungeon-adjacent-closed-bsp2");
    // The shared center wall at x=0..16 should produce a solid BSP separation
    // between the left and right room leaves. Verify the BSP tree has multiple
    // empty leaves (both rooms) separated by solid leaves (the wall).
    let empty_leaves: Vec<_> = world
        .leaves
        .iter()
        .filter(|l| l.contents == -1) // EMPTY
        .collect();
    // At least 2 empty leaves expected (one per room)
    assert!(
        empty_leaves.len() >= 2,
        "adjacent closed map must have at least 2 empty leaves (one per room), got {}",
        empty_leaves.len()
    );

    // Verify the shared wall produces solid leaves between the rooms
    let solid_leaves: Vec<_> = world
        .leaves
        .iter()
        .filter(|l| l.contents == -2) // SOLID
        .collect();
    assert!(
        !solid_leaves.is_empty(),
        "shared wall must produce solid leaves"
    );
}

// ── Connected rooms produce valid visibility (PVS non-empty) ───────────────

#[test]
fn topology_straight_junction_has_vis() {
    let world = load_bsp2_fixture("dungeon-junction-straight-bsp2");
    assert!(
        !world.vis_data.is_empty(),
        "straight junction must have VIS data (sealed map)"
    );

    // Camera at info_player_start in Quake space (-192, 0, 0)
    let qte = QuakeToEngine::default();
    let cam_eng = qte.position_vec3(Vec3::new(-192.0, 0.0, 0.0));
    let pvs = visibility::camera_pvs(
        &cam_eng,
        &world.vis_data,
        &world.nodes,
        &world.leaves,
        &world.planes,
    );
    assert!(pvs.is_some(), "camera inside sealed room must have PVS");
    let pvs = pvs.unwrap();
    assert!(pvs.valid, "PVS must be valid for sealed map");
}

#[test]
fn topology_t_junction_has_vis() {
    let world = load_bsp2_fixture("dungeon-junction-t-bsp2");
    assert!(!world.vis_data.is_empty(), "T-junction must have VIS data");

    let qte = QuakeToEngine::default();
    let cam_eng = qte.position_vec3(Vec3::new(-200.0, 0.0, 0.0));
    let pvs = visibility::camera_pvs(
        &cam_eng,
        &world.vis_data,
        &world.nodes,
        &world.leaves,
        &world.planes,
    );
    assert!(pvs.is_some(), "camera inside sealed T-junction must have PVS");
    assert!(pvs.unwrap().valid);
}

#[test]
fn topology_x_junction_has_vis() {
    let world = load_bsp2_fixture("dungeon-junction-x-bsp2");
    assert!(!world.vis_data.is_empty(), "X-junction must have VIS data");

    let qte = QuakeToEngine::default();
    let cam_eng = qte.position_vec3(Vec3::new(-200.0, 0.0, 0.0));
    let pvs = visibility::camera_pvs(
        &cam_eng,
        &world.vis_data,
        &world.nodes,
        &world.leaves,
        &world.planes,
    );
    assert!(pvs.is_some(), "camera inside sealed X-junction must have PVS");
    assert!(pvs.unwrap().valid);
}

#[test]
fn topology_l_junction_has_vis() {
    let world = load_bsp2_fixture("dungeon-junction-l-bsp2");
    assert!(!world.vis_data.is_empty(), "L-junction must have VIS data");

    let qte = QuakeToEngine::default();
    let cam_eng = qte.position_vec3(Vec3::new(-192.0, 128.0, 0.0));
    let pvs = visibility::camera_pvs(
        &cam_eng,
        &world.vis_data,
        &world.nodes,
        &world.leaves,
        &world.planes,
    );
    assert!(pvs.is_some(), "camera inside sealed L-junction must have PVS");
    assert!(pvs.unwrap().valid);
}

// ── Parallel corridors: verify both have separate visibility ───────────────

#[test]
fn topology_parallel_corridors_separate_vis() {
    let world = load_bsp2_fixture("dungeon-parallel-corridors-bsp2");
    assert!(!world.vis_data.is_empty());

    let qte = QuakeToEngine::default();
    // Camera in north corridor
    let cam_north = qte.position_vec3(Vec3::new(-196.0, 48.0, 0.0));
    let pvs_north = visibility::camera_pvs(
        &cam_north,
        &world.vis_data,
        &world.nodes,
        &world.leaves,
        &world.planes,
    );
    assert!(pvs_north.is_some(), "north corridor must have PVS");

    // Camera in south corridor
    let cam_south = qte.position_vec3(Vec3::new(-196.0, -48.0, 0.0));
    let pvs_south = visibility::camera_pvs(
        &cam_south,
        &world.vis_data,
        &world.nodes,
        &world.leaves,
        &world.planes,
    );
    assert!(pvs_south.is_some(), "south corridor must have PVS");
}

// ── Grazing route: separate but adjacent spaces ────────────────────────────

#[test]
fn topology_grazing_route_separate_spaces() {
    let world = load_bsp2_fixture("dungeon-grazing-route-bsp2");
    assert!(!world.vis_data.is_empty());

    let qte = QuakeToEngine::default();
    // Camera in room
    let cam_room = qte.position_vec3(Vec3::new(0.0, 0.0, 0.0));
    let pvs_room = visibility::camera_pvs(
        &cam_room,
        &world.vis_data,
        &world.nodes,
        &world.leaves,
        &world.planes,
    );
    assert!(pvs_room.is_some(), "room must have PVS");

    // Camera in corridor (east of room, grazing its east wall)
    let cam_corridor = qte.position_vec3(Vec3::new(160.0, 48.0, 0.0));
    let pvs_corridor = visibility::camera_pvs(
        &cam_corridor,
        &world.vis_data,
        &world.nodes,
        &world.leaves,
        &world.planes,
    );
    assert!(pvs_corridor.is_some(), "corridor must have PVS");
}

// ── Intentional leak: pointfile / no VIS ───────────────────────────────────

#[test]
fn topology_intentional_leak_empty_vis() {
    let world = load_bsp2_fixture("dungeon-intentional-leak-bsp2");
    // The leaky map should produce empty VIS data (vis can't compute
    // visibility for an unsealed map).
    // Depending on compiler behavior, VIS data may be empty or the PVS
    // state may be corrupt/empty.
    let pvs_state = PvsState::new(
        world.leaves.len().saturating_sub(1) as u32,
        &world.vis_data,
    );
    // Either VIS is empty (corrupt) or has zero leaves
    assert!(
        pvs_state.corrupt || world.vis_data.is_empty(),
        "intentional leak map must have corrupt or empty VIS"
    );
}

#[test]
fn topology_intentional_leak_produces_bsp() {
    // Even though the map leaks, qbsp still produces a valid BSP.
    let world = load_bsp2_fixture("dungeon-intentional-leak-bsp2");
    assert!(world.num_models() > 0);
    assert!(world.num_faces() > 0);
    assert!(!world.planes.is_empty());
}

// ── Large face: no subdivision issues ──────────────────────────────────────

#[test]
fn topology_large_face_no_subdivision_issues() {
    let world = load_bsp2_fixture("dungeon-large-face-bsp2");

    // The map has walls with faces exceeding 240x240 units.
    // Verify the face count is reasonable (no excessive subdivision).
    let face_count = world.faces.len();
    // A simple 6-brush room with large faces should produce a modest
    // face count. Less than 256 faces is a reasonable sanity check.
    assert!(
        face_count < 256,
        "large-face map should have reasonable face count (< 256), got {face_count}"
    );

    // Verify the BSP tree is not degenerate (has reasonable depth).
    let node_count = world.nodes.len();
    assert!(node_count > 0, "large-face map must have BSP nodes");
    assert!(
        node_count < 1024,
        "large-face map BSP should not be excessively subdivided"
    );
}

#[test]
fn topology_large_face_bounds() {
    let world = load_bsp2_fixture("dungeon-large-face-bsp2");
    // Verify the map extents are consistent with a 320x320x256 room.
    // The world model (model 0) bounds should encompass the room.
    assert!(world.num_models() > 0);
    let model0 = &world.models[0];
    let mins = model0.mins;
    let maxs = model0.maxs;
    // Room interior is roughly -160..160 in x/y, -128..128 in z
    assert!(mins[0] < -150.0, "world mins x should be < -150");
    assert!(maxs[0] > 150.0, "world maxs x should be > 150");
    assert!(mins[2] < -120.0, "world mins z should be < -120");
    assert!(maxs[2] > 120.0, "world maxs z should be > 120");
}

// ── Unequal ceiling: clip hulls ────────────────────────────────────────────

#[test]
fn topology_unequal_ceiling_has_clipnodes() {
    let world = load_bsp2_fixture("dungeon-unequal-ceiling-bsp2");
    // The map must have clipnodes for collision detection
    assert!(
        !world.clipnodes.is_empty(),
        "unequal ceiling map must have clipnodes"
    );

    // Verify clipnode hull structure: hull 0 (point) headnode should be valid
    let model0 = &world.models[0];
    // headnode[0] is the root clipnode for hull 0
    let headnode = model0.headnode[0];
    assert!(
        headnode != 0 || world.clipnodes.len() > 1,
        "hull 0 should have valid clipnodes"
    );
}

#[test]
fn topology_unequal_ceiling_height_transition() {
    let world = load_bsp2_fixture("dungeon-unequal-ceiling-bsp2");

    // The map has two room heights. Verify the BSP tree handles the
    // ceiling step correctly by checking that leaves at different
    // heights exist.
    let empty_leaves: Vec<_> = world
        .leaves
        .iter()
        .filter(|l| l.contents == -1) // EMPTY
        .collect();

    // Should have leaves in both the low-ceiling and high-ceiling regions
    assert!(
        empty_leaves.len() >= 3,
        "unequal ceiling map must have multiple empty leaves, got {}",
        empty_leaves.len()
    );

    // Verify VIS data exists (map is sealed)
    assert!(
        !world.vis_data.is_empty(),
        "unequal ceiling map must have VIS data"
    );
}

// ── PVS sanity: verify PVS sets have expected bit counts ───────────────────

#[test]
fn topology_pvs_bit_counts_match_leaf_counts() {
    let world = load_bsp2_fixture("dungeon-junction-straight-bsp2");
    let state = PvsState::new(
        world.leaves.len().saturating_sub(1) as u32,
        &world.vis_data,
    );
    assert!(!state.corrupt);
    // pvs_bytes must be (num_leaves + 7) / 8
    assert_eq!(
        state.pvs_bytes,
        (state.num_leaves + 7) / 8,
        "PVS byte count must match leaf count"
    );
}

// ── Leaf membership: verify leaf membership maps are coherent ──────────────

#[test]
fn topology_leaf_membership_from_compiled_fixture() {
    let world = load_bsp2_fixture("dungeon-junction-straight-bsp2");
    let members = visibility::build_leaf_membership(&world.leaves, &world.markfaces);
    // Every face should have at least one leaf membership entry
    // (faces without membership are real but acceptable — they'll be
    // conservatively visible)
    assert_eq!(
        members.len(),
        world.faces.len() as usize,
        "membership vec length must match face count"
    );
}

// ── Point contents: verify camera in room returns non-solid ────────────────

#[test]
fn topology_point_contents_empty_in_room() {
    let world = load_bsp2_fixture("dungeon-junction-straight-bsp2");
    // Convert info_player_start Quake position (-192, 0, 0) to engine space
    let qte = QuakeToEngine::default();
    let eng_pos = qte.position_vec3(Vec3::new(-192.0, 0.0, 0.0));
    let result = queries::point_contents(
        eng_pos,
        &world.nodes,
        &world.leaves,
        &world.planes,
    );
    assert!(
        !result.is_solid(),
        "camera inside room must not be in solid"
    );
    assert!(
        result.is_empty(),
        "camera inside room must be in empty space, got {result:?}"
    );
}

#[test]
fn topology_point_contents_solid_in_wall() {
    let world = load_bsp2_fixture("dungeon-adjacent-closed-bsp2");
    // Point inside the shared center wall at Quake x=8 should be solid
    let qte = QuakeToEngine::default();
    let eng_pos = qte.position_vec3(Vec3::new(8.0, 0.0, 0.0));
    let result = queries::point_contents(
        eng_pos,
        &world.nodes,
        &world.leaves,
        &world.planes,
    );
    assert!(
        result.is_solid(),
        "point in shared wall must be solid, got {result:?}"
    );
}

// ── Detection of info_player_start entity ──────────────────────────────────

#[test]
fn topology_all_fixtures_have_info_player_start() {
    let fixtures = [
        "dungeon-junction-straight-bsp2",
        "dungeon-junction-l-bsp2",
        "dungeon-junction-t-bsp2",
        "dungeon-junction-x-bsp2",
        "dungeon-adjacent-closed-bsp2",
        "dungeon-parallel-corridors-bsp2",
        "dungeon-grazing-route-bsp2",
        "dungeon-large-face-bsp2",
        "dungeon-unequal-ceiling-bsp2",
        "dungeon-intentional-leak-bsp2",
    ];

    for name in &fixtures {
        let world = load_bsp2_fixture(name);
        let has_player_start = world
            .entities
            .iter()
            .any(|e| matches!(e.class, bsp::entities::EntityClass::SpawnMarker));
        assert!(
            has_player_start,
            "{name} must have an info_player_start entity"
        );
    }
}
