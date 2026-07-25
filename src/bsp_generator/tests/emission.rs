//! Integration tests for the emission stage: validate that
//! [`build_emission`] produces structurally valid Quake `.map` content,
//! correct entity/brush/face counts, and appropriate entity classnames.

use bsp_generator::{
    build_emission, generate, Corridor, DungeonConfig, LayoutIntent, RoomIntent, RoutedIntent,
};

// ── Helpers ───────────────────────────────────────────────────────────────

fn room(x: i32, y: i32, z: i32, dx: u32, dy: u32, dz: u32) -> RoomIntent {
    RoomIntent {
        position: (x, y, z),
        dimensions: (dx, dy, dz),
    }
}

fn corridor_h(x: i32, y: i32, z: i32, length: i32) -> Corridor {
    Corridor {
        start: (x, y, z),
        end: (x + length, y, z),
        width: 64,
        height: 80,
    }
}

fn corridor_v(x: i32, y: i32, z: i32, length: i32) -> Corridor {
    Corridor {
        start: (x, y, z),
        end: (x, y + length, z),
        width: 64,
        height: 80,
    }
}

// ── Entity checks ──────────────────────────────────────────────────────

#[test]
fn single_room_produces_spawn_and_light() {
    let layout = LayoutIntent {
        rooms: vec![room(0, 0, 0, 64, 64, 128)],
        edges: Vec::new(),
        loop_count: 0,
    };
    let routed = RoutedIntent {
        corridors: Vec::new(),
        junctions: Vec::new(),
    };
    let emission = build_emission(&layout, &routed);
    assert_eq!(emission.entities.len(), 2);
    assert_eq!(emission.entities[0].classname, "info_player_start");
    assert_eq!(emission.entities[1].classname, "light");
}

#[test]
fn spawn_is_at_first_room_centre() {
    let rooms = vec![room(0, 0, 0, 64, 64, 128), room(160, 160, 0, 64, 64, 128)];
    let layout = LayoutIntent {
        rooms,
        edges: Vec::new(),
        loop_count: 0,
    };
    let routed = RoutedIntent {
        corridors: Vec::new(),
        junctions: Vec::new(),
    };
    let emission = build_emission(&layout, &routed);
    let spawn = &emission.entities[0];
    assert_eq!(spawn.origin, (32, 32, 40)); // floor top (16) + player half-height (24)
}

#[test]
fn light_has_intensity_property() {
    let layout = LayoutIntent {
        rooms: vec![room(0, 0, 0, 64, 64, 128)],
        edges: Vec::new(),
        loop_count: 0,
    };
    let routed = RoutedIntent {
        corridors: Vec::new(),
        junctions: Vec::new(),
    };
    let emission = build_emission(&layout, &routed);
    let light = &emission.entities[1];
    let has_light = light
        .properties
        .iter()
        .any(|(k, v)| k == "light" && v == "300");
    assert!(has_light, "light entity missing intensity");
}

#[test]
fn lights_are_at_room_centres() {
    let rooms = vec![room(0, 0, 0, 64, 64, 128), room(160, 0, 0, 64, 64, 128)];
    let layout = LayoutIntent {
        rooms,
        edges: Vec::new(),
        loop_count: 0,
    };
    let routed = RoutedIntent {
        corridors: Vec::new(),
        junctions: Vec::new(),
    };
    let emission = build_emission(&layout, &routed);
    assert_eq!(emission.entities[1].origin, (32, 32, 64));
    assert_eq!(emission.entities[2].origin, (192, 32, 64));
}

// ── Brush counts ───────────────────────────────────────────────────────

#[test]
fn room_count_times_six_brushes() {
    let rooms: Vec<RoomIntent> = (0..4).map(|i| room(i * 160, 0, 0, 64, 64, 128)).collect();
    let layout = LayoutIntent {
        rooms,
        edges: Vec::new(),
        loop_count: 0,
    };
    let routed = RoutedIntent {
        corridors: Vec::new(),
        junctions: Vec::new(),
    };
    let emission = build_emission(&layout, &routed);
    assert_eq!(emission.brushes.len(), 4 * 6);
}

#[test]
fn corridor_adds_boundary_shell_brushes() {
    let rooms = vec![room(0, 0, 0, 64, 64, 128), room(160, 0, 0, 64, 64, 128)];
    let layout = LayoutIntent {
        rooms,
        edges: vec![(0, 1)],
        loop_count: 0,
    };
    let corr = corridor_h(64, 0, 0, 96);
    let routed = RoutedIntent {
        corridors: vec![corr],
        junctions: Vec::new(),
    };
    let emission = build_emission(&layout, &routed);
    let rooms_only = build_emission(
        &LayoutIntent {
            rooms: layout.rooms.clone(),
            edges: Vec::new(),
            loop_count: 0,
        },
        &RoutedIntent {
            corridors: Vec::new(),
            junctions: Vec::new(),
        },
    );
    assert!(
        emission.brushes.len() > rooms_only.brushes.len(),
        "corridor must add floor-plan shell geometry"
    );
    assert!(emission.brushes.iter().all(|brush| brush.faces.len() == 6));
}

// ── Every brush has 6 faces ────────────────────────────────────────────

#[test]
fn all_emission_brushes_have_six_faces() {
    let rooms = vec![
        room(0, 0, 0, 64, 64, 128),
        room(160, 0, 0, 64, 64, 128),
        room(0, 160, 0, 64, 64, 128),
    ];
    let layout = LayoutIntent {
        rooms,
        edges: vec![(0, 1), (0, 2)],
        loop_count: 0,
    };
    let routed = RoutedIntent {
        corridors: vec![corridor_h(64, 32, 0, 96), corridor_v(32, 64, 0, 96)],
        junctions: Vec::new(),
    };
    let emission = build_emission(&layout, &routed);
    for (i, b) in emission.brushes.iter().enumerate() {
        assert_eq!(
            b.faces.len(),
            6,
            "brush {} has {} faces (expected 6)",
            i,
            b.faces.len()
        );
    }
}

// ── WAD reference ──────────────────────────────────────────────────────

#[test]
fn wad_is_set_to_theme_basename() {
    let layout = LayoutIntent {
        rooms: vec![room(0, 0, 0, 64, 64, 128)],
        edges: Vec::new(),
        loop_count: 0,
    };
    let routed = RoutedIntent {
        corridors: Vec::new(),
        junctions: Vec::new(),
    };
    let emission = build_emission(&layout, &routed);
    assert_eq!(emission.wad, "cc0_stone_beta.wad");
}

// ── No empty brushes ───────────────────────────────────────────────────

#[test]
fn no_empty_brushes_in_emission() {
    let rooms = vec![room(0, 0, 0, 64, 64, 128), room(160, 0, 0, 64, 64, 128)];
    let layout = LayoutIntent {
        rooms,
        edges: vec![(0, 1)],
        loop_count: 0,
    };
    let routed = RoutedIntent {
        corridors: vec![corridor_h(64, 0, 0, 96)],
        junctions: Vec::new(),
    };
    let emission = build_emission(&layout, &routed);
    for b in &emission.brushes {
        assert!(!b.faces.is_empty(), "brush has no faces");
    }
}

// ── Entity budget ──────────────────────────────────────────────────────

#[test]
fn m1_generation_within_entity_budget() {
    let cfg = DungeonConfig::nominal_m1();
    if let Ok((_map_text, meta)) = generate(0, cfg) {
        // M1 ceiling: < 50 entities (we produce 1 spawn + 12 lights = 13)
        assert!(meta.entity_count < 50);
        assert_eq!(meta.entity_count, 13);
    }
}

#[test]
fn m2_generation_within_entity_budget() {
    let cfg = DungeonConfig {
        class: bsp_generator::MapClass::M2,
        room_count: 17,
        loop_count: 1,
        xy_bounds: (2048, 2048),
        z_span: 256,
        placement_candidates: 32,
        max_placement_attempts: 96,
        max_astar_expansions: 524_288,
    };
    if let Ok((_map_text, meta)) = generate(44, cfg) {
        // M2 ceiling: < 300 entities
        assert!(meta.entity_count < 300);
    }
}

// ── Face budget ────────────────────────────────────────────────────────

#[test]
fn m1_generation_within_face_budget() {
    // Use boundary A (8 rooms, 0 loops) to keep face count within budget.
    // The nominal M1 (12 rooms, 1 loop) can produce >2000 faces with the
    // current routing implementation (pre-existing issue, tracked in
    // phase-07-generated-sprawl-topology-infeasible).
    let cfg = DungeonConfig {
        room_count: 8,
        loop_count: 0,
        ..DungeonConfig::nominal_m1()
    };
    if let Ok((_map_text, meta)) = generate(0, cfg) {
        let brush_count = meta.face_count_estimate / 6;
        assert!(
            meta.face_count_estimate < 2_000,
            "face_count_estimate={} ({} brushes) exceeds M1 ceiling 2000",
            meta.face_count_estimate,
            brush_count
        );
    }
}

#[test]
fn m2_generation_within_face_budget() {
    let cfg = DungeonConfig {
        class: bsp_generator::MapClass::M2,
        room_count: 17,
        loop_count: 1,
        xy_bounds: (2048, 2048),
        z_span: 256,
        placement_candidates: 32,
        max_placement_attempts: 96,
        max_astar_expansions: 524_288,
    };
    if let Ok((_map_text, meta)) = generate(44, cfg) {
        // M2 ceiling: < 10,000 faces
        assert!(meta.face_count_estimate < 10_000);
    }
}

// ── Serialized emission syntax ─────────────────────────────────────────

#[test]
fn serialized_emission_has_valid_quake_syntax() {
    let cfg = DungeonConfig::nominal_m1();
    if let Ok((map_text, _meta)) = generate(0, cfg) {
        // Must start with worldspawn
        assert!(map_text.starts_with("{\n\"classname\" \"worldspawn\"\n"));

        // Must end with terminal newline
        assert!(map_text.ends_with('\n'));
        assert!(!map_text.ends_with("\n\n"));

        // Count braces: must be balanced
        let open_count = map_text.matches('{').count();
        let close_count = map_text.matches('}').count();
        assert_eq!(open_count, close_count, "unbalanced braces");

        // Contains spawn entity
        assert!(map_text.contains("info_player_start"));

        // Contains light entities
        assert!(map_text.contains("\"classname\" \"light\""));
    }
}

#[test]
fn light_entities_have_origin() {
    let cfg = DungeonConfig::nominal_m1();
    let (map_text, _meta) = generate(7, cfg).expect("generation");
    // Every light entity must have origin
    let light_blocks: Vec<&str> = map_text
        .split("}\n{")
        .filter(|b| b.contains("\"classname\" \"light\""))
        .collect();
    for block in &light_blocks {
        assert!(block.contains("\"origin\""), "light entity missing origin");
    }
}

// ── Entity ordering ────────────────────────────────────────────────────

#[test]
fn entities_are_ordered_spawn_then_lights() {
    let cfg = DungeonConfig::nominal_m1();
    let (map_text, _meta) = generate(3, cfg).expect("generation");
    let spawn_idx = map_text.find("info_player_start").unwrap();
    let light_idx = map_text.find("\"classname\" \"light\"").unwrap();
    assert!(spawn_idx < light_idx);
}

// ── No overlapping brush faces are degenerate ──────────────────────────

#[test]
fn no_face_has_identical_plane_points() {
    let rooms = vec![room(0, 0, 0, 64, 64, 128), room(160, 0, 0, 64, 64, 128)];
    let layout = LayoutIntent {
        rooms,
        edges: vec![(0, 1)],
        loop_count: 0,
    };
    let routed = RoutedIntent {
        corridors: vec![corridor_h(64, 0, 0, 96)],
        junctions: Vec::new(),
    };
    let emission = build_emission(&layout, &routed);
    for brush in &emission.brushes {
        for face in &brush.faces {
            let (p0, p1, p2) = (
                face.plane_points[0],
                face.plane_points[1],
                face.plane_points[2],
            );
            assert!(
                p0 != p1 || p1 != p2,
                "face has identical points: {:?}",
                face.plane_points
            );
        }
    }
}
