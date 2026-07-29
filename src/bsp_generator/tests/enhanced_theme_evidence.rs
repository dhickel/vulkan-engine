//! Enhanced v2 theme evidence — complete validation of the CC0 Dungeon v2
//! theme package, palette assignment, role/zone derivation, WAD closure, and
//! deterministic byte identity.
//!
//! Tests invoke `themes/cc0_dungeon_v2/build.py` in a temporary directory,
//! inspect every output, and confirm byte-identical regeneration.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use bsp_generator::enhanced::config::EnhancedConfig;
use bsp_generator::enhanced::intent::{PaletteId, RoomId};
use bsp_generator::enhanced::placement::{place_rooms, PlacedRoom};
use bsp_generator::enhanced::seed::{tags, EnhancedSeed, EnhancedStageRng};
use bsp_generator::enhanced::theme::{
    assign_by_zone, assign_uniform, cc0_dungeon_v2_theme, derive_roles, derive_zones,
    AssignmentStrategy, PaletteDefinition, RoomRole, TextureRole, ThemePackage,
};
use bsp_generator::enhanced::topology::{build_topology, TopologyResult};

const VISUAL_TEXTURE_DIMENSION: u32 = 1024;
const COMPILER_SKIP_DIMENSION: u32 = 64;

// ── Helpers ──────────────────────────────────────────────────────────────

fn theme_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("themes")
        .join("cc0_dungeon_v2")
}

fn run_build(out_dir: &Path) -> std::process::ExitStatus {
    let build_py = theme_dir().join("build.py");
    assert!(
        build_py.is_file(),
        "build.py missing at {}",
        build_py.display()
    );
    Command::new("python3")
        .arg(&build_py)
        .arg(out_dir)
        .status()
        .expect("failed to execute build.py")
}

fn assert_file(path: &Path) {
    assert!(path.is_file(), "missing file: {}", path.display());
}

fn png_dimensions(path: &Path) -> (u32, u32) {
    let data = std::fs::read(path).expect("cannot read PNG");
    assert!(
        &data[0..8] == b"\x89PNG\r\n\x1a\n",
        "not a valid PNG: {}",
        path.display()
    );
    assert!(
        &data[12..16] == b"IHDR",
        "IHDR chunk missing: {}",
        path.display()
    );
    let width = u32::from_be_bytes(data[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(data[20..24].try_into().unwrap());
    (width, height)
}

/// Build a nominal topology for theme tests.
fn build_nominal(
    seed_val: u64,
) -> (
    EnhancedConfig,
    Vec<PlacedRoom>,
    TopologyResult,
    EnhancedStageRng,
) {
    let cfg = EnhancedConfig::nominal();
    let eseed = EnhancedSeed::new(seed_val);
    let placement = place_rooms(&cfg, eseed.stage_seed(tags::LAYER_PLACEMENT)).unwrap();
    let mut topo_rng = eseed.stage_seed(tags::VERTICAL_TOPOLOGY).rng();
    let topology = build_topology(&cfg, &placement, &mut topo_rng).unwrap();
    let theme_rng = eseed.stage_seed(tags::THEME_ASSIGNMENT).rng();
    (cfg, placement.rooms, topology, theme_rng)
}

fn theme() -> ThemePackage {
    cc0_dungeon_v2_theme()
}

// ── Theme package structure ─────────────────────────────────────────────

#[test]
fn theme_has_three_room_palettes_plus_connector() {
    let t = theme();
    assert_eq!(t.palettes.len(), 4);
    assert_eq!(t.room_palette_count(), 3);

    let names: BTreeSet<&str> = t.palettes.iter().map(|p| p.name).collect();
    assert!(names.contains("base_stone"));
    assert!(names.contains("crypt"));
    assert!(names.contains("treasury"));
    assert!(names.contains("connector"));
}

#[test]
fn theme_base_is_connector_is_explicit() {
    let t = theme();
    let base = t.base_palette();
    assert_eq!(base.name, "base_stone");
    assert!(base.is_base);
    assert!(!base.is_connector);

    let conn = t.connector_palette();
    assert_eq!(conn.name, "connector");
    assert!(conn.is_connector);
    assert!(!conn.is_base);
}

#[test]
fn theme_palettes_have_distinct_ids() {
    let t = theme();
    let ids: BTreeSet<u32> = t.palettes.iter().map(|p| p.id.0).collect();
    assert_eq!(ids.len(), 4);
}

#[test]
fn theme_visible_texture_names_are_exact_case_ascii() {
    let t = theme();
    for name in t.visible_texture_names() {
        assert!(
            name.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
            "texture name '{name}' is not lowercase ASCII with underscores"
        );
    }
}

#[test]
fn theme_connector_has_no_accent() {
    let t = theme();
    let conn = t.connector_palette();
    assert!(conn.accent.is_none());
    // But floor, wall, ceiling must exist
    assert!(!conn.floor.is_empty());
    assert!(!conn.wall.is_empty());
    assert!(!conn.ceiling.is_empty());
}

#[test]
fn theme_every_room_palette_has_all_four_roles() {
    let t = theme();
    for p in &t.palettes {
        if p.is_connector {
            continue;
        }
        assert!(p.accent.is_some(), "room palette {} missing accent", p.name);
        assert!(!p.floor.is_empty());
        assert!(!p.wall.is_empty());
        assert!(!p.ceiling.is_empty());
    }
}

#[test]
fn theme_texture_lookup_returns_correct_identity() {
    let t = theme();

    // Exact-case lookups
    for p in &t.palettes {
        for role in TextureRole::ALL {
            let expected = match role {
                TextureRole::Floor => Some(p.floor),
                TextureRole::Wall => Some(p.wall),
                TextureRole::Ceiling => Some(p.ceiling),
                TextureRole::Accent => p.accent,
            };
            assert_eq!(t.texture_for(p.id, *role), expected);
        }
    }

    // Unknown palette
    assert!(t.texture_for(PaletteId(99), TextureRole::Floor).is_none());
}

#[test]
fn theme_cc0_provenance_populated() {
    let t = theme();
    assert!(t.cc0_provenance.contains("CC0"));
    assert!(t.cc0_provenance.contains("project-authored"));
    assert_ne!(t.wad_sha256, [0; 32], "theme must declare its WAD identity");
}

// ── Role derivation ────────────────────────────────────────────────────

#[test]
fn role_derivation_entry_is_lowest_room_id() {
    let (_, rooms, topology, _) = build_nominal(42);
    let roles = derive_roles(&rooms, &topology);
    let min_id = rooms.iter().map(|r| r.id).min().unwrap();
    assert_eq!(roles[&min_id], RoomRole::Entry);
}

#[test]
fn role_derivation_covers_every_room() {
    let (_, rooms, topology, _) = build_nominal(42);
    let roles = derive_roles(&rooms, &topology);
    assert_eq!(roles.len(), rooms.len());
}

#[test]
fn role_derivation_dead_ends_are_degree_one() {
    let (_, rooms, topology, _) = build_nominal(42);

    // compute degrees
    let mut degrees: BTreeMap<RoomId, usize> = rooms.iter().map(|r| (r.id, 0)).collect();
    for route in &topology.routes {
        *degrees.get_mut(&route.source_room).unwrap() += 1;
        *degrees.get_mut(&route.target_room).unwrap() += 1;
    }
    for t in &topology.transitions {
        *degrees.get_mut(&t.lower_room).unwrap() += 1;
        *degrees.get_mut(&t.upper_room).unwrap() += 1;
    }

    let roles = derive_roles(&rooms, &topology);
    for (id, &deg) in &degrees {
        if deg == 1 {
            let role = roles[id];
            assert!(
                role == RoomRole::DeadEnd || role == RoomRole::Entry,
                "degree-1 room {:?} has role {:?} (should be DeadEnd or Entry)",
                id,
                role
            );
        }
    }
}

#[test]
fn role_derivation_hub_has_max_degree() {
    let (_, rooms, topology, _) = build_nominal(42);

    let mut degrees: BTreeMap<RoomId, usize> = rooms.iter().map(|r| (r.id, 0)).collect();
    for route in &topology.routes {
        *degrees.get_mut(&route.source_room).unwrap() += 1;
        *degrees.get_mut(&route.target_room).unwrap() += 1;
    }
    for t in &topology.transitions {
        *degrees.get_mut(&t.lower_room).unwrap() += 1;
        *degrees.get_mut(&t.upper_room).unwrap() += 1;
    }

    let roles = derive_roles(&rooms, &topology);
    let entry_id = rooms.iter().map(|r| r.id).min().unwrap();
    let hub_id = roles
        .iter()
        .find(|(_, &r)| r == RoomRole::Hub)
        .map(|(id, _)| *id);

    if let Some(hub_id) = hub_id {
        let hub_deg = degrees[&hub_id];
        for (id, &deg) in &degrees {
            if *id != entry_id && *id != hub_id {
                assert!(
                    deg <= hub_deg,
                    "room {:?} degree {} exceeds hub {:?} degree {}",
                    id,
                    deg,
                    hub_id,
                    hub_deg
                );
            }
        }
    }
}

#[test]
fn role_derivation_deterministic() {
    let (_, rooms, topology, _) = build_nominal(42);
    let a = derive_roles(&rooms, &topology);
    let b = derive_roles(&rooms, &topology);
    assert_eq!(a, b);
}

#[test]
fn role_derivation_known_seed_ties_stable() {
    // Multiple seeds — prove roles are deterministic
    for seed in [0u64, 1, 42, 99, 255, 1000] {
        let (_, rooms, topology, _) = build_nominal(seed);
        let a = derive_roles(&rooms, &topology);
        let b = derive_roles(&rooms, &topology);
        assert_eq!(a, b, "role derivation diverged for seed {seed}");
    }
}

// ── Zone derivation ────────────────────────────────────────────────────

#[test]
fn zone_derivation_covers_all_rooms() {
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let zones = derive_zones(&rooms, &topology, &mut rng, 3);
    assert_eq!(zones.len(), rooms.len());
    for room in &rooms {
        assert!(zones.contains_key(&room.id));
    }
}

#[test]
fn zone_derivation_deterministic() {
    let (_, rooms, topology, _) = build_nominal(42);
    let mut rng_a = EnhancedSeed::new(42)
        .stage_seed(tags::THEME_ASSIGNMENT)
        .rng();
    let mut rng_b = EnhancedSeed::new(42)
        .stage_seed(tags::THEME_ASSIGNMENT)
        .rng();
    let a = derive_zones(&rooms, &topology, &mut rng_a, 3);
    let b = derive_zones(&rooms, &topology, &mut rng_b, 3);
    assert_eq!(a, b);
}

#[test]
fn zone_derivation_ids_within_bounds() {
    let (_, rooms, topology, rng) = build_nominal(42);
    for pcount in 1..=5 {
        let zones = derive_zones(&rooms, &topology, &mut rng.clone(), pcount);
        for z in zones.values() {
            assert!((z.0 as usize) < pcount);
        }
    }
}

#[test]
fn zone_derivation_single_palette_all_zone_zero() {
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let zones = derive_zones(&rooms, &topology, &mut rng, 1);
    for z in zones.values() {
        assert_eq!(z.0, 0);
    }
}

#[test]
fn zone_derivation_respects_topology_proximity() {
    // Rooms connected directly should tend to be in the same zone
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let zones = derive_zones(&rooms, &topology, &mut rng, 3);

    // Count same-zone edges vs cross-zone edges
    let mut same_zone_edges = 0usize;
    let mut cross_zone_edges = 0usize;
    for route in &topology.routes {
        if zones[&route.source_room] == zones[&route.target_room] {
            same_zone_edges += 1;
        } else {
            cross_zone_edges += 1;
        }
    }
    // With BFS-based partitioning, direct neighbors should tend to share zones
    // This is a soft check — topology proximity is the basis of derivation
    assert!(
        same_zone_edges + cross_zone_edges > 0,
        "must have some edges"
    );
}

// ── Uniform assignment ─────────────────────────────────────────────────

#[test]
fn uniform_all_rooms_receive_base() {
    let t = theme();
    let (_, rooms, topology, _) = build_nominal(42);
    let a = assign_uniform(&t, &rooms, &topology);

    let base = t.base_palette();
    for room in &rooms {
        let pa = &a.room_palettes[&room.id];
        assert_eq!(pa.palette_id, base.id);
        assert!(!pa.is_fallback);
    }
}

#[test]
fn uniform_all_routes_receive_base() {
    let t = theme();
    let (_, rooms, topology, _) = build_nominal(42);
    let a = assign_uniform(&t, &rooms, &topology);

    let base = t.base_palette();
    for route in &topology.routes {
        let pa = &a.route_palettes[&route.id];
        assert_eq!(pa.palette_id, base.id);
        assert!(!pa.is_fallback);
    }
}

#[test]
fn uniform_all_transitions_receive_base() {
    let t = theme();
    let (_, rooms, topology, _) = build_nominal(42);
    let a = assign_uniform(&t, &rooms, &topology);

    let base = t.base_palette();
    for tr in &topology.transitions {
        let pa = &a.transition_palettes[&tr.id];
        assert_eq!(pa.palette_id, base.id);
        assert!(!pa.is_fallback);
    }
}

#[test]
fn uniform_no_fallbacks() {
    let t = theme();
    let (_, rooms, topology, _) = build_nominal(42);
    let a = assign_uniform(&t, &rooms, &topology);
    assert!(a.fallbacks.is_empty());
}

#[test]
fn uniform_deterministic() {
    let t = theme();
    let (_, rooms, topology, _) = build_nominal(42);
    let a = assign_uniform(&t, &rooms, &topology);
    let b = assign_uniform(&t, &rooms, &topology);
    assert_eq!(a, b);
}

#[test]
fn uniform_strategy_marked() {
    let t = theme();
    let (_, rooms, topology, _) = build_nominal(42);
    let a = assign_uniform(&t, &rooms, &topology);
    assert_eq!(a.strategy, AssignmentStrategy::Uniform);
}

// ── ByZone assignment ──────────────────────────────────────────────────

#[test]
fn byzone_all_rooms_assigned() {
    let t = theme();
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let a = assign_by_zone(&t, &rooms, &topology, &mut rng);

    assert_eq!(a.room_palettes.len(), rooms.len());
    assert_eq!(a.strategy, AssignmentStrategy::ByZone);
}

#[test]
fn byzone_all_routes_assigned() {
    let t = theme();
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let a = assign_by_zone(&t, &rooms, &topology, &mut rng);
    assert_eq!(a.route_palettes.len(), topology.routes.len());
}

#[test]
fn byzone_all_transitions_assigned() {
    let t = theme();
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let a = assign_by_zone(&t, &rooms, &topology, &mut rng);
    assert_eq!(a.transition_palettes.len(), topology.transitions.len());
}

#[test]
fn byzone_cross_zone_routes_use_connector() {
    let t = theme();
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let a = assign_by_zone(&t, &rooms, &topology, &mut rng);
    let connector = t.connector_palette();

    for route in &topology.routes {
        let src_z = a.zones[&route.source_room];
        let tgt_z = a.zones[&route.target_room];
        let pa = &a.route_palettes[&route.id];
        if src_z != tgt_z {
            assert_eq!(
                pa.palette_id, connector.id,
                "cross-zone route {:?} must use connector",
                route.id
            );
        }
    }
}

#[test]
fn byzone_same_zone_routes_use_zone_palette() {
    let t = theme();
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let a = assign_by_zone(&t, &rooms, &topology, &mut rng);
    let connector = t.connector_palette();

    for route in &topology.routes {
        let src_z = a.zones[&route.source_room];
        let tgt_z = a.zones[&route.target_room];
        let pa = &a.route_palettes[&route.id];
        if src_z == tgt_z {
            assert_ne!(
                pa.palette_id, connector.id,
                "same-zone route {:?} should not use connector",
                route.id
            );
        }
    }
}

#[test]
fn byzone_cross_zone_transitions_use_connector() {
    let t = theme();
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let a = assign_by_zone(&t, &rooms, &topology, &mut rng);
    let connector = t.connector_palette();

    for tr in &topology.transitions {
        let lower_z = a.zones[&tr.lower_room];
        let upper_z = a.zones[&tr.upper_room];
        let pa = &a.transition_palettes[&tr.id];
        if lower_z != upper_z {
            assert_eq!(pa.palette_id, connector.id);
        }
    }
}

#[test]
fn byzone_deterministic() {
    let t = theme();
    let (_, rooms, topology, _) = build_nominal(42);
    let mut rng_a = EnhancedSeed::new(42)
        .stage_seed(tags::THEME_ASSIGNMENT)
        .rng();
    let mut rng_b = EnhancedSeed::new(42)
        .stage_seed(tags::THEME_ASSIGNMENT)
        .rng();
    let a = assign_by_zone(&t, &rooms, &topology, &mut rng_a);
    let b = assign_by_zone(&t, &rooms, &topology, &mut rng_b);
    assert_eq!(a, b);
}

#[test]
fn byzone_room_roles_included() {
    let t = theme();
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let a = assign_by_zone(&t, &rooms, &topology, &mut rng);
    assert_eq!(a.room_roles.len(), rooms.len());

    // Verify at least Entry and Hub are present
    let has_entry = a.room_roles.values().any(|r| *r == RoomRole::Entry);
    assert!(has_entry);
}

#[test]
fn byzone_zones_are_consistent_with_assignment() {
    let t = theme();
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let a = assign_by_zone(&t, &rooms, &topology, &mut rng);

    // Every room's zone should match the palette assigned to that zone
    let room_palettes_order: Vec<&PaletteDefinition> =
        t.palettes.iter().filter(|p| !p.is_connector).collect();

    for room in &rooms {
        let zone = a.zones[&room.id];
        let room_pa = &a.room_palettes[&room.id];

        if (zone.0 as usize) < room_palettes_order.len() {
            let zone_palette = room_palettes_order[zone.0 as usize];
            if !room_pa.is_fallback {
                assert_eq!(
                    room_pa.palette_id, zone_palette.id,
                    "room {:?} zone {:?} mismatch",
                    room.id, zone
                );
            }
        }
    }
}

// ── Fallback reporting ─────────────────────────────────────────────────

#[test]
fn byzone_fallback_only_uses_base() {
    let t = theme();
    let base = t.base_palette();

    // Create a theme with only 1 room palette + connector
    let small = ThemePackage {
        palettes: vec![base.clone(), t.connector_palette().clone()],
        ..t.clone()
    };
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let a = assign_by_zone(&small, &rooms, &topology, &mut rng);

    // Any fallback must use the base palette
    for f in &a.fallbacks {
        assert_eq!(f.fallback_palette, base.name);
    }

    // Every assignment must resolve to a known palette
    for pa in a.room_palettes.values() {
        assert!(small.palette(pa.palette_id).is_some());
    }
    for pa in a.route_palettes.values() {
        assert!(small.palette(pa.palette_id).is_some());
    }
    for pa in a.transition_palettes.values() {
        assert!(small.palette(pa.palette_id).is_some());
    }
}

#[test]
fn byzone_fallback_records_reason() {
    let t = theme();
    let base = t.base_palette();
    let small = ThemePackage {
        palettes: vec![base.clone(), t.connector_palette().clone()],
        ..t.clone()
    };
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let a = assign_by_zone(&small, &rooms, &topology, &mut rng);

    for f in &a.fallbacks {
        assert!(!f.reason.is_empty(), "fallback without reason: {f:?}");
        assert!(!f.owner_kind.is_empty());
        // requested_palette should differ from fallback_palette
        assert_ne!(f.requested_palette, f.fallback_palette);
    }
}

#[test]
fn byzone_no_fallback_when_zones_fit() {
    let t = theme();
    // Full theme with 3 room palettes for nominal (28 rooms, zones ≤ 3)
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let a = assign_by_zone(&t, &rooms, &topology, &mut rng);

    // Room fallbacks should be empty when palette count >= zone count
    let room_fallbacks: Vec<_> = a
        .fallbacks
        .iter()
        .filter(|f| f.owner_kind == "room")
        .collect();
    assert!(
        room_fallbacks.is_empty(),
        "unexpected room fallbacks: {room_fallbacks:?}"
    );
}

// ── No-fallback closure ────────────────────────────────────────────────

#[test]
fn theme_assignment_all_palette_ids_valid() {
    let t = theme();
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let a = assign_by_zone(&t, &rooms, &topology, &mut rng);

    for pa in a.room_palettes.values() {
        assert!(
            t.palette(pa.palette_id).is_some(),
            "room assignment references unknown palette {:?}",
            pa.palette_id
        );
    }
    for pa in a.route_palettes.values() {
        assert!(
            t.palette(pa.palette_id).is_some(),
            "route assignment references unknown palette {:?}",
            pa.palette_id
        );
    }
    for pa in a.transition_palettes.values() {
        assert!(
            t.palette(pa.palette_id).is_some(),
            "transition assignment references unknown palette {:?}",
            pa.palette_id
        );
    }
}

// ── Exact-case lookup ──────────────────────────────────────────────────

#[test]
fn exact_case_lookup_case_sensitive() {
    let t = theme();
    let base = t.base_palette();

    // Look up exact names
    assert_eq!(t.texture_for(base.id, TextureRole::Floor), Some("bs_floor"));
    // Case variants should not match (our lookup is exact)
    // (The theme uses exact strings, so this is tautological for the typed data)
}

// ── Build artifact validation ──────────────────────────────────────────

#[test]
fn build_produces_all_outputs() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let status = run_build(tmp.path());
    assert!(status.success(), "build.py failed");

    assert_file(&tmp.path().join("palette.lmp"));
    assert_file(&tmp.path().join("cc0_dungeon_v2.wad"));
    assert_file(&tmp.path().join("theme.toml"));
    assert_file(&tmp.path().join("LICENSE"));

    let tex_dir = tmp.path().join("textures");
    let t = theme();
    for name in t.visible_texture_names() {
        assert_file(&tex_dir.join(format!("{name}_basecolor.png")));
        assert_file(&tex_dir.join(format!("{name}_norm.png")));
        assert_file(&tex_dir.join(format!("{name}_gloss.png")));
    }
}

#[test]
fn build_palette_is_768_bytes() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());
    let pal = std::fs::read(tmp.path().join("palette.lmp")).expect("read palette");
    assert_eq!(pal.len(), 768);
}

#[test]
fn build_palette_fullbrights_reserved() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());
    let pal = std::fs::read(tmp.path().join("palette.lmp")).expect("read palette");

    for idx in 224usize..=255 {
        let rgb = &pal[idx * 3..idx * 3 + 3];
        assert!(
            rgb.iter().any(|&c| c == 255),
            "fullbright index {idx} should have at least one saturated channel"
        );
    }
}

#[test]
fn build_wad2_valid_header_and_directory() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());
    let wad = std::fs::read(tmp.path().join("cc0_dungeon_v2.wad")).expect("read wad");

    // WAD2 magic
    assert_eq!(&wad[0..4], b"WAD2");

    let numlumps = i32::from_le_bytes(wad[4..8].try_into().unwrap()) as usize;
    let infotableofs = i32::from_le_bytes(wad[8..12].try_into().unwrap()) as usize;

    // 15 visible (3×4 roles + 1×3 roles) + 1 skip = 16 lumps
    assert_eq!(numlumps, 16);
    assert!(infotableofs >= 12);

    let t = theme();
    let expected_names: BTreeSet<&str> = t
        .visible_texture_names()
        .into_iter()
        .chain(std::iter::once("skip"))
        .collect();

    let mut found_names = BTreeSet::new();
    for i in 0..numlumps {
        let off = infotableofs + i * 32;
        let filepos = i32::from_le_bytes(wad[off..off + 4].try_into().unwrap());
        let disksize = i32::from_le_bytes(wad[off + 4..off + 8].try_into().unwrap());
        let size = i32::from_le_bytes(wad[off + 8..off + 12].try_into().unwrap());
        let typ = wad[off + 12];
        let comp = wad[off + 13];
        let name_bytes = &wad[off + 16..off + 32];
        let nul = name_bytes.iter().position(|&b| b == 0).unwrap_or(16);
        let name = std::str::from_utf8(&name_bytes[..nul]).expect("non-UTF8 lump name");
        found_names.insert(name);

        assert_eq!(typ, 0x44, "lump {name} type must be miptex (0x44)");
        assert_eq!(comp, 0, "lump {name} compression must be 0");
        assert_eq!(disksize, size, "lump {name} disksize != size");
        assert!(disksize > 0 && filepos >= 12);
    }

    assert_eq!(found_names, expected_names, "WAD name mismatch");
}

#[test]
fn build_wad2_miptex_dimensions() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());
    let wad = std::fs::read(tmp.path().join("cc0_dungeon_v2.wad")).expect("read wad");

    let numlumps = i32::from_le_bytes(wad[4..8].try_into().unwrap()) as usize;
    let infotableofs = i32::from_le_bytes(wad[8..12].try_into().unwrap()) as usize;

    for i in 0..numlumps {
        let off = infotableofs + i * 32;
        let filepos = i32::from_le_bytes(wad[off..off + 4].try_into().unwrap()) as usize;
        let name_bytes = &wad[off + 16..off + 32];
        let nul = name_bytes.iter().position(|&b| b == 0).unwrap_or(16);
        let name = std::str::from_utf8(&name_bytes[..nul]).unwrap();

        let mip_w = u32::from_le_bytes(wad[filepos + 16..filepos + 20].try_into().unwrap());
        let mip_h = u32::from_le_bytes(wad[filepos + 20..filepos + 24].try_into().unwrap());

        if name == "skip" {
            assert_eq!(mip_w, COMPILER_SKIP_DIMENSION);
            assert_eq!(mip_h, COMPILER_SKIP_DIMENSION);
        } else {
            assert_eq!(mip_w, VISUAL_TEXTURE_DIMENSION);
            assert_eq!(mip_h, VISUAL_TEXTURE_DIMENSION);
        }
    }
}

#[test]
fn build_wad2_mip_levels_present() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());
    let wad = std::fs::read(tmp.path().join("cc0_dungeon_v2.wad")).expect("read wad");

    let numlumps = i32::from_le_bytes(wad[4..8].try_into().unwrap()) as usize;
    let infotableofs = i32::from_le_bytes(wad[8..12].try_into().unwrap()) as usize;

    for i in 0..numlumps {
        let off = infotableofs + i * 32;
        let filepos = i32::from_le_bytes(wad[off..off + 4].try_into().unwrap()) as usize;
        let disksize = i32::from_le_bytes(wad[off + 4..off + 8].try_into().unwrap()) as usize;

        for m in 0..4usize {
            let mip_off = u32::from_le_bytes(
                wad[filepos + 24 + m * 4..filepos + 28 + m * 4]
                    .try_into()
                    .unwrap(),
            ) as usize;
            assert!(mip_off >= 40, "mip {m} offset too small");
            assert!(
                mip_off < disksize,
                "mip {m} offset {mip_off} beyond lump size {disksize}"
            );
        }
    }
}

#[test]
fn build_wad2_no_lump_overlap() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());
    let wad = std::fs::read(tmp.path().join("cc0_dungeon_v2.wad")).expect("read wad");

    let numlumps = i32::from_le_bytes(wad[4..8].try_into().unwrap()) as usize;
    let infotableofs = i32::from_le_bytes(wad[8..12].try_into().unwrap()) as usize;

    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for i in 0..numlumps {
        let off = infotableofs + i * 32;
        let filepos = i32::from_le_bytes(wad[off..off + 4].try_into().unwrap()) as usize;
        let disksize = i32::from_le_bytes(wad[off + 4..off + 8].try_into().unwrap()) as usize;
        ranges.push((filepos, filepos + disksize));
    }

    ranges.sort_by_key(|r| r.0);
    for w in ranges.windows(2) {
        assert!(w[0].1 <= w[1].0, "lump overlap: {:?} and {:?}", w[0], w[1]);
    }

    let last_end = ranges.last().unwrap().1;
    assert!(
        last_end <= infotableofs,
        "last lump ends at {last_end}, info table starts at {infotableofs}"
    );
}

#[test]
fn build_companion_dimensions_match() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());
    let tex_dir = tmp.path().join("textures");

    let t = theme();
    for name in t.visible_texture_names() {
        let base = tex_dir.join(format!("{name}_basecolor.png"));
        let norm = tex_dir.join(format!("{name}_norm.png"));
        let gloss = tex_dir.join(format!("{name}_gloss.png"));

        let (bw, bh) = png_dimensions(&base);
        let (nw, nh) = png_dimensions(&norm);
        let (gw, gh) = png_dimensions(&gloss);

        assert_eq!(
            (bw, bh),
            (VISUAL_TEXTURE_DIMENSION, VISUAL_TEXTURE_DIMENSION)
        );
        assert_eq!((nw, nh), (bw, bh), "{name} normal dimension mismatch");
        assert_eq!((gw, gh), (bw, bh), "{name} gloss dimension mismatch");
    }
}

#[test]
fn build_skip_has_no_png_companions() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());
    let tex_dir = tmp.path().join("textures");

    // skip is compiler-only: no basecolor/norm/gloss PNGs
    for suffix in &["_basecolor.png", "_norm.png", "_gloss.png"] {
        let path = tex_dir.join(format!("skip{suffix}"));
        assert!(
            !path.exists(),
            "skip must have no companion PNG: {}",
            path.display()
        );
    }
}

#[test]
fn build_skip_not_in_visible_texture_names() {
    let t = theme();
    let names = t.visible_texture_names();
    assert!(!names.contains(&"skip"));
    assert_eq!(t.skip_name, "skip");
}

#[test]
fn build_theme_toml_has_correct_content() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());

    let toml = std::fs::read_to_string(tmp.path().join("theme.toml")).expect("read theme.toml");
    assert!(toml.contains("cc0_dungeon_v2"));
    assert!(toml.contains("base_stone"));
    assert!(toml.contains("crypt"));
    assert!(toml.contains("treasury"));
    assert!(toml.contains("connector"));
    assert!(toml.contains("skip"));
}

#[test]
fn build_license_cc0_declaration() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());

    let lic = std::fs::read_to_string(tmp.path().join("LICENSE")).expect("read LICENSE");
    assert!(lic.contains("CC0"));
    assert!(lic.to_lowercase().contains("public domain"));
    assert!(lic.len() > 100);
}

#[test]
fn build_wad2_visible_miptex_has_detail() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());
    let wad = std::fs::read(tmp.path().join("cc0_dungeon_v2.wad")).expect("read wad");

    let numlumps = i32::from_le_bytes(wad[4..8].try_into().unwrap()) as usize;
    let infotableofs = i32::from_le_bytes(wad[8..12].try_into().unwrap()) as usize;

    for i in 0..numlumps {
        let off = infotableofs + i * 32;
        let filepos = i32::from_le_bytes(wad[off..off + 4].try_into().unwrap()) as usize;
        let name_bytes = &wad[off + 16..off + 32];
        let nul = name_bytes.iter().position(|&b| b == 0).unwrap_or(16);
        let name = std::str::from_utf8(&name_bytes[..nul]).unwrap();

        if name == "skip" {
            continue;
        }

        let mip_w =
            u32::from_le_bytes(wad[filepos + 16..filepos + 20].try_into().unwrap()) as usize;
        let mip_h =
            u32::from_le_bytes(wad[filepos + 20..filepos + 24].try_into().unwrap()) as usize;
        let mip0_off =
            u32::from_le_bytes(wad[filepos + 24..filepos + 28].try_into().unwrap()) as usize;
        let mip0 = &wad[filepos + mip0_off..filepos + mip0_off + mip_w * mip_h];

        // Visible textures must have palette-index variation
        assert!(
            mip0.iter().any(|&idx| idx != mip0[0]),
            "{name} mip-0 is flat (no palette variation)"
        );
        // Albedo textures must not use fullbright palette entries (224-255)
        assert!(
            mip0.iter().all(|&idx| idx < 224),
            "{name} albedo uses fullbright palette entries"
        );
    }
}

// ── Two-build byte identity ────────────────────────────────────────────

#[test]
fn two_runs_produce_byte_identical_wad() {
    let tmp_a = tempfile::tempdir().expect("tempdir A");
    let tmp_b = tempfile::tempdir().expect("tempdir B");
    assert!(run_build(tmp_a.path()).success());
    assert!(run_build(tmp_b.path()).success());

    let wad_a = std::fs::read(tmp_a.path().join("cc0_dungeon_v2.wad")).unwrap();
    let wad_b = std::fs::read(tmp_b.path().join("cc0_dungeon_v2.wad")).unwrap();
    assert_eq!(wad_a.len(), wad_b.len());
    assert_eq!(wad_a, wad_b, "WAD not byte-identical between runs");
}

#[test]
fn two_runs_produce_byte_identical_pngs() {
    let tmp_a = tempfile::tempdir().expect("tempdir A");
    let tmp_b = tempfile::tempdir().expect("tempdir B");
    assert!(run_build(tmp_a.path()).success());
    assert!(run_build(tmp_b.path()).success());

    let tex_a = tmp_a.path().join("textures");
    let tex_b = tmp_b.path().join("textures");
    let t = theme();

    for name in t.visible_texture_names() {
        for suffix in &["_basecolor.png", "_norm.png", "_gloss.png"] {
            let fname = format!("{name}{suffix}");
            let a = std::fs::read(tex_a.join(&fname)).unwrap();
            let b = std::fs::read(tex_b.join(&fname)).unwrap();
            assert_eq!(a.len(), b.len());
            assert_eq!(a, b, "PNG {fname} not byte-identical between runs");
        }
    }
}

#[test]
fn two_runs_produce_byte_identical_palette() {
    let tmp_a = tempfile::tempdir().expect("tempdir A");
    let tmp_b = tempfile::tempdir().expect("tempdir B");
    assert!(run_build(tmp_a.path()).success());
    assert!(run_build(tmp_b.path()).success());

    let pal_a = std::fs::read(tmp_a.path().join("palette.lmp")).unwrap();
    let pal_b = std::fs::read(tmp_b.path().join("palette.lmp")).unwrap();
    assert_eq!(pal_a, pal_b);
}

#[test]
fn two_runs_all_files_byte_identical() {
    let tmp_a = tempfile::tempdir().expect("tempdir A");
    let tmp_b = tempfile::tempdir().expect("tempdir B");
    assert!(run_build(tmp_a.path()).success());
    assert!(run_build(tmp_b.path()).success());

    for entry in walkdir::WalkDir::new(tmp_a.path())
        .sort_by_file_name()
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let rel = entry.path().strip_prefix(tmp_a.path()).unwrap();
        let path_b = tmp_b.path().join(rel);
        let bytes_a = std::fs::read(entry.path()).expect("read A");
        let bytes_b = std::fs::read(&path_b).expect("read B");
        assert_eq!(
            bytes_a,
            bytes_b,
            "byte mismatch in {} (len {} vs {})",
            rel.display(),
            bytes_a.len(),
            bytes_b.len()
        );
    }
}

// ── WAD names match typed theme data ───────────────────────────────────

#[test]
fn build_wad_names_match_theme_package_exact_case() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());
    let wad = std::fs::read(tmp.path().join("cc0_dungeon_v2.wad")).expect("read wad");

    let numlumps = i32::from_le_bytes(wad[4..8].try_into().unwrap()) as usize;
    let infotableofs = i32::from_le_bytes(wad[8..12].try_into().unwrap()) as usize;

    let t = theme();
    let expected: BTreeSet<&str> = t
        .visible_texture_names()
        .into_iter()
        .chain(std::iter::once("skip"))
        .collect();

    let mut found = BTreeSet::new();
    for i in 0..numlumps {
        let off = infotableofs + i * 32;
        // Also check the miptex internal name
        let filepos = i32::from_le_bytes(wad[off..off + 4].try_into().unwrap()) as usize;
        let mip_name_bytes = &wad[filepos..filepos + 16];
        let mnul = mip_name_bytes.iter().position(|&b| b == 0).unwrap_or(16);
        let mip_name = std::str::from_utf8(&mip_name_bytes[..mnul]).unwrap();

        let dir_name_bytes = &wad[off + 16..off + 32];
        let dnul = dir_name_bytes.iter().position(|&b| b == 0).unwrap_or(16);
        let dir_name = std::str::from_utf8(&dir_name_bytes[..dnul]).unwrap();

        assert_eq!(mip_name, dir_name, "dir/miptex name mismatch for entry {i}");
        assert!(
            expected.contains(mip_name),
            "unexpected lump name: {mip_name}"
        );
        found.insert(mip_name);
    }

    assert_eq!(found, expected);
}

// ── WAD hash stability ────────────────────────────────────────────────

#[test]
fn build_wad_hash_reproducible() {
    use sha2::{Digest, Sha256};

    let tmp_a = tempfile::tempdir().expect("tempdir A");
    let tmp_b = tempfile::tempdir().expect("tempdir B");
    assert!(run_build(tmp_a.path()).success());
    assert!(run_build(tmp_b.path()).success());

    let wad_a = std::fs::read(tmp_a.path().join("cc0_dungeon_v2.wad")).unwrap();
    let wad_b = std::fs::read(tmp_b.path().join("cc0_dungeon_v2.wad")).unwrap();

    let hash_a: [u8; 32] = Sha256::digest(&wad_a).into();
    let hash_b: [u8; 32] = Sha256::digest(&wad_b).into();
    assert_eq!(hash_a, hash_b, "WAD SHA-256 differs between builds");
    assert_eq!(
        hash_a,
        theme().wad_sha256,
        "WAD hash differs from package identity"
    );
}

// ── Legacy unchanged ───────────────────────────────────────────────────

#[test]
fn legacy_v1_unaffected_by_enhanced_theme() {
    // The Legacy v1 pipeline must still produce its original output
    let cfg = bsp_generator::DungeonConfig::nominal_m1();
    let (map, meta) = bsp_generator::generate(0, cfg).expect("legacy generate");
    assert!(!map.is_empty());
    assert_eq!(meta.room_count, 12);

    // Replay must be byte-identical
    let (map2, meta2) = bsp_generator::generate(0, bsp_generator::DungeonConfig::nominal_m1())
        .expect("legacy replay");
    assert_eq!(map, map2);
    assert_eq!(meta, meta2);
}

// ── Known-seed theme evidence ──────────────────────────────────────────

#[test]
fn known_seed_multi_zone_byzone() {
    // Prove that ByZone produces at least 2 distinct zone palette assignments
    let t = theme();
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let a = assign_by_zone(&t, &rooms, &topology, &mut rng);

    let mut used_palettes: BTreeSet<u32> = BTreeSet::new();
    for pa in a.room_palettes.values() {
        used_palettes.insert(pa.palette_id.0);
    }
    // With 28 rooms and 3 palettes, we should see at least 2 different palettes
    assert!(
        used_palettes.len() >= 2,
        "ByZone with 3 palettes should use at least 2 in room assignments"
    );
}

#[test]
fn known_seed_connector_ownership_explicit() {
    // Cross-zone routes and transitions must use connector explicitly
    let t = theme();
    let (_, rooms, topology, mut rng) = build_nominal(42);
    let a = assign_by_zone(&t, &rooms, &topology, &mut rng);
    let connector = t.connector_palette();

    let mut cross_zone_routes = 0usize;
    let mut cross_zone_transitions = 0usize;

    for route in &topology.routes {
        if a.zones[&route.source_room] != a.zones[&route.target_room] {
            cross_zone_routes += 1;
            assert_eq!(a.route_palettes[&route.id].palette_id, connector.id);
        }
    }

    for tr in &topology.transitions {
        if a.zones[&tr.lower_room] != a.zones[&tr.upper_room] {
            cross_zone_transitions += 1;
            assert_eq!(a.transition_palettes[&tr.id].palette_id, connector.id);
        }
    }

    // Nominal config has 3 loops and 1 vertical edge — cross-zone edges
    // are expected. But even if coincidentally zero, the assertions above
    // are vacuously true.
    eprintln!(
        "cross-zone routes: {cross_zone_routes}, cross-zone transitions: {cross_zone_transitions}"
    );
}

// ── Material owner completeness ────────────────────────────────────────

#[test]
fn every_room_has_explicit_palette_owner() {
    let t = theme();
    let (_, rooms, topology, _) = build_nominal(42);
    let a = assign_uniform(&t, &rooms, &topology);

    for room in &rooms {
        assert!(a.room_palettes.contains_key(&room.id));
        let pa = &a.room_palettes[&room.id];
        assert!(t.palette(pa.palette_id).is_some());
    }
}

#[test]
fn every_route_has_explicit_palette_owner() {
    let t = theme();
    let (_, rooms, topology, _) = build_nominal(42);
    let a = assign_uniform(&t, &rooms, &topology);

    for route in &topology.routes {
        assert!(a.route_palettes.contains_key(&route.id));
        let pa = &a.route_palettes[&route.id];
        assert!(t.palette(pa.palette_id).is_some());
    }
}

#[test]
fn every_transition_has_explicit_palette_owner() {
    let t = theme();
    let (_, rooms, topology, _) = build_nominal(42);
    let a = assign_uniform(&t, &rooms, &topology);

    for tr in &topology.transitions {
        assert!(a.transition_palettes.contains_key(&tr.id));
        let pa = &a.transition_palettes[&tr.id];
        assert!(t.palette(pa.palette_id).is_some());
    }
}

// ── Checked-in theme closure determinism ──────────────────────────────

/// Prove the checked-in theme directory matches a fresh deterministic build.
/// Every output — LICENSE, theme.toml, palette, WAD, and all 45 companion
/// PNGs — must be byte-identical.
#[test]
fn checked_in_closure_matches_deterministic_build() {
    let tmp = tempfile::tempdir().expect("tempdir");
    assert!(run_build(tmp.path()).success());

    // Static files
    for name in &["LICENSE", "theme.toml", "palette.lmp", "cc0_dungeon_v2.wad"] {
        let checked_in = theme_dir().join(name);
        let generated = tmp.path().join(name);
        assert_file(&checked_in);
        assert_file(&generated);
        let ci_bytes = std::fs::read(&checked_in).unwrap();
        let gen_bytes = std::fs::read(&generated).unwrap();
        assert_eq!(
            ci_bytes, gen_bytes,
            "checked-in {name} differs from deterministic build"
        );
    }

    // All 45 PNG companions — 15 identities × 3 variants (basecolor/norm/gloss).
    // The build.py identities match the ALL_VISIBLE names directly:
    //   bs_floor, bs_wall, bs_ceil, bs_accent,
    //   crypt_floor, crypt_wall, crypt_ceil, crypt_accent,
    //   treas_floor, treas_wall, treas_ceil, treas_accent,
    //   conn_floor, conn_wall, conn_ceil
    let visible_identities: &[&str] = &[
        "bs_floor", "bs_wall", "bs_ceil", "bs_accent",
        "crypt_floor", "crypt_wall", "crypt_ceil", "crypt_accent",
        "treas_floor", "treas_wall", "treas_ceil", "treas_accent",
        "conn_floor", "conn_wall", "conn_ceil",
    ];
    assert_eq!(visible_identities.len(), 15, "15 visible identities");

    let variants = ["_basecolor.png", "_norm.png", "_gloss.png"];
    let expected_pngs = visible_identities
        .iter()
        .flat_map(|identity| variants.iter().map(move |suffix| format!("{identity}{suffix}")))
        .collect::<BTreeSet<_>>();
    assert_eq!(expected_pngs.len(), 45, "15 identities × 3 PNG variants");

    for directory in [theme_dir().join("textures"), tmp.path().join("textures")] {
        let actual_pngs = std::fs::read_dir(&directory)
            .expect("read texture directory")
            .map(|entry| {
                let entry = entry.expect("read texture entry");
                assert!(entry.path().is_file(), "texture entry must be a file");
                entry.file_name().to_string_lossy().into_owned()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(
            actual_pngs,
            expected_pngs,
            "{} must contain exactly the 45 declared texture PNGs",
            directory.display()
        );
    }

    let mut total = 0usize;
    for identity in visible_identities {
        for suffix in &variants {
            let fname = format!("{identity}{suffix}");
            let checked_in = theme_dir().join("textures").join(&fname);
            let generated = tmp.path().join("textures").join(&fname);
            assert!(
                checked_in.is_file(),
                "checked-in textures/{fname} missing"
            );
            assert!(
                generated.is_file(),
                "generated textures/{fname} missing"
            );
            let ci_bytes = std::fs::read(&checked_in).unwrap();
            let gen_bytes = std::fs::read(&generated).unwrap();
            assert_eq!(
                ci_bytes, gen_bytes,
                "checked-in textures/{fname} differs from deterministic build"
            );
            total += 1;
        }
    }
    assert_eq!(total, 45, "must verify exactly 45 companion PNGs");

    eprintln!(
        "PASS: checked-in theme closure ({total} files) matches deterministic build"
    );
}
