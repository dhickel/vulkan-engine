//! Phase 11 EnhancedV3 Richness CarvedGrotto compiler qualification.
//!
//! One sealed cave fixture per theme: a both-band host mass with a carved
//! cave void (64x80 passage, chamber, south gallery) and the emitted solid
//! complement. Each fixture crosses the pinned warning-fatal
//! `qbsp -> vis -> light` boundary, strict-reloads, and is queried through
//! hull-0 contents and the compiled player stored hull. Missing fixtures,
//! theme assets, compiler profile, tools, or pinned executable identities
//! are hard failures.

#[path = "support/conventions_compiler.rs"]
mod compiler_support;

use compiler_support as cc;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
struct CaveCase {
    theme: &'static str,
    fixture: &'static str,
}

const CAVE_CASES: [CaveCase; 3] = [
    CaveCase {
        theme: "ancient",
        fixture: "ancient_cave",
    },
    CaveCase {
        theme: "egyptian",
        fixture: "egyptian_cave",
    },
    CaveCase {
        theme: "brutalist",
        fixture: "brutalist_cave",
    },
];

const LEGAL_MIPTEX: [&str; 4] = ["bs_floor", "bs_wall", "bs_ceil", "bs_accent"];

fn fixture_path(case: CaveCase) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/enhanced_v3_richness_cave")
        .join(format!("{}.map", case.fixture))
}

fn source_brush_count(map: &str) -> usize {
    map.lines().filter(|line| line.trim() == "{").count() - 1 // worldspawn
}

fn quake_to_engine(point: (i32, i32, i32), transform: &bsp::QuakeToEngine) -> glam::Vec3 {
    transform.position(point.0 as f32, point.1 as f32, point.2 as f32)
}

fn point_contents(world: &bsp::BspWorld, point: (i32, i32, i32)) -> bsp::PointContents {
    let transform = bsp::QuakeToEngine::default();
    bsp::point_contents(
        quake_to_engine(point, &transform),
        &world.nodes,
        &world.leaves,
        &world.planes,
    )
}

fn player_trace(
    world: &bsp::BspWorld,
    start: (i32, i32, i32),
    end: (i32, i32, i32),
) -> bsp::TraceResult {
    let transform = bsp::QuakeToEngine::default();
    bsp::trace_line(
        quake_to_engine(start, &transform),
        quake_to_engine(end, &transform),
        bsp::StoredHull::Player,
        &world.clipnodes,
        &world.planes,
        &world.models,
        &transform,
    )
}

fn assert_point_solid(world: &bsp::BspWorld, label: &str, point: (i32, i32, i32)) {
    let contents = point_contents(world, point);
    assert!(
        contents.is_solid(),
        "{label}: hull-0 point {point:?} lacks solid structure: {contents:?}"
    );
}

fn assert_player_clear(
    world: &bsp::BspWorld,
    label: &str,
    start: (i32, i32, i32),
    end: (i32, i32, i32),
) {
    let trace = player_trace(world, start, end);
    let completed_fraction = bsp::TraceResult::no_hit().hit_fraction;
    assert!(
        trace.hit_fraction == completed_fraction && !trace.starts_solid && !trace.all_solid,
        "{label}: player stored-hull trace {start:?}->{end:?} blocked: {trace:?}"
    );
}

fn assert_player_blocked(
    world: &bsp::BspWorld,
    label: &str,
    start: (i32, i32, i32),
    end: (i32, i32, i32),
) {
    let trace = player_trace(world, start, end);
    assert!(
        trace.hit_fraction < bsp::TraceResult::no_hit().hit_fraction || trace.all_solid,
        "{label}: player stored-hull trace {start:?}->{end:?} unexpectedly clear"
    );
}

fn assert_source_contract(case: CaveCase, map: &str) {
    assert!(
        map.contains(&format!("\"richness_theme\" \"{}\"", case.theme)),
        "{}: missing theme identity",
        case.theme
    );
    let mut brushes = 0;
    for line in map.lines() {
        let line = line.trim();
        if line == "{" {
            brushes += 1;
        }
        if line.starts_with('(') {
            let tex = line.split('"').nth(1).unwrap_or("");
            assert!(
                LEGAL_MIPTEX.contains(&tex),
                "{}: illegal texture {tex}",
                case.theme
            );
        }
    }
    assert!(
        brushes >= 9,
        "{}: cave fixture too sparse ({brushes} brushes)",
        case.theme
    );
}

fn assert_compiled_texture_closure(world: &bsp::BspWorld, theme: &str) {
    let slots = bsp::resources::parse_miptex_slots(&world.miptex_data);
    for (face_index, face) in world.faces.iter().enumerate() {
        let texinfo = &world.texinfos[face.texinfo_id as usize];
        let identity = slots[texinfo.miptex as usize]
            .identity
            .as_deref()
            .unwrap_or_else(|| panic!("{theme}: face {face_index} has no miptex identity"));
        assert!(
            LEGAL_MIPTEX.contains(&identity),
            "{theme}: compiled face {face_index} uses unauthorized miptex {identity}"
        );
    }
}

/// Compile every cave fixture and qualify the carved void witnesses.
#[test]
fn cave_fixtures_compile_and_strict_reload_in_all_themes() {
    let profile = cc::load_compiler_profile().expect("required pinned compiler profile");
    let tool_dir = cc::resolve_tool_dir();
    assert!(
        cc::tools_available(&tool_dir),
        "required pinned ericw-tools unavailable at {}",
        tool_dir.display()
    );
    cc::verify_executable_hashes(&tool_dir, &profile)
        .unwrap_or_else(|errors| panic!("pinned ericw-tools hash mismatch: {errors:?}"));

    let (wad, palette) = cc::theme_paths();
    assert!(wad.is_file(), "required WAD missing at {}", wad.display());
    assert!(
        palette.is_file(),
        "required palette missing at {}",
        palette.display()
    );

    for case in CAVE_CASES {
        let path = fixture_path(case);
        let map = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("required fixture {}: {error}", path.display()));
        assert_source_contract(case, &map);

        let staging = cc::create_staging_dir(case.fixture)
            .unwrap_or_else(|error| panic!("{}: staging: {error}", case.theme));
        let compiled = cc::compile_map(&path, staging.path(), &tool_dir, &wad, &palette, &profile)
            .unwrap_or_else(|error| panic!("{}: pinned compiler pipeline: {error}", case.theme));

        assert_eq!(&compiled.bsp_data[..4], b"BSP2", "{}: BSP2", case.theme);
        assert_eq!(&compiled.lit_data[..4], b"QLIT", "{}: QLIT", case.theme);
        assert!(
            compiled.qbsp_output.diagnostics.is_empty(),
            "{}: qbsp diagnostics: {:?}",
            case.theme,
            compiled.qbsp_output.diagnostics
        );
        assert!(
            compiled.vis_output.diagnostics.is_empty(),
            "{}: vis diagnostics: {:?}",
            case.theme,
            compiled.vis_output.diagnostics
        );
        assert!(
            compiled.light_output.diagnostics.is_empty(),
            "{}: light diagnostics: {:?}",
            case.theme,
            compiled.light_output.diagnostics
        );

        let (world, reload) =
            cc::strict_reload_with_paths(&compiled.bsp_data, &compiled.lit_data, &wad, &palette)
                .unwrap_or_else(|error| panic!("{}: strict reload: {error}", case.theme));
        assert_eq!(reload.profile, "bsp2", "{}: strict profile", case.theme);
        assert_eq!(reload.diagnostics, 0, "{}: strict diagnostics", case.theme);
        assert!(
            reload.faces > 100,
            "{}: compiled fixture is sparse",
            case.theme
        );
        assert!(
            reload.faces < 15_000,
            "{}: Richness face ceiling",
            case.theme
        );
        assert!(
            reload.clipnodes > 0,
            "{}: missing collision hulls",
            case.theme
        );
        assert!(
            reload.empty_leaves > 0,
            "{}: missing clear leaves",
            case.theme
        );
        assert!(
            reload.solid_leaves > 0,
            "{}: missing solid leaves",
            case.theme
        );
        assert!(
            reload.lightdata_bytes > 0,
            "{}: missing baked lightdata",
            case.theme
        );
        assert!(!world.vis_data.is_empty(), "{}: missing PVS", case.theme);

        // Carved void witnesses (passage 64x80, chamber, south gallery).
        // Passage: x 64..128, y 240..304, z 16..96 -> center (96, 272, 48).
        assert_player_clear(
            &world,
            &format!("{} passage center standing", case.theme),
            (96, 272, 41),
            (96, 272, 41),
        );
        assert_player_clear(
            &world,
            &format!("{} passage headroom", case.theme),
            (96, 272, 41),
            (96, 272, 70),
        );
        // Passage mouth -> chamber.
        assert_player_clear(
            &world,
            &format!("{} passage-to-chamber route", case.theme),
            (96, 272, 48),
            (160, 272, 48),
        );
        // Chamber center.
        assert_player_clear(
            &world,
            &format!("{} chamber center", case.theme),
            (256, 256, 41),
            (256, 256, 41),
        );
        // Chamber headroom to 200 (pendant bottom at 224).
        assert_player_clear(
            &world,
            &format!("{} chamber headroom", case.theme),
            (232, 232, 41),
            (232, 232, 200),
        );
        // South gallery route around the stalagmite.
        assert_player_clear(
            &world,
            &format!("{} south gallery route", case.theme),
            (256, 256, 48),
            (256, 176, 48),
        );
        // Cave solids present (hull-0 point queries).
        assert_point_solid(
            &world,
            &format!("{} west complement mass", case.theme),
            (40, 200, 200),
        );
        assert_point_solid(
            &world,
            &format!("{} chamber stalagmite", case.theme),
            (176, 176, 80),
        );
        assert_point_solid(
            &world,
            &format!("{} chamber pillar", case.theme),
            (368, 368, 80),
        );
        assert_point_solid(
            &world,
            &format!("{} ceiling pendant", case.theme),
            (256, 176, 248),
        );
        // Solids block the player hull.
        assert_player_blocked(
            &world,
            &format!("{} pillar blocks", case.theme),
            (256, 368, 48),
            (368, 368, 48),
        );

        assert_compiled_texture_closure(&world, case.theme);

        eprintln!(
            "{}: {} source brushes -> {} faces, {} leaves, {} clipnodes; BSP={} LIT={}",
            case.theme,
            source_brush_count(&map),
            reload.faces,
            reload.leaves,
            reload.clipnodes,
            compiled.bsp_sha256,
            compiled.lit_sha256,
        );
    }
}
