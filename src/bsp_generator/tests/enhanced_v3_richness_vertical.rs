//! Phase 10 EnhancedV3 Richness vertical-architecture compiler qualification.
//!
//! Three sealed additive suites cover every vertical primitive once per theme.
//! Each suite crosses the pinned warning-fatal `qbsp -> vis -> light` boundary,
//! strict-reloads, and is queried through both hull-0 contents and the compiled
//! player stored hull. Missing fixtures, theme assets, compiler profile, tools,
//! or pinned executable identities are hard failures.

#[path = "support/conventions_compiler.rs"]
mod compiler_support;

use compiler_support as cc;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
struct ThemeCase {
    theme: &'static str,
    fixture: &'static str,
}

const THEME_CASES: [ThemeCase; 3] = [
    ThemeCase {
        theme: "ancient",
        fixture: "ancient_vertical_suite",
    },
    ThemeCase {
        theme: "egyptian",
        fixture: "egyptian_vertical_suite",
    },
    ThemeCase {
        theme: "brutalist",
        fixture: "brutalist_vertical_suite",
    },
];

const PRIMITIVE_MARKERS: [&str; 11] = [
    "grand_multi_storey_shell",
    "balcony_mezzanine",
    "catwalk_bridge_committed_void",
    "overlook_sill_no_floor_hole",
    "pit_chasm_paired_omissions_accessible_bottom",
    "ladder_shaft_climb_descriptor",
    "integer_12_step_spiral",
    "guarded_one_way_drop",
    "vertical_arena",
    "straight_stairwell",
    "two_band_open_stairwell",
];

const LEGAL_MIPTEX: [&str; 4] = ["bs_floor", "bs_wall", "bs_ceil", "bs_accent"];

#[derive(Clone, Copy, Debug)]
struct StepWitness {
    center: (i32, i32),
    top: i32,
}

const PIT_ACCESS_STEPS: [StepWitness; 12] = [
    StepWitness {
        center: (1712, 88),
        top: -160,
    },
    StepWitness {
        center: (1712, 136),
        top: -144,
    },
    StepWitness {
        center: (1712, 184),
        top: -128,
    },
    StepWitness {
        center: (1712, 232),
        top: -112,
    },
    StepWitness {
        center: (1712, 280),
        top: -96,
    },
    StepWitness {
        center: (1712, 328),
        top: -80,
    },
    StepWitness {
        center: (1872, 328),
        top: -64,
    },
    StepWitness {
        center: (1872, 280),
        top: -48,
    },
    StepWitness {
        center: (1872, 232),
        top: -32,
    },
    StepWitness {
        center: (1872, 184),
        top: -16,
    },
    StepWitness {
        center: (1872, 136),
        top: 0,
    },
    StepWitness {
        center: (1872, 88),
        top: 16,
    },
];

const SPIRAL_STEPS: [StepWitness; 12] = [
    StepWitness {
        center: (864, 672),
        top: 32,
    },
    StepWitness {
        center: (864, 736),
        top: 48,
    },
    StepWitness {
        center: (864, 800),
        top: 64,
    },
    StepWitness {
        center: (864, 864),
        top: 80,
    },
    StepWitness {
        center: (800, 864),
        top: 96,
    },
    StepWitness {
        center: (736, 864),
        top: 112,
    },
    StepWitness {
        center: (672, 864),
        top: 128,
    },
    StepWitness {
        center: (672, 800),
        top: 144,
    },
    StepWitness {
        center: (672, 736),
        top: 160,
    },
    StepWitness {
        center: (672, 672),
        top: 176,
    },
    StepWitness {
        center: (736, 672),
        top: 192,
    },
    StepWitness {
        center: (832, 672),
        top: 208,
    },
];

const STRAIGHT_STEPS: [StepWitness; 12] = [
    StepWitness {
        center: (79, 1224),
        top: 32,
    },
    StepWitness {
        center: (111, 1224),
        top: 48,
    },
    StepWitness {
        center: (143, 1224),
        top: 64,
    },
    StepWitness {
        center: (175, 1224),
        top: 80,
    },
    StepWitness {
        center: (207, 1224),
        top: 96,
    },
    StepWitness {
        center: (239, 1224),
        top: 112,
    },
    StepWitness {
        center: (271, 1224),
        top: 128,
    },
    StepWitness {
        center: (303, 1224),
        top: 144,
    },
    StepWitness {
        center: (335, 1224),
        top: 160,
    },
    StepWitness {
        center: (367, 1224),
        top: 176,
    },
    StepWitness {
        center: (399, 1224),
        top: 192,
    },
    StepWitness {
        center: (431, 1224),
        top: 208,
    },
];

const OPEN_STEPS: [StepWitness; 12] = [
    StepWitness {
        center: (591, 1160),
        top: 32,
    },
    StepWitness {
        center: (623, 1160),
        top: 48,
    },
    StepWitness {
        center: (655, 1160),
        top: 64,
    },
    StepWitness {
        center: (687, 1160),
        top: 80,
    },
    StepWitness {
        center: (719, 1160),
        top: 96,
    },
    StepWitness {
        center: (751, 1160),
        top: 112,
    },
    StepWitness {
        center: (753, 1336),
        top: 128,
    },
    StepWitness {
        center: (721, 1336),
        top: 144,
    },
    StepWitness {
        center: (689, 1336),
        top: 160,
    },
    StepWitness {
        center: (657, 1336),
        top: 176,
    },
    StepWitness {
        center: (625, 1336),
        top: 192,
    },
    StepWitness {
        center: (593, 1336),
        top: 208,
    },
];

fn crate_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn fixture_path(case: ThemeCase) -> PathBuf {
    crate_dir().join(format!(
        "tests/fixtures/enhanced_v3_richness_vertical/{}.map",
        case.fixture
    ))
}

fn source_brush_count(map: &str) -> usize {
    let face_lines = map
        .lines()
        .filter(|line| line.trim_start().starts_with('('))
        .count();
    assert_eq!(
        face_lines % 6,
        0,
        "fixture must contain six plane lines per axis-aligned brush"
    );
    face_lines / 6
}

fn assert_source_contract(case: ThemeCase, map: &str) {
    assert!(map.ends_with('\n'), "{}: missing terminal LF", case.theme);
    assert!(!map.contains('\r'), "{}: non-LF line ending", case.theme);
    assert!(
        map.contains("\"wad\" \"cc0_dungeon_v2.wad\""),
        "{}: fixture must select the qualified WAD",
        case.theme
    );
    assert!(
        map.contains(&format!("\"richness_theme\" \"{}\"", case.theme)),
        "{}: explicit theme marker missing",
        case.theme
    );
    assert!(
        source_brush_count(map) >= 250,
        "{}: fixture is not sufficiently dense",
        case.theme
    );

    for primitive in PRIMITIVE_MARKERS {
        let marker = format!("// primitive: {primitive}");
        assert_eq!(
            map.match_indices(&marker).count(),
            1,
            "{}: primitive marker {primitive} must occur exactly once",
            case.theme
        );
    }

    for line in map
        .lines()
        .filter(|line| line.trim_start().starts_with('('))
    {
        let miptex = line
            .split('"')
            .nth(1)
            .unwrap_or_else(|| panic!("{}: malformed face line: {line}", case.theme));
        assert!(
            LEGAL_MIPTEX.contains(&miptex),
            "{}: unauthorized cc0_dungeon_v2 miptex {miptex}",
            case.theme
        );
    }
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

fn assert_point_clear(world: &bsp::BspWorld, label: &str, point: (i32, i32, i32)) {
    let contents = point_contents(world, point);
    assert!(
        !contents.is_solid(),
        "{label}: hull-0 point {point:?} is solid: {contents:?}"
    );
}

fn assert_point_solid(world: &bsp::BspWorld, label: &str, point: (i32, i32, i32)) {
    let contents = point_contents(world, point);
    assert!(
        contents.is_solid(),
        "{label}: hull-0 point {point:?} lacks solid structure: {contents:?}"
    );
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
        !trace.starts_solid,
        "{label}: guard probe starts solid at {start:?}: {trace:?}"
    );
    assert!(
        trace.hit_fraction != bsp::TraceResult::no_hit().hit_fraction,
        "{label}: player stored hull passed through required solid structure {start:?}->{end:?}"
    );
}

fn assert_player_standing(world: &bsp::BspWorld, label: &str, point: (i32, i32, i32)) {
    assert_player_clear(world, label, point, point);
}

fn assert_eighty_headroom(world: &bsp::BspWorld, label: &str, center: (i32, i32), floor_top: i32) {
    assert_player_clear(
        world,
        label,
        (center.0, center.1, floor_top + 25),
        (center.0, center.1, floor_top + 56),
    );
}

fn assert_step_support_and_clearance(world: &bsp::BspWorld, family: &str, steps: &[StepWitness]) {
    for (index, step) in steps.iter().enumerate() {
        let support = (step.center.0, step.center.1, step.top - 8);
        let surface_clear = (step.center.0, step.center.1, step.top + 1);
        // Stay one unit above exact hull/floor contact; clip-plane equality is
        // compiler-tree dependent and does not prove usable clearance.
        let standing = (step.center.0, step.center.1, step.top + 25);
        assert_point_solid(world, &format!("{family} tread {index} support"), support);
        assert_point_clear(
            world,
            &format!("{family} tread {index} surface"),
            surface_clear,
        );
        assert_player_standing(world, &format!("{family} tread {index} standing"), standing);
        assert_eighty_headroom(
            world,
            &format!("{family} tread {index} headroom"),
            step.center,
            step.top,
        );
    }
}

fn assert_step_transitions(world: &bsp::BspWorld, family: &str, steps: &[StepWitness]) {
    for (index, pair) in steps.windows(2).enumerate() {
        let from = pair[0];
        let to = pair[1];
        let from_origin = (from.center.0, from.center.1, from.top + 25);
        let raised_origin = (from.center.0, from.center.1, to.top + 25);
        let to_origin = (to.center.0, to.center.1, to.top + 25);
        assert_player_clear(
            world,
            &format!("{family} step {index} raise"),
            from_origin,
            raised_origin,
        );
        assert_player_clear(
            world,
            &format!("{family} step {index} cross"),
            raised_origin,
            to_origin,
        );
    }
}

fn assert_matching_hole(
    world: &bsp::BspWorld,
    label: &str,
    xy: (i32, i32),
    clear_z: &[i32],
    solid_points: &[(i32, i32, i32)],
) {
    for z in clear_z {
        assert_point_clear(world, label, (xy.0, xy.1, *z));
    }
    for point in solid_points {
        assert_point_solid(world, label, *point);
    }
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

fn qualify_grand_shell(world: &bsp::BspWorld, theme: &str) {
    assert_point_solid(
        world,
        &format!("{theme} grand lower floor"),
        (1280, 1408, 8),
    );
    assert_point_solid(
        world,
        &format!("{theme} grand upper gallery"),
        (1280, 1248, 200),
    );
    assert_point_solid(world, &format!("{theme} grand cap"), (1280, 1408, 504));
    assert_point_solid(world, &format!("{theme} grand shell wall"), (8, 1408, 232));
    assert_player_clear(
        world,
        &format!("{theme} grand lower 64x80 route"),
        (1088, 1408, 41),
        (1472, 1408, 41),
    );
    assert_player_clear(
        world,
        &format!("{theme} grand upper 64x80 route"),
        (1120, 1248, 233),
        (1440, 1248, 233),
    );
    assert_player_clear(
        world,
        &format!("{theme} grand committed vertical volume"),
        (1280, 1408, 41),
        (1280, 1408, 400),
    );
    assert_player_blocked(
        world,
        &format!("{theme} grand wall guard"),
        (64, 1408, 232),
        (8, 1408, 232),
    );
}

fn qualify_balcony(world: &bsp::BspWorld, theme: &str) {
    assert_point_solid(world, &format!("{theme} balcony slab"), (256, 96, 200));
    assert_point_solid(world, &format!("{theme} balcony corbel"), (264, 32, 184));
    assert_point_solid(world, &format!("{theme} balcony rail"), (256, 168, 232));
    assert_player_clear(
        world,
        &format!("{theme} balcony standing route"),
        (96, 96, 233),
        (416, 96, 233),
    );
    assert_player_clear(
        world,
        &format!("{theme} balcony lower clear route"),
        (96, 96, 41),
        (416, 96, 41),
    );
    assert_eighty_headroom(world, &format!("{theme} balcony headroom"), (256, 96), 208);
    assert_player_blocked(
        world,
        &format!("{theme} balcony collision guard"),
        (256, 96, 233),
        (256, 168, 233),
    );
}

fn qualify_catwalk(world: &bsp::BspWorld, theme: &str) {
    assert_point_clear(
        world,
        &format!("{theme} catwalk committed void"),
        (768, 160, 40),
    );
    assert_point_clear(world, &format!("{theme} catwalk slab hole"), (768, 160, 8));
    assert_point_solid(
        world,
        &format!("{theme} catwalk void bottom"),
        (768, 160, -184),
    );
    assert_point_solid(world, &format!("{theme} catwalk deck"), (768, 256, 200));
    assert_point_solid(world, &format!("{theme} catwalk rail"), (768, 216, 232));
    assert_player_clear(
        world,
        &format!("{theme} catwalk 64x80 route"),
        (656, 256, 233),
        (880, 256, 233),
    );
    assert_player_clear(
        world,
        &format!("{theme} catwalk protected turn"),
        (608, 160, 233),
        (608, 256, 233),
    );
    assert_eighty_headroom(world, &format!("{theme} catwalk headroom"), (768, 256), 208);
    assert_player_blocked(
        world,
        &format!("{theme} catwalk collision guard"),
        (768, 256, 233),
        (768, 216, 233),
    );
}

fn qualify_overlook(world: &bsp::BspWorld, theme: &str) {
    assert_point_solid(world, &format!("{theme} overlook sill"), (1280, 264, 40));
    assert_point_clear(
        world,
        &format!("{theme} overlook aperture"),
        (1280, 264, 128),
    );
    assert_point_solid(
        world,
        &format!("{theme} overlook near floor"),
        (1280, 224, 8),
    );
    assert_point_solid(
        world,
        &format!("{theme} overlook far floor"),
        (1280, 304, 8),
    );
    assert_player_clear(
        world,
        &format!("{theme} overlook aperture headroom"),
        (1280, 224, 96),
        (1280, 304, 96),
    );
    assert_player_blocked(
        world,
        &format!("{theme} overlook sill guard"),
        (1280, 223, 41),
        (1280, 304, 41),
    );
}

fn qualify_pit(world: &bsp::BspWorld, theme: &str) {
    assert_matching_hole(
        world,
        &format!("{theme} pit matching XY omission"),
        (1792, 256),
        &[8, 168, 200],
        &[(1600, 256, 8), (1600, 256, 168), (1600, 256, 200)],
    );
    assert_point_solid(
        world,
        &format!("{theme} pit accessible bottom"),
        (1792, 256, -184),
    );
    assert_player_standing(
        world,
        &format!("{theme} pit bottom standing"),
        (1792, 256, -151),
    );
    assert_step_support_and_clearance(world, &format!("{theme} pit access"), &PIT_ACCESS_STEPS);
    assert_step_transitions(
        world,
        &format!("{theme} pit lower flight"),
        &PIT_ACCESS_STEPS[..6],
    );
    assert_player_clear(
        world,
        &format!("{theme} pit first landing leg"),
        (1712, 328, -55),
        (1712, 384, -55),
    );
    assert_player_clear(
        world,
        &format!("{theme} pit protected turn"),
        (1712, 384, -55),
        (1872, 384, -55),
    );
    assert_player_clear(
        world,
        &format!("{theme} pit turn raise"),
        (1872, 384, -55),
        (1872, 384, -39),
    );
    assert_player_clear(
        world,
        &format!("{theme} pit turn to upper flight"),
        (1872, 384, -39),
        (1872, 328, -39),
    );
    assert_step_transitions(
        world,
        &format!("{theme} pit upper flight"),
        &PIT_ACCESS_STEPS[6..],
    );
    assert_player_clear(
        world,
        &format!("{theme} pit lower landing"),
        (1712, 48, -135),
        (1712, 88, -135),
    );
    assert_player_clear(
        world,
        &format!("{theme} pit upper landing"),
        (1872, 88, 41),
        (1872, 48, 41),
    );
}

fn qualify_ladder(world: &bsp::BspWorld, theme: &str) {
    assert_matching_hole(
        world,
        &format!("{theme} ladder paired slab holes"),
        (256, 768),
        &[168, 200],
        &[(200, 768, 168), (200, 768, 200)],
    );
    assert_point_solid(
        world,
        &format!("{theme} ladder lower landing"),
        (256, 768, 8),
    );
    assert_point_solid(world, &format!("{theme} ladder rung"), (300, 768, 34));
    assert_player_clear(
        world,
        &format!("{theme} ladder lower offset approach"),
        (256, 672, 41),
        (256, 768, 41),
    );
    assert_player_clear(
        world,
        &format!("{theme} ladder 64x80 climb route"),
        (256, 768, 41),
        (256, 768, 233),
    );
    assert_player_clear(
        world,
        &format!("{theme} ladder upper landing"),
        (256, 768, 233),
        (256, 856, 233),
    );
    assert_player_blocked(
        world,
        &format!("{theme} ladder shaft wall"),
        (256, 768, 120),
        (200, 768, 120),
    );
}

fn qualify_spiral(world: &bsp::BspWorld, theme: &str) {
    assert_eq!(SPIRAL_STEPS.len(), 12, "integer spiral template changed");
    assert_matching_hole(
        world,
        &format!("{theme} spiral paired slab holes"),
        (816, 768),
        &[168, 200],
        &[(616, 768, 168), (616, 768, 200)],
    );
    assert_point_solid(
        world,
        &format!("{theme} spiral center column"),
        (768, 768, 120),
    );
    assert_step_support_and_clearance(world, &format!("{theme} spiral"), &SPIRAL_STEPS);
    assert_step_transitions(world, &format!("{theme} spiral"), &SPIRAL_STEPS);
    assert_player_clear(
        world,
        &format!("{theme} spiral lower landing"),
        (864, 608, 41),
        (864, 608, 57),
    );
    assert_player_clear(
        world,
        &format!("{theme} spiral lower entry"),
        (864, 608, 57),
        (864, 672, 57),
    );
    assert_player_clear(
        world,
        &format!("{theme} spiral upper landing"),
        (832, 672, 233),
        (928, 672, 233),
    );
}

fn qualify_drop(world: &bsp::BspWorld, theme: &str) {
    assert_matching_hole(
        world,
        &format!("{theme} drop matching XY omissions"),
        (1280, 768),
        &[8, 168, 200],
        &[(1200, 768, 168), (1200, 768, 200)],
    );
    assert_point_solid(
        world,
        &format!("{theme} drop upper guard"),
        (1280, 728, 232),
    );
    assert_point_solid(
        world,
        &format!("{theme} drop lower floor"),
        (1280, 768, -184),
    );
    assert_player_clear(
        world,
        &format!("{theme} drop protected upper entry"),
        (1184, 768, 233),
        (1280, 768, 233),
    );
    assert_player_clear(
        world,
        &format!("{theme} upper-to-lower drop route"),
        (1280, 768, 233),
        (1280, 768, -151),
    );
    assert_player_clear(
        world,
        &format!("{theme} drop offset lower egress"),
        (1280, 768, -151),
        (1408, 768, -151),
    );
    assert_player_blocked(
        world,
        &format!("{theme} drop upper collision guard"),
        (1280, 768, 233),
        (1280, 720, 233),
    );
    assert_player_blocked(
        world,
        &format!("{theme} structural non-return headroom"),
        (1408, 768, -151),
        (1408, 768, -71),
    );
    assert_point_solid(
        world,
        &format!("{theme} structural return wall"),
        (1336, 768, 0),
    );
}

fn qualify_arena(world: &bsp::BspWorld, theme: &str) {
    for (index, (start, end)) in [
        ((1520, 720, 41), (1616, 720, 41)),
        ((1888, 832, 41), (1984, 832, 41)),
        ((1520, 720, 233), (1616, 720, 233)),
        ((1888, 832, 233), (1984, 832, 233)),
    ]
    .into_iter()
    .enumerate()
    {
        assert_player_clear(
            world,
            &format!("{theme} arena controlled entry {index}"),
            start,
            end,
        );
    }
    assert_player_clear(
        world,
        &format!("{theme} arena balcony route"),
        (1680, 624, 233),
        (1904, 624, 233),
    );
    assert_player_clear(
        world,
        &format!("{theme} arena east-west catwalk"),
        (1680, 768, 233),
        (1840, 768, 233),
    );
    assert_player_clear(
        world,
        &format!("{theme} arena north-south catwalk"),
        (1760, 688, 233),
        (1760, 848, 233),
    );
    assert_player_clear(
        world,
        &format!("{theme} arena crossing turn"),
        (1680, 768, 233),
        (1760, 768, 233),
    );
    assert_player_clear(
        world,
        &format!("{theme} arena crossing exit"),
        (1760, 768, 233),
        (1760, 848, 233),
    );
    assert_player_clear(
        world,
        &format!("{theme} arena internal access"),
        (1632, 624, 41),
        (1632, 624, 233),
    );
    assert_point_solid(
        world,
        &format!("{theme} arena catwalk deck"),
        (1760, 768, 200),
    );
    assert_point_clear(
        world,
        &format!("{theme} arena committed void"),
        (1760, 720, 40),
    );
    assert_point_solid(
        world,
        &format!("{theme} arena void bottom"),
        (1760, 720, -184),
    );
    assert_point_solid(world, &format!("{theme} arena shell"), (1576, 800, 120));
    assert_player_blocked(
        world,
        &format!("{theme} arena catwalk guard"),
        (1688, 768, 233),
        (1688, 720, 233),
    );
}

fn qualify_straight_stairwell(world: &bsp::BspWorld, theme: &str) {
    assert_eq!(
        STRAIGHT_STEPS.len(),
        12,
        "straight stair tread count changed"
    );
    assert_matching_hole(
        world,
        &format!("{theme} straight stair slab opening"),
        (240, 1224),
        &[168, 200],
        &[(16, 1224, 168), (16, 1224, 200)],
    );
    assert_step_support_and_clearance(world, &format!("{theme} straight stair"), &STRAIGHT_STEPS);
    assert_step_transitions(world, &format!("{theme} straight stair"), &STRAIGHT_STEPS);
    assert_player_clear(
        world,
        &format!("{theme} straight lower landing raise"),
        (40, 1224, 41),
        (40, 1224, 57),
    );
    assert_player_clear(
        world,
        &format!("{theme} straight lower landing entry"),
        (40, 1224, 57),
        (79, 1224, 57),
    );
    assert_player_clear(
        world,
        &format!("{theme} straight upper landing"),
        (431, 1224, 233),
        (464, 1224, 233),
    );
    assert_point_solid(world, &format!("{theme} straight guard"), (239, 1160, 120));
    assert_player_blocked(
        world,
        &format!("{theme} straight stair guard"),
        (239, 1224, 137),
        (239, 1160, 137),
    );
}

fn qualify_open_stairwell(world: &bsp::BspWorld, theme: &str) {
    assert_eq!(OPEN_STEPS.len(), 12, "open stair tread count changed");
    assert_matching_hole(
        world,
        &format!("{theme} open stair paired slab hole"),
        (736, 1248),
        &[168, 200],
        &[(536, 1248, 168), (536, 1248, 200)],
    );
    assert_step_support_and_clearance(world, &format!("{theme} open stair"), &OPEN_STEPS);
    assert_step_transitions(
        world,
        &format!("{theme} open lower flight"),
        &OPEN_STEPS[..6],
    );
    assert_player_clear(
        world,
        &format!("{theme} open stair first landing leg"),
        (751, 1160, 137),
        (792, 1160, 137),
    );
    assert_player_clear(
        world,
        &format!("{theme} open stair protected turn"),
        (792, 1160, 137),
        (792, 1336, 137),
    );
    assert_player_clear(
        world,
        &format!("{theme} open stair turn raise"),
        (792, 1336, 137),
        (792, 1336, 153),
    );
    assert_player_clear(
        world,
        &format!("{theme} open stair turn to upper flight"),
        (792, 1336, 153),
        (753, 1336, 153),
    );
    assert_step_transitions(
        world,
        &format!("{theme} open upper flight"),
        &OPEN_STEPS[6..],
    );
    assert_player_clear(
        world,
        &format!("{theme} open stair upper landing"),
        (593, 1336, 233),
        (552, 1336, 233),
    );
    assert_point_solid(
        world,
        &format!("{theme} open stair guard"),
        (688, 1112, 120),
    );
    assert_player_blocked(
        world,
        &format!("{theme} open stair collision guard"),
        (687, 1160, 105),
        (687, 1112, 105),
    );
}

fn qualify_all_primitives(world: &bsp::BspWorld, theme: &str) {
    qualify_grand_shell(world, theme);
    qualify_balcony(world, theme);
    qualify_catwalk(world, theme);
    qualify_overlook(world, theme);
    qualify_pit(world, theme);
    qualify_ladder(world, theme);
    qualify_spiral(world, theme);
    qualify_drop(world, theme);
    qualify_arena(world, theme);
    qualify_straight_stairwell(world, theme);
    qualify_open_stairwell(world, theme);
}

#[test]
fn fixture_matrix_is_dense_complete_and_legally_textured() {
    for case in THEME_CASES {
        let path = fixture_path(case);
        let map = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("required fixture {}: {error}", path.display()));
        assert_source_contract(case, &map);
    }
}

#[test]
fn all_vertical_primitives_compile_and_strict_reload_in_all_themes() {
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

    for case in THEME_CASES {
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
            reload.faces > 1_000,
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

        let entity_text = String::from_utf8_lossy(&world.entity_raw);
        assert!(
            entity_text.contains(&format!("\"richness_theme\" \"{}\"", case.theme)),
            "{}: theme identity did not survive compilation",
            case.theme
        );
        for required in [
            "\"classname\" \"info_climb_descriptor\"",
            "\"climb_descriptor\" \"1\"",
            "\"climb_normal\" \"1 0 0\"",
            "\"climb_priority\" \"1\"",
            "\"mins\" \"-32 -32 -24\"",
            "\"maxs\" \"32 32 168\"",
            "\"richness_volume\" \"one_way_drop\"",
            "\"one_way\" \"1\"",
        ] {
            assert!(
                entity_text.contains(required),
                "{}: compiler stripped required descriptor cell {required}",
                case.theme
            );
        }

        assert_compiled_texture_closure(&world, case.theme);
        qualify_all_primitives(&world, case.theme);

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
