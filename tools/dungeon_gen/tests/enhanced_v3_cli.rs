//! Enhanced V3 CLI production integration tests.
//!
//! These invoke the released `dungeon_gen` binary, rather than testing the
//! library directly, so m3 argument parsing, defaults, summaries, and output
//! bytes remain part of the production contract.

use std::collections::BTreeSet;
use std::process::{Command, Output};

const BINARY: &str = env!("CARGO_BIN_EXE_dungeon_gen");

#[derive(Debug)]
struct M3Run {
    map: Vec<u8>,
    stderr: String,
}

fn invoke(args: &[String]) -> Output {
    Command::new(BINARY)
        .args(args)
        .output()
        .expect("run dungeon_gen")
}

fn run_m3(seed: u64, preset: &str, extent: Option<u32>) -> M3Run {
    let mut args = vec![
        "--class".into(),
        "m3".into(),
        "--seed".into(),
        seed.to_string(),
        "--preset".into(),
        preset.into(),
    ];
    if let Some(extent) = extent {
        args.extend(["--extent".into(), extent.to_string()]);
    }
    args.extend(["--out".into(), "/dev/stdout".into()]);
    let output = invoke(&args);
    assert!(
        output.status.success(),
        "m3 seed={seed} preset={preset} extent={extent:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    M3Run {
        map: output.stdout,
        stderr: String::from_utf8(output.stderr).expect("UTF-8 stderr"),
    }
}

fn run_m3_default(seed: u64, preset: Option<&str>) -> M3Run {
    let mut args = vec![
        "--class".into(),
        "m3".into(),
        "--seed".into(),
        seed.to_string(),
    ];
    if let Some(preset) = preset {
        args.extend(["--preset".into(), preset.into()]);
    }
    args.extend(["--out".into(), "/dev/stdout".into()]);
    let output = invoke(&args);
    assert!(
        output.status.success(),
        "m3 default seed={seed} preset={preset:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    M3Run {
        map: output.stdout,
        stderr: String::from_utf8(output.stderr).expect("UTF-8 stderr"),
    }
}

fn summary_value(stderr: &str, key: &str) -> u32 {
    stderr
        .split_whitespace()
        .find_map(|word| word.strip_prefix(&format!("{key}=")))
        .and_then(|value| value.trim_end_matches(',').parse().ok())
        .unwrap_or_else(|| panic!("missing {key} summary in stderr: {stderr}"))
}

fn assert_valid_map(label: &str, map: &[u8]) {
    let map = std::str::from_utf8(map).unwrap_or_else(|_| panic!("{label}: map is not UTF-8"));
    assert!(!map.is_empty(), "{label}: empty map");
    assert!(map.contains("worldspawn"), "{label}: no worldspawn");
    assert!(map.contains("info_player_start"), "{label}: no spawn");
    assert!(
        map.contains("\"classname\" \"light\""),
        "{label}: no lights"
    );
    assert!(
        map.contains("cc0_dungeon_v2.wad"),
        "{label}: no WAD reference"
    );
    assert!(map.contains("_minlight"), "{label}: no _minlight");
    assert!(map.contains("bs_wall"), "{label}: no bs_wall");
    assert!(map.contains("bs_floor"), "{label}: no bs_floor");
    assert!(map.contains("bs_ceil"), "{label}: no bs_ceil");
    assert_eq!(
        map.matches('{').count(),
        map.matches('}').count(),
        "{label}: mismatched braces"
    );
    assert!(map.ends_with('\n'), "{label}: no trailing newline");
    assert!(!map.contains('\r'), "{label}: contains CR bytes");
}

fn assert_m3_contract(run: &M3Run, label: &str, rooms: u32, routes: u32) {
    assert_valid_map(label, &run.map);
    assert_eq!(summary_value(&run.stderr, "rooms"), rooms, "{label}");
    assert_eq!(summary_value(&run.stderr, "corridors"), routes, "{label}");
}

fn expected(preset: &str) -> (u32, u32, u32) {
    match preset {
        "sparse" => (12, 10, 2048),
        "moderate" => (20, 20, 2048),
        "rich" => (28, 30, 3072),
        _ => unreachable!("test preset"),
    }
}

#[test]
fn cli_m3_default_is_sparse_at_2048() {
    let default = run_m3_default(42, None);
    let explicit = run_m3(42, "sparse", Some(2048));
    assert_m3_contract(&default, "m3 default", 12, 10);
    assert_eq!(default.map, explicit.map, "default must be Sparse@2048");
}

#[test]
fn cli_m3_rich_default_extent_is_3072() {
    let default = run_m3_default(42, Some("rich"));
    let explicit = run_m3(42, "rich", Some(3072));
    assert_m3_contract(&default, "m3 rich default", 28, 30);
    assert_eq!(default.map, explicit.map, "Rich default must be 3072");
}

#[test]
fn cli_m3_seed_preset_matrix_has_exact_topology_counts() {
    for preset in ["sparse", "moderate", "rich"] {
        let (rooms, routes, extent) = expected(preset);
        for seed in [0, 42, 99, 255] {
            let run = run_m3(seed, preset, Some(extent));
            assert_m3_contract(&run, &format!("{preset}/{seed}"), rooms, routes);
        }
    }
}

#[test]
fn cli_m3_seed_preset_matrix_replays_byte_identically() {
    for preset in ["sparse", "moderate", "rich"] {
        let (_, _, extent) = expected(preset);
        for seed in [0, 42, 99, 255] {
            let first = run_m3(seed, preset, Some(extent));
            let replay = run_m3(seed, preset, Some(extent));
            assert_eq!(first.map, replay.map, "{preset}/{seed}: map replay drift");
            assert_eq!(
                first.stderr, replay.stderr,
                "{preset}/{seed}: summary replay drift"
            );
        }
    }
}

#[test]
fn cli_m3_seed_preset_matrix_has_distinct_seed_outputs() {
    for preset in ["sparse", "moderate", "rich"] {
        let (_, _, extent) = expected(preset);
        let maps: BTreeSet<_> = [0, 42, 99, 255]
            .into_iter()
            .map(|seed| run_m3(seed, preset, Some(extent)).map)
            .collect();
        assert_eq!(
            maps.len(),
            4,
            "{preset}: required seed outputs must all differ"
        );
    }
}

#[test]
fn cli_m3_rejects_invalid_preset_and_extent() {
    let bad_preset = invoke(&[
        "--class".into(),
        "m3".into(),
        "--preset".into(),
        "dense".into(),
    ]);
    assert!(!bad_preset.status.success());
    assert!(String::from_utf8_lossy(&bad_preset.stderr).contains("dense"));

    for extent in ["2047", "512", "4096"] {
        let output = invoke(&[
            "--class".into(),
            "m3".into(),
            "--extent".into(),
            extent.into(),
        ]);
        assert!(!output.status.success(), "m3 accepted extent {extent}");
    }
}

#[test]
fn cli_m1_m2_reject_m3_only_flags() {
    for class in ["m1", "m2"] {
        for (flag, value) in [("--preset", "sparse"), ("--extent", "2048")] {
            let output = invoke(&["--class".into(), class.into(), flag.into(), value.into()]);
            assert!(!output.status.success(), "{class} accepted m3-only {flag}");
        }
    }
}

#[test]
fn cli_v1_v2_remain_available() {
    for class in ["m1", "m2"] {
        let output = invoke(&[
            "--class".into(),
            class.into(),
            "--seed".into(),
            "0".into(),
            "--out".into(),
            "/dev/stdout".into(),
        ]);
        assert!(
            output.status.success(),
            "{class}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("worldspawn"));
    }
}

#[test]
fn cli_help_and_unknown_class_report_the_production_interface() {
    let help = invoke(&["--help".into()]);
    assert!(help.status.success());
    let help = String::from_utf8_lossy(&help.stderr);
    for required in ["m1", "m2", "m3", "--preset", "--extent"] {
        assert!(help.contains(required), "help lacks {required}: {help}");
    }

    let unknown = invoke(&["--class".into(), "v3".into()]);
    assert!(!unknown.status.success());
}
