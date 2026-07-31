//! Phase 05 — Enhanced V3 CLI integration tests.
//!
//! Validates that `dungeon_gen --class m3` produces valid .map output
//! through the binary. Uses `std::process::Command` to invoke the binary
//! via `cargo run`.

use std::process::Command;

const BINARY: &str = env!("CARGO_BIN_EXE_dungeon_gen");

#[test]
fn cli_m3_produces_valid_map() {
    let output = Command::new(BINARY)
        .args(["--class", "m3", "--seed", "42", "--out", "/dev/stdout"])
        .output()
        .expect("failed to run dungeon_gen");

    assert!(
        output.status.success(),
        "dungeon_gen exited with error: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let map = String::from_utf8_lossy(&output.stdout);
    assert!(!map.is_empty());
    assert!(map.contains("worldspawn"));
    assert!(map.contains("info_player_start"));
    assert!(map.contains("light"));
    assert!(map.contains("cc0_dungeon_v2.wad"));
}

#[test]
fn cli_m3_with_different_seeds_both_valid() {
    let output_a = Command::new(BINARY)
        .args(["--class", "m3", "--seed", "0", "--out", "/dev/stdout"])
        .output()
        .expect("failed to run dungeon_gen seed 0");
    assert!(output_a.status.success());

    let output_b = Command::new(BINARY)
        .args(["--class", "m3", "--seed", "42", "--out", "/dev/stdout"])
        .output()
        .expect("failed to run dungeon_gen seed 42");
    assert!(output_b.status.success());

    let map_a = String::from_utf8_lossy(&output_a.stdout);
    let map_b = String::from_utf8_lossy(&output_b.stdout);
    assert!(!map_a.is_empty());
    assert!(!map_b.is_empty());

    // Both produce valid Quake .map structure
    for map in &[&map_a, &map_b] {
        assert!(map.contains("worldspawn"));
        assert!(map.contains("info_player_start"));
        assert!(map.contains("light"));
        let open = map.matches('{').count();
        let close = map.matches('}').count();
        assert_eq!(open, close, "mismatched braces");
    }
}

#[test]
fn cli_m3_deterministic() {
    let run = |seed: u64| -> Vec<u8> {
        let output = Command::new(BINARY)
            .args([
                "--class",
                "m3",
                "--seed",
                &seed.to_string(),
                "--out",
                "/dev/stdout",
            ])
            .output()
            .unwrap();
        assert!(output.status.success());
        output.stdout
    };

    let run1 = run(42);
    let run2 = run(42);
    assert_eq!(run1, run2, "same seed should produce identical output");
}

#[test]
fn cli_m3_balanced_braces() {
    let output = Command::new(BINARY)
        .args(["--class", "m3", "--seed", "42", "--out", "/dev/stdout"])
        .output()
        .expect("failed to run dungeon_gen");

    assert!(output.status.success());
    let map = String::from_utf8_lossy(&output.stdout);
    let open = map.matches('{').count();
    let close = map.matches('}').count();
    assert_eq!(open, close, "mismatched braces");
    assert!(open > 2, "expected multiple brush blocks");
}

#[test]
fn cli_m3_map_has_worldspawn_first() {
    let output = Command::new(BINARY)
        .args(["--class", "m3", "--seed", "42", "--out", "/dev/stdout"])
        .output()
        .expect("failed to run dungeon_gen");

    assert!(output.status.success());
    let map = String::from_utf8_lossy(&output.stdout);
    let worldspawn_pos = map.find("worldspawn").unwrap();
    let spawn_pos = map.find("info_player_start").unwrap();
    assert!(
        worldspawn_pos < spawn_pos,
        "worldspawn must precede spawn entity"
    );
}

#[test]
fn cli_m3_map_has_approved_textures() {
    let output = Command::new(BINARY)
        .args(["--class", "m3", "--seed", "42", "--out", "/dev/stdout"])
        .output()
        .expect("failed to run dungeon_gen");

    assert!(output.status.success());
    let map = String::from_utf8_lossy(&output.stdout);
    assert!(map.contains("bs_wall"));
    assert!(map.contains("bs_floor"));
    assert!(map.contains("bs_ceil"));
}

#[test]
fn cli_m3_help_mentions_m3() {
    let output = Command::new(BINARY)
        .args(["--help"])
        .output()
        .expect("failed to run dungeon_gen --help");

    // --help exits with 0 but writes to stderr
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("m3"),
        "help should mention m3 class: {stderr}"
    );
}

#[test]
fn cli_legacy_m1_still_works() {
    let output = Command::new(BINARY)
        .args(["--class", "m1", "--seed", "0", "--out", "/dev/stdout"])
        .output()
        .expect("failed to run dungeon_gen m1");

    assert!(output.status.success());
    let map = String::from_utf8_lossy(&output.stdout);
    assert!(!map.is_empty());
    assert!(map.contains("worldspawn"));
}

#[test]
fn cli_enhanced_m2_still_works() {
    let output = Command::new(BINARY)
        .args(["--class", "m2", "--seed", "0", "--out", "/dev/stdout"])
        .output()
        .expect("failed to run dungeon_gen m2");

    assert!(output.status.success());
    let map = String::from_utf8_lossy(&output.stdout);
    assert!(!map.is_empty());
    assert!(map.contains("worldspawn"));
}

#[test]
fn cli_rejects_unknown_class() {
    let output = Command::new(BINARY)
        .args(["--class", "v3", "--seed", "0", "--out", "/dev/stdout"])
        .output()
        .expect("failed to run dungeon_gen v3");

    assert!(!output.status.success(), "v3 class should be rejected");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("v3"), "error should mention v3: {stderr}");
}
