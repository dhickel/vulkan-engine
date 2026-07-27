//! Canonical `.map` serializer producing deterministic, byte-identical output
//! for semantically equivalent [`EmissionIntent`] values.
//!
//! # Serialization grammar
//!
//! The approved grammar is **Standard Quake** offset/rotation/scale per
//! `DECISION-20260726-01`. Every face line uses the canonical form:
//!
//! ```text
//! ( x y z ) ( x y z ) ( x y z ) "texture" 0 0 0 0.25 0.25
//! ```
//!
//! Valve 220 bracket axes (`[ 1 0 0 0 ]`) are **not** implemented. The
//! `BrushFace` IR carries only plane points and texture identity — no
//! `u_axis`/`v_axis` fields exist in the approved IR.
//!
//! The full serialization contract is frozen by `DECISION-20260724-08`:
//!
//! | rule                  | value                                               |
//! |-----------------------|-----------------------------------------------------|
//! | entity order          | worldspawn first, then creation-index               |
//! | key order             | alphabetical (ASCII byte order) per entity          |
//! | brush order           | by creation index                                   |
//! | face order per brush  | bottom, top, north, south, west, east               |
//! | integer formatting    | decimal, no scientific notation                     |
//! | texture axes          | Standard Quake `0 0 0 0.25 0.25` offset/rotation/scale|
//! | line endings          | `\n` (LF)                                           |
//! | terminal newline      | exactly one trailing `\n`                            |

use std::fmt::Write;

use crate::intent::{Brush, BrushFace, EmissionIntent, EntityIntent};

// ── Public API ────────────────────────────────────────────────────────────

/// Serialize an [`EmissionIntent`] to a canonical `.map` string.
///
/// The output is guaranteed to be byte-identical for logically equivalent
/// inputs when the construction pipeline is deterministic.
pub fn serialize(emission: &EmissionIntent) -> String {
    let mut out = String::new();

    // ── worldspawn ────────────────────────────────────────────────────
    out.push_str("{\n");
    emit_key_value(&mut out, "classname", "worldspawn");
    emit_key_value(&mut out, "wad", &emission.wad);
    for brush in &emission.brushes {
        emit_brush(&mut out, brush);
    }
    out.push_str("}\n");

    // ── Non-worldspawn entities ───────────────────────────────────────
    for entity in &emission.entities {
        emit_entity(&mut out, entity);
    }

    // The last `}\n` already provides exactly one terminal newline.
    out
}

// ── Entity emission ───────────────────────────────────────────────────────

fn emit_entity(out: &mut String, entity: &EntityIntent) {
    out.push_str("{\n");

    // Collect all key-value pairs, then sort alphabetically.
    // The dedicated `classname` and `origin` fields are the authoritative
    // values; properties that duplicate them are ignored.
    let mut pairs: Vec<(&str, &str)> = Vec::new();
    pairs.push(("classname", &entity.classname));
    let origin_str = format_origin(entity.origin);
    // We need to own origin_str for the lifetime; push after formatting
    pairs.push(("origin", "")); // placeholder

    for (k, v) in &entity.properties {
        if k != "classname" && k != "origin" {
            pairs.push((k.as_str(), v.as_str()));
        }
    }

    // Sort alphabetically by key (ASCII byte order)
    pairs.sort_by(|a, b| a.0.cmp(b.0));

    for (key, value) in &pairs {
        if *key == "origin" {
            emit_key_value(out, "origin", &origin_str);
        } else {
            emit_key_value(out, key, value);
        }
    }

    for brush in &entity.brushes {
        emit_brush(out, brush);
    }

    out.push_str("}\n");
}

// ── Brush emission ────────────────────────────────────────────────────────

fn emit_brush(out: &mut String, brush: &Brush) {
    out.push_str("{\n");
    for face in &brush.faces {
        emit_face(out, face);
    }
    out.push_str("}\n");
}

fn emit_face(out: &mut String, face: &BrushFace) {
    let (p0, p1, p2) = (
        face.plane_points[0],
        face.plane_points[1],
        face.plane_points[2],
    );

    // Plane points: ( x y z ) ( x y z ) ( x y z )
    write!(
        out,
        "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) ",
        p0.0, p0.1, p0.2, p1.0, p1.1, p1.2, p2.0, p2.1, p2.2,
    )
    .unwrap();

    // Texture name + standard offset/rotation/scale (no explicit axis vectors)
    // Format: "name" x_off y_off rotation x_scale y_scale
    writeln!(out, "\"{}\" 0 0 0 0.25 0.25", face.texture).unwrap();
}

fn emit_key_value(out: &mut String, key: &str, value: &str) {
    out.push('"');
    out.push_str(key);
    out.push_str("\" \"");
    out.push_str(value);
    out.push_str("\"\n");
}

fn format_origin(origin: (i32, i32, i32)) -> String {
    format!("{} {} {}", origin.0, origin.1, origin.2)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent::{BrushFace, EntityIntent};
    use crate::junction;

    fn make_test_emission() -> EmissionIntent {
        let brush = junction::make_brush((0, 0, 0), (64, 64, 128), "test_wall");

        EmissionIntent {
            brushes: vec![brush.clone()],
            entities: vec![
                EntityIntent {
                    classname: "info_player_start".to_string(),
                    origin: (32, 32, 0),
                    properties: vec![
                        ("classname".to_string(), "info_player_start".to_string()),
                        ("origin".to_string(), "32 32 0".to_string()),
                    ],
                    brushes: Vec::new(),
                },
                EntityIntent {
                    classname: "light".to_string(),
                    origin: (32, 32, 64),
                    properties: vec![
                        ("classname".to_string(), "light".to_string()),
                        ("origin".to_string(), "32 32 64".to_string()),
                        ("light".to_string(), "300".to_string()),
                    ],
                    brushes: Vec::new(),
                },
            ],
            wad: "test.wad".to_string(),
        }
    }

    // ── Formatting ────────────────────────────────────────────────────

    #[test]
    fn output_starts_with_worldspawn() {
        let emission = make_test_emission();
        let s = serialize(&emission);
        assert!(
            s.starts_with("{\n\"classname\" \"worldspawn\"\n"),
            "got: {:?}",
            s
        );
    }

    #[test]
    fn output_ends_with_terminal_newline() {
        let emission = make_test_emission();
        let s = serialize(&emission);
        assert!(
            s.ends_with('\n'),
            "must end with exactly one terminal newline"
        );
        // Only one trailing \n (the last `}\n` should be the only one)
        assert!(
            !s.ends_with("\n\n"),
            "must not have double trailing newline"
        );
    }

    #[test]
    fn contains_wad_reference() {
        let emission = make_test_emission();
        let s = serialize(&emission);
        assert!(s.contains("\"wad\" \"test.wad\""));
    }

    #[test]
    fn all_line_endings_are_lf() {
        let emission = make_test_emission();
        let s = serialize(&emission);
        assert!(!s.contains('\r'), "must not contain CR");
        assert!(s.contains('\n'), "must contain LF");
    }

    #[test]
    fn integer_format_is_decimal_no_scientific() {
        let emission = make_test_emission();
        let s = serialize(&emission);
        // No scientific notation: check that coordinate numbers are pure
        // integers with no decimal point.  Texture names may contain 'e'.
        for line in s.lines() {
            if line.contains('(') {
                // Extract all numeric tokens from parentheses
                for chunk in line.split('(') {
                    if let Some(inner) = chunk.split(')').next() {
                        for token in inner.split_whitespace() {
                            // Numeric token: must not contain '.' (float) or
                            // be in scientific form (digit followed by 'e'/'E').
                            if token.chars().any(|c| c.is_ascii_digit()) {
                                assert!(
                                    !token.contains('.'),
                                    "float in coordinate: {} in line: {}",
                                    token,
                                    line
                                );
                                // Check for scientific: digit then 'e'/'E'
                                let bytes = token.as_bytes();
                                for i in 0..bytes.len().saturating_sub(1) {
                                    if bytes[i].is_ascii_digit()
                                        && (bytes[i + 1] == b'e' || bytes[i + 1] == b'E')
                                    {
                                        panic!(
                                            "scientific notation in: {} (token: {})",
                                            line, token
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn texture_axes_use_standard_format() {
        let emission = make_test_emission();
        let s = serialize(&emission);
        // Every face line should contain standard offset/rotation/scale pattern
        assert!(
            s.contains("0 0 0 0.25 0.25"),
            "missing texture coordinate format"
        );
        // Verify no bracket format remains
        let face_lines: Vec<&str> = s.lines().filter(|l| l.contains('[')).collect();
        assert!(face_lines.is_empty(), "no bracket format expected");
    }

    // ── Entity ordering ───────────────────────────────────────────────

    #[test]
    fn worldspawn_is_first_entity() {
        let emission = make_test_emission();
        let s = serialize(&emission);
        let first_entity_start = s.find("{\n\"classname\"").unwrap();
        let after_worldspawn: &str = &s[first_entity_start..];
        assert!(
            after_worldspawn.starts_with("{\n\"classname\" \"worldspawn\""),
            "first entity must be worldspawn"
        );
    }

    #[test]
    fn entity_order_is_worldspawn_then_spawn_then_light() {
        let emission = make_test_emission();
        let s = serialize(&emission);
        // Find entity blocks
        let worldspawn_pos = s.find("\"classname\" \"worldspawn\"").unwrap();
        let spawn_pos = s.find("\"classname\" \"info_player_start\"").unwrap();
        let light_pos = s.find("\"classname\" \"light\"").unwrap();
        assert!(worldspawn_pos < spawn_pos);
        assert!(spawn_pos < light_pos);
    }

    // ── Key ordering ─────────────────────────────────────────────────

    #[test]
    fn light_entity_keys_are_alphabetical() {
        let emission = make_test_emission();
        let s = serialize(&emission);
        // Find the light entity block
        let light_start = s.find("\"classname\" \"light\"").unwrap();
        let after = &s[light_start..];
        // Keys should appear in alphabetical order: classname, light, origin
        let classname_pos = after.find("\"classname\"").unwrap();
        let light_pos = after.find("\"light\"").unwrap();
        let origin_pos = after.find("\"origin\"").unwrap();
        assert!(classname_pos < light_pos, "classname before light");
        assert!(light_pos < origin_pos, "light before origin");
    }

    #[test]
    fn worldspawn_keys_are_alphabetical() {
        let emission = make_test_emission();
        let s = serialize(&emission);
        // worldspawn: "classname" before "wad"
        let class_pos = s.find("\"classname\" \"worldspawn\"").unwrap();
        let wad_pos = s.find("\"wad\"").unwrap();
        assert!(class_pos < wad_pos, "classname before wad");
    }

    // ── Brush ordering ────────────────────────────────────────────────

    #[test]
    fn multiple_brushes_preserve_creation_order() {
        let b0 = junction::make_brush((0, 0, 0), (16, 16, 16), "first");
        let b1 = junction::make_brush((32, 32, 0), (48, 48, 16), "second");
        let b2 = junction::make_brush((64, 64, 0), (80, 80, 16), "third");

        let emission = EmissionIntent {
            brushes: vec![b0.clone(), b1.clone(), b2.clone()],
            entities: Vec::new(),
            wad: "test.wad".to_string(),
        };
        let s = serialize(&emission);

        let first_pos = s.find("\"first\"").unwrap();
        let second_pos = s.find("\"second\"").unwrap();
        let third_pos = s.find("\"third\"").unwrap();
        assert!(first_pos < second_pos);
        assert!(second_pos < third_pos);
    }

    // ── Face ordering ─────────────────────────────────────────────────

    #[test]
    fn brush_faces_are_emitted_in_creation_order() {
        // Build a brush with explicit faces in known order.
        // Per DECISION-20260726-01, BrushFace has no u_axis/v_axis fields.
        let brush = Brush {
            faces: vec![
                BrushFace {
                    plane_points: [(0, 0, 0), (64, 0, 0), (0, 64, 0)],
                    texture: "bottom".to_string(),
                },
                BrushFace {
                    plane_points: [(0, 64, 128), (64, 64, 128), (0, 0, 128)],
                    texture: "top".to_string(),
                },
                BrushFace {
                    plane_points: [(0, 64, 0), (64, 64, 0), (0, 64, 128)],
                    texture: "north".to_string(),
                },
                BrushFace {
                    plane_points: [(0, 0, 128), (64, 0, 128), (0, 0, 0)],
                    texture: "south".to_string(),
                },
                BrushFace {
                    plane_points: [(0, 0, 128), (0, 64, 128), (0, 0, 0)],
                    texture: "west".to_string(),
                },
                BrushFace {
                    plane_points: [(64, 0, 0), (64, 64, 0), (64, 0, 128)],
                    texture: "east".to_string(),
                },
            ],
        };

        let emission = EmissionIntent {
            brushes: vec![brush],
            entities: Vec::new(),
            wad: "test.wad".to_string(),
        };
        let s = serialize(&emission);

        let bottom_pos = s.find("\"bottom\"").unwrap();
        let top_pos = s.find("\"top\"").unwrap();
        let north_pos = s.find("\"north\"").unwrap();
        let south_pos = s.find("\"south\"").unwrap();
        let west_pos = s.find("\"west\"").unwrap();
        let east_pos = s.find("\"east\"").unwrap();

        assert!(bottom_pos < top_pos);
        assert!(top_pos < north_pos);
        assert!(north_pos < south_pos);
        assert!(south_pos < west_pos);
        assert!(west_pos < east_pos);
    }

    // ── Deterministic output ──────────────────────────────────────────

    #[test]
    fn same_input_produces_identical_output() {
        let emission = make_test_emission();
        let a = serialize(&emission);
        let b = serialize(&emission);
        assert_eq!(a, b);
    }

    #[test]
    fn output_is_valid_utf8() {
        let emission = make_test_emission();
        let s = serialize(&emission);
        // String is already guaranteed UTF-8 in Rust; verify no null bytes
        assert!(!s.contains('\0'));
    }

    // ── Empty emission ────────────────────────────────────────────────

    #[test]
    fn empty_emission_produces_minimal_valid_map() {
        let emission = EmissionIntent {
            brushes: Vec::new(),
            entities: Vec::new(),
            wad: "empty.wad".to_string(),
        };
        let s = serialize(&emission);
        // Should contain worldspawn entity with wad reference
        assert!(s.contains("worldspawn"));
        assert!(s.contains("empty.wad"));
        assert!(s.ends_with('\n'));
    }
}
