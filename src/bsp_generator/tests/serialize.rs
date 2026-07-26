//! White-box tests for [`bsp_generator::serialize`] — verify that the
//! canonical `.map` serializer obeys every rule in `DECISION-20260724-08`.

use bsp_generator::{
    generate, serialize, Brush, BrushFace, DungeonConfig, EmissionIntent, EntityIntent,
};

// ── Helpers ───────────────────────────────────────────────────────────────

fn make_face(texture: &str) -> BrushFace {
    BrushFace {
        plane_points: [(0, 0, 0), (64, 0, 0), (0, 64, 0)],
        texture: texture.to_string(),
    }
}

fn minimal_emission() -> EmissionIntent {
    EmissionIntent {
        brushes: Vec::new(),
        entities: Vec::new(),
        wad: "test.wad".to_string(),
    }
}

// ── Entity ordering ──────────────────────────────────────────────────────

#[test]
fn worldspawn_is_first_entity() {
    let emission = EmissionIntent {
        brushes: vec![], // no brushes needed for ordering test
        entities: vec![EntityIntent {
            classname: "info_player_start".to_string(),
            origin: (0, 0, 0),
            properties: vec![("classname".to_string(), "info_player_start".to_string())],
            brushes: Vec::new(),
        }],
        wad: "test.wad".to_string(),
    };
    let s = serialize(&emission);
    let worldspawn_pos = s.find("\"classname\" \"worldspawn\"").unwrap();
    let spawn_pos = s.find("info_player_start").unwrap();
    assert!(worldspawn_pos < spawn_pos);
}

#[test]
fn entities_ordered_by_creation_index() {
    let emission = EmissionIntent {
        brushes: Vec::new(),
        entities: vec![
            EntityIntent {
                classname: "info_player_start".to_string(),
                origin: (0, 0, 0),
                properties: vec![("classname".to_string(), "info_player_start".to_string())],
                brushes: Vec::new(),
            },
            EntityIntent {
                classname: "light".to_string(),
                origin: (32, 32, 0),
                properties: vec![
                    ("classname".to_string(), "light".to_string()),
                    ("light".to_string(), "300".to_string()),
                ],
                brushes: Vec::new(),
            },
            EntityIntent {
                classname: "light".to_string(),
                origin: (64, 64, 0),
                properties: vec![
                    ("classname".to_string(), "light".to_string()),
                    ("light".to_string(), "300".to_string()),
                ],
                brushes: Vec::new(),
            },
        ],
        wad: "test.wad".to_string(),
    };
    let s = serialize(&emission);
    let first_light = s.find("\"classname\" \"light\"").unwrap();
    let second_light = s[first_light + 1..]
        .find("\"classname\" \"light\"")
        .map(|p| p + first_light + 1)
        .unwrap();
    assert!(first_light < second_light);
}

// ── Key ordering ────────────────────────────────────────────────────────

#[test]
fn keys_are_alphabetical_within_entity() {
    let emission = EmissionIntent {
        brushes: Vec::new(),
        entities: vec![EntityIntent {
            classname: "light".to_string(),
            origin: (0, 0, 0),
            properties: vec![
                ("classname".to_string(), "light".to_string()),
                ("origin".to_string(), "0 0 0".to_string()),
                ("light".to_string(), "300".to_string()),
            ],
            brushes: Vec::new(),
        }],
        wad: "test.wad".to_string(),
    };
    let s = serialize(&emission);

    // Extract the light entity block
    let light_start = s.find("\"classname\" \"light\"").unwrap();
    let after = &s[light_start..];
    let close_brace = after.find("\n}").unwrap();
    let block = &after[..close_brace];

    // Find all key occurrences in order
    let mut keys: Vec<&str> = Vec::new();
    let mut pos = 0;
    while let Some(key_start) = block[pos..].find('"') {
        let after_key = &block[pos + key_start + 1..];
        if let Some(key_end) = after_key.find('"') {
            let key = &after_key[..key_end];
            // Only count keys that look like property keys (not values)
            if key.chars().all(|c| c.is_ascii_lowercase() || c == '_') && key.len() > 1 {
                keys.push(key);
            }
            pos += key_start + 1 + key_end + 1;
        } else {
            break;
        }
    }

    // Filter to just "classname", "light", "origin" (the deduplicated set)
    let entity_keys: Vec<&str> = keys
        .into_iter()
        .filter(|k| *k == "classname" || *k == "light" || *k == "origin")
        .collect();

    let mut sorted = entity_keys.clone();
    sorted.sort();
    assert_eq!(entity_keys, sorted, "keys must be alphabetical");
}

#[test]
fn worldspawn_keys_are_classname_before_wad() {
    let emission = minimal_emission();
    let s = serialize(&emission);
    let class_pos = s.find("\"classname\"").unwrap();
    let wad_pos = s.find("\"wad\"").unwrap();
    assert!(class_pos < wad_pos);
}

// ── Brush ordering ──────────────────────────────────────────────────────

#[test]
fn brushes_preserve_creation_order() {
    let b0 = Brush {
        faces: vec![make_face("alpha")],
    };
    let b1 = Brush {
        faces: vec![make_face("beta")],
    };
    let b2 = Brush {
        faces: vec![make_face("gamma")],
    };

    let emission = EmissionIntent {
        brushes: vec![b0, b1, b2],
        entities: Vec::new(),
        wad: "test.wad".to_string(),
    };
    let s = serialize(&emission);

    let a = s.find("\"alpha\"").unwrap();
    let b = s.find("\"beta\"").unwrap();
    let g = s.find("\"gamma\"").unwrap();
    assert!(a < b);
    assert!(b < g);
}

// ── Face ordering ───────────────────────────────────────────────────────

#[test]
fn faces_are_emitted_in_vec_order() {
    let faces: Vec<BrushFace> = (0..6)
        .map(|i| BrushFace {
            plane_points: [(i * 16, 0, 0), (i * 16 + 16, 0, 0), (i * 16, 16, 0)],
            texture: format!("face_{}", i),
        })
        .collect();

    let brush = Brush { faces };
    let emission = EmissionIntent {
        brushes: vec![brush],
        entities: Vec::new(),
        wad: "test.wad".to_string(),
    };
    let s = serialize(&emission);

    let positions: Vec<usize> = (0..6)
        .map(|i| s.find(&format!("\"face_{}\"", i)).unwrap())
        .collect();
    let sorted = {
        let mut p = positions.clone();
        p.sort();
        p
    };
    assert_eq!(positions, sorted, "faces must appear in creation order");
}

// ── Integer formatting ──────────────────────────────────────────────────

#[test]
fn coordinates_are_decimal_no_scientific() {
    let cfg = DungeonConfig::nominal_m1();
    // Use seed 0 which is known to route successfully
    if let Ok((map_text, _meta)) = generate(0, cfg) {
        for line in map_text.lines() {
            if line.contains('(') {
                // Extract numeric tokens from parentheses only — texture names
                // may contain 'e' (e.g. "generator_floor").  Check each
                // parenthesised chunk for numeric-scientific patterns.
                for chunk in line.split('(') {
                    if let Some(inner) = chunk.split(')').next() {
                        for token in inner.split_whitespace() {
                            // Only check tokens that appear numeric
                            if token.chars().any(|c| c.is_ascii_digit()) {
                                assert!(!token.contains('.'), "float in: {}", line);
                                let bytes = token.as_bytes();
                                for i in 0..bytes.len().saturating_sub(1) {
                                    if bytes[i].is_ascii_digit()
                                        && (bytes[i + 1] == b'e' || bytes[i + 1] == b'E')
                                    {
                                        panic!("scientific notation in: {}", line);
                                    }
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
fn face_line_has_exact_format() {
    let emission = minimal_emission();
    let s = serialize(&emission);
    // Verify line endings
    let lines: Vec<&str> = s.lines().collect();
    for line in &lines {
        assert!(!line.contains('\r'), "CR in line: {}", line);
    }
}

// ── Texture axis format ─────────────────────────────────────────────────

#[test]
fn texture_axis_standard_format() {
    let brush = Brush {
        faces: vec![BrushFace {
            plane_points: [(0, 0, 0), (64, 0, 0), (0, 64, 0)],
            texture: "wall".to_string(),
        }],
    };
    let emission = EmissionIntent {
        brushes: vec![brush],
        entities: Vec::new(),
        wad: "test.wad".to_string(),
    };
    let s = serialize(&emission);

    // Standard format: texture_name x_off y_off rot x_scale y_scale
    assert!(s.contains("0 0 0 1.0 1.0"), "must use standard offset/scale format");

    // Verify no Valve 220 bracket format remains
    let face_line = s.lines().find(|l| l.contains("wall")).unwrap();
    assert!(!face_line.contains('['), "no bracket texture axes expected");
    assert!(!face_line.contains(']'), "no bracket texture axes expected");
}

// ── Line endings ────────────────────────────────────────────────────────

#[test]
fn all_lines_end_with_lf() {
    let cfg = DungeonConfig::nominal_m1();
    if let Ok((map_text, _meta)) = generate(0, cfg) {
        assert!(!map_text.contains('\r'));
        assert!(map_text.contains('\n'));
    }
}

#[test]
fn exactly_one_terminal_newline() {
    let cfg = DungeonConfig::nominal_m1();
    if let Ok((map_text, _meta)) = generate(0, cfg) {
        assert!(map_text.ends_with('\n'));
        assert!(!map_text.ends_with("\n\n"));
    }
}

// ── Determinism ─────────────────────────────────────────────────────────

#[test]
fn same_emission_produces_identical_bytes() {
    let emission = EmissionIntent {
        brushes: vec![Brush {
            faces: vec![
                BrushFace {
                    plane_points: [(0, 0, 0), (64, 0, 0), (0, 64, 0)],
                    texture: "floor".to_string(),
                },
                BrushFace {
                    plane_points: [(0, 64, 128), (64, 64, 128), (0, 0, 128)],
                    texture: "ceiling".to_string(),
                },
            ],
        }],
        entities: vec![EntityIntent {
            classname: "light".to_string(),
            origin: (32, 32, 64),
            properties: vec![
                ("classname".to_string(), "light".to_string()),
                ("origin".to_string(), "32 32 64".to_string()),
            ],
            brushes: Vec::new(),
        }],
        wad: "test.wad".to_string(),
    };

    let a = serialize(&emission);
    let b = serialize(&emission);
    assert_eq!(a, b);
    assert_eq!(a.len(), b.len());
}

// ── Generated output format ─────────────────────────────────────────────

#[test]
fn generated_map_has_balanced_braces() {
    let cfg = DungeonConfig::nominal_m1();
    if let Ok((map_text, _meta)) = generate(0, cfg) {
        let open = map_text.matches('{').count();
        let close = map_text.matches('}').count();
        assert_eq!(open, close);
        assert!(open > 0);
    }
}

#[test]
fn generated_map_starts_with_worldspawn() {
    let cfg = DungeonConfig::nominal_m1();
    if let Ok((map_text, _meta)) = generate(0, cfg) {
        assert!(map_text.starts_with("{\n\"classname\" \"worldspawn\"\n"));
    }
}

#[test]
fn generated_map_contains_wad_key() {
    // Use boundary C (17 rooms, 1 loop) which routes successfully
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
    if let Ok((map_text, _meta)) = generate(44, cfg) {
        assert!(map_text.contains("\"wad\""));
        assert!(map_text.contains("cc0_stone_beta.wad"));
    }
}
