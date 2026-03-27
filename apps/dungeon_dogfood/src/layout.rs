use std::path::Path;
use thiserror::Error;

/// Tile size in world units (locked convention for all phases)
pub const TILE_SIZE: f32 = 1.0;

/// Convert ASCII tile coordinates to world-space position
///
/// Convention (locked):
/// - ASCII +X -> world +X
/// - ASCII +Y -> world -Z (since world Y is UP)
pub fn tile_to_world(x: usize, y: usize) -> glam::Vec3 {
    glam::Vec3::new(x as f32 * TILE_SIZE, 0.0, -(y as f32 * TILE_SIZE))
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Tile {
    Wall,
    Floor,
    Void,
    RampNorth(u8), // R^ - level 0..N
    RampEast(u8),
    RampSouth(u8),
    RampWest(u8),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub layer: usize,
    pub x: usize,
    pub y: usize,
}

#[derive(Clone, Debug)]
pub struct ParsedLevel {
    pub width: usize,
    pub height: usize,
    pub layers: Vec<Vec<Tile>>, // Each vec is width * height
    pub spawn: TileCoord,
    pub model_markers: Vec<TileCoord>,
    pub light_markers: Vec<TileCoord>,
}

impl ParsedLevel {
    pub fn tile_at(&self, x: usize, y: usize) -> Tile {
        self.tile_at_3d(0, x, y)
    }

    pub fn tile_at_3d(&self, layer: usize, x: usize, y: usize) -> Tile {
        assert!(layer < self.layers.len(), "layer index out of bounds");
        assert!(
            x < self.width && y < self.height,
            "tile coordinates out of bounds"
        );
        self.layers[layer][y * self.width + x]
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("unknown token '{token}' at line {line}, column {column}")]
    UnknownToken {
        line: usize,
        column: usize,
        token: char,
    },
    #[error("incomplete ramp token at line {line}, column {column}")]
    IncompleteRamp { line: usize, column: usize },
    #[error("invalid ramp direction '{token}' at line {line}, column {column}")]
    InvalidRampDir {
        line: usize,
        column: usize,
        token: char,
    },
    #[error("map must be rectangular (expected {expected} tiles, got {actual} on line {line})")]
    NonRectangular {
        line: usize,
        expected: usize,
        actual: usize,
    },
    #[error("empty map")]
    Empty,
    #[error("failed to read level file: {0}")]
    FileRead(#[from] std::io::Error),
    #[error("level must have exactly one spawn marker 'S' (found {count})")]
    SpawnCardinality { count: usize },
    #[error("layer dimensions must match (layer {layer} has {actual_width}x{actual_height}, expected {expected_width}x{expected_height})")]
    LayerDimensionMismatch {
        layer: usize,
        expected_width: usize,
        expected_height: usize,
        actual_width: usize,
        actual_height: usize,
    },
}

/// Parse an ASCII level file
pub fn parse_level(content: &str) -> Result<ParsedLevel, LayoutError> {
    let mut layers_tiles = Vec::new();
    let mut model_markers = Vec::new();
    let mut light_markers = Vec::new();
    let mut spawn_markers = Vec::new();

    let mut expected_width = 0;
    let mut expected_height = 0;

    // Split by layer separator
    let layer_blocks: Vec<&str> = content
        .split("---")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if layer_blocks.is_empty() {
        return Err(LayoutError::Empty);
    }

    for (layer_idx, block) in layer_blocks.iter().enumerate() {
        let mut layer_all_tiles = Vec::new();
        let lines: Vec<&str> = block
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();
        if lines.is_empty() {
            continue;
        }

        let layer_height = lines.len();
        let mut layer_width = 0;

        for (y, line) in lines.iter().enumerate() {
            let (tiles, markers, spawns) = parse_line(line, y)?;
            if y == 0 {
                layer_width = tiles.len();
            } else if tiles.len() != layer_width {
                return Err(LayoutError::NonRectangular {
                    line: y + 1,
                    expected: layer_width,
                    actual: tiles.len(),
                });
            }

            for (x, mtype) in markers {
                match mtype {
                    MarkerType::Model => model_markers.push(TileCoord {
                        layer: layer_idx,
                        x,
                        y,
                    }),
                    MarkerType::Light => light_markers.push(TileCoord {
                        layer: layer_idx,
                        x,
                        y,
                    }),
                }
            }
            for x in spawns {
                spawn_markers.push(TileCoord {
                    layer: layer_idx,
                    x,
                    y,
                });
            }

            layer_all_tiles.extend(tiles);
        }

        if layer_idx == 0 {
            expected_width = layer_width;
            expected_height = layer_height;
        } else if layer_width != expected_width || layer_height != expected_height {
            return Err(LayoutError::LayerDimensionMismatch {
                layer: layer_idx,
                expected_width,
                expected_height,
                actual_width: layer_width,
                actual_height: layer_height,
            });
        }

        layers_tiles.push(layer_all_tiles);
    }

    if spawn_markers.len() != 1 {
        return Err(LayoutError::SpawnCardinality {
            count: spawn_markers.len(),
        });
    }

    Ok(ParsedLevel {
        width: expected_width,
        height: expected_height,
        layers: layers_tiles,
        spawn: spawn_markers[0],
        model_markers,
        light_markers,
    })
}

pub fn load_level_file(path: impl AsRef<Path>) -> Result<ParsedLevel, LayoutError> {
    let content = std::fs::read_to_string(path)?;
    parse_level(&content)
}

#[derive(Debug, Clone, Copy)]
enum MarkerType {
    Model,
    Light,
}

/// Parse a single line, returning (tiles, markers, spawns)
fn parse_line(
    line: &str,
    line_idx: usize,
) -> Result<(Vec<Tile>, Vec<(usize, MarkerType)>, Vec<usize>), LayoutError> {
    let mut tiles = Vec::new();
    let mut markers = Vec::new();
    let mut spawns = Vec::new();

    let chars: Vec<char> = line.chars().collect();
    let mut col = 0;
    let mut tile_x = 0;

    while col < chars.len() {
        let ch = chars[col];

        // Validate ASCII
        if !ch.is_ascii() {
            return Err(LayoutError::UnknownToken {
                line: line_idx + 1,
                column: col + 1,
                token: ch,
            });
        }

        // Parse token
        match ch {
            '#' => {
                tiles.push(Tile::Wall);
                col += 1;
                tile_x += 1;
            }
            '.' => {
                tiles.push(Tile::Floor);
                col += 1;
                tile_x += 1;
            }
            '_' => {
                tiles.push(Tile::Void);
                col += 1;
                tile_x += 1;
            }
            'S' => {
                tiles.push(Tile::Floor);
                spawns.push(tile_x);
                col += 1;
                tile_x += 1;
            }
            'M' => {
                tiles.push(Tile::Floor);
                markers.push((tile_x, MarkerType::Model));
                col += 1;
                tile_x += 1;
            }
            'L' => {
                tiles.push(Tile::Floor);
                markers.push((tile_x, MarkerType::Light));
                col += 1;
                tile_x += 1;
            }
            'R' => {
                // Multi-character ramp token
                if col + 1 >= chars.len() {
                    return Err(LayoutError::IncompleteRamp {
                        line: line_idx + 1,
                        column: col + 1,
                    });
                }

                let mut next_idx = col + 1;
                let level = if chars[next_idx].is_ascii_digit() {
                    let level = chars[next_idx].to_digit(10).unwrap() as u8;
                    next_idx += 1;
                    level
                } else {
                    0
                };

                if next_idx >= chars.len() {
                    return Err(LayoutError::IncompleteRamp {
                        line: line_idx + 1,
                        column: next_idx + 1,
                    });
                }

                let dir = chars[next_idx];
                let tile = match dir {
                    '^' => Tile::RampNorth(level),
                    '>' => Tile::RampEast(level),
                    'v' => Tile::RampSouth(level),
                    '<' => Tile::RampWest(level),
                    _ => {
                        return Err(LayoutError::InvalidRampDir {
                            line: line_idx + 1,
                            column: next_idx + 1,
                            token: dir,
                        });
                    }
                };

                tiles.push(tile);
                col = next_idx + 1; // Skip all chars consumed
                tile_x += 1;
            }
            ' ' | '\t' => {
                // Reject whitespace inside map body for v1
                return Err(LayoutError::UnknownToken {
                    line: line_idx + 1,
                    column: col + 1,
                    token: ch,
                });
            }
            _ => {
                return Err(LayoutError::UnknownToken {
                    line: line_idx + 1,
                    column: col + 1,
                    token: ch,
                });
            }
        }
    }

    Ok((tiles, markers, spawns))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_minimal_level() {
        let input = "####\n#S.#\n####";
        let level = parse_level(input).unwrap();
        assert_eq!(level.width, 4);
        assert_eq!(level.height, 3);
        assert_eq!(
            level.spawn,
            TileCoord {
                layer: 0,
                x: 1,
                y: 1
            }
        );
        assert_eq!(level.tile_at(1, 1), Tile::Floor);
    }

    #[test]
    fn multi_layered_level() {
        let input = "####\n#S.#\n####\n---\n####\n#..#\n####";
        let level = parse_level(input).unwrap();
        assert_eq!(level.layer_count(), 2);
        assert_eq!(level.tile_at_3d(0, 1, 1), Tile::Floor);
        assert_eq!(level.tile_at_3d(1, 1, 1), Tile::Floor);
    }

    #[test]
    fn tile_to_world_conversion() {
        let origin = tile_to_world(1, 1);
        assert_eq!(origin.x, TILE_SIZE);
        assert_eq!(origin.z, -TILE_SIZE);
    }

    #[test]
    fn reject_multiple_spawns() {
        let input = "####\n#S.#\n#S.#\n####";
        let err = parse_level(input).unwrap_err();
        match err {
            LayoutError::SpawnCardinality { count } => {
                assert_eq!(count, 2);
            }
            _ => panic!("expected SpawnCardinality error"),
        }
    }

    #[test]
    fn reject_no_spawn() {
        let input = "####\n#..#\n####";
        let err = parse_level(input).unwrap_err();
        match err {
            LayoutError::SpawnCardinality { count } => {
                assert_eq!(count, 0);
            }
            _ => panic!("expected SpawnCardinality error"),
        }
    }

    #[test]
    fn reject_non_rectangular() {
        let input = "###\n#S..\n###";
        let err = parse_level(input).unwrap_err();
        match err {
            LayoutError::NonRectangular { .. } => {}
            _ => panic!("expected NonRectangular error"),
        }
    }

    #[test]
    fn parse_ramps_all_directions() {
        let input = "#####\n#SR^.#\n#R>Rv.#\n#R<..#\n#####";
        let level = parse_level(input).unwrap();
        assert_eq!(level.width, 5);
        assert_eq!(level.tile_at(2, 1), Tile::RampNorth(0));
        assert_eq!(level.tile_at(1, 2), Tile::RampEast(0));
        assert_eq!(level.tile_at(2, 2), Tile::RampSouth(0));
        assert_eq!(level.tile_at(1, 3), Tile::RampWest(0));
    }

    #[test]
    fn collect_markers() {
        let input = "#####\n#S.M#\n#L..#\n#####";
        let level = parse_level(input).unwrap();
        assert_eq!(level.model_markers.len(), 1);
        assert_eq!(level.light_markers.len(), 1);
        assert_eq!(
            level.model_markers[0],
            TileCoord {
                layer: 0,
                x: 3,
                y: 1
            }
        );
        assert_eq!(
            level.light_markers[0],
            TileCoord {
                layer: 0,
                x: 1,
                y: 2
            }
        );
    }

    #[test]
    fn handle_crlf() {
        let input = "####\r\n#S.#\r\n####";
        let level = parse_level(input).unwrap();
        assert_eq!(level.tile_at(1, 1), Tile::Floor);
    }

    #[test]
    fn handle_leading_trailing_blank_lines() {
        let input = "\n\n####\n#S.#\n####\n\n";
        let level = parse_level(input).unwrap();
        assert_eq!(level.tile_at(1, 1), Tile::Floor);
    }

    #[test]
    fn reject_whitespace_in_map() {
        let input = "###\n#S #\n###";
        let err = parse_level(input).unwrap_err();
        match err {
            LayoutError::UnknownToken { token, .. } => {
                assert_eq!(token, ' ');
            }
            _ => panic!("expected UnknownToken for whitespace"),
        }
    }

    #[test]
    fn level_pack_files_parse_cleanly() {
        parse_level(include_str!("../assets/levels/level_01.txt")).unwrap();
        parse_level(include_str!("../assets/levels/level_02_ramps.txt")).unwrap();
        parse_level(include_str!("../assets/levels/level_03_lighting.txt")).unwrap();
    }

    #[test]
    fn level_02_contains_all_ramp_tokens() {
        let level = parse_level(include_str!("../assets/levels/level_02_ramps.txt"))
            .expect("level_02_ramps should parse");

        assert!(level.layers[0].contains(&Tile::RampNorth(0)));
        assert!(level.layers[0].contains(&Tile::RampEast(0)));
        assert!(level.layers[0].contains(&Tile::RampSouth(0)));
        assert!(level.layers[0].contains(&Tile::RampWest(0)));
    }

    #[test]
    fn level_03_has_dense_light_markers() {
        let level = parse_level(include_str!("../assets/levels/level_03_lighting.txt"))
            .expect("level_03_lighting should parse");

        assert!(level.light_markers.len() >= 10);
    }

    #[test]
    fn level_01_has_intro_markers_reachable_from_spawn() {
        let level = parse_level(include_str!("../assets/levels/level_01.txt")).unwrap();
        let reachable = reachable_tiles(&level);

        assert!(!level.model_markers.is_empty());
        assert!(!level.light_markers.is_empty());
        assert_markers_reachable(&level.model_markers, &reachable);
        assert_markers_reachable(&level.light_markers, &reachable);
    }

    #[test]
    fn level_02_ramps_are_reachable_from_spawn() {
        let level = parse_level(include_str!("../assets/levels/level_02_ramps.txt")).unwrap();
        let reachable = reachable_tiles(&level);

        let mut ramp_tiles = Vec::new();
        for y in 0..level.height {
            for x in 0..level.width {
                let tile = level.tile_at(x, y);
                if matches!(
                    tile,
                    Tile::RampNorth(_) | Tile::RampEast(_) | Tile::RampSouth(_) | Tile::RampWest(_)
                ) {
                    ramp_tiles.push((x, y));
                }
            }
        }

        assert_eq!(ramp_tiles.len(), 4);
        for ramp in ramp_tiles {
            assert!(
                reachable.contains(&(0, ramp.0, ramp.1)),
                "ramp tile at ({}, {}) should be reachable from spawn",
                ramp.0,
                ramp.1
            );
        }
    }

    fn reachable_tiles(level: &ParsedLevel) -> std::collections::HashSet<(usize, usize, usize)> {
        let mut reachable = std::collections::HashSet::new();
        let mut stack = vec![level.spawn];

        while let Some(TileCoord { layer, x, y }) = stack.pop() {
            if !reachable.insert((layer, x, y)) {
                continue;
            }

            let neighbors = [
                (layer, x as isize + 1, y as isize),
                (layer, x as isize - 1, y as isize),
                (layer, x as isize, y as isize + 1),
                (layer, x as isize, y as isize - 1),
            ];

            for (layer, nx, ny) in neighbors {
                if nx >= 0 && ny >= 0 && nx < level.width as isize && ny < level.height as isize {
                    let nx = nx as usize;
                    let ny = ny as usize;
                    if !matches!(level.tile_at_3d(layer, nx, ny), Tile::Wall | Tile::Void) {
                        stack.push(TileCoord { layer, x: nx, y: ny });
                    }
                }
            }
        }

        reachable
    }

    fn assert_markers_reachable(
        markers: &[TileCoord],
        reachable: &std::collections::HashSet<(usize, usize, usize)>,
    ) {
        for marker in markers {
            assert!(
                reachable.contains(&(marker.layer, marker.x, marker.y)),
                "marker at ({}, {}, {}) should be reachable from spawn",
                marker.layer,
                marker.x,
                marker.y
            );
        }
    }

}
