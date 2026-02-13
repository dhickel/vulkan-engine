use std::path::Path;
use thiserror::Error;

/// Tile size in world units (locked convention for all phases)
pub const TILE_SIZE: f32 = 1.0;

/// Convert ASCII tile coordinates to world-space position
///
/// Convention (locked):
/// - ASCII +X -> world +X
/// - ASCII +Y (down rows) -> world -Z
/// - world +Y is up
pub fn tile_to_world(x: usize, y: usize) -> glam::Vec3 {
    glam::Vec3::new(x as f32 * TILE_SIZE, 0.0, -(y as f32) * TILE_SIZE)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Tile {
    Wall,
    Floor,
    RampNorth, // R^ - ascending northward (up in ASCII)
    RampEast,  // R> - ascending eastward (right)
    RampSouth, // Rv - ascending southward (down in ASCII)
    RampWest,  // R< - ascending westward (left)
}

#[derive(Clone, Debug)]
pub struct ParsedLevel {
    pub width: usize,
    pub height: usize,
    pub tiles: Vec<Tile>, // Row-major: tiles[y * width + x]
    pub spawn: (usize, usize),
    pub model_markers: Vec<(usize, usize)>,
    pub light_markers: Vec<(usize, usize)>,
}

impl ParsedLevel {
    pub fn tile_at(&self, x: usize, y: usize) -> Tile {
        assert!(x < self.width && y < self.height, "tile coordinates out of bounds");
        self.tiles[y * self.width + x]
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
    #[error("non-rectangular map: line {line} has width {actual}, expected {expected}")]
    NonRectangular {
        line: usize,
        expected: usize,
        actual: usize,
    },
    #[error("expected exactly one spawn marker, found {count}")]
    SpawnCardinality { count: usize },
    #[error("map is empty")]
    EmptyMap,
    #[error("failed to read level file: {0}")]
    FileRead(#[from] std::io::Error),
}

/// Parse ASCII level from string
pub fn parse_level(input: &str) -> Result<ParsedLevel, LayoutError> {
    // Collect non-empty lines, supporting both \n and \r\n
    let raw_lines: Vec<&str> = input.lines().collect();

    // Trim leading and trailing empty lines
    let mut start = 0;
    let mut end = raw_lines.len();

    while start < end && raw_lines[start].trim().is_empty() {
        start += 1;
    }
    while end > start && raw_lines[end - 1].trim().is_empty() {
        end -= 1;
    }

    if start >= end {
        return Err(LayoutError::EmptyMap);
    }

    let lines = &raw_lines[start..end];

    // Validate all lines are non-empty
    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            return Err(LayoutError::EmptyMap);
        }
    }

    // Parse first line to get expected width
    let (first_tiles, _, _) = parse_line(lines[0], 0)?;
    let expected_width = first_tiles.len();

    if expected_width == 0 {
        return Err(LayoutError::EmptyMap);
    }

    // Parse all lines and validate rectangularity
    let mut all_tiles = Vec::new();
    let mut spawn_markers = Vec::new();
    let mut model_markers = Vec::new();
    let mut light_markers = Vec::new();

    for (line_idx, line) in lines.iter().enumerate() {
        let (tiles, markers, spawns) = parse_line(line, line_idx)?;

        if tiles.len() != expected_width {
            return Err(LayoutError::NonRectangular {
                line: line_idx + 1, // 1-indexed for display
                expected: expected_width,
                actual: tiles.len(),
            });
        }

        all_tiles.extend(tiles);

        // Collect markers with absolute positions
        for (col, marker_type) in markers {
            match marker_type {
                MarkerType::Model => model_markers.push((col, line_idx)),
                MarkerType::Light => light_markers.push((col, line_idx)),
            }
        }

        // Collect spawns
        for col in spawns {
            spawn_markers.push((col, line_idx));
        }
    }

    // Validate spawn cardinality
    if spawn_markers.len() != 1 {
        return Err(LayoutError::SpawnCardinality {
            count: spawn_markers.len(),
        });
    }

    Ok(ParsedLevel {
        width: expected_width,
        height: lines.len(),
        tiles: all_tiles,
        spawn: spawn_markers[0],
        model_markers,
        light_markers,
    })
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
            }
            '.' => {
                tiles.push(Tile::Floor);
                col += 1;
            }
            'S' => {
                tiles.push(Tile::Floor);
                spawns.push(col);
                col += 1;
            }
            'M' => {
                tiles.push(Tile::Floor);
                markers.push((col, MarkerType::Model));
                col += 1;
            }
            'L' => {
                tiles.push(Tile::Floor);
                markers.push((col, MarkerType::Light));
                col += 1;
            }
            'R' => {
                // Multi-character ramp token
                if col + 1 >= chars.len() {
                    return Err(LayoutError::IncompleteRamp {
                        line: line_idx + 1,
                        column: col + 1,
                    });
                }

                let dir = chars[col + 1];
                let tile = match dir {
                    '^' => Tile::RampNorth,
                    '>' => Tile::RampEast,
                    'v' => Tile::RampSouth,
                    '<' => Tile::RampWest,
                    _ => {
                        return Err(LayoutError::InvalidRampDir {
                            line: line_idx + 1,
                            column: col + 2,
                            token: dir,
                        });
                    }
                };

                tiles.push(tile);
                col += 2; // Skip both 'R' and direction character
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

/// Load and parse level from file
pub fn load_level_file<P: AsRef<Path>>(path: P) -> Result<ParsedLevel, LayoutError> {
    let contents = std::fs::read_to_string(path)?;
    parse_level(&contents)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_minimal_level() {
        let input = "#####\n#.S.#\n#####";
        let level = parse_level(input).unwrap();
        assert_eq!(level.width, 5);
        assert_eq!(level.height, 3);
        assert_eq!(level.spawn, (2, 1));
        assert_eq!(level.tile_at(2, 1), Tile::Floor);
        assert_eq!(level.tile_at(0, 0), Tile::Wall);
    }

    #[test]
    fn reject_unknown_token() {
        let input = "#####\n#.X.#\n#####";
        let err = parse_level(input).unwrap_err();
        match err {
            LayoutError::UnknownToken { token, line, column } => {
                assert_eq!(token, 'X');
                assert_eq!(line, 2);
                assert_eq!(column, 3);
            }
            _ => panic!("expected UnknownToken error"),
        }
    }

    #[test]
    fn reject_non_rectangular() {
        let input = "#####\n#..#\n#####";
        let err = parse_level(input).unwrap_err();
        match err {
            LayoutError::NonRectangular { line, expected, actual } => {
                assert_eq!(line, 2);
                assert_eq!(expected, 5);
                assert_eq!(actual, 4);
            }
            _ => panic!("expected NonRectangular error"),
        }
    }

    #[test]
    fn reject_multiple_spawns() {
        let input = "#####\n#S.S#\n#####";
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
        let input = "#####\n#...#\n#####";
        let err = parse_level(input).unwrap_err();
        match err {
            LayoutError::SpawnCardinality { count } => {
                assert_eq!(count, 0);
            }
            _ => panic!("expected SpawnCardinality error"),
        }
    }

    #[test]
    fn parse_ramps_all_directions() {
        // Each ramp token is 2 characters but produces 1 tile
        // All lines must produce same number of tiles (4 tiles each)
        let input = "####\nR^R>##\nRvR<##\n#S##\n####";
        let level = parse_level(input).unwrap();
        // Line 2: "R^R>##" = R^ (1 tile) + R> (1 tile) + # + # = 4 tiles
        assert_eq!(level.tile_at(0, 1), Tile::RampNorth);
        assert_eq!(level.tile_at(1, 1), Tile::RampEast);
        assert_eq!(level.tile_at(0, 2), Tile::RampSouth);
        assert_eq!(level.tile_at(1, 2), Tile::RampWest);
    }

    #[test]
    fn collect_markers() {
        let input = "#####\n#S.M#\n#L..#\n#####";
        let level = parse_level(input).unwrap();
        assert_eq!(level.spawn, (1, 1));
        assert_eq!(level.model_markers, vec![(3, 1)]);
        assert_eq!(level.light_markers, vec![(1, 2)]);
    }

    #[test]
    fn handle_crlf() {
        let input = "#####\r\n#.S.#\r\n#####";
        let level = parse_level(input).unwrap();
        assert_eq!(level.width, 5);
        assert_eq!(level.height, 3);
        assert_eq!(level.spawn, (2, 1));
    }

    #[test]
    fn reject_incomplete_ramp() {
        // R at end of line with no direction character
        let input = "#####\n#..R\n#S..#\n#####";
        let err = parse_level(input).unwrap_err();
        match err {
            LayoutError::IncompleteRamp { line, column } => {
                assert_eq!(line, 2);
                assert_eq!(column, 4);
            }
            _ => panic!("expected IncompleteRamp error, got: {:?}", err),
        }
    }

    #[test]
    fn reject_invalid_ramp_direction() {
        let input = "#####\n#RX.#\n#S..#\n#####";
        let err = parse_level(input).unwrap_err();
        match err {
            LayoutError::InvalidRampDir { line, column, token } => {
                assert_eq!(line, 2);
                assert_eq!(column, 3);
                assert_eq!(token, 'X');
            }
            _ => panic!("expected InvalidRampDir error"),
        }
    }

    #[test]
    fn reject_empty_map() {
        let input = "\n\n\n";
        let err = parse_level(input).unwrap_err();
        assert!(matches!(err, LayoutError::EmptyMap));
    }

    #[test]
    fn handle_leading_trailing_blank_lines() {
        let input = "\n\n#####\n#.S.#\n#####\n\n";
        let level = parse_level(input).unwrap();
        assert_eq!(level.height, 3);
        assert_eq!(level.spawn, (2, 1));
    }

    #[test]
    fn reject_whitespace_in_map() {
        let input = "#####\n# S #\n#####";
        let err = parse_level(input).unwrap_err();
        match err {
            LayoutError::UnknownToken { token, .. } => {
                assert_eq!(token, ' ');
            }
            _ => panic!("expected UnknownToken error for space"),
        }
    }

    #[test]
    fn tile_to_world_conversion() {
        // Test coordinate system convention
        let pos = tile_to_world(0, 0);
        assert_eq!(pos, glam::Vec3::new(0.0, 0.0, 0.0));

        let pos = tile_to_world(1, 0);
        assert_eq!(pos, glam::Vec3::new(1.0, 0.0, 0.0));

        let pos = tile_to_world(0, 1);
        assert_eq!(pos, glam::Vec3::new(0.0, 0.0, -1.0));

        let pos = tile_to_world(5, 3);
        assert_eq!(pos, glam::Vec3::new(5.0, 0.0, -3.0));
    }
}
