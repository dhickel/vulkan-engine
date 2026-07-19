//! Canonical `ParsedLevel` ASCII serialization and exact parser round-trip.

use std::collections::BTreeMap;

use crate::layout::{ParsedLevel, Tile, TileCoord};

use super::error::{ErrorStage, GeneratorError};

/// Serialize a marker-complete level using the authored parser's vocabulary.
///
/// Marker collisions and markers on non-floor tiles are rejected because the
/// single-token grammar cannot represent them without losing information.
pub(super) fn serialize_level(level: &ParsedLevel) -> Result<String, GeneratorError> {
    validate_level_shape(level)?;

    let mut markers = BTreeMap::<(usize, usize, usize), char>::new();
    insert_marker(level, &mut markers, level.spawn, 'S')?;
    validate_canonical_marker_order(&level.model_markers, "model")?;
    validate_canonical_marker_order(&level.light_markers, "light")?;
    for &marker in &level.model_markers {
        insert_marker(level, &mut markers, marker, 'M')?;
    }
    for &marker in &level.light_markers {
        insert_marker(level, &mut markers, marker, 'L')?;
    }

    let mut output = String::new();
    for (layer_idx, layer) in level.layers.iter().enumerate() {
        if layer_idx > 0 {
            output.push_str("---\n");
        }
        for y in 0..level.height {
            for x in 0..level.width {
                if let Some(token) = markers.get(&(layer_idx, x, y)) {
                    output.push(*token);
                } else {
                    output.push_str(tile_to_ascii_token(layer[y * level.width + x]));
                }
            }
            output.push('\n');
        }
    }
    Ok(output)
}

fn validate_level_shape(level: &ParsedLevel) -> Result<(), GeneratorError> {
    let expected = level
        .width
        .checked_mul(level.height)
        .ok_or(GeneratorError::ArithmeticOverflow {
            stage: ErrorStage::Ir,
            operation: "ascii_level_area",
        })?;
    if level.width == 0
        || level.height == 0
        || level.layers.is_empty()
        || level.layers.iter().any(|layer| layer.len() != expected)
    {
        return Err(ascii_error("non_rectangular_or_empty_level"));
    }
    if level.layers.iter().flatten().any(|tile| {
        matches!(
            tile,
            Tile::RampNorth(level)
                | Tile::RampEast(level)
                | Tile::RampSouth(level)
                | Tile::RampWest(level)
                if *level > 2
        )
    }) {
        return Err(ascii_error("unsupported_ramp_substep"));
    }
    Ok(())
}

fn validate_canonical_marker_order(
    markers: &[TileCoord],
    kind: &'static str,
) -> Result<(), GeneratorError> {
    if markers
        .windows(2)
        .any(|pair| marker_key(pair[0]) >= marker_key(pair[1]))
    {
        return Err(ascii_error(format!("{kind}_markers_not_canonical")));
    }
    Ok(())
}

fn marker_key(marker: TileCoord) -> (usize, usize, usize) {
    (marker.layer, marker.y, marker.x)
}

fn insert_marker(
    level: &ParsedLevel,
    markers: &mut BTreeMap<(usize, usize, usize), char>,
    marker: TileCoord,
    token: char,
) -> Result<(), GeneratorError> {
    if marker.layer >= level.layer_count() || marker.x >= level.width || marker.y >= level.height {
        return Err(ascii_error(format!(
            "marker_oob token={token} l={} x={} y={}",
            marker.layer, marker.x, marker.y
        )));
    }
    if level.tile_at_3d(marker.layer, marker.x, marker.y) != Tile::Floor {
        return Err(ascii_error(format!(
            "marker_not_floor token={token} l={} x={} y={}",
            marker.layer, marker.x, marker.y
        )));
    }
    if let Some(previous) = markers.insert((marker.layer, marker.x, marker.y), token) {
        return Err(ascii_error(format!(
            "marker_collision tokens={previous},{token} l={} x={} y={}",
            marker.layer, marker.x, marker.y
        )));
    }
    Ok(())
}

fn tile_to_ascii_token(tile: Tile) -> &'static str {
    match tile {
        Tile::Wall => "#",
        Tile::Floor => ".",
        Tile::Void => "_",
        Tile::RampNorth(0) => "R0^",
        Tile::RampNorth(1) => "R1^",
        Tile::RampNorth(2) => "R2^",
        Tile::RampEast(0) => "R0>",
        Tile::RampEast(1) => "R1>",
        Tile::RampEast(2) => "R2>",
        Tile::RampSouth(0) => "R0v",
        Tile::RampSouth(1) => "R1v",
        Tile::RampSouth(2) => "R2v",
        Tile::RampWest(0) => "R0<",
        Tile::RampWest(1) => "R1<",
        Tile::RampWest(2) => "R2<",
        _ => "",
    }
}

/// Serialize, parse, and require exact `ParsedLevel` equality.
pub(super) fn round_trip_exact(level: &ParsedLevel) -> Result<String, GeneratorError> {
    let serialized = serialize_level(level)?;
    let reparsed = crate::layout::parse_level(&serialized)
        .map_err(|error| ascii_error(format!("parse_failed reason={error}")))?;
    if &reparsed != level {
        return Err(ascii_error("round_trip_mismatch"));
    }
    Ok(serialized)
}

fn ascii_error(detail: impl Into<String>) -> GeneratorError {
    GeneratorError::IrInvariant {
        stage: ErrorStage::Ir,
        detail: format!("[ascii] {}", detail.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::parse_level;

    #[test]
    fn exact_round_trip_preserves_layers_ramps_and_markers() {
        let level = parse_level(
            "######\n#SM.L#\n#R0>R1>R2>.#\n######\n---\n######\n#....#\n#....#\n######",
        )
        .unwrap();
        let ascii = round_trip_exact(&level).unwrap();
        assert!(ascii.contains("---"));
        assert!(ascii.contains("R0>R1>R2>"));
        assert_eq!(parse_level(&ascii).unwrap(), level);
    }

    #[test]
    fn all_bundled_levels_round_trip_exactly() {
        for level_name in ["level_01", "level_02_ramps", "level_03_lighting"] {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join(format!("assets/levels/{level_name}.txt"));
            let level = parse_level(&std::fs::read_to_string(path).unwrap()).unwrap();
            round_trip_exact(&level)
                .unwrap_or_else(|error| panic!("round-trip failed for {level_name}: {error}"));
        }
    }

    #[test]
    fn marker_collision_is_rejected_instead_of_dropped() {
        let mut level = parse_level("####\n#SM#\n####").unwrap();
        level.model_markers[0] = level.spawn;
        let error = serialize_level(&level).unwrap_err();
        assert!(error.to_string().contains("marker_collision"));
    }

    #[test]
    fn noncanonical_marker_order_is_rejected() {
        let mut level = parse_level("#####\n#SMM#\n#####").unwrap();
        level.model_markers.reverse();
        let error = serialize_level(&level).unwrap_err();
        assert!(error.to_string().contains("not_canonical"));
    }

    #[test]
    fn marker_on_ramp_is_rejected() {
        let mut level = parse_level("####\n#SR0>#\n####").unwrap();
        level.model_markers.push(TileCoord { layer: 0, x: 2, y: 1 });
        let error = serialize_level(&level).unwrap_err();
        assert!(error.to_string().contains("marker_not_floor"));
    }
}
