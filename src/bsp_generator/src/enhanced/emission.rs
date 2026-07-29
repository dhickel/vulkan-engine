//! Enhanced v2 emission — produce canonical map text from intents.
//!
//! Consumes placement, topology, and theme data; emits worldspawn brushes
//! and entities in canonical Standard Quake format.

use super::config::EnhancedConfig;
use super::error::EnhancedError;
use super::intent::TransitionIntent;
use super::placement::PlacedRoom;
use super::topology::TopologyResult;

const Q: i32 = 16;

/// Emit a complete Enhanced v2 .map file.
pub fn emit_map(
    config: &EnhancedConfig,
    rooms: &[PlacedRoom],
    topology: &TopologyResult,
    _wad_basename: &str,
) -> Result<String, EnhancedError> {
    let mut out = String::new();
    out.push_str("{\n\"classname\" \"worldspawn\"\n\"wad\" \"cc0_stone_beta.wad\"\n");

    // Emit rooms
    for room in rooms {
        emit_room(&mut out, room)?;
    }

    // Emit stairs for each transition
    for t in &topology.transitions {
        emit_stairs(&mut out, t, rooms)?;
    }

    out.push_str("}\n");

    // Spawn entity
    if let Some(first) = rooms.first() {
        let cx = (first.shell.0 + first.shell.2) / 2;
        let cy = (first.shell.1 + first.shell.3) / 2;
        let cz = first.floor_z + Q + 24;
        out.push_str(&format!(
            "{{\n\"classname\" \"info_player_start\"\n\"origin\" \"{} {} {}\"\n}}\n",
            cx, cy, cz
        ));
    }

    Ok(out)
}

fn emit_room(out: &mut String, room: &PlacedRoom) -> Result<(), EnhancedError> {
    let (x0, y0, x1, y1) = room.shell;
    let z0 = room.floor_z;
    let zh = room.dims.2 as i32;
    let z1 = z0 + zh;

    // Floor
    emit_brush(out, x0, y0, z0, x1, y1, z0 + Q, "stone_floor");
    // Ceiling
    emit_brush(out, x0, y0, z1 - Q, x1, y1, z1, "stone_ceiling");
    // Walls (solid — no apertures; Phase 06 adds variance)
    emit_brush(out, x0, y0, z0, x1, y0 + Q, z1, "stone_wall"); // south
    emit_brush(out, x0, y1 - Q, z0, x1, y1, z1, "stone_wall"); // north
    emit_brush(out, x0, y0, z0, x0 + Q, y1, z1, "stone_wall"); // west
    emit_brush(out, x1 - Q, y0, z0, x1, y1, z1, "stone_wall"); // east
    Ok(())
}

fn emit_stairs(
    out: &mut String,
    t: &TransitionIntent,
    rooms: &[PlacedRoom],
) -> Result<(), EnhancedError> {
    let lower =
        rooms
            .iter()
            .find(|r| r.id == t.lower_room)
            .ok_or(EnhancedError::ContractViolation {
                detail: "lower room not found".into(),
            })?;
    let upper =
        rooms
            .iter()
            .find(|r| r.id == t.upper_room)
            .ok_or(EnhancedError::ContractViolation {
                detail: "upper room not found".into(),
            })?;

    // Place stairwell between rooms (simplified)
    let sw_x0 = lower.shell.2; // right of lower room
    let sw_x1 = upper.shell.0; // left of upper room
    let sw_y0 = (lower.shell.1 + lower.shell.3) / 2 - 48;
    let sw_y1 = sw_y0 + 96;
    let sw_z0 = lower.floor_z;
    let sw_z1 = upper.floor_z + Q;

    // Stairwell shell
    emit_brush(
        out,
        sw_x0,
        sw_y0,
        sw_z0,
        sw_x1,
        sw_y1,
        sw_z0 + Q,
        "stone_floor",
    );
    emit_brush(
        out,
        sw_x0,
        sw_y0,
        sw_z1 - Q,
        sw_x1,
        sw_y1,
        sw_z1,
        "stone_ceiling",
    );
    emit_brush(
        out,
        sw_x0,
        sw_y0,
        sw_z0,
        sw_x1,
        sw_y0 + Q,
        sw_z1,
        "stone_wall",
    );
    emit_brush(
        out,
        sw_x0,
        sw_y1 - Q,
        sw_z0,
        sw_x1,
        sw_y1,
        sw_z1,
        "stone_wall",
    );

    // Steps
    let rise = upper.floor_z - lower.floor_z;
    let steps = rise / Q;
    let tread = Q; // default 16
    for i in 0..steps {
        let sx0 = sw_x0 + Q + i * tread;
        let sz0 = lower.floor_z + i * Q;
        emit_brush(
            out,
            sx0,
            sw_y0 + Q,
            sz0,
            sx0 + tread,
            sw_y1 - Q,
            sz0 + Q,
            "stone_floor",
        );
    }

    Ok(())
}

fn emit_brush(out: &mut String, x0: i32, y0: i32, z0: i32, x1: i32, y1: i32, z1: i32, tex: &str) {
    // Canonical face order matching make_brush
    out.push_str("{\n");
    out.push_str(&format!(
        "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n",
        x0, y1, z0, x0, y0, z0, x1, y0, z0, tex
    ));
    out.push_str(&format!(
        "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n",
        x0, y1, z1, x1, y1, z1, x1, y0, z1, tex
    ));
    out.push_str(&format!(
        "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n",
        x0, y1, z1, x0, y1, z0, x1, y1, z0, tex
    ));
    out.push_str(&format!(
        "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n",
        x0, y0, z1, x1, y0, z1, x1, y0, z0, tex
    ));
    out.push_str(&format!(
        "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n",
        x0, y1, z1, x0, y0, z1, x0, y0, z0, tex
    ));
    out.push_str(&format!(
        "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n",
        x1, y1, z0, x1, y0, z0, x1, y0, z1, tex
    ));
    out.push_str("}\n");
}

#[cfg(test)]
mod tests {
    use super::super::config::EnhancedConfig;
    use super::super::placement::place_rooms;
    use super::super::seed::EnhancedSeed;
    use super::super::topology::build_topology;
    use super::*;

    #[test]
    fn emit_nominal_map() {
        let cfg = EnhancedConfig::nominal();
        let seed = EnhancedSeed::new(99);
        let placement = place_rooms(
            &cfg,
            seed.stage_seed(super::super::seed::tags::LAYER_PLACEMENT),
        )
        .unwrap();
        let mut topo_rng = seed.stage_seed(super::super::seed::tags::VERTICAL_TOPOLOGY);
        let topo = build_topology(&cfg, &placement, &mut topo_rng).unwrap();
        let map = emit_map(&cfg, &placement.rooms, &topo, "cc0_stone_beta.wad").unwrap();
        assert!(!map.is_empty());
        assert!(map.contains("worldspawn"));
        assert!(map.contains("info_player_start"));
    }
}
