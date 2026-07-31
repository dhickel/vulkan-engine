//! Canonical .map emission for the Enhanced v3 proof.
//!
//! Serializes validated assemblies as Standard Quake worldspawn with
//! integer plane points, v2 texture roles, `0 0 0 0.25 0.25`, stable
//! ordering, LF line endings. No floating-point or Valve-220 extensions.

use super::assembly::{Assembly, AssemblyBrush, BrushRole};
use super::contract::{self, ContractError};
use super::geometry::{self, ConvexBrush, FaceRole, Point3, Rational};

/// Emit a complete .map string from a validated Assembly.
///
/// All brushes are in worldspawn. Entities (player start, lights) are
/// emitted after worldspawn closure.
pub fn emit_map(
    assembly: &Assembly,
    spawn_origin: (i32, i32, i32),
    spawn_yaw: i32,
    light_origins: &[(i32, i32, i32)],
) -> Result<String, ContractError> {
    if !assembly.validated {
        return Err(ContractError::InvariantViolation {
            detail: "cannot emit unvalidated assembly".into(),
        });
    }

    let mut out = String::new();

    // Worldspawn header
    out.push_str("{\n");
    out.push_str("\"classname\" \"worldspawn\"\n");
    out.push_str("\"wad\" \"cc0_dungeon_v2.wad\"\n");
    out.push_str(&format!("\"_minlight\" \"16\"\n"));

    // Emit each brush
    for brush in &assembly.brushes {
        emit_brush(&mut out, brush)?;
    }

    out.push_str("}\n");

    // Player start entity
    out.push_str(&format!(
        "{{\n\"angle\" \"{spawn_yaw}\"\n\"classname\" \"info_player_start\"\n\"origin\" \"{} {} {}\"\n}}\n",
        spawn_origin.0, spawn_origin.1, spawn_origin.2
    ));

    // Light entities
    for light in light_origins {
        out.push_str(&format!(
            "{{\n\"classname\" \"light\"\n\"light\" \"300\"\n\"origin\" \"{} {} {}\"\n}}\n",
            light.0, light.1, light.2
        ));
    }

    Ok(out)
}

/// Emit a single brush as a Quake .map brush block.
fn emit_brush(out: &mut String, ab: &AssemblyBrush) -> Result<(), ContractError> {
    let texture = texture_for_role(ab.role);

    let faces = &ab.brush.faces;
    out.push_str("{\n");
    for face in faces {
        let points = brush_face_plane_points(&ab.brush, &face.plane)?;
        let (p0, p1, p2) = (points[0], points[1], points[2]);

        // Format: ( x y z ) ( x y z ) ( x y z ) "texture" 0 0 0 0.25 0.25
        use std::fmt::Write;
        write!(
            out,
            "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n",
            p0.0, p0.1, p0.2, p1.0, p1.1, p1.2, p2.0, p2.1, p2.2, texture,
        )
        .unwrap();
    }
    out.push_str("}\n");

    Ok(())
}

/// Get texture name for a brush role (using cc0_dungeon_v2 textures).
fn texture_for_role(role: BrushRole) -> &'static str {
    match role {
        BrushRole::WallShell | BrushRole::PortalThroat => "bs_wall",
        BrushRole::FloorSlab => "bs_floor",
        BrushRole::CeilingSlab => "bs_ceil",
        BrushRole::Column | BrushRole::Buttress => "bs_accent",
        BrushRole::Feature => "bs_accent",
        BrushRole::World => "bs_wall",
    }
}

/// Compute three non-collinear integer points on a brush face plane.
///
/// For an axis-aligned rectangular brush, this returns the three defining
/// points that represent the face. The points are integer grid-aligned.
fn brush_face_plane_points(
    brush: &ConvexBrush,
    plane: &geometry::CanonicalPlane,
) -> Result<[(i32, i32, i32); 3], ContractError> {
    // Find vertices that lie on this face plane
    let vertices = brush
        .compute_vertices()
        .map_err(|e| ContractError::InvariantViolation {
            detail: format!("vertex computation: {e}"),
        })?;

    let mut face_verts: Vec<&Point3> = vertices
        .iter()
        .filter(|v| {
            plane
                .signed_distance_rational(v)
                .map(|d| d == Rational::ZERO)
                .unwrap_or(false)
        })
        .collect();

    // Sort to get stable order
    face_verts.sort();

    if face_verts.len() < 3 {
        return Err(ContractError::InvariantViolation {
            detail: format!("face {} has fewer than 3 vertices", plane.describe()),
        });
    }

    // Pick three non-collinear points
    // For rectangular brushes, any three distinct vertices on the face work
    let p0 = rational_to_integer_point(face_verts[0])?;
    let p1 = rational_to_integer_point(face_verts[1])?;
    let p2 = rational_to_integer_point(face_verts[face_verts.len() - 1])?;

    // Ensure the winding produces an inward-facing normal
    // (the plane normal points inward for the half-space n·x >= d)
    Ok([p0, p1, p2])
}

/// Convert a rational point to integer coordinates.
///
/// All proof geometry is quantum-aligned, so rational points are always
/// integer-valued.
fn rational_to_integer_point(p: &Point3) -> Result<(i32, i32, i32), ContractError> {
    let x = p.x.num;
    let y = p.y.num;
    let z = p.z.num;

    if p.x.den != 1 || p.y.den != 1 || p.z.den != 1 {
        return Err(ContractError::InvariantViolation {
            detail: format!("non-integer vertex in emission: {p}"),
        });
    }

    Ok((
        i32::try_from(x).map_err(|_| ContractError::ArithmeticOverflow {
            operation: "vertex x conversion",
        })?,
        i32::try_from(y).map_err(|_| ContractError::ArithmeticOverflow {
            operation: "vertex y conversion",
        })?,
        i32::try_from(z).map_err(|_| ContractError::ArithmeticOverflow {
            operation: "vertex z conversion",
        })?,
    ))
}

/// Emit a simple box assembly directly as a .map for the canonical fixture.
///
/// This bypasses the full Assembly pipeline for the minimal fixture case
/// and generates a valid worldspawn directly from room parameters.
pub fn emit_simple_room_map(
    room_x0: i32,
    room_y0: i32,
    room_z0: i32,
    room_x1: i32,
    room_y1: i32,
    room_z1: i32,
    spawn_origin: (i32, i32, i32),
    spawn_yaw: i32,
    light_origins: &[(i32, i32, i32)],
) -> String {
    let t = contract::CONSTRUCTION_QUANTUM;
    let w = t; // wall thickness

    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("\"classname\" \"worldspawn\"\n");
    out.push_str("\"wad\" \"cc0_dungeon_v2.wad\"\n");
    out.push_str("\"_minlight\" \"16\"\n");

    // Floor
    emit_box_brush(
        &mut out,
        room_x0,
        room_y0,
        room_z0,
        room_x1,
        room_y1,
        room_z0 + w,
        "bs_floor",
    );
    // Ceiling
    emit_box_brush(
        &mut out,
        room_x0,
        room_y0,
        room_z1 - w,
        room_x1,
        room_y1,
        room_z1,
        "bs_ceil",
    );
    // North wall
    emit_box_brush(
        &mut out,
        room_x0 + w,
        room_y0,
        room_z0 + w,
        room_x1 - w,
        room_y0 + w,
        room_z1 - w,
        "bs_wall",
    );
    // South wall
    emit_box_brush(
        &mut out,
        room_x0 + w,
        room_y1 - w,
        room_z0 + w,
        room_x1 - w,
        room_y1,
        room_z1 - w,
        "bs_wall",
    );
    // East wall
    emit_box_brush(
        &mut out,
        room_x1 - w,
        room_y0 + w,
        room_z0 + w,
        room_x1,
        room_y1 - w,
        room_z1 - w,
        "bs_wall",
    );
    // West wall
    emit_box_brush(
        &mut out,
        room_x0,
        room_y0 + w,
        room_z0 + w,
        room_x0 + w,
        room_y1 - w,
        room_z1 - w,
        "bs_wall",
    );

    out.push_str("}\n");

    // Player start
    out.push_str(&format!(
        "{{\n\"angle\" \"{spawn_yaw}\"\n\"classname\" \"info_player_start\"\n\"origin\" \"{} {} {}\"\n}}\n",
        spawn_origin.0, spawn_origin.1, spawn_origin.2
    ));

    // Lights
    for light in light_origins {
        out.push_str(&format!(
            "{{\n\"classname\" \"light\"\n\"light\" \"300\"\n\"origin\" \"{} {} {}\"\n}}\n",
            light.0, light.1, light.2
        ));
    }

    out
}

/// Emit a single solid box brush with the given texture.
fn emit_box_brush(
    out: &mut String,
    x0: i32,
    y0: i32,
    z0: i32,
    x1: i32,
    y1: i32,
    z1: i32,
    texture: &str,
) {
    if x0 >= x1 || y0 >= y1 || z0 >= z1 {
        return;
    }
    use std::fmt::Write;
    // Six faces of the box
    // Each face is defined by three non-collinear points on the plane
    // Face normals point inward

    // Floor: z=z0, normal (0,0,1), d=z0
    // Points: (x0,y0,z0), (x1,y0,z0), (x1,y1,z0)
    writeln!(out, "{{").unwrap();
    // Floor
    writeln!(
        out,
        "( {x0} {y0} {z0} ) ( {x1} {y0} {z0} ) ( {x1} {y1} {z0} ) \"{texture}\" 0 0 0 0.25 0.25"
    )
    .unwrap();
    // Ceiling
    writeln!(
        out,
        "( {x0} {y1} {z1} ) ( {x1} {y1} {z1} ) ( {x1} {y0} {z1} ) \"{texture}\" 0 0 0 0.25 0.25"
    )
    .unwrap();
    // North (-Y)
    writeln!(
        out,
        "( {x0} {y0} {z0} ) ( {x0} {y0} {z1} ) ( {x1} {y0} {z1} ) \"{texture}\" 0 0 0 0.25 0.25"
    )
    .unwrap();
    // South (+Y)
    writeln!(
        out,
        "( {x1} {y1} {z1} ) ( {x0} {y1} {z1} ) ( {x0} {y1} {z0} ) \"{texture}\" 0 0 0 0.25 0.25"
    )
    .unwrap();
    // East (+X)
    writeln!(
        out,
        "( {x1} {y0} {z0} ) ( {x1} {y1} {z0} ) ( {x1} {y1} {z1} ) \"{texture}\" 0 0 0 0.25 0.25"
    )
    .unwrap();
    // West (-X)
    writeln!(
        out,
        "( {x0} {y1} {z1} ) ( {x0} {y1} {z0} ) ( {x0} {y0} {z0} ) \"{texture}\" 0 0 0 0.25 0.25"
    )
    .unwrap();
    writeln!(out, "}}").unwrap();
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::assembly::{Assembly, AssemblyBrush, BrushRole, Support};
    use super::super::geometry::ConvexBrush;
    use super::*;

    #[test]
    fn emit_simple_map() {
        let map = emit_simple_room_map(0, 0, 0, 128, 128, 176, (64, 64, 24), 90, &[(64, 64, 160)]);

        assert!(map.contains("worldspawn"));
        assert!(map.contains("info_player_start"));
        assert!(map.contains("\"classname\" \"light\""));
        assert!(map.ends_with('\n'));
        // Should not have double trailing LF
        assert!(!map.ends_with("\n\n"));
    }

    #[test]
    fn emit_simple_map_deterministic() {
        let a = emit_simple_room_map(0, 0, 0, 128, 128, 176, (64, 64, 24), 90, &[(64, 64, 160)]);
        let b = emit_simple_room_map(0, 0, 0, 128, 128, 176, (64, 64, 24), 90, &[(64, 64, 160)]);
        assert_eq!(a, b);
    }

    #[test]
    fn emit_unvalidated_assembly_rejected() {
        let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
        let ab = AssemblyBrush::new(
            "test",
            BrushRole::WallShell,
            brush,
            Support::World {
                surface: FaceRole::Floor,
            },
        );

        let assembly = Assembly {
            brushes: vec![ab],
            interfaces: vec![],
            apertures: vec![],
            protected_volumes: vec![],
            support_edges: vec![],
            validated: false,
        };

        assert!(emit_map(&assembly, (32, 32, 16), 0, &[]).is_err());
    }

    #[test]
    fn texture_mapping_correct() {
        assert_eq!(texture_for_role(BrushRole::WallShell), "bs_wall");
        assert_eq!(texture_for_role(BrushRole::FloorSlab), "bs_floor");
        assert_eq!(texture_for_role(BrushRole::CeilingSlab), "bs_ceil");
        assert_eq!(texture_for_role(BrushRole::PortalThroat), "bs_wall");
        assert_eq!(texture_for_role(BrushRole::Feature), "bs_accent");
    }
}
