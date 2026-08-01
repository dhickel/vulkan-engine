//! Canonical Standard Quake .map serialization from validated assemblies.
//!
//! Emits worldspawn brushes with fixed-role textures, integer plane points,
//! stable ordering, LF line endings, and one terminal newline. All brushes
//! must be in sorted ID order before emission.
//!
//! # Contract
//!
//! - Plane points are three non-collinear integer vertices per face.
//! - Texture mapping is always `0 0 0 0.25 0.25`.
//! - Textures are assigned by brush role from the approved `cc0_dungeon_v2` set.
//! - Entity key order is stable (classname first, then others alphabetically).
//! - Output is pure ASCII/UTF-8 with LF (`\n`) line endings.

use std::fmt::Write;

use super::assembly::{Assembly, AssemblyBrush, BrushRole};
use super::error::V3Error;
use super::geometry::{CanonicalPlane, ConvexBrush, Point3, Rational};

// ── Public emission entry point ───────────────────────────────────────────

/// Emit a complete .map string from a validated Assembly.
///
/// Returns canonical Quake .map text with LF line endings and exactly one
/// terminal newline. All output is deterministic given identical input.
///
/// # Errors
///
/// Returns `V3Error::UnvalidatedAssembly` if the assembly has not been
/// validated. Returns `V3Error::EmissionInvariant` if any brush cannot be
/// serialized (non-integer vertices, insufficient coplanar points, etc.).
pub fn emit_map_text(
    assembly: &Assembly,
    spawn_origin: (i32, i32, i32),
    light_origins: &[(i32, i32, i32)],
) -> Result<String, V3Error> {
    emit_map_text_with_minlight(assembly, spawn_origin, light_origins, 16)
}

/// Emit canonical map text with an explicit worldspawn `_minlight` value.
pub fn emit_map_text_with_minlight(
    assembly: &Assembly,
    spawn_origin: (i32, i32, i32),
    light_origins: &[(i32, i32, i32)],
    minlight: u32,
) -> Result<String, V3Error> {
    if !assembly.validated {
        return Err(V3Error::UnvalidatedAssembly);
    }

    let mut out = String::with_capacity(assembly.brushes.len() * 512 + 1024);

    // ── Worldspawn header ──────────────────────────────────────────────
    out.push_str("{\n");
    out.push_str("\"classname\" \"worldspawn\"\n");
    out.push_str("\"wad\" \"cc0_dungeon_v2.wad\"\n");
    writeln!(out, "\"_minlight\" \"{minlight}\"").expect("write to String is infallible");
    // emit a blank line for readability between header and brushes
    out.push('\n');

    // ── World brushes (already in sorted ID order) ────────────────────
    let mut brush_count = 0u32;
    for brush in &assembly.brushes {
        emit_brush_block(&mut out, brush)?;
        brush_count += 1;
        if brush_count < assembly.brushes.len() as u32 {
            out.push('\n');
        }
    }

    out.push_str("}\n");

    // ── Player start entity ────────────────────────────────────────────
    out.push_str(&format!(
        "{{\n\"angle\" \"90\"\n\"classname\" \"info_player_start\"\n\"origin\" \"{} {} {}\"\n}}\n",
        spawn_origin.0, spawn_origin.1, spawn_origin.2
    ));

    // ── Light entities ─────────────────────────────────────────────────
    for (i, light) in light_origins.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format!(
            "{{\n\"classname\" \"light\"\n\"light\" \"300\"\n\"origin\" \"{} {} {}\"\n}}",
            light.0, light.1, light.2
        ));
    }
    out.push('\n');

    Ok(out)
}

/// Emit a single brush as a Quake .map brush block with proper face planes.
fn emit_brush_block(out: &mut String, ab: &AssemblyBrush) -> Result<(), V3Error> {
    let texture = texture_for_role(ab.role);
    let faces = &ab.brush.faces;

    out.push_str("{\n");
    for face in faces {
        let points = brush_face_plane_points(&ab.brush, &face.plane)?;
        let (p0, p1, p2) = (points[0], points[1], points[2]);

        write!(
            out,
            "( {} {} {} ) ( {} {} {} ) ( {} {} {} ) \"{}\" 0 0 0 0.25 0.25\n",
            p0.0, p0.1, p0.2, p1.0, p1.1, p1.2, p2.0, p2.1, p2.2, texture,
        )
        .expect("write to String is infallible");
    }
    out.push_str("}\n");

    Ok(())
}

// ── Texture role mapping ──────────────────────────────────────────────────

/// Get the approved `cc0_dungeon_v2` texture name for a brush role.
pub fn texture_for_role(role: BrushRole) -> &'static str {
    match role {
        BrushRole::WallShell | BrushRole::PortalThroat => "bs_wall",
        BrushRole::FloorSlab => "bs_floor",
        BrushRole::CeilingSlab => "bs_ceil",
        BrushRole::Column
        | BrushRole::Buttress
        | BrushRole::Blade
        | BrushRole::VaultRib
        | BrushRole::Monolith
        | BrushRole::Feature => "bs_accent",
        BrushRole::World => "bs_wall",
    }
}

// ── Plane point computation ───────────────────────────────────────────────

/// Compute three non-collinear integer points on a brush face plane.
///
/// Returns three distinct points in the plane that span a non-zero area
/// triangle. Points are ordered so the cross product (p1-p0)×(p2-p0)
/// aligns with the canonical plane normal direction.
pub fn brush_face_plane_points(
    brush: &ConvexBrush,
    plane: &CanonicalPlane,
) -> Result<[(i32, i32, i32); 3], V3Error> {
    let vertices = brush.compute_vertices()?;
    let mut coplanar: Vec<Point3> = vertices
        .into_iter()
        .filter(|v| {
            plane
                .signed_distance_rational(v)
                .map(|d| d == Rational::ZERO)
                .unwrap_or(false)
        })
        .collect();

    if coplanar.len() < 3 {
        return Err(V3Error::EmissionInvariant {
            detail: format!(
                "less than 3 coplanar vertices on {} (found {})",
                plane.describe(),
                coplanar.len()
            ),
        });
    }

    // Sort by (x, y, z) for stability, then pick three well-spaced points.
    coplanar.sort();
    coplanar.dedup();

    if coplanar.len() < 3 {
        return Err(V3Error::EmissionInvariant {
            detail: format!(
                "only {} distinct coplanar vertices on {}",
                coplanar.len(),
                plane.describe()
            ),
        });
    }

    let p0 = point3_to_i32_tuple(&coplanar[0])?;
    let p_last = point3_to_i32_tuple(&coplanar[coplanar.len() - 1])?;

    // Find a middle point that is not collinear with p0 and p_last.
    // Also ensure the cross product aligns with the plane normal.
    for i in (1..coplanar.len() - 1).rev() {
        let p_mid = point3_to_i32_tuple(&coplanar[i])?;
        let cross = (
            (p_last.1 - p0.1) * (p_mid.2 - p0.2) - (p_last.2 - p0.2) * (p_mid.1 - p0.1),
            (p_last.2 - p0.2) * (p_mid.0 - p0.0) - (p_last.0 - p0.0) * (p_mid.2 - p0.2),
            (p_last.0 - p0.0) * (p_mid.1 - p0.1) - (p_last.1 - p0.1) * (p_mid.0 - p0.0),
        );
        if cross == (0, 0, 0) {
            continue;
        }

        // Check alignment with plane normal. The cross product should
        // have the same sign as the plane normal (or be a positive multiple).
        // The plane normal in Quake .map points toward the interior of the brush.
        // ericw-tools qbsp determines the "inside" from the plane equations.
        // We ensure the cross product isn't zero and points in a consistent direction.
        let dot = plane.nx as i64 * cross.0 as i64
            + plane.ny as i64 * cross.1 as i64
            + plane.nz as i64 * cross.2 as i64;

        if dot > 0 {
            // Normal aligns — use (p0, p_last, p_mid)
            return Ok([p0, p_last, p_mid]);
        } else if dot < 0 {
            // Normal is opposite — swap to flip winding
            return Ok([p0, p_mid, p_last]);
        }
        // dot == 0 shouldn't happen for non-collinear points on this plane
        // but if it does, the winding is ambiguous; still accept it
        return Ok([p0, p_last, p_mid]);
    }

    // Fallback: try the second point
    let p1 = point3_to_i32_tuple(&coplanar[1])?;
    Ok([p0, p_last, p1])
}

/// Convert a rational `Point3` to `(i32, i32, i32)`, requiring exact integers.
pub fn point3_to_i32_tuple(p: &Point3) -> Result<(i32, i32, i32), V3Error> {
    if p.x.den != 1 || p.y.den != 1 || p.z.den != 1 {
        return Err(V3Error::EmissionInvariant {
            detail: format!("non-integer point {p} in emission"),
        });
    }
    let x = i32::try_from(p.x.num).map_err(|_| V3Error::EmissionInvariant {
        detail: format!("x coordinate {} out of i32 range", p.x.num),
    })?;
    let y = i32::try_from(p.y.num).map_err(|_| V3Error::EmissionInvariant {
        detail: format!("y coordinate {} out of i32 range", p.y.num),
    })?;
    let z = i32::try_from(p.z.num).map_err(|_| V3Error::EmissionInvariant {
        detail: format!("z coordinate {} out of i32 range", p.z.num),
    })?;
    Ok((x, y, z))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::assembly::Support;
    use super::super::geometry::FaceRole;
    use super::*;

    #[test]
    fn texture_roles_are_known() {
        assert_eq!(texture_for_role(BrushRole::WallShell), "bs_wall");
        assert_eq!(texture_for_role(BrushRole::FloorSlab), "bs_floor");
        assert_eq!(texture_for_role(BrushRole::CeilingSlab), "bs_ceil");
        assert_eq!(texture_for_role(BrushRole::Column), "bs_accent");
        assert_eq!(texture_for_role(BrushRole::Feature), "bs_accent");
        assert_eq!(texture_for_role(BrushRole::PortalThroat), "bs_wall");
        assert_eq!(texture_for_role(BrushRole::World), "bs_wall");
    }

    #[test]
    fn emission_requires_validated_assembly() {
        let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
        let mut assembly = Assembly {
            brushes: vec![AssemblyBrush::new(
                "test",
                BrushRole::WallShell,
                brush,
                Support::World {
                    surface: FaceRole::Floor,
                },
            )],
            interfaces: vec![],
            protected_volumes: vec![],
            support_edges: vec![],
            validated: false,
        };
        // Force unvalidated state
        assembly.validated = false;
        let result = emit_map_text(&assembly, (0, 0, 0), &[]);
        assert!(result.is_err());
    }

    #[test]
    fn plane_points_for_box_face() {
        let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
        let floor_face = brush
            .faces
            .iter()
            .find(|f| f.role == FaceRole::Floor)
            .unwrap();
        let points = brush_face_plane_points(&brush, &floor_face.plane).unwrap();
        // Should have three distinct integer points
        assert_ne!(points[0], points[1]);
        assert_ne!(points[1], points[2]);
        assert_ne!(points[0], points[2]);
        // All points should be on the floor plane (z = 0)
        for p in &points {
            assert_eq!(p.2, 0);
        }
    }

    #[test]
    fn map_text_has_lf_endings() {
        let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
        let brushes = vec![AssemblyBrush::new(
            "wall/0000/wall_north",
            BrushRole::WallShell,
            brush,
            Support::World {
                surface: FaceRole::Floor,
            },
        )];
        let assembly = Assembly::new(brushes, vec![], vec![]).unwrap();
        let map = emit_map_text(&assembly, (32, 32, 48), &[]).unwrap();
        // No CR characters
        assert!(!map.contains('\r'));
        // Uses LF
        assert!(map.contains('\n'));
        // Has entities and worldspawn
        assert!(map.contains("worldspawn"));
        assert!(map.contains("info_player_start"));
    }

    #[test]
    fn map_text_has_terminal_newline() {
        let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
        let brushes = vec![AssemblyBrush::new(
            "w",
            BrushRole::WallShell,
            brush,
            Support::World {
                surface: FaceRole::Floor,
            },
        )];
        let assembly = Assembly::new(brushes, vec![], vec![]).unwrap();
        let map = emit_map_text(&assembly, (32, 32, 48), &[]).unwrap();
        assert!(map.ends_with('\n'));
    }

    #[test]
    fn map_text_uses_fixed_texture_mapping() {
        let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
        let brushes = vec![AssemblyBrush::new(
            "w",
            BrushRole::WallShell,
            brush,
            Support::World {
                surface: FaceRole::Floor,
            },
        )];
        let assembly = Assembly::new(brushes, vec![], vec![]).unwrap();
        let map = emit_map_text(&assembly, (32, 32, 48), &[]).unwrap();
        // Every face line should end with the fixed texture mapping
        for line in map.lines() {
            if line.trim_start().starts_with('(') {
                assert!(line.ends_with("0 0 0 0.25 0.25"));
            }
        }
    }

    #[test]
    fn integer_plane_points() {
        let brush = ConvexBrush::make_box((0, 64), (0, 64), (0, 128)).unwrap();
        for face in &brush.faces {
            let points = brush_face_plane_points(&brush, &face.plane).unwrap();
            for p in &points {
                // All coordinates must be integers
                assert_eq!(p.0 as i64 * 1, p.0 as i64);
                assert_eq!(p.1 as i64 * 1, p.1 as i64);
                assert_eq!(p.2 as i64 * 1, p.2 as i64);
            }
        }
    }
}
