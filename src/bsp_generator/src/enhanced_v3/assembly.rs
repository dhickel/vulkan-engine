//! Exact brush assembly and solid-intersection kernel for Enhanced V3.
//!
//! Validates assembly ordering, interface ownership, protected-volume
//! immutability, exact intersection proofs, and support graphs.
//! All arithmetic is exact — no floats, no snapping.
//!
//! # Design contract
//!
//! - Protected volumes are immutable.
//! - Positive-volume overlap between brushes is an error.
//! - Zero-volume contact is permitted only at declared interfaces.
//! - Results are always canonically ordered.
//! - Support graph must be acyclic; every dependent reaches a world surface.

use std::collections::{BTreeMap, BTreeSet};

use super::error::V3Error;
use super::geometry::{self, ConvexBrush, FaceRole, Point3, Rational, Vector3};

// ── Assembly role ──────────────────────────────────────────────────────────

/// Role of a brush in the assembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BrushRole {
    WallShell,
    FloorSlab,
    CeilingSlab,
    Column,
    Buttress,
    Blade,
    VaultRib,
    Monolith,
    PortalThroat,
    Feature,
    World,
}

impl BrushRole {
    pub fn tag(self) -> &'static str {
        match self {
            Self::WallShell => "wall_shell",
            Self::FloorSlab => "floor_slab",
            Self::CeilingSlab => "ceiling_slab",
            Self::Column => "column",
            Self::Buttress => "buttress",
            Self::Blade => "blade",
            Self::VaultRib => "vault_rib",
            Self::Monolith => "monolith",
            Self::PortalThroat => "portal_throat",
            Self::Feature => "feature",
            Self::World => "world",
        }
    }
}

// ── Interface ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interface {
    pub id: String,
    pub brush_a: String,
    pub brush_b: String,
    pub face_role_a: FaceRole,
    pub face_role_b: FaceRole,
}

impl Interface {
    pub fn new(
        id: impl Into<String>,
        brush_a: impl Into<String>,
        brush_b: impl Into<String>,
        face_role_a: FaceRole,
        face_role_b: FaceRole,
    ) -> Self {
        Self {
            id: id.into(),
            brush_a: brush_a.into(),
            brush_b: brush_b.into(),
            face_role_a,
            face_role_b,
        }
    }
}

// ── Support ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Support {
    World {
        surface: FaceRole,
    },
    SupportedBy {
        brush_id: String,
        interface_id: String,
    },
}

impl Support {
    pub fn is_world(&self) -> bool {
        matches!(self, Self::World { .. })
    }
}

// ── Assembly brush ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssemblyBrush {
    pub id: String,
    pub role: BrushRole,
    pub brush: ConvexBrush,
    pub support: Support,
}

impl AssemblyBrush {
    pub fn new(
        id: impl Into<String>,
        role: BrushRole,
        brush: ConvexBrush,
        support: Support,
    ) -> Self {
        Self {
            id: id.into(),
            role,
            brush,
            support,
        }
    }
}

// ── Assembly ───────────────────────────────────────────────────────────────

/// A validated assembly of convex brushes with interfaces, apertures,
/// and protected volumes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assembly {
    pub brushes: Vec<AssemblyBrush>,
    pub interfaces: Vec<Interface>,
    pub protected_volumes: Vec<ProtectedVolume>,
    pub support_edges: Vec<(String, String)>,
    pub validated: bool,
}

/// An immutable protected volume within the assembly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtectedVolume {
    pub id: String,
    pub brush: ConvexBrush,
}

impl Assembly {
    pub fn new(
        brushes: Vec<AssemblyBrush>,
        interfaces: Vec<Interface>,
        protected_volumes: Vec<ProtectedVolume>,
    ) -> Result<Self, V3Error> {
        let mut assembly = Self {
            brushes,
            interfaces,
            protected_volumes,
            support_edges: Vec::new(),
            validated: false,
        };
        assembly.validate()?;
        Ok(assembly)
    }

    pub fn validate(&mut self) -> Result<(), V3Error> {
        self.validate_ordering()?;
        self.check_protected_volume_intrusion()?;
        self.check_pairwise_intersections()?;
        self.validate_interfaces()?;
        self.build_support_graph()?;
        self.validate_support_graph()?;

        self.validated = true;
        Ok(())
    }

    fn validate_ordering(&self) -> Result<(), V3Error> {
        let mut ids = BTreeSet::new();
        for brush in &self.brushes {
            if !ids.insert(brush.id.clone()) {
                return Err(V3Error::DuplicateBrushId {
                    id: brush.id.clone(),
                });
            }
        }
        for w in self.brushes.windows(2) {
            if w[0].id >= w[1].id {
                return Err(V3Error::AssemblyValidation {
                    detail: format!("{} precedes {}", w[0].id, w[1].id),
                });
            }
        }
        Ok(())
    }

    fn find_brush(&self, id: &str) -> Option<&AssemblyBrush> {
        self.brushes.iter().find(|b| b.id == id)
    }

    fn check_pairwise_intersections(&self) -> Result<(), V3Error> {
        // Computing an exact brush AABB requires enumerating every triple-plane
        // vertex.  Cache that exact broad-phase result once per brush; repeating
        // it for every pair turned a semantically small Rich map into O(n²)
        // vertex enumerations.  The cache rejects only strictly disjoint boxes,
        // so coplanar contacts still reach the exact intersection test.
        let aabbs: Vec<_> = self
            .brushes
            .iter()
            .map(|brush| {
                brush.brush.aabb().map_err(|e| V3Error::AssemblyValidation {
                    detail: format!("invalid brush {}: {e}", brush.id),
                })
            })
            .collect::<Result<_, _>>()?;

        for i in 0..self.brushes.len() {
            for j in (i + 1)..self.brushes.len() {
                let a = &self.brushes[i];
                let b = &self.brushes[j];
                let aabb_a = aabbs[i];
                let aabb_b = aabbs[j];

                if aabb_a.0 .0 > aabb_b.1 .0
                    || aabb_b.0 .0 > aabb_a.1 .0
                    || aabb_a.0 .1 > aabb_b.1 .1
                    || aabb_b.0 .1 > aabb_a.1 .1
                    || aabb_a.0 .2 > aabb_b.1 .2
                    || aabb_b.0 .2 > aabb_a.1 .2
                {
                    continue;
                }

                let result = exact_intersection_volume(&a.brush, &b.brush)?;
                match result {
                    IntersectionResult::Disjoint => {}
                    IntersectionResult::ZeroVolumeContact(plane) => {
                        if !self.has_declared_interface(&a.id, &b.id) {
                            return Err(V3Error::UndeclaredContact {
                                brush_a: a.id.clone(),
                                brush_b: b.id.clone(),
                                plane,
                            });
                        }
                    }
                    IntersectionResult::PositiveVolume => {
                        return Err(V3Error::PositiveVolumeOverlap {
                            brush_a: a.id.clone(),
                            brush_b: b.id.clone(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn has_declared_interface(&self, a: &str, b: &str) -> bool {
        self.interfaces.iter().any(|iface| {
            (iface.brush_a == a && iface.brush_b == b) || (iface.brush_a == b && iface.brush_b == a)
        })
    }

    fn validate_interfaces(&self) -> Result<(), V3Error> {
        for iface in &self.interfaces {
            let brush_a = self
                .find_brush(&iface.brush_a)
                .ok_or_else(|| V3Error::UnknownBrush {
                    id: iface.brush_a.clone(),
                })?;
            let brush_b = self
                .find_brush(&iface.brush_b)
                .ok_or_else(|| V3Error::UnknownBrush {
                    id: iface.brush_b.clone(),
                })?;

            let face_a = brush_a
                .brush
                .faces
                .iter()
                .find(|f| f.role == iface.face_role_a)
                .ok_or_else(|| V3Error::MissingInterface {
                    interface_id: iface.id.clone(),
                    brush_a: iface.brush_a.clone(),
                    brush_b: iface.brush_b.clone(),
                })?;
            let face_b = brush_b
                .brush
                .faces
                .iter()
                .find(|f| f.role == iface.face_role_b)
                .ok_or_else(|| V3Error::MissingInterface {
                    interface_id: iface.id.clone(),
                    brush_a: iface.brush_a.clone(),
                    brush_b: iface.brush_b.clone(),
                })?;

            if !face_a.plane.is_coincident_with(&face_b.plane)? {
                return Err(V3Error::AssemblyValidation {
                    detail: format!(
                        "interface {} not coplanar: {} vs {}",
                        iface.id, face_a.plane, face_b.plane
                    ),
                });
            }
        }
        Ok(())
    }

    fn build_support_graph(&mut self) -> Result<(), V3Error> {
        self.support_edges.clear();
        for brush in &self.brushes {
            if let Support::SupportedBy {
                brush_id,
                interface_id,
            } = &brush.support
            {
                if self.find_brush(brush_id).is_none() {
                    return Err(V3Error::UnknownBrush {
                        id: brush_id.clone(),
                    });
                }
                let interface = self
                    .interfaces
                    .iter()
                    .find(|iface| iface.id == *interface_id)
                    .ok_or_else(|| V3Error::MissingInterface {
                        interface_id: interface_id.clone(),
                        brush_a: brush.id.clone(),
                        brush_b: brush_id.clone(),
                    })?;
                if !((interface.brush_a == brush.id && interface.brush_b == *brush_id)
                    || (interface.brush_b == brush.id && interface.brush_a == *brush_id))
                {
                    return Err(V3Error::MissingInterface {
                        interface_id: interface_id.clone(),
                        brush_a: brush.id.clone(),
                        brush_b: brush_id.clone(),
                    });
                }
                self.validate_positive_support_contact(
                    brush,
                    self.find_brush(brush_id)
                        .ok_or_else(|| V3Error::UnknownBrush {
                            id: brush_id.clone(),
                        })?,
                )?;
                self.support_edges
                    .push((brush.id.clone(), brush_id.clone()));
            }
        }
        Ok(())
    }

    fn validate_positive_support_contact(
        &self,
        child: &AssemblyBrush,
        parent: &AssemblyBrush,
    ) -> Result<(), V3Error> {
        if !matches!(
            exact_intersection_volume(&child.brush, &parent.brush)?,
            IntersectionResult::ZeroVolumeContact(_)
        ) {
            return Err(V3Error::AssemblyValidation {
                detail: format!(
                    "support {} -> {} does not contact at zero volume",
                    child.id, parent.id
                ),
            });
        }
        let (child_min, child_max) = child.brush.aabb()?;
        let (parent_min, parent_max) = parent.brush.aabb()?;
        let overlaps = [
            child_min.0 < parent_max.0 && child_max.0 > parent_min.0,
            child_min.1 < parent_max.1 && child_max.1 > parent_min.1,
            child_min.2 < parent_max.2 && child_max.2 > parent_min.2,
        ]
        .into_iter()
        .filter(|overlap| *overlap)
        .count();
        if overlaps < 2 {
            return Err(V3Error::AssemblyValidation {
                detail: format!(
                    "support {} -> {} has no positive-area face contact",
                    child.id, parent.id
                ),
            });
        }
        Ok(())
    }

    fn validate_support_graph(&self) -> Result<(), V3Error> {
        let mut adj: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (child, parent) in &self.support_edges {
            adj.entry(child).or_default().push(parent);
        }

        let world_supported: BTreeSet<&str> = self
            .brushes
            .iter()
            .filter(|b| b.support.is_world())
            .map(|b| b.id.as_str())
            .collect();

        // Every brush reaches world
        for brush in &self.brushes {
            if world_supported.contains(brush.id.as_str()) {
                continue;
            }
            let mut visited = BTreeSet::new();
            let mut stack = vec![brush.id.as_str()];
            let mut reaches = false;
            while let Some(current) = stack.pop() {
                if world_supported.contains(current) {
                    reaches = true;
                    break;
                }
                if !visited.insert(current) {
                    continue;
                }
                if let Some(parents) = adj.get(current) {
                    stack.extend(parents);
                }
            }
            if !reaches {
                return Err(V3Error::UnsupportedBrush {
                    id: brush.id.clone(),
                });
            }
        }

        // Cycle detection
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Color {
            White,
            Gray,
            Black,
        }

        let mut colors: BTreeMap<&str, Color> = BTreeMap::new();
        for brush in &self.brushes {
            colors.insert(&brush.id, Color::White);
        }

        fn dfs_cycle<'a>(
            node: &'a str,
            adj: &BTreeMap<&str, Vec<&'a str>>,
            colors: &mut BTreeMap<&'a str, Color>,
            path: &mut Vec<String>,
        ) -> Result<(), Vec<String>> {
            colors.insert(node, Color::Gray);
            path.push(node.to_string());
            if let Some(neighbors) = adj.get(node) {
                for &n in neighbors {
                    match colors.get(n) {
                        Some(Color::Gray) => {
                            path.push(n.to_string());
                            return Err(path.clone());
                        }
                        Some(Color::White) => {
                            dfs_cycle(n, adj, colors, path)?;
                        }
                        _ => {}
                    }
                }
            }
            colors.insert(node, Color::Black);
            path.pop();
            Ok(())
        }

        let mut path = Vec::new();
        for brush in &self.brushes {
            if colors[brush.id.as_str()] == Color::White {
                if let Err(cycle) = dfs_cycle(&brush.id, &adj, &mut colors, &mut path) {
                    return Err(V3Error::SupportGraphCycle { members: cycle });
                }
            }
        }
        Ok(())
    }

    fn check_protected_volume_intrusion(&self) -> Result<(), V3Error> {
        let brush_aabbs = self
            .brushes
            .iter()
            .map(|brush| brush.brush.aabb())
            .collect::<Result<Vec<_>, _>>()?;
        let protected_aabbs = self
            .protected_volumes
            .iter()
            .map(|volume| volume.brush.aabb())
            .collect::<Result<Vec<_>, _>>()?;

        for (brush_index, brush) in self.brushes.iter().enumerate() {
            let brush_aabb = brush_aabbs[brush_index];
            for (protected_index, pv) in self.protected_volumes.iter().enumerate() {
                let protected_aabb = protected_aabbs[protected_index];
                // Protected-volume validation rejects positive volume only;
                // plane contact is permitted. Equality therefore proves that
                // an axis has no positive overlap and can skip the exact test.
                if brush_aabb.1 .0 <= protected_aabb.0 .0
                    || protected_aabb.1 .0 <= brush_aabb.0 .0
                    || brush_aabb.1 .1 <= protected_aabb.0 .1
                    || protected_aabb.1 .1 <= brush_aabb.0 .1
                    || brush_aabb.1 .2 <= protected_aabb.0 .2
                    || protected_aabb.1 .2 <= brush_aabb.0 .2
                {
                    continue;
                }
                if matches!(
                    exact_intersection_volume(&brush.brush, &pv.brush)?,
                    IntersectionResult::PositiveVolume
                ) {
                    return Err(V3Error::ProtectedVolumeIntrusion {
                        brush_id: brush.id.clone(),
                        protected_id: pv.id.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

// ── Exact intersection ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
enum IntersectionResult {
    Disjoint,
    ZeroVolumeContact(String),
    PositiveVolume,
}

/// Compute the exact intersection of two convex brushes.
fn exact_intersection_volume(
    a: &ConvexBrush,
    b: &ConvexBrush,
) -> Result<IntersectionResult, V3Error> {
    let mut all_faces: Vec<geometry::BrushFace> = Vec::with_capacity(a.faces.len() + b.faces.len());
    for face in &a.faces {
        all_faces.push(face.clone());
    }
    for face in &b.faces {
        all_faces.push(face.clone());
    }

    if all_faces.len() < 4 {
        return Ok(IntersectionResult::Disjoint);
    }

    let vertices = geometry::half_space_vertices(&all_faces)?;

    if vertices.len() >= 4 {
        // Four or more vertices could still be coplanar (zero volume).
        // Test by computing the tetrahedral volume of the first 4 vertices.
        if vertices_span_volume(&vertices)? {
            return Ok(IntersectionResult::PositiveVolume);
        }
        // Coplanar: treat as zero-volume contact
        Ok(IntersectionResult::ZeroVolumeContact(
            "coplanar-contact".into(),
        ))
    } else if vertices.len() >= 3 {
        Ok(IntersectionResult::ZeroVolumeContact(
            "coplanar-contact".into(),
        ))
    } else {
        Ok(IntersectionResult::Disjoint)
    }
}

/// Check if a set of points spans a 3D volume (non-zero tetrahedral volume).
fn vertices_span_volume(vertices: &[Point3]) -> Result<bool, V3Error> {
    if vertices.len() < 4 {
        return Ok(false);
    }
    // Compute signed volume of tetrahedron formed by first 4 non-coplanar points
    let p0 = vertices[0];
    let v1 = vertices[1].checked_sub(p0)?;
    for i in 2..vertices.len() {
        let v2 = vertices[i].checked_sub(p0)?;
        for j in (i + 1)..vertices.len() {
            let v3 = vertices[j].checked_sub(p0)?;
            let det = scalar_triple_product(&v1, &v2, &v3)?;
            if det != Rational::ZERO {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn scalar_triple_product(a: &Vector3, b: &Vector3, c: &Vector3) -> Result<Rational, V3Error> {
    // a · (b × c)
    let cx = b.y.checked_mul(c.z)?.checked_sub(b.z.checked_mul(c.y)?)?;
    let cy = b.z.checked_mul(c.x)?.checked_sub(b.x.checked_mul(c.z)?)?;
    let cz = b.x.checked_mul(c.y)?.checked_sub(b.y.checked_mul(c.x)?)?;
    a.x.checked_mul(cx)?
        .checked_add(a.y.checked_mul(cy)?)?
        .checked_add(a.z.checked_mul(cz)?)
}

/// Build a structural wall shell for a room.
pub fn build_wall_shell(
    x_range: (i128, i128),
    y_range: (i128, i128),
    z_range: (i128, i128),
    id: &str,
) -> Result<AssemblyBrush, V3Error> {
    let brush = ConvexBrush::make_box(x_range, y_range, z_range)?;
    Ok(AssemblyBrush::new(
        id,
        BrushRole::WallShell,
        brush,
        Support::World {
            surface: FaceRole::Floor,
        },
    ))
}

/// Build a floor slab brush.
pub fn build_floor_slab(
    x_range: (i128, i128),
    y_range: (i128, i128),
    z_range: (i128, i128),
    id: &str,
) -> Result<AssemblyBrush, V3Error> {
    let brush = ConvexBrush::make_box(x_range, y_range, z_range)?;
    Ok(AssemblyBrush::new(
        id,
        BrushRole::FloorSlab,
        brush,
        Support::World {
            surface: FaceRole::Floor,
        },
    ))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::geometry::ConvexBrush;
    use super::*;

    fn make_box(x0: i128, y0: i128, z0: i128, x1: i128, y1: i128, z1: i128) -> ConvexBrush {
        ConvexBrush::make_box((x0, x1), (y0, y1), (z0, z1)).unwrap()
    }

    #[test]
    fn disjoint_brushes_pass_validation() {
        let b1 = make_box(0, 0, 0, 64, 64, 128);
        let b2 = make_box(128, 0, 0, 192, 64, 128);

        let brushes = vec![
            AssemblyBrush::new(
                "wall_a",
                BrushRole::WallShell,
                b1,
                Support::World {
                    surface: FaceRole::Floor,
                },
            ),
            AssemblyBrush::new(
                "wall_b",
                BrushRole::WallShell,
                b2,
                Support::World {
                    surface: FaceRole::Floor,
                },
            ),
        ];

        let assembly = Assembly::new(brushes, vec![], vec![]).unwrap();
        assert!(assembly.validated);
    }

    #[test]
    fn overlapping_brushes_rejected() {
        let b1 = make_box(0, 0, 0, 128, 128, 128);
        let b2 = make_box(64, 64, 64, 192, 192, 192);

        let brushes = vec![
            AssemblyBrush::new(
                "wall_a",
                BrushRole::WallShell,
                b1,
                Support::World {
                    surface: FaceRole::Floor,
                },
            ),
            AssemblyBrush::new(
                "wall_b",
                BrushRole::WallShell,
                b2,
                Support::World {
                    surface: FaceRole::Floor,
                },
            ),
        ];

        let result = Assembly::new(brushes, vec![], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn duplicate_id_rejected() {
        let b1 = make_box(0, 0, 0, 64, 64, 128);
        let b2 = make_box(128, 0, 0, 192, 64, 128);
        let b3 = make_box(256, 0, 0, 320, 64, 128);

        let brushes = vec![
            AssemblyBrush::new(
                "wall_a",
                BrushRole::WallShell,
                b1,
                Support::World {
                    surface: FaceRole::Floor,
                },
            ),
            AssemblyBrush::new(
                "wall_a", // duplicate
                BrushRole::WallShell,
                b2,
                Support::World {
                    surface: FaceRole::Floor,
                },
            ),
            AssemblyBrush::new(
                "wall_c",
                BrushRole::WallShell,
                b3,
                Support::World {
                    surface: FaceRole::Floor,
                },
            ),
        ];

        let result = Assembly::new(brushes, vec![], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn protected_volume_intrusion_rejected() {
        let b1 = make_box(0, 0, 0, 128, 128, 128);
        let pv = ProtectedVolume {
            id: "spawn_zone".into(),
            brush: make_box(32, 32, 32, 96, 96, 96),
        };

        let brushes = vec![AssemblyBrush::new(
            "wall_a",
            BrushRole::WallShell,
            b1,
            Support::World {
                surface: FaceRole::Floor,
            },
        )];

        let result = Assembly::new(brushes, vec![], vec![pv]);
        assert!(result.is_err());
    }

    #[test]
    fn build_wall_shell_creates_valid_brush() {
        let brush = build_wall_shell((0, 256), (0, 16), (0, 176), "wall_north").unwrap();
        assert_eq!(brush.role, BrushRole::WallShell);
        assert!(brush.support.is_world());
    }

    #[test]
    fn build_floor_slab_creates_valid_brush() {
        let brush = build_floor_slab((0, 256), (0, 256), (0, 16), "floor_0").unwrap();
        assert_eq!(brush.role, BrushRole::FloorSlab);
    }
}
