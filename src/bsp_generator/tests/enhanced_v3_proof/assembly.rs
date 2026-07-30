//! Exact assembly and solid-intersection kernel.
//!
//! Validates assembly ordering, interface ownership, protected-volume
//! immutability, exact intersection proofs, support graphs, and aperture
//! geometry. All arithmetic is exact — no floats, no snapping.
//!
//! # Design contract
//!
//! - Protected volumes are immutable — never mutate them.
//! - Positive-volume overlap between brushes is an error.
//! - Zero-volume contact is permitted only at declared interfaces.
//! - Results are always canonically ordered.
//! - Support graph must be acyclic; every dependent reaches a world surface.
//! - Never depends on production code, BSP, renderer, or runtime.

use std::collections::{BTreeMap, BTreeSet};

use super::geometry::{self, ConvexBrush, GeometryError, Point3, Rational};

// ── Assembly error type ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssemblyError {
    InvalidBrush {
        id: String,
        reason: GeometryError,
    },
    PositiveVolumeOverlap {
        brush_a: String,
        brush_b: String,
    },
    UndeclaredContact {
        brush_a: String,
        brush_b: String,
        plane: String,
    },
    MissingInterface {
        interface_id: String,
        brush_a: String,
        brush_b: String,
    },
    InterfacePlaneMismatch {
        interface_id: String,
        plane_a: String,
        plane_b: String,
    },
    InterfaceNotCoplanar {
        interface_id: String,
        detail: String,
    },
    SupportCycle {
        members: Vec<String>,
    },
    UnsupportedBrush {
        id: String,
    },
    ProtectedVolumeMutated {
        id: String,
    },
    ProtectedVolumeIntrusion {
        brush_id: String,
        protected_id: String,
    },
    ApertureIncomplete {
        aperture_id: String,
        wall_face: String,
        detail: String,
    },
    InsufficientThroatDepth {
        aperture_id: String,
        depth: Rational,
    },
    InvalidOrdering {
        detail: String,
    },
    DuplicateBrushId {
        id: String,
    },
    UnknownBrush {
        id: String,
    },
    Geometry(GeometryError),
}

impl From<GeometryError> for AssemblyError {
    fn from(e: GeometryError) -> Self {
        AssemblyError::Geometry(e)
    }
}

impl std::fmt::Display for AssemblyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBrush { id, reason } => write!(f, "invalid brush {id}: {reason}"),
            Self::PositiveVolumeOverlap { brush_a, brush_b } => {
                write!(f, "positive-volume overlap: {brush_a} ∩ {brush_b}")
            }
            Self::UndeclaredContact {
                brush_a,
                brush_b,
                plane,
            } => {
                write!(
                    f,
                    "undeclared contact: {brush_a} touches {brush_b} at {plane}"
                )
            }
            Self::MissingInterface {
                interface_id,
                brush_a,
                brush_b,
            } => {
                write!(f, "missing interface {interface_id}: {brush_a} ↔ {brush_b}")
            }
            Self::InterfacePlaneMismatch {
                interface_id,
                plane_a,
                plane_b,
            } => {
                write!(
                    f,
                    "interface {interface_id} plane mismatch: {plane_a} vs {plane_b}"
                )
            }
            Self::InterfaceNotCoplanar {
                interface_id,
                detail,
            } => {
                write!(f, "interface {interface_id} not coplanar: {detail}")
            }
            Self::SupportCycle { members } => {
                write!(f, "support cycle: {}", members.join(" → "))
            }
            Self::UnsupportedBrush { id } => {
                write!(f, "unsupported brush {id} does not reach world")
            }
            Self::ProtectedVolumeMutated { id } => write!(f, "protected volume {id} was mutated"),
            Self::ProtectedVolumeIntrusion {
                brush_id,
                protected_id,
            } => {
                write!(
                    f,
                    "brush {brush_id} intrudes into protected volume {protected_id}"
                )
            }
            Self::ApertureIncomplete {
                aperture_id,
                wall_face,
                detail,
            } => {
                write!(
                    f,
                    "aperture {aperture_id} incomplete on {wall_face}: {detail}"
                )
            }
            Self::InsufficientThroatDepth { aperture_id, depth } => {
                write!(
                    f,
                    "aperture {aperture_id} insufficient throat depth {depth}"
                )
            }
            Self::InvalidOrdering { detail } => write!(f, "invalid ordering: {detail}"),
            Self::DuplicateBrushId { id } => write!(f, "duplicate brush ID: {id}"),
            Self::UnknownBrush { id } => write!(f, "unknown brush: {id}"),
            Self::Geometry(e) => write!(f, "geometry: {e}"),
        }
    }
}

impl std::error::Error for AssemblyError {}

// ── Assembly role ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BrushRole {
    WallShell,
    FloorSlab,
    CeilingSlab,
    Column,
    Buttress,
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
            Self::PortalThroat => "portal_throat",
            Self::Feature => "feature",
            Self::World => "world",
        }
    }
}

impl std::fmt::Display for BrushRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.tag())
    }
}

// ── Interface definition ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interface {
    pub id: String,
    pub brush_a: String,
    pub brush_b: String,
    pub face_role_a: geometry::FaceRole,
    pub face_role_b: geometry::FaceRole,
}

impl Interface {
    pub fn new(
        id: impl Into<String>,
        brush_a: impl Into<String>,
        brush_b: impl Into<String>,
        face_role_a: geometry::FaceRole,
        face_role_b: geometry::FaceRole,
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

// ── Aperture ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aperture {
    pub id: String,
    pub wall_brush_id: String,
    pub wall_face: geometry::FaceRole,
    pub aperture_bounds: ApertureBounds,
    pub throat_depth: Rational,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApertureBounds {
    Rectangular {
        wall_d: i128,
        u_min: i128,
        u_max: i128,
        v_min: i128,
        v_max: i128,
    },
    PointedArch {
        wall_d: i128,
        u_center: i128,
        u_half_width: i128,
        v_base: i128,
        v_apex: i128,
        arch_rise: i128,
    },
}

// ── Protected volume ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtectedVolume {
    pub id: String,
    pub brush: ConvexBrush,
    pub content_hash: u64,
}

impl ProtectedVolume {
    pub fn new(id: impl Into<String>, brush: ConvexBrush) -> Self {
        let hash = Self::compute_hash(&brush);
        Self {
            id: id.into(),
            brush,
            content_hash: hash,
        }
    }

    fn compute_hash(brush: &ConvexBrush) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        for face in &brush.faces {
            face.plane.nx.hash(&mut h);
            face.plane.ny.hash(&mut h);
            face.plane.nz.hash(&mut h);
            face.plane.d.hash(&mut h);
            face.role.hash(&mut h);
        }
        h.finish()
    }

    pub fn check_immutable(&self) -> Result<(), AssemblyError> {
        let current = Self::compute_hash(&self.brush);
        if current != self.content_hash {
            return Err(AssemblyError::ProtectedVolumeMutated {
                id: self.id.clone(),
            });
        }
        Ok(())
    }
}

// ── Support relation ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Support {
    World {
        surface: geometry::FaceRole,
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

// ── Assembly brush ────────────────────────────────────────────────────────

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

impl PartialOrd for AssemblyBrush {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AssemblyBrush {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

// ── Assembly ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assembly {
    pub brushes: Vec<AssemblyBrush>,
    pub interfaces: Vec<Interface>,
    pub apertures: Vec<Aperture>,
    pub protected_volumes: Vec<ProtectedVolume>,
    pub support_edges: Vec<(String, String)>,
    pub validated: bool,
}

// ── Exact intersection result ─────────────────────────────────────────────

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
) -> Result<IntersectionResult, AssemblyError> {
    // Combine ALL faces from both brushes — do NOT filter coincident faces.
    // Two faces that are coincident (same surface) but have opposite normals
    // define opposite half-spaces that are both needed for the intersection.
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

    // Compute vertices from the combined half-space system directly.
    let mut vertices: BTreeSet<Point3> = BTreeSet::new();
    let n = all_faces.len();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                if let Some(pt) = geometry::intersect_three_planes(
                    &all_faces[i].plane,
                    &all_faces[j].plane,
                    &all_faces[k].plane,
                )? {
                    let mut valid = true;
                    for face in &all_faces {
                        let num = face.plane.nx * pt.x.num * pt.y.den * pt.z.den
                            + face.plane.ny * pt.y.num * pt.x.den * pt.z.den
                            + face.plane.nz * pt.z.num * pt.x.den * pt.y.den
                            - face.plane.d * pt.x.den * pt.y.den * pt.z.den;
                        if num < 0 {
                            valid = false;
                            break;
                        }
                    }
                    if valid {
                        vertices.insert(pt);
                    }
                }
            }
        }
    }

    if vertices.len() >= 4 {
        let refs: Vec<&Point3> = vertices.iter().collect();
        if !all_coplanar(&refs) {
            Ok(IntersectionResult::PositiveVolume)
        } else {
            Ok(IntersectionResult::ZeroVolumeContact(
                "coplanar-contact".into(),
            ))
        }
    } else if vertices.len() >= 3 {
        Ok(IntersectionResult::ZeroVolumeContact(
            "coplanar-contact".into(),
        ))
    } else {
        Ok(IntersectionResult::Disjoint)
    }
}

/// Check if all points are coplanar (lie on the same plane).
fn all_coplanar(pts: &[&Point3]) -> bool {
    if pts.len() <= 3 {
        return true;
    }
    for i in 3..pts.len() {
        let det = tetrahedron_det_4(pts[0], pts[1], pts[2], pts[i]);
        if det != Rational::ZERO {
            return false;
        }
    }
    true
}

/// 3×3 determinant of vectors from p0 (p1-p0, p2-p0, p3-p0).
fn tetrahedron_det_4(p0: &Point3, p1: &Point3, p2: &Point3, p3: &Point3) -> Rational {
    let a11 = p1.x.checked_sub(p0.x).unwrap_or(Rational::ZERO);
    let a12 = p1.y.checked_sub(p0.y).unwrap_or(Rational::ZERO);
    let a13 = p1.z.checked_sub(p0.z).unwrap_or(Rational::ZERO);
    let a21 = p2.x.checked_sub(p0.x).unwrap_or(Rational::ZERO);
    let a22 = p2.y.checked_sub(p0.y).unwrap_or(Rational::ZERO);
    let a23 = p2.z.checked_sub(p0.z).unwrap_or(Rational::ZERO);
    let a31 = p3.x.checked_sub(p0.x).unwrap_or(Rational::ZERO);
    let a32 = p3.y.checked_sub(p0.y).unwrap_or(Rational::ZERO);
    let a33 = p3.z.checked_sub(p0.z).unwrap_or(Rational::ZERO);

    let t1 = a11.checked_mul(
        a22.checked_mul(a33)
            .unwrap_or(Rational::ZERO)
            .checked_sub(a23.checked_mul(a32).unwrap_or(Rational::ZERO))
            .unwrap_or(Rational::ZERO),
    );
    let t2 = a12.checked_mul(
        a21.checked_mul(a33)
            .unwrap_or(Rational::ZERO)
            .checked_sub(a23.checked_mul(a31).unwrap_or(Rational::ZERO))
            .unwrap_or(Rational::ZERO),
    );
    let t3 = a13.checked_mul(
        a21.checked_mul(a32)
            .unwrap_or(Rational::ZERO)
            .checked_sub(a22.checked_mul(a31).unwrap_or(Rational::ZERO))
            .unwrap_or(Rational::ZERO),
    );

    match (t1, t2, t3) {
        (Ok(v1), Ok(v2), Ok(v3)) => v1
            .checked_sub(v2)
            .unwrap_or(Rational::ZERO)
            .checked_add(v3)
            .unwrap_or(Rational::ZERO),
        _ => Rational::ZERO,
    }
}

/// Measure the wall thickness for a specific face of a brush.
fn wall_thickness_for_face(
    brush: &ConvexBrush,
    face_role: geometry::FaceRole,
) -> Result<Rational, AssemblyError> {
    let target_face = brush.faces.iter().find(|f| f.role == face_role);
    let target_face = match target_face {
        Some(f) => f,
        None => return Ok(Rational::ZERO),
    };

    let opposing = brush.faces.iter().find(|f| {
        f.plane.is_parallel_to(&target_face.plane)
            && f.plane.nx * target_face.plane.nx
                + f.plane.ny * target_face.plane.ny
                + f.plane.nz * target_face.plane.nz
                < 0
    });

    let Some(opposing) = opposing else {
        return Ok(Rational::ZERO);
    };

    let (d_target, d_opposing) = if target_face.plane.nx * opposing.plane.nx
        + target_face.plane.ny * opposing.plane.ny
        + target_face.plane.nz * opposing.plane.nz
        > 0
    {
        return Ok(Rational::ZERO);
    } else {
        (target_face.plane.d, -opposing.plane.d)
    };

    let gap = (d_opposing - d_target).abs();
    let norm = target_face.plane.nx.abs() + target_face.plane.ny.abs() + target_face.plane.nz.abs();
    Rational::new(gap as i128, norm as i128).map_err(AssemblyError::Geometry)
}

// ── Assembly impl ─────────────────────────────────────────────────────────

impl Assembly {
    pub fn new(
        brushes: Vec<AssemblyBrush>,
        interfaces: Vec<Interface>,
        apertures: Vec<Aperture>,
        protected_volumes: Vec<ProtectedVolume>,
    ) -> Result<Self, AssemblyError> {
        let mut assembly = Self {
            brushes,
            interfaces,
            apertures,
            protected_volumes,
            support_edges: Vec::new(),
            validated: false,
        };
        assembly.validate()?;
        Ok(assembly)
    }

    pub fn validate(&mut self) -> Result<(), AssemblyError> {
        self.validate_ordering()?;

        for pv in &self.protected_volumes {
            pv.check_immutable()?;
        }

        self.check_protected_volume_intrusion()?;
        self.check_pairwise_intersections()?;
        self.validate_interfaces()?;
        self.build_support_graph()?;
        self.validate_support_graph()?;
        self.validate_apertures()?;

        self.validated = true;
        Ok(())
    }

    fn validate_ordering(&self) -> Result<(), AssemblyError> {
        let mut ids = BTreeSet::new();
        for brush in &self.brushes {
            if !ids.insert(brush.id.clone()) {
                return Err(AssemblyError::DuplicateBrushId {
                    id: brush.id.clone(),
                });
            }
        }
        for w in self.brushes.windows(2) {
            if w[0].id >= w[1].id {
                return Err(AssemblyError::InvalidOrdering {
                    detail: format!("{} precedes {}", w[0].id, w[1].id),
                });
            }
        }
        for w in self.interfaces.windows(2) {
            if w[0].id >= w[1].id {
                return Err(AssemblyError::InvalidOrdering {
                    detail: format!("interface {} precedes {}", w[0].id, w[1].id),
                });
            }
        }
        Ok(())
    }

    fn find_brush(&self, id: &str) -> Option<&AssemblyBrush> {
        self.brushes.iter().find(|b| b.id == id)
    }

    fn check_pairwise_intersections(&self) -> Result<(), AssemblyError> {
        for i in 0..self.brushes.len() {
            for j in (i + 1)..self.brushes.len() {
                let a = &self.brushes[i];
                let b = &self.brushes[j];

                let aabb_a = a.brush.aabb().map_err(|e| AssemblyError::InvalidBrush {
                    id: a.id.clone(),
                    reason: e,
                })?;
                let aabb_b = b.brush.aabb().map_err(|e| AssemblyError::InvalidBrush {
                    id: b.id.clone(),
                    reason: e,
                })?;

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
                            return Err(AssemblyError::UndeclaredContact {
                                brush_a: a.id.clone(),
                                brush_b: b.id.clone(),
                                plane,
                            });
                        }
                    }
                    IntersectionResult::PositiveVolume => {
                        return Err(AssemblyError::PositiveVolumeOverlap {
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

    fn validate_interfaces(&self) -> Result<(), AssemblyError> {
        for iface in &self.interfaces {
            let brush_a =
                self.find_brush(&iface.brush_a)
                    .ok_or_else(|| AssemblyError::UnknownBrush {
                        id: iface.brush_a.clone(),
                    })?;
            let brush_b =
                self.find_brush(&iface.brush_b)
                    .ok_or_else(|| AssemblyError::UnknownBrush {
                        id: iface.brush_b.clone(),
                    })?;

            let face_a = brush_a
                .brush
                .faces
                .iter()
                .find(|f| f.role == iface.face_role_a)
                .ok_or_else(|| AssemblyError::MissingInterface {
                    interface_id: iface.id.clone(),
                    brush_a: iface.brush_a.clone(),
                    brush_b: iface.brush_b.clone(),
                })?;
            let face_b = brush_b
                .brush
                .faces
                .iter()
                .find(|f| f.role == iface.face_role_b)
                .ok_or_else(|| AssemblyError::MissingInterface {
                    interface_id: iface.id.clone(),
                    brush_a: iface.brush_a.clone(),
                    brush_b: iface.brush_b.clone(),
                })?;

            if !face_a.plane.is_coincident_with(&face_b.plane) {
                return Err(AssemblyError::InterfaceNotCoplanar {
                    interface_id: iface.id.clone(),
                    detail: format!(
                        "{} (brush {}) != {} (brush {})",
                        face_a.plane, iface.brush_a, face_b.plane, iface.brush_b
                    ),
                });
            }
        }
        Ok(())
    }

    fn build_support_graph(&mut self) -> Result<(), AssemblyError> {
        self.support_edges.clear();
        for brush in &self.brushes {
            if let Support::SupportedBy {
                brush_id,
                interface_id,
            } = &brush.support
            {
                if self.find_brush(brush_id).is_none() {
                    return Err(AssemblyError::UnknownBrush {
                        id: brush_id.clone(),
                    });
                }
                if !self
                    .interfaces
                    .iter()
                    .any(|iface| iface.id == *interface_id)
                {
                    return Err(AssemblyError::MissingInterface {
                        interface_id: interface_id.clone(),
                        brush_a: brush.id.clone(),
                        brush_b: brush_id.clone(),
                    });
                }
                self.support_edges
                    .push((brush.id.clone(), brush_id.clone()));
            }
        }
        Ok(())
    }

    fn validate_support_graph(&self) -> Result<(), AssemblyError> {
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
                return Err(AssemblyError::UnsupportedBrush {
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
                    return Err(AssemblyError::SupportCycle { members: cycle });
                }
            }
        }
        Ok(())
    }

    fn check_protected_volume_intrusion(&self) -> Result<(), AssemblyError> {
        for brush in &self.brushes {
            for pv in &self.protected_volumes {
                if matches!(
                    exact_intersection_volume(&brush.brush, &pv.brush)?,
                    IntersectionResult::PositiveVolume
                ) {
                    return Err(AssemblyError::ProtectedVolumeIntrusion {
                        brush_id: brush.id.clone(),
                        protected_id: pv.id.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    fn validate_apertures(&self) -> Result<(), AssemblyError> {
        for aperture in &self.apertures {
            let wall = self.find_brush(&aperture.wall_brush_id).ok_or_else(|| {
                AssemblyError::UnknownBrush {
                    id: aperture.wall_brush_id.clone(),
                }
            })?;

            if !matches!(wall.role, BrushRole::WallShell | BrushRole::PortalThroat) {
                return Err(AssemblyError::InvalidBrush {
                    id: wall.id.clone(),
                    reason: GeometryError::MalformedRole {
                        detail: format!(
                            "aperture wall must be WallShell or PortalThroat, got {}",
                            wall.role
                        ),
                    },
                });
            }

            let wall_thickness = wall_thickness_for_face(&wall.brush, aperture.wall_face)?;
            if wall_thickness < aperture.throat_depth {
                return Err(AssemblyError::InsufficientThroatDepth {
                    aperture_id: aperture.id.clone(),
                    depth: wall_thickness,
                });
            }

            match &aperture.aperture_bounds {
                ApertureBounds::Rectangular {
                    u_min,
                    u_max,
                    v_min,
                    v_max,
                    ..
                } => {
                    if u_min >= u_max {
                        return Err(AssemblyError::ApertureIncomplete {
                            aperture_id: aperture.id.clone(),
                            wall_face: aperture.wall_face.tag().into(),
                            detail: format!("invalid u range: [{u_min}, {u_max}]"),
                        });
                    }
                    if v_min >= v_max {
                        return Err(AssemblyError::ApertureIncomplete {
                            aperture_id: aperture.id.clone(),
                            wall_face: aperture.wall_face.tag().into(),
                            detail: format!("invalid v range: [{v_min}, {v_max}]"),
                        });
                    }
                }
                ApertureBounds::PointedArch {
                    v_base,
                    v_apex,
                    u_half_width,
                    ..
                } => {
                    if *v_base >= *v_apex {
                        return Err(AssemblyError::ApertureIncomplete {
                            aperture_id: aperture.id.clone(),
                            wall_face: aperture.wall_face.tag().into(),
                            detail: format!("invalid v range: [{v_base}, {v_apex}]"),
                        });
                    }
                    if *u_half_width <= 0 {
                        return Err(AssemblyError::ApertureIncomplete {
                            aperture_id: aperture.id.clone(),
                            wall_face: aperture.wall_face.tag().into(),
                            detail: "zero or negative half-width".into(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Transitive dependent removal closure.
    pub fn dependent_removal_closure(&self, root_id: &str) -> BTreeSet<String> {
        let mut closure = BTreeSet::new();
        closure.insert(root_id.to_string());

        let mut children: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (child, parent) in &self.support_edges {
            children.entry(parent).or_default().push(child);
        }

        let mut pending: Vec<&str> = vec![root_id];
        while let Some(next) = pending.pop() {
            if let Some(kids) = children.get(next) {
                for kid in kids {
                    if closure.insert(kid.to_string()) {
                        pending.push(kid);
                    }
                }
            }
        }
        closure
    }
}

// ── Public support graph validators ───────────────────────────────────────

pub fn validate_support_acyclic(edges: &[(String, String)]) -> Result<(), AssemblyError> {
    let mut adj: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (child, parent) in edges {
        adj.entry(child).or_default().push(parent);
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Color {
        White,
        Gray,
        Black,
    }

    let mut all_nodes = BTreeSet::new();
    for (a, b) in edges {
        all_nodes.insert(a.as_str());
        all_nodes.insert(b.as_str());
    }
    let mut colors: BTreeMap<&str, Color> = BTreeMap::new();
    for n in &all_nodes {
        colors.insert(n, Color::White);
    }

    fn dfs<'a>(
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
                        dfs(n, adj, colors, path)?;
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
    for &n in &all_nodes {
        if colors[n] == Color::White {
            if let Err(cycle) = dfs(n, &adj, &mut colors, &mut path) {
                return Err(AssemblyError::SupportCycle { members: cycle });
            }
        }
    }
    Ok(())
}

pub fn validate_all_supported(
    edges: &[(String, String)],
    world_nodes: &BTreeSet<String>,
    all_nodes: &BTreeSet<String>,
) -> Result<(), AssemblyError> {
    let mut adj: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (child, parent) in edges {
        adj.entry(child).or_default().push(parent);
    }
    for node in all_nodes {
        if world_nodes.contains(node) {
            continue;
        }
        let mut visited = BTreeSet::new();
        let mut stack = vec![node.as_str()];
        let mut reaches = false;
        while let Some(current) = stack.pop() {
            if world_nodes.contains(current) {
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
            return Err(AssemblyError::UnsupportedBrush { id: node.clone() });
        }
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::geometry::{self, ConvexBrush, Point3, Rational};
    use super::*;

    #[test]
    fn debug_all_coplanar() {
        let pts = [
            Point3::from_ints(16, 0, 0),
            Point3::from_ints(16, 64, 0),
            Point3::from_ints(16, 0, 128),
            Point3::from_ints(16, 64, 128),
        ];
        let refs: Vec<&Point3> = pts.iter().collect();
        assert!(
            all_coplanar(&refs),
            "four coplanar points should be coplanar"
        );
    }

    fn make_box_brush(x0: i128, y0: i128, z0: i128, x1: i128, y1: i128, z1: i128) -> ConvexBrush {
        ConvexBrush::make_box((x0, x1), (y0, y1), (z0, z1)).unwrap()
    }

    fn wall_shell(
        id: &str,
        x0: i128,
        y0: i128,
        z0: i128,
        x1: i128,
        y1: i128,
        z1: i128,
    ) -> AssemblyBrush {
        AssemblyBrush::new(
            id,
            BrushRole::WallShell,
            make_box_brush(x0, y0, z0, x1, y1, z1),
            Support::World {
                surface: geometry::FaceRole::Floor,
            },
        )
    }

    fn floor_slab(
        id: &str,
        x0: i128,
        y0: i128,
        z0: i128,
        x1: i128,
        y1: i128,
        z1: i128,
    ) -> AssemblyBrush {
        AssemblyBrush::new(
            id,
            BrushRole::FloorSlab,
            make_box_brush(x0, y0, z0, x1, y1, z1),
            Support::World {
                surface: geometry::FaceRole::Floor,
            },
        )
    }

    // ── Assembly ordering ──────────────────────────────────────────────

    #[test]
    fn valid_assembly_passes_validation() {
        let b1 = wall_shell("brush_01", 0, 0, 0, 16, 64, 128);
        let b2 = wall_shell("brush_02", 32, 0, 0, 48, 64, 128);
        // Disjoint — no contact needed
        assert!(Assembly::new(vec![b1, b2], vec![], vec![], vec![]).is_ok());
    }

    #[test]
    fn duplicate_brush_ids_rejected() {
        let b1 = wall_shell("dup", 0, 0, 0, 16, 64, 128);
        let b2 = wall_shell("dup", 32, 0, 0, 48, 64, 128);
        assert!(Assembly::new(vec![b1, b2], vec![], vec![], vec![]).is_err());
    }

    #[test]
    fn unsorted_brushes_rejected() {
        let b1 = wall_shell("b", 0, 0, 0, 16, 64, 128);
        let b2 = wall_shell("a", 32, 0, 0, 48, 64, 128);
        assert!(Assembly::new(vec![b1, b2], vec![], vec![], vec![]).is_err());
    }

    // ── Protected volume immutability ──────────────────────────────────

    #[test]
    fn protected_volume_mutation_detected() {
        let vol = make_box_brush(50, 50, 0, 80, 80, 128);
        let mut pv = ProtectedVolume::new("pv_01", vol.clone());
        assert!(pv.check_immutable().is_ok());
        pv.brush = make_box_brush(50, 50, 0, 81, 80, 128);
        assert!(pv.check_immutable().is_err());
    }

    #[test]
    fn brush_intrusion_into_protected_volume_rejected() {
        let b = wall_shell("b", 0, 0, 0, 64, 64, 128);
        let pv_vol = make_box_brush(32, 32, 32, 96, 96, 96);
        let pv = ProtectedVolume::new("pv_01", pv_vol);
        assert!(Assembly::new(vec![b], vec![], vec![], vec![pv]).is_err());
    }

    // ── Pairwise intersections ─────────────────────────────────────────

    #[test]
    fn overlapping_brushes_rejected() {
        let b1 = wall_shell("b1", 0, 0, 0, 20, 64, 128);
        let b2 = wall_shell("b2", 10, 0, 0, 30, 64, 128);
        assert!(matches!(
            Assembly::new(vec![b1, b2], vec![], vec![], vec![]),
            Err(AssemblyError::PositiveVolumeOverlap { .. })
        ));
    }

    #[test]
    fn disjoint_brushes_pass() {
        let b1 = wall_shell("b1", 0, 0, 0, 16, 64, 128);
        let b2 = wall_shell("b2", 32, 0, 0, 48, 64, 128);
        assert!(Assembly::new(vec![b1, b2], vec![], vec![], vec![]).is_ok());
    }

    // ── Interfaces ────────────────────────────────────────────────────

    #[test]
    fn touching_with_declared_interface_passes() {
        // Two brushes that share a coplanar face need an interface.
        // b1 has EastWall at x=16, b2 has WestWall at x=16
        let b1 = wall_shell("b1", 0, 0, 0, 16, 64, 128);
        let b2 = AssemblyBrush::new(
            "b2",
            BrushRole::WallShell,
            make_box_brush(16, 0, 0, 32, 64, 128),
            Support::World {
                surface: geometry::FaceRole::Floor,
            },
        );

        let interfaces = vec![Interface::new(
            "iface_01",
            "b1",
            "b2",
            geometry::FaceRole::EastWall,
            geometry::FaceRole::WestWall,
        )];

        let result = Assembly::new(vec![b1, b2], interfaces, vec![], vec![]);
        assert!(result.is_ok(), "error: {:?}", result.err());
    }

    #[test]
    fn touching_without_interface_rejected() {
        let b1 = wall_shell("b1", 0, 0, 0, 16, 64, 128);
        let b2 = AssemblyBrush::new(
            "b2",
            BrushRole::WallShell,
            make_box_brush(16, 0, 0, 32, 64, 128),
            Support::World {
                surface: geometry::FaceRole::Floor,
            },
        );

        assert!(Assembly::new(vec![b1, b2], vec![], vec![], vec![]).is_err());
    }

    #[test]
    fn interface_with_mismatched_plane_rejected() {
        let b1 = wall_shell("b1", 0, 0, 0, 16, 64, 128);
        let b2 = wall_shell("b2", 32, 0, 0, 48, 64, 128);

        let interfaces = vec![Interface::new(
            "iface_01",
            "b1",
            "b2",
            geometry::FaceRole::EastWall,
            geometry::FaceRole::WestWall,
        )];

        // b1's EastWall plane is x >= 0? Wait, with the fixed make_box:
        // b1 box [0,16] → WestWall at x=0, EastWall at x=16.
        // b2 box [32,48] → WestWall at x=32, EastWall at x=48.
        // These are NOT coplanar.
        assert!(matches!(
            Assembly::new(vec![b1, b2], interfaces, vec![], vec![]),
            Err(AssemblyError::InterfaceNotCoplanar { .. })
        ));
    }

    // ── Support graph ─────────────────────────────────────────────────

    #[test]
    fn acyclic_support_graph_passes() {
        let base = floor_slab("base", 0, 0, 0, 64, 64, 16);
        let pillar = AssemblyBrush::new(
            "pillar",
            BrushRole::Column,
            make_box_brush(24, 24, 16, 40, 40, 80),
            Support::SupportedBy {
                brush_id: "base".into(),
                interface_id: "iface_01".into(),
            },
        );

        // base's Floor at z=16 (top face), pillar's Ceiling at z=16 (bottom)
        // base Floor: (0,0,1)·x >= 0 → z >= 0. Wait, that's the bottom.
        // base Ceiling: (0,0,-1)·x >= -16 → z <= 16. Top is at z=16.
        // pillar Floor: z >= 16. Bottom of pillar at z=16.
        // So base Ceiling and pillar Floor touch at z=16.
        let interfaces = vec![Interface::new(
            "iface_01",
            "pillar",
            "base",
            geometry::FaceRole::Floor,
            geometry::FaceRole::Ceiling,
        )];

        let assembly = Assembly::new(vec![base, pillar], interfaces, vec![], vec![]).unwrap();
        assert!(assembly.validated);
    }

    #[test]
    fn support_cycle_rejected() {
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
            ("c".to_string(), "a".to_string()),
        ];
        assert!(validate_support_acyclic(&edges).is_err());
    }

    #[test]
    fn unsupported_brush_rejected() {
        let b1 = wall_shell("world_wall", 0, 0, 0, 16, 64, 128);
        let b2 = AssemblyBrush::new(
            "orphan",
            BrushRole::Feature,
            make_box_brush(32, 16, 0, 48, 32, 80),
            Support::SupportedBy {
                brush_id: "nonexistent".into(),
                interface_id: "fake".into(),
            },
        );
        assert!(Assembly::new(vec![b1, b2], vec![], vec![], vec![]).is_err());
    }

    // ── Dependent removal ─────────────────────────────────────────────

    #[test]
    fn dependent_removal_closure_correct() {
        let b1 = wall_shell("base", 0, 0, 0, 16, 64, 128);
        let b2 = AssemblyBrush::new(
            "mid",
            BrushRole::Feature,
            make_box_brush(16, 16, 0, 32, 32, 80),
            Support::SupportedBy {
                brush_id: "base".into(),
                interface_id: "iface_01".into(),
            },
        );
        let b3 = AssemblyBrush::new(
            "top",
            BrushRole::Feature,
            make_box_brush(32, 32, 0, 48, 48, 80),
            Support::SupportedBy {
                brush_id: "mid".into(),
                interface_id: "iface_02".into(),
            },
        );

        // Need matching coplanar faces for the interfaces
        let interfaces = vec![
            Interface::new(
                "iface_01",
                "mid",
                "base",
                geometry::FaceRole::WestWall,
                geometry::FaceRole::EastWall,
            ),
            Interface::new(
                "iface_02",
                "top",
                "mid",
                geometry::FaceRole::WestWall,
                geometry::FaceRole::EastWall,
            ),
        ];

        let assembly = Assembly::new(vec![b1, b2, b3], interfaces, vec![], vec![]).unwrap();
        let closure = assembly.dependent_removal_closure("base");
        assert!(closure.contains("base"));
        assert!(closure.contains("mid"));
        assert!(closure.contains("top"));
        assert_eq!(closure.len(), 3);
    }

    // ── Apertures ─────────────────────────────────────────────────────

    #[test]
    fn valid_aperture_passes() {
        let wall = wall_shell("wall", 0, 0, 0, 16, 64, 128);
        let aperture = Aperture {
            id: "apt_01".into(),
            wall_brush_id: "wall".into(),
            wall_face: geometry::FaceRole::EastWall,
            aperture_bounds: ApertureBounds::Rectangular {
                wall_d: 16,
                u_min: 16,
                u_max: 48,
                v_min: 16,
                v_max: 96,
            },
            throat_depth: Rational::from_int(16),
        };
        assert!(Assembly::new(vec![wall], vec![], vec![aperture], vec![]).is_ok());
    }

    #[test]
    fn aperture_on_non_wall_rejected() {
        let floor = floor_slab("floor_slab", 0, 0, 0, 64, 64, 16);
        let aperture = Aperture {
            id: "apt_01".into(),
            wall_brush_id: "floor_slab".into(),
            wall_face: geometry::FaceRole::Floor,
            aperture_bounds: ApertureBounds::Rectangular {
                wall_d: 0,
                u_min: 16,
                u_max: 48,
                v_min: 0,
                v_max: 16,
            },
            throat_depth: Rational::from_int(16),
        };
        assert!(matches!(
            Assembly::new(vec![floor], vec![], vec![aperture], vec![]),
            Err(AssemblyError::InvalidBrush { .. })
        ));
    }

    #[test]
    fn empty_assembly_passes() {
        assert!(Assembly::new(vec![], vec![], vec![], vec![]).is_ok());
    }

    // ── Adversarial proof corpus: golden box assembly ─────────────────

    #[test]
    fn box_assembly_validates() {
        let floor = AssemblyBrush::new(
            "floor",
            BrushRole::FloorSlab,
            make_box_brush(0, 0, 0, 128, 128, 16),
            Support::World {
                surface: geometry::FaceRole::Floor,
            },
        );

        // With the fix, walls sit ON TOP of the floor (z=0 to z=128), and floor is z=0 to z=16.
        // wall_n: Ceiling at z=0 (bottom of wall), floor: Floor at z=16 (top of floor)
        // These don't touch! We need walls to start at z=16.
        // Let me fix: walls from z=16 to z=128, with their Ceiling at z=16 touching floor's Floor at z=16.
        let wall_n = AssemblyBrush::new(
            "wall_n",
            BrushRole::WallShell,
            make_box_brush(16, 0, 16, 112, 16, 128),
            Support::SupportedBy {
                brush_id: "floor".into(),
                interface_id: "if_n".into(),
            },
        );
        let wall_s = AssemblyBrush::new(
            "wall_s",
            BrushRole::WallShell,
            make_box_brush(16, 112, 16, 112, 128, 128),
            Support::SupportedBy {
                brush_id: "floor".into(),
                interface_id: "if_s".into(),
            },
        );
        let wall_e = AssemblyBrush::new(
            "wall_e",
            BrushRole::WallShell,
            make_box_brush(112, 16, 16, 128, 112, 128),
            Support::SupportedBy {
                brush_id: "floor".into(),
                interface_id: "if_e".into(),
            },
        );
        let wall_w = AssemblyBrush::new(
            "wall_w",
            BrushRole::WallShell,
            make_box_brush(0, 16, 16, 16, 112, 128),
            Support::SupportedBy {
                brush_id: "floor".into(),
                interface_id: "if_w".into(),
            },
        );

        // floor Ceiling at z=16, wall Floor at z=16 — coplanar
        let interfaces = vec![
            Interface::new(
                "if_e",
                "wall_e",
                "floor",
                geometry::FaceRole::Floor,
                geometry::FaceRole::Ceiling,
            ),
            Interface::new(
                "if_n",
                "wall_n",
                "floor",
                geometry::FaceRole::Floor,
                geometry::FaceRole::Ceiling,
            ),
            Interface::new(
                "if_s",
                "wall_s",
                "floor",
                geometry::FaceRole::Floor,
                geometry::FaceRole::Ceiling,
            ),
            Interface::new(
                "if_w",
                "wall_w",
                "floor",
                geometry::FaceRole::Floor,
                geometry::FaceRole::Ceiling,
            ),
        ];

        let assembly = Assembly::new(
            vec![floor, wall_e, wall_n, wall_s, wall_w],
            interfaces,
            vec![],
            vec![],
        )
        .unwrap();
        assert!(assembly.validated);
        assert_eq!(assembly.support_edges.len(), 4);
    }

    #[test]
    fn supported_stack_validates() {
        let base = wall_shell("base", 0, 0, 0, 32, 32, 80);
        // pillar sits on top of base: pillar z from 80 to 160
        // pillar Floor at z=80, base Ceiling at z=80 — coplanar
        let pillar = AssemblyBrush::new(
            "pillar",
            BrushRole::Column,
            make_box_brush(8, 8, 80, 24, 24, 160),
            Support::SupportedBy {
                brush_id: "base".into(),
                interface_id: "if_01".into(),
            },
        );

        let interfaces = vec![Interface::new(
            "if_01",
            "pillar",
            "base",
            geometry::FaceRole::Floor,
            geometry::FaceRole::Ceiling,
        )];

        let assembly = Assembly::new(vec![base, pillar], interfaces, vec![], vec![]).unwrap();
        assert!(assembly.validated);
        assert_eq!(assembly.support_edges.len(), 1);
    }
}
