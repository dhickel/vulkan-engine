//! Complete transitive support DAG validation for every visible/colliding brush.
//!
//! Every brush must have a positive-area, orientation-valid support path to
//! world support. Exact AABB contacts are insufficient for diagonal brushes;
//! this module uses plane orientation + positive-area polygon intersection
//! for support contacts.
//!
//! # Contract
//!
//! - Every brush in the assembly must be transitively supported by world.
//! - Support contacts require positive-area intersection on a plane.
//! - Diagonal brushes need orientation-aware contact detection.
//! - Rejects unsupported pieces (any brush whose support chain doesn't reach world).
//! - Floor/ceiling slabs are world-supported or chain-supported.
//! - Crate-private; canonical ordering; no baseline changes.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::assembly::{AssemblyIR, BrushAssembly, BrushAssemblyRole, SupportTarget};
use super::error::{RichnessError, RichnessErrorCategory, RichnessErrorCode};
use super::geometry::{exact_face_contact, face_contact_bounds};
use super::ids::BrushAssemblyId;
use crate::enhanced_v3::geometry::{
    half_space_vertices, polygon_area_squared, CanonicalPlane, ConvexBrush, Rational,
};

// ── Support contact ───────────────────────────────────────────────────────

/// A positive-area orientation-valid support contact between two brushes.
///
/// The contact must have positive area on the supporting plane and the
/// supporting plane must be oriented correctly (supporting brush's top
/// face must be upward-facing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SupportContact {
    /// The brush being supported (child).
    pub child: BrushAssemblyId,
    /// The brush providing support (parent).
    pub parent: BrushAssemblyId,
    /// The contact area bounds (x0, y0, z0, x1, y1, z1).
    pub contact_bounds: (i128, i128, i128, i128, i128, i128),
    /// The plane on which contact occurs (the supporting plane of parent).
    pub contact_plane: (i128, i128, i128, i128), // (nx, ny, nz, d)
    /// Whether the contact is orientation-valid (parent's face points upward).
    pub orientation_valid: bool,
    /// Exact squared contact area. A positive value is the support proof.
    pub contact_area_squared: Rational,
}

// ── Support DAG ───────────────────────────────────────────────────────────

/// A directed acyclic graph of support relationships.
///
/// World is the root. Every brush is a node; edges point from supported
/// (child) to supporter (parent). The DAG is transitively closed and
/// validated for acyclic property.
#[derive(Debug, Clone)]
pub(crate) struct SupportDag {
    /// Adjacency: child -> set of parents (supporters).
    /// World-supported brushes have an entry with World as sole parent.
    pub edges: BTreeMap<BrushAssemblyId, BTreeSet<SupportTarget>>,
    /// Reverse adjacency: parent -> set of children.
    pub reverse_edges: BTreeMap<BrushAssemblyId, BTreeSet<BrushAssemblyId>>,
    /// All support contacts with computed area/orientation data.
    pub contacts: Vec<SupportContact>,
    /// Whether the DAG is complete (all brushes reached).
    pub complete: bool,
    /// Unsupported brush IDs (no geometrically valid path to world).
    pub unsupported: Vec<BrushAssemblyId>,
    /// Declared brush parents with no positive-area face contact.
    pub missing_contacts: Vec<(BrushAssemblyId, BrushAssemblyId)>,
}

impl SupportDag {
    /// Build a support DAG from an assembly IR.
    ///
    /// Every brush that has a support target is recorded. The DAG is then
    /// transitively validated.
    pub fn build(ir: &AssemblyIR) -> Self {
        let mut edges: BTreeMap<BrushAssemblyId, BTreeSet<SupportTarget>> = BTreeMap::new();
        let mut reverse_edges: BTreeMap<BrushAssemblyId, BTreeSet<BrushAssemblyId>> =
            BTreeMap::new();
        let mut contacts = Vec::new();
        let mut missing_contacts = Vec::new();

        for brush in ir.brushes.values() {
            let mut parents = BTreeSet::new();
            match brush.support {
                SupportTarget::World => {
                    parents.insert(SupportTarget::World);
                }
                SupportTarget::Brush(parent_id) => {
                    let contact = ir
                        .brushes
                        .get(&parent_id)
                        .and_then(|parent| compute_support_contact(brush, parent));
                    match contact {
                        Some(contact)
                            if contact.orientation_valid
                                && contact.contact_area_squared > Rational::ZERO =>
                        {
                            parents.insert(SupportTarget::Brush(parent_id));
                            reverse_edges.entry(parent_id).or_default().insert(brush.id);
                            contacts.push(contact);
                        }
                        Some(contact) => {
                            contacts.push(contact);
                        }
                        None => missing_contacts.push((brush.id, parent_id)),
                    }
                }
            }
            edges.insert(brush.id, parents);
        }

        let (complete, unsupported) = Self::validate_transitive_support(&edges);

        Self {
            edges,
            reverse_edges,
            contacts,
            complete,
            unsupported,
            missing_contacts,
        }
    }

    /// Validate that every brush has a transitive support path to World.
    ///
    /// Returns (all_reachable, unsupported_ids).
    fn validate_transitive_support(
        edges: &BTreeMap<BrushAssemblyId, BTreeSet<SupportTarget>>,
    ) -> (bool, Vec<BrushAssemblyId>) {
        let mut reachable: BTreeSet<BrushAssemblyId> = BTreeSet::new();
        let mut queue: VecDeque<BrushAssemblyId> = VecDeque::new();

        // Seed: brushes directly supported by World
        for (brush_id, parents) in edges {
            if parents.iter().any(|p| p.is_world()) {
                reachable.insert(*brush_id);
                queue.push_back(*brush_id);
            }
        }

        // BFS: brushes supported by already-reachable brushes become reachable
        // We need a reverse lookup: parent -> children
        let mut parent_to_children: BTreeMap<BrushAssemblyId, Vec<BrushAssemblyId>> =
            BTreeMap::new();
        for (child, parents) in edges {
            for parent in parents {
                if let SupportTarget::Brush(pid) = parent {
                    parent_to_children.entry(*pid).or_default().push(*child);
                }
            }
        }

        while let Some(parent) = queue.pop_front() {
            if let Some(children) = parent_to_children.get(&parent) {
                for child in children {
                    if !reachable.contains(child) {
                        reachable.insert(*child);
                        queue.push_back(*child);
                    }
                }
            }
        }

        let all_ids: BTreeSet<_> = edges.keys().copied().collect();
        let unsupported: Vec<BrushAssemblyId> = all_ids.difference(&reachable).copied().collect();

        (unsupported.is_empty(), unsupported)
    }

    /// Whether all brushes are transitively supported.
    pub fn all_supported(&self) -> bool {
        self.complete && self.unsupported.is_empty()
    }

    /// All brush IDs in the DAG.
    pub fn all_brush_ids(&self) -> BTreeSet<BrushAssemblyId> {
        self.edges.keys().copied().collect()
    }

    /// Get the direct parents of a brush.
    pub fn parents_of(&self, brush_id: BrushAssemblyId) -> BTreeSet<SupportTarget> {
        self.edges.get(&brush_id).cloned().unwrap_or_default()
    }

    /// Get the direct children of a brush (what it supports).
    pub fn children_of(&self, brush_id: BrushAssemblyId) -> BTreeSet<BrushAssemblyId> {
        self.reverse_edges
            .get(&brush_id)
            .cloned()
            .unwrap_or_default()
    }

    /// All support contacts that are orientation-invalid.
    pub fn invalid_contacts(&self) -> Vec<&SupportContact> {
        self.contacts
            .iter()
            .filter(|c| !c.orientation_valid)
            .collect()
    }

    /// Validate that the DAG is acyclic.
    ///
    /// Uses DFS with cycle detection. World is not a node in the DAG
    /// (it's the terminal).
    pub fn is_acyclic(&self) -> bool {
        let mut visited = BTreeSet::new();
        let mut in_stack = BTreeSet::new();

        for &node in self.edges.keys() {
            if !visited.contains(&node) {
                if has_cycle(node, &self.edges, &mut visited, &mut in_stack) {
                    return false;
                }
            }
        }
        true
    }
}

/// DFS cycle detection helper.
fn has_cycle(
    node: BrushAssemblyId,
    edges: &BTreeMap<BrushAssemblyId, BTreeSet<SupportTarget>>,
    visited: &mut BTreeSet<BrushAssemblyId>,
    in_stack: &mut BTreeSet<BrushAssemblyId>,
) -> bool {
    visited.insert(node);
    in_stack.insert(node);

    if let Some(parents) = edges.get(&node) {
        for parent in parents {
            if let SupportTarget::Brush(pid) = parent {
                if !visited.contains(pid) {
                    if has_cycle(*pid, edges, visited, in_stack) {
                        return true;
                    }
                } else if in_stack.contains(pid) {
                    return true;
                }
            }
        }
    }

    in_stack.remove(&node);
    false
}

// ── Support contact computation ────────────────────────────────────────────

/// Compute a support contact between a child brush and its supporting parent.
///
/// Uses plane orientation and positive-area polygon intersection.
/// Exact AABB contacts alone are insufficient for diagonal brushes.
pub(crate) fn compute_support_contact(
    child: &BrushAssembly,
    parent: &BrushAssembly,
) -> Option<SupportContact> {
    let contact = exact_face_contact(&child.brush, &parent.brush)?;
    let bounds = face_contact_bounds(&child.brush, &parent.brush)?;

    // Half-space normals point into a brush. A child bottom therefore has
    // +Z, while the coincident parent top has -Z. Split masonry can also
    // transfer load laterally into a same-wall segment whose base is lower;
    // the strict base ordering keeps those structural edges acyclic.
    let gravity = contact.plane.nx == 0 && contact.plane.ny == 0 && contact.plane.nz > 0;
    let child_min_z = child.brush.aabb().ok()?.0 .2;
    let parent_min_z = parent.brush.aabb().ok()?.0 .2;
    let masonry_transfer = contact.plane.nz == 0
        && child.role.is_wall()
        && parent.role.is_wall()
        && parent_min_z < child_min_z;
    let orientation_valid = gravity || masonry_transfer;
    let parent_plane = (
        -contact.plane.nx,
        -contact.plane.ny,
        -contact.plane.nz,
        -contact.plane.d,
    );

    Some(SupportContact {
        child: child.id,
        parent: parent.id,
        contact_bounds: bounds,
        contact_plane: parent_plane,
        orientation_valid,
        contact_area_squared: contact.area_squared,
    })
}

/// Compute the exact squared area of the intersection polygon on `plane`.
/// Handles cardinal and approved XY-45 diagonal contact planes.
pub(crate) fn plane_contact_polygon_area(
    a: &ConvexBrush,
    b: &ConvexBrush,
    plane: &CanonicalPlane,
) -> Option<Rational> {
    let mut faces = a.faces.clone();
    faces.extend(b.faces.iter().cloned());
    let vertices: Vec<_> = half_space_vertices(&faces)
        .ok()?
        .into_iter()
        .filter(|vertex| {
            plane
                .signed_distance_rational(vertex)
                .is_ok_and(|distance| distance == Rational::ZERO)
        })
        .collect();
    if vertices.len() < 3 {
        return None;
    }
    let area_squared = polygon_area_squared(&vertices, plane).ok()?;
    (area_squared > Rational::ZERO).then_some(area_squared)
}

/// Derive one exact gravity-support parent for every emitted brush.
///
/// Floor slabs are explicit world anchors. Every other brush selects the
/// positive-area, orientation-valid contact with the greatest exact area;
/// brush ID breaks ties canonically. Existing declarative records are rebuilt
/// from the resulting single source of truth.
pub(crate) fn derive_support_records(ir: &mut AssemblyIR) -> Result<(), RichnessError> {
    let brush_ids: Vec<_> = ir.brushes.keys().copied().collect();
    let mut targets = BTreeMap::new();

    for child_id in &brush_ids {
        let child = &ir.brushes[child_id];
        if child.role == BrushAssemblyRole::FloorSlab {
            targets.insert(*child_id, SupportTarget::World);
            continue;
        }

        let parent = brush_ids
            .iter()
            .filter(|parent_id| *parent_id != child_id)
            .filter_map(|parent_id| {
                let contact = compute_support_contact(child, &ir.brushes[parent_id])?;
                (contact.orientation_valid && contact.contact_area_squared > Rational::ZERO)
                    .then_some((
                        *parent_id,
                        contact.contact_area_squared,
                        contact.contact_plane.2 < 0,
                    ))
            })
            .max_by(
                |(left_id, left_area, left_gravity), (right_id, right_area, right_gravity)| {
                    left_gravity
                        .cmp(right_gravity)
                        .then_with(|| left_area.cmp(right_area))
                        .then_with(|| right_id.cmp(left_id))
                },
            )
            .map(|(parent_id, _, _)| parent_id)
            .ok_or_else(|| {
                support_error(
                    "support.derive",
                    format!(
                        "brush {} ({}) has no positive-area gravity support contact",
                        child_id.raw(),
                        child.role.tag()
                    ),
                )
            })?;
        targets.insert(*child_id, SupportTarget::Brush(parent));
    }

    ir.supports.clear();
    for child_id in brush_ids {
        let parent = targets[&child_id].clone();
        let Some(brush) = ir.brushes.get_mut(&child_id) else {
            return Err(support_error(
                "support.derive",
                format!(
                    "brush {} disappeared while deriving support",
                    child_id.raw()
                ),
            ));
        };
        brush.support = parent.clone();
        let id = ir.alloc_support_id();
        ir.insert_support(super::assembly::SupportRecord {
            id,
            child: child_id,
            parent,
        });
    }
    Ok(())
}

// ── Support validator ──────────────────────────────────────────────────────

/// Validate that every brush in the assembly has a complete transitive
/// support path to world.
pub(crate) fn validate_support_dag(ir: &AssemblyIR) -> Result<SupportDag, RichnessError> {
    let dag = SupportDag::build(ir);

    if !dag.missing_contacts.is_empty() {
        let contacts = dag
            .missing_contacts
            .iter()
            .map(|(child, parent)| format!("child={} parent={}", child.raw(), parent.raw()))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(support_error(
            "support.contact",
            format!("declared support edges lack positive-area contact: {contacts}"),
        ));
    }

    let invalid = dag.invalid_contacts();
    if !invalid.is_empty() {
        let contacts = invalid
            .iter()
            .map(|contact| {
                format!(
                    "child={} parent={} plane={:?}",
                    contact.child.raw(),
                    contact.parent.raw(),
                    contact.contact_plane
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(support_error(
            "support.orientation",
            format!("orientation-invalid support contacts: {contacts}"),
        ));
    }

    if !dag.complete {
        let unsupported = dag
            .unsupported
            .iter()
            .map(|id| id.raw().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(support_error(
            "support.dag",
            format!("unsupported brushes without a valid path to world: {unsupported}"),
        ));
    }

    if !dag.is_acyclic() {
        return Err(support_error(
            "support.cycle",
            "support DAG contains a cycle",
        ));
    }

    Ok(dag)
}

fn support_error(path: &str, context: impl Into<String>) -> RichnessError {
    RichnessError::new(
        RichnessErrorCode::SemanticInfeasible,
        0,
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        "?",
        path,
        RichnessErrorCategory::SemanticInfeasibility,
        context,
    )
}

/// Quick check: does a brush have a valid support path to world?
pub(crate) fn is_transitively_supported(brush_id: BrushAssemblyId, ir: &AssemblyIR) -> bool {
    let mut visited = BTreeSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(brush_id);

    while let Some(current) = queue.pop_front() {
        if !visited.insert(current) {
            continue;
        }
        if let Some(brush) = ir.brushes.get(&current) {
            if brush.support.is_world() {
                return true;
            }
            if let SupportTarget::Brush(parent) = brush.support {
                queue.push_back(parent);
            }
        }
    }
    false
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enhanced_v3::geometry::ConvexBrush;
    use crate::enhanced_v3::richness::assembly::{
        AssemblyIR, BrushAssembly, BrushAssemblyRole, BudgetDimension, CostSource,
        SemanticAttribution, SupportRecord, SupportTarget,
    };
    use crate::enhanced_v3::richness::ids::{
        ArchetypeIndex, ArchetypeRequestId, BeatId, ReservationId, ZoneId,
    };

    fn make_attr() -> SemanticAttribution {
        SemanticAttribution::from_reservation(
            ReservationId::new(0),
            Some(ArchetypeRequestId::new(0)),
            Some(ArchetypeIndex::new(0)),
            Some(BeatId::new(0)),
            Some(ZoneId::new(0)),
        )
    }

    fn make_cost() -> CostSource {
        CostSource {
            dimension: BudgetDimension::SourceFaces,
            face_count: 6,
        }
    }

    #[test]
    fn world_supported_brush_is_transitively_supported() {
        let mut ir = AssemblyIR::new();
        let attr = make_attr();
        let cost = make_cost();

        let floor_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: floor_id,
            brush: ConvexBrush::make_box((0, 256), (0, 256), (0, 16)).unwrap(),
            role: BrushAssemblyRole::FloorSlab,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let support_id = ir.alloc_support_id();
        ir.insert_support(SupportRecord {
            id: support_id,
            child: floor_id,
            parent: SupportTarget::World,
        });

        assert!(is_transitively_supported(floor_id, &ir));
    }

    #[test]
    fn chain_supported_brush_is_transitively_supported() {
        let mut ir = AssemblyIR::new();
        let attr = make_attr();
        let cost = make_cost();

        let floor_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: floor_id,
            brush: ConvexBrush::make_box((0, 256), (0, 256), (0, 16)).unwrap(),
            role: BrushAssemblyRole::FloorSlab,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let wall_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: wall_id,
            brush: ConvexBrush::make_box((0, 256), (240, 256), (16, 160)).unwrap(),
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::Brush(floor_id),
        });

        let ceil_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: ceil_id,
            brush: ConvexBrush::make_box((0, 256), (0, 256), (160, 176)).unwrap(),
            role: BrushAssemblyRole::CeilingSlab,
            owner: attr,
            cost,
            support: SupportTarget::Brush(wall_id),
        });

        assert!(is_transitively_supported(wall_id, &ir));
        assert!(is_transitively_supported(ceil_id, &ir));
    }

    #[test]
    fn support_dag_is_acyclic_for_chain() {
        let mut ir = AssemblyIR::new();
        let attr = make_attr();
        let cost = make_cost();

        let floor_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: floor_id,
            brush: ConvexBrush::make_box((0, 256), (0, 256), (0, 16)).unwrap(),
            role: BrushAssemblyRole::FloorSlab,
            owner: attr.clone(),
            cost,
            support: SupportTarget::World,
        });

        let wall_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: wall_id,
            brush: ConvexBrush::make_box((0, 256), (240, 256), (16, 160)).unwrap(),
            role: BrushAssemblyRole::NorthWall,
            owner: attr,
            cost,
            support: SupportTarget::Brush(floor_id),
        });

        let dag = SupportDag::build(&ir);
        assert!(dag.is_acyclic());
        assert!(dag.all_supported());
    }

    #[test]
    fn validate_support_dag_rejects_unsupported() {
        let mut ir = AssemblyIR::new();
        let attr = make_attr();
        let cost = make_cost();

        // Brush supported by a non-existent brush
        let orphan_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: orphan_id,
            brush: ConvexBrush::make_box((0, 64), (0, 64), (16, 160)).unwrap(),
            role: BrushAssemblyRole::NorthWall,
            owner: attr.clone(),
            cost,
            support: SupportTarget::Brush(BrushAssemblyId::new(99)), // non-existent
        });

        // Also record a support edge in the supports map
        let sup_id = ir.alloc_support_id();
        ir.insert_support(SupportRecord {
            id: sup_id,
            child: orphan_id,
            parent: SupportTarget::Brush(BrushAssemblyId::new(99)),
        });

        let result = validate_support_dag(&ir);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .context
            .contains("lack positive-area contact"));
    }

    #[test]
    fn support_contact_computed_for_valid_pair() {
        let floor = BrushAssembly {
            id: BrushAssemblyId::new(0),
            brush: ConvexBrush::make_box((0, 256), (0, 256), (0, 16)).unwrap(),
            role: BrushAssemblyRole::FloorSlab,
            owner: make_attr(),
            cost: make_cost(),
            support: SupportTarget::World,
        };

        let wall = BrushAssembly {
            id: BrushAssemblyId::new(1),
            brush: ConvexBrush::make_box((0, 256), (240, 256), (16, 160)).unwrap(),
            role: BrushAssemblyRole::NorthWall,
            owner: make_attr(),
            cost: make_cost(),
            support: SupportTarget::Brush(BrushAssemblyId::new(0)),
        };

        let contact = compute_support_contact(&wall, &floor);
        assert!(contact.is_some());
        let contact = contact.unwrap();
        assert!(contact.contact_area_squared > Rational::ZERO);
        assert!(contact.orientation_valid);
    }

    #[test]
    fn diagonal_footprint_has_exact_positive_support_polygon() {
        let floor = ConvexBrush::make_box((0, 128), (0, 128), (0, 16)).unwrap();
        let diagonal =
            crate::enhanced_v3::geometry::make_diagonal_wall((0, 128), (0, 128), 16, 160, 1, 1, 64)
                .unwrap();
        let floor = BrushAssembly {
            id: BrushAssemblyId::new(0),
            brush: floor,
            role: BrushAssemblyRole::FloorSlab,
            owner: make_attr(),
            cost: make_cost(),
            support: SupportTarget::World,
        };
        let wall = BrushAssembly {
            id: BrushAssemblyId::new(1),
            brush: diagonal,
            role: BrushAssemblyRole::DiagNEWall,
            owner: make_attr(),
            cost: make_cost(),
            support: SupportTarget::Brush(floor.id),
        };
        let contact = compute_support_contact(&wall, &floor).unwrap();
        assert!(contact.orientation_valid);
        assert!(contact.contact_area_squared > Rational::ZERO);
        let plane = CanonicalPlane::new(0, 0, 1, 16).unwrap();
        assert!(plane_contact_polygon_area(&wall.brush, &floor.brush, &plane).is_some());
    }

    #[test]
    fn unsupported_brush_detected() {
        let mut ir = AssemblyIR::new();
        let attr = make_attr();
        let cost = make_cost();

        // Brush supported by a brush that itself is unsupported (points to non-existent)
        let a_id = ir.alloc_brush_id();
        ir.insert_brush(BrushAssembly {
            id: a_id,
            brush: ConvexBrush::make_box((0, 64), (0, 64), (16, 160)).unwrap(),
            role: BrushAssemblyRole::NorthWall,
            owner: attr,
            cost,
            support: SupportTarget::Brush(BrushAssemblyId::new(99)),
        });

        let dag = SupportDag::build(&ir);
        assert!(!dag.all_supported());
        assert_eq!(dag.unsupported, vec![a_id]);
    }

    #[test]
    fn support_dag_cycle_detected() {
        let a_id = BrushAssemblyId::new(0);
        let b_id = BrushAssemblyId::new(1);
        let dag = SupportDag {
            edges: [
                (a_id, [SupportTarget::Brush(b_id)].into_iter().collect()),
                (b_id, [SupportTarget::Brush(a_id)].into_iter().collect()),
            ]
            .into_iter()
            .collect(),
            reverse_edges: BTreeMap::new(),
            contacts: Vec::new(),
            complete: false,
            unsupported: vec![a_id, b_id],
            missing_contacts: Vec::new(),
        };
        assert!(!dag.is_acyclic());
    }
}
